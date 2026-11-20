"""Radiosonde (ADPUPA) profile observation reader for the normalized,
cycle-partitioned PREPBUFR parquet archive -- temperature, horizontal wind,
and specific humidity. `vertical_interpolation` ('exact' default, or
'nearest'/'log-linear') controls whether a level is kept only if it's an
exact mandatory-level match (the original design, no vertical
interpolation) or snapped to its nearest model level regardless of
distance (optionally vertically corrected to its own real pressure by the
forward operator) -- see docs/integrate_multiple_parquetobs.md for the
design and why, and `psobs_parquet.py` for the sibling surface-pressure
reader this shares I/O helpers with (`prepbufr_parquet_common.py`).

Ported from a sibling repo's `long-window-4dvar` (JAX/NeuralGCM) reference
implementation, via `/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/
lw4dvar-torch-dev/add-amv-gpsro/lw4dvar-torch` (branch `add-raobs-parquet`),
restricted to the ADPUPA class, with specific humidity added -- the
reference never implemented moisture for radiosondes at all.

Temperature is sensible temperature: the archive's current TOB is virtual
temperature wherever moisture was reported (VIRTMP), so `t` is taken from the
first non-VIRTMP event (`common.sensible_temperature`).
"""

import numpy as np

import prepbufr_parquet_common as common


# Bit masks for the returned per-(variable, level, station, slot) QC flag.
# Availability is recorded separately (see `mask` vs a hypothetical
# "available" array) -- zero means an available observation passed every
# check.
QC_PHYSICAL_HEIGHT = 1
QC_NON_MONOTONIC_HEIGHT = 2
QC_BELOW_STATION = 4
QC_PREPBUFR_QUALITY = 8
QC_PHYSICAL_BOUNDS = 16
QC_FLAG_MASKS = np.array(
    [QC_PHYSICAL_HEIGHT, QC_NON_MONOTONIC_HEIGHT, QC_BELOW_STATION, QC_PREPBUFR_QUALITY, QC_PHYSICAL_BOUNDS],
    dtype=np.int16,
)
QC_FLAG_MEANINGS = (
    'physical_height non_monotonic_height below_station prepbufr_quality physical_bounds'
)

DEFAULT_RAOBS_CLASSES = ('ADPUPA',)

# Physical sanity bounds on the VALUE itself (model units: K, m/s, kg/kg),
# same numbers as UFO's GFS sonde QC config (PR #29's review comment --
# test/testinput/instrumentTests/Sonde/sonde_gfs_qc.yaml): T 195-327 K, q
# 1e-8-0.035 kg/kg, |u|,|v| <= 130 m/s. A read-time check, like the
# existing height-based ones -- independent of the background/model
# state, so it belongs here rather than in the forward operator.
_RAOBS_PHYSICAL_BOUNDS = {
    't': (195., 327.),
    'q': (1.e-8, 0.035),
    'u': (-130., 130.),
    'v': (-130., 130.),
}

# model-base-name -> (value column, quality column) in the archive's
# `levels` table. 'height' is always read (for QC only -- station-elevation
# and monotonicity checks); it is never itself assimilated here.
_VARIABLE_COLUMNS = {
    't': ('temperature_c', 'temperature_quality'),
    'u': ('u_wind_ms', 'wind_quality'),
    'v': ('v_wind_ms', 'wind_quality'),
    'q': ('specific_humidity_mg_kg', 'humidity_quality'),
}
_HEIGHT_COLUMNS = ('height_m', 'height_quality')

# model-base-name -> source observation-error column in the archive's
# `levels` table (the PREPBUFR TOE/WOE mnemonics, the same embedded-BUFR
# kind of value `psobs_parquet.py`'s `pressure_error_mb`/`use_source_error`
# already reads for ps). NOT GSI's actual DA error: GSI's read_prepbufr.F90
# reads this identical embedded value (ufbint/oestr) but then
# unconditionally overwrites it from the prepobs_errtable.global fix table
# (etabl) for every conventional type before assimilating -- confirmed by
# tracing read_prepbufr.F90 line-by-line, not just comparing magnitudes
# (0.8-1.5 K for raob T, 1.4-3.2 m/s for raob wind land in a plausible
# range, but that is not the same as being the value GSI's Jo actually
# weights by). See docs/integrate_multiple_parquetobs.md. 'u' and 'v'
# share one column -- GSI's wind error is one vector quantity, not separate
# per component. 'q' is handled separately (see `_saturation_specific_humidity`
# below): its source column is RH-based (percent/10, confirmed against
# convert_prepbufr_to_parquet.py's own field metadata), not kg/kg, and needs
# a qsat conversion -- using the archive's own static `background_*` fields
# (the historical GFS background at archive time), not this run's own
# model state; see docs/integrate_multiple_parquetobs.md for why that's a
# known, deliberate simplification, not a GSI match.
_VARIABLE_ERROR_COLUMNS = {
    't': 'temperature_error_c',
    'u': 'wind_error_ms',
    'v': 'wind_error_ms',
}

# GSI's own x1.2 error inflation (read_prepbufr.F90), applied to the
# source error itself (not just a gross-check comparison like the
# ermin/ermax clamp in long_window_4dvar_utils.py) -- confirmed by
# reading the source: `cdata_all(1,iout)=toe` writes out the ALREADY-
# inflated value, which setupt/setupw/setupq.f90 then use directly for
# both the Jo weight and (via errormod) the gross check. Two independent,
# multiplicative triggers:
# - quality mark 3 or 7 (suspect but not rejected): x1.2, t/u/v/q alike
#   (`if (inflate_error) toe/woe/qoe=.../.../...*r1_2`).
# - above a variable-specific pressure: x1.2 more, t only above 100 hPa,
#   u/v only above 50 hPa (`if(ppb < r100)toe=toe*r1_2`,
#   `if(obsdat(1,k) < r50)woe=woe*r1_2`) -- q has no pressure-based
#   inflation in GSI.
_RAOBS_INFLATE_QM = (3, 7)
_RAOBS_INFLATE_PRESSURE_HPA = {'t': 100., 'u': 50., 'v': 50.}


def _gsi_inflate_error(raw, quality, pressure=None, pressure_threshold=None):
    factor = np.where(np.isin(quality, _RAOBS_INFLATE_QM), 1.2, 1.0)
    if pressure_threshold is not None:
        factor = factor * np.where(pressure < pressure_threshold, 1.2, 1.0)
    return raw * factor

# Saturation vapor pressure (Pa) from temperature (K): the ECMWF/IFS
# Magnus-Tetens form, identical to backends/ace2/ace2_ic.py's
# `_saturation_vapor_pressure` -- duplicated rather than imported, since
# src/ must stay usable without any backend's dependencies (see
# _ensure_backend_on_path in long_window_4dvar_utils.py).
_SVP_A1, _SVP_A2, _SVP_A3, _SVP_A4 = 611.21, 273.16, 17.502, 32.19
_EPS_RD_RV = 0.621981  # Rd/Rv


def _saturation_specific_humidity(temperature_k, pressure_mb):
    """Saturation specific humidity (kg/kg) from temperature (K) and total
    pressure (hPa) -- same Magnus-Tetens + eps formula as
    `backends/ace2/ace2_ic.py`'s `_saturation_vapor_pressure`/
    `_specific_humidity_from_dewpoint`, with temperature standing in for
    its own dewpoint (RH = 100%)."""
    e_sat_pa = _SVP_A1 * np.exp(_SVP_A3 * (temperature_k - _SVP_A2) / (temperature_k - _SVP_A4))
    pressure_pa = pressure_mb * 100.
    return _EPS_RD_RV * e_sat_pa / (pressure_pa - (1. - _EPS_RD_RV) * e_sat_pa)


def _interp_rh_percent_table(table, pressure_mb):
    """ln(p) interpolation of a {hPa: percent} table to arbitrary pressures
    (vectorized over `pressure_mb`) -- same formula as long_window_4dvar_
    utils.py's `_interp_level_table`, duplicated rather than imported
    since this module must stay usable with no solver-layer dependency."""
    tp = np.array(sorted(float(k) for k in table))
    tv = np.array([float(table[k]) for k in sorted(table, key=float)])
    return np.interp(np.log(np.asarray(pressure_mb, dtype=float)), np.log(tp), tv)

# Unit conversions from the archive's native column units to the model's
# decode_state units (Kelvin for 't', kg/kg for 'q' -- 'u'/'v' are already
# m/s, matching the model, so no entry needed for them).
_VARIABLE_UNIT_CONVERT = {
    't': lambda value: value + 273.15,
    'q': lambda value: value / 1.e6,
}


def _quality_passes(value, maximum):
    return common.quality_passes(value, maximum)


def _profile_records(
    directory,
    verification_times,
    levels_hpa,
    variables,
    time_tolerance_hours,
    prepbufr_classes,
    quality_mark_max,
    pressure_match_tolerance_hpa,
    geopotential_height_min_m,
    geopotential_height_max_m,
    below_station_tolerance_m,
    use_source_error=False,
    use_drift_time=False,
    vertical_interpolation='exact',
    use_drift_position=False,
    q_error_rh_percent=None,
):
    """Return {(slot_index, station_key): record} before fixed-shape padding.

    `q_error_rh_percent`: {hPa: percent} table (sorted by pressure, any
    order) -- the DEFAULT q observation error, independent of
    `use_source_error`: this level's own RH-error-percent (ln(p)-
    interpolated to its real reported pressure) times qsat computed at
    the archive's own static `background_temperature_c`/
    `background_pressure_mb` (same qsat machinery `use_source_error`'s q
    path already uses -- see `_saturation_specific_humidity`). Recorded as
    `record['default_error']` whenever 'q' is requested, regardless of
    `use_source_error` -- GSI's own real mechanism is `RH_fraction *
    qsat(background)` (confirmed directly from GSI's source and
    `prepobs_errtable.global`'s RH column, PR #29's review discussion),
    not a flat per-level kg/kg guess. None (q not requested, or no table
    given) skips this entirely. See docs/integrate_multiple_parquetobs.md.

    `use_drift_position`: record EACH LEVEL's own balloon-drift-corrected
    horizontal position (`drift_longitude`/`drift_latitude`, archive
    columns -- absolute degrees, the same convention as `latitude`/
    `longitude`; confirmed against real data: a 218-level sounding drifts
    from its launch position 100.27E/5.30N to 99.94E/5.45N by 6 hPa)
    instead of every level sharing the report's single launch position.
    Falls back to the launch position wherever drift_longitude/
    drift_latitude is missing/non-finite for a given level. Independent of
    `use_drift_time` (that corrects WHICH slot a level is compared
    against; this corrects WHERE, horizontally) -- either, both, or
    neither may be set. See docs/integrate_multiple_parquetobs.md.

    `vertical_interpolation`: 'exact' (default) keeps today's behavior --
    a level is kept only within `pressure_match_tolerance_hpa` of one of
    the variable's own model levels, everything else discarded. 'nearest'/
    'log-linear' drop that discard entirely: every level is kept, assigned
    to its single nearest model level regardless of distance (so a model
    with few/coarse levels still gets an observation at each one), and its
    own real `pressure_mb` is additionally recorded (`record['pressure']`)
    for the forward operator to use -- ignored by 'nearest', consumed by
    'log-linear' for the actual vertical correction (long_window_4dvar_
    utils.py's `_compute_raobs_observation_diagnostics_at_time`). See
    docs/integrate_multiple_parquetobs.md.

    `use_drift_time`: assign EACH LEVEL to its own nearest slot using that
    level's own balloon-drift-corrected time (`drift_time_hours`, an
    archive column -- confirmed against real data, PR #29's drift-time
    discussion: near 0 at the surface, growing smoothly to +1.6h by 5 hPa
    for a real deep sounding) instead of the report's single launch time
    for every level. Falls back to the report's own time wherever
    `drift_time_hours` is missing/non-finite for a given level. This can
    split one profile's levels across multiple slots -- no special
    handling needed: the (slot_index, station_key) keying below already
    treats each slot independently.

    `levels_hpa`: {model_base_name: 1D array of exact pressures (hPa) to
    match} -- a level row is kept only when its `pressure_mb` is within
    `pressure_match_tolerance_hpa` of one of THAT variable's own levels (so
    `u`/`v` can share one level set while `q` uses a shorter one, matching
    each variable's actual model pressure-level axis).

    Duplicate (variable, level_index) rows for the same station/slot are
    ranked by actual QC eligibility, not just the raw PREPBUFR quality
    mark: a row whose own quality fails `quality_mark_max`, OR whose
    (quality-passing, non-category-4) height fails the physical-height-
    bounds/below-station checks, ranks behind any row that passes both --
    otherwise picking the single numerically-best-quality-mark duplicate
    (the only criterion previously used) could retain a row with a
    garbage height over an available row that would have passed every
    check, which then wrongly fails height-based QC downstream. Within
    the same pass/fail bucket, lower (better) quality mark still wins, so
    an all-bad set of duplicates still keeps its best candidate
    (`load_raobs_parquet` still flags it bad via its own QC pass over the
    final selection -- this only changes WHICH duplicate gets selected).
    """
    variables = tuple(variables)
    unsupported = set(variables) - set(_VARIABLE_COLUMNS)
    if unsupported:
        raise ValueError('Unsupported raobs variables: '+str(sorted(unsupported)))
    prepbufr_classes = tuple(str(c).upper() for c in prepbufr_classes)

    cycles = common.cycle_directories(directory, verification_times, time_tolerance_hours)
    verification = np.asarray([
        np.datetime64(common.to_datetime(value), 's') for value in verification_times
    ])
    tolerance_seconds = float(time_tolerance_hours) * 3600.

    report_columns = [
        'report_id', 'observation_time', 'station_id', 'latitude',
        'longitude', 'elevation_m', 'report_type',
    ]
    level_columns = ['report_id', 'level_index', 'category', 'pressure_mb'] + list(_HEIGHT_COLUMNS)
    for variable in variables:
        level_columns.extend(_VARIABLE_COLUMNS[variable])
    if use_source_error:
        for variable in variables:
            column = _VARIABLE_ERROR_COLUMNS.get(variable)
            if column is not None:
                level_columns.append(column)
        if 'q' in variables:
            level_columns.append('humidity_error_tenths_percent')
    # background_temperature_c/background_pressure_mb feed the qsat
    # conversion for q's error -- needed whenever q is requested, not just
    # under use_source_error, since the DEFAULT q error is now also
    # RH_percent * qsat (see q_error_rh_percent above).
    if 'q' in variables and (use_source_error or q_error_rh_percent):
        level_columns.extend(('background_temperature_c', 'background_pressure_mb'))
    if use_drift_time:
        level_columns.append('drift_time_hours')
    if use_drift_position:
        level_columns.extend(('drift_longitude', 'drift_latitude'))
    level_columns = list(dict.fromkeys(level_columns))

    records = {}
    for cycle in cycles:
        for prepbufr_class in prepbufr_classes:
            reports = common.read_partition(
                common.class_file(cycle, 'reports', prepbufr_class), report_columns
            )
            levels = common.read_partition(
                common.class_file(cycle, 'levels', prepbufr_class), level_columns
            )
            report_ids = common.column(reports, 'report_id', np.int64)
            report_rows = {
                int(report_id): row for row, report_id in enumerate(report_ids)
            }
            report_times = common.timestamps(reports)
            station_ids = reports['station_id'].to_pylist()
            latitude = common.column(reports, 'latitude')
            longitude = common.column(reports, 'longitude')
            elevation = common.column(reports, 'elevation_m')

            level_report_ids = common.column(levels, 'report_id', np.int64)
            category = common.column(levels, 'category', np.int16)
            pressure = common.column(levels, 'pressure_mb')
            height = common.column(levels, 'height_m')
            height_quality = common.column(levels, 'height_quality')
            drift_time_hours = common.column(levels, 'drift_time_hours') if use_drift_time else None
            drift_longitude = common.column(levels, 'drift_longitude') if use_drift_position else None
            drift_latitude = common.column(levels, 'drift_latitude') if use_drift_position else None
            level_values = {
                v: common.column(levels, _VARIABLE_COLUMNS[v][0]) for v in variables
            }
            level_quality = {
                v: common.column(levels, _VARIABLE_COLUMNS[v][1]) for v in variables
            }
            level_source_error = {
                v: _gsi_inflate_error(
                    common.column(levels, _VARIABLE_ERROR_COLUMNS[v]), level_quality[v],
                    pressure, _RAOBS_INFLATE_PRESSURE_HPA.get(v))
                for v in variables if use_source_error and v in _VARIABLE_ERROR_COLUMNS
            }
            level_source_qsat = {}
            level_default_error = {}
            if 'q' in variables and (use_source_error or q_error_rh_percent):
                # qsat at the archive's own STATIC background T/p (not
                # this run's live model state -- see _VARIABLE_ERROR_
                # COLUMNS's docstring above) -- needed by BOTH the
                # archive-sourced path below and the default table path,
                # so computed once, unconditionally, whenever either is
                # in play.
                background_t_k = common.column(levels, 'background_temperature_c') + 273.15
                background_p_mb = common.column(levels, 'background_pressure_mb')
                qsat = _saturation_specific_humidity(background_t_k, background_p_mb)
                if use_source_error:
                    # RH-based source error (percent/10, per
                    # convert_prepbufr_to_parquet.py's own field metadata)
                    # -> absolute kg/kg via the SAME qsat. Kept separately
                    # (per level) so the gross check can apply GSI's own
                    # RH-percent ermin/ermax bounds later (long_window_
                    # 4dvar_utils.py) -- NOT inflated/clamped itself (a
                    # physical qsat value, not an error); only the error
                    # value below is. The ermin/ermax clamp is gross-
                    # check-only (see GSI's setupq.f90 -- its clamped
                    # `obserror` never feeds the Jo weight, confirmed by
                    # reading the source), unlike this x1.2 inflation,
                    # which GSI applies directly to the value it writes
                    # out and weights with.
                    rh_error_percent = common.column(levels, 'humidity_error_tenths_percent') * 10.
                    raw_q_error = (rh_error_percent / 100.) * qsat
                    level_source_error['q'] = _gsi_inflate_error(raw_q_error, level_quality['q'])
                    level_source_qsat['q'] = qsat
                if q_error_rh_percent:
                    # The DEFAULT q error (used wherever use_source_error
                    # is off, or its own archive value is missing): GSI's
                    # real mechanism, RH_fraction * qsat(background), NOT
                    # inflated -- inflation is specific to the genuine
                    # archive-sourced value above, not a configured table.
                    # Interpolated to each level's own REAL pressure
                    # (`pressure`, not the matched model level), same
                    # ln(p) convention as error_std's own per-level table.
                    default_rh_percent = _interp_rh_percent_table(q_error_rh_percent, pressure)
                    level_default_error['q'] = (default_rh_percent / 100.) * qsat
            if 't' in level_values:
                # current TOB is virtual temperature where moisture was
                # reported (VIRTMP); the model's t is sensible temperature
                level_values['t'] = common.sensible_temperature(
                    cycle, prepbufr_class, level_report_ids,
                    common.column(levels, 'level_index', np.int64), level_values['t'],
                )

            for level_row, report_id in enumerate(level_report_ids):
                report_row = report_rows.get(int(report_id))
                if report_row is None:
                    continue
                lat = latitude[report_row]
                lon = longitude[report_row]
                pressure_value = pressure[level_row]
                if not (
                    np.isfinite(lat) and np.isfinite(lon)
                    and -90. <= lat <= 90. and np.isfinite(pressure_value)
                ):
                    continue
                report_time = report_times[report_row]
                if np.isnat(report_time):
                    continue
                level_lat, level_lon = lat, lon
                if drift_longitude is not None and drift_latitude is not None:
                    drift_lat = drift_latitude[level_row]
                    drift_lon = drift_longitude[level_row]
                    if np.isfinite(drift_lat) and np.isfinite(drift_lon):
                        level_lat, level_lon = drift_lat, drift_lon
                level_time = report_time
                if drift_time_hours is not None:
                    drift_h = drift_time_hours[level_row]
                    if np.isfinite(drift_h):
                        level_time = report_time + np.timedelta64(int(round(drift_h * 3600.)), 's')
                delta_seconds = np.abs(
                    (verification - level_time).astype('timedelta64[s]')
                    .astype(np.int64)
                )
                slot_index = int(np.argmin(delta_seconds))
                if delta_seconds[slot_index] > tolerance_seconds:
                    continue

                station_key = common.station_key(
                    station_ids[report_row], lat, lon,
                    str(cycle)+':'+str(report_id),
                )
                key = (slot_index, station_key)
                elevation_m = elevation[report_row]
                record = records.setdefault(key, {
                    'lat': lat, 'lon': lon, 'elev': elevation_m,
                    'values': {}, 'quality': {}, 'height': {}, 'height_quality': {}, 'priority': {},
                    'source_error': {}, 'source_qsat': {}, 'default_error': {},
                    'pressure': {}, 'level_position': {},
                })

                for variable in variables:
                    variable_levels = levels_hpa[variable]
                    differences = np.abs(variable_levels - pressure_value)
                    level_index = int(np.argmin(differences))
                    if vertical_interpolation == 'exact':
                        if differences[level_index] > pressure_match_tolerance_hpa:
                            continue
                    # 'nearest'/'log-linear': no distance cutoff -- always
                    # keep the nearest level, so a coarse model's levels
                    # still get an observation instead of possibly none.
                    value = level_values[variable][level_row]
                    if not np.isfinite(value):
                        continue
                    convert = _VARIABLE_UNIT_CONVERT.get(variable)
                    if convert is not None:
                        value = convert(value)
                    quality = level_quality[variable][level_row]
                    height_m = height[level_row]
                    height_quality_value = height_quality[level_row]
                    # Only a height that would actually be attached (see
                    # fix below: category 4 / failing its own quality mark
                    # never is) can count against this row -- an absent or
                    # ineligible height can't be judged either way.
                    height_eligible = (
                        np.isfinite(height_m) and category[level_row] != 4
                        and _quality_passes(height_quality_value, quality_mark_max)
                    )
                    height_bad = height_eligible and (
                        not (float(geopotential_height_min_m) <= height_m <= float(geopotential_height_max_m))
                        or (
                            np.isfinite(elevation_m)
                            and height_m < elevation_m - float(below_station_tolerance_m)
                        )
                    )
                    quality_bad = not _quality_passes(quality, quality_mark_max)
                    # (worse-bucket-first, then numeric quality mark -- a
                    # non-finite quality sorts last within its bucket, same
                    # as the pre-existing tie-break). 'nearest'/'log-linear'
                    # add distance-to-level as a third tie-break: with the
                    # exact-match discard gone, several genuinely different
                    # raw levels can now compete for one bucket, and among
                    # QC-equal candidates the physically closest one should
                    # win (irrelevant for 'exact', where the tight tolerance
                    # already made this moot).
                    priority = (
                        quality_bad or height_bad,
                        quality if np.isfinite(quality) else np.inf,
                        differences[level_index] if vertical_interpolation != 'exact' else 0.,
                    )

                    existing_priority = record['priority'].get((variable, level_index))
                    if existing_priority is None or priority < existing_priority:
                        record['values'][(variable, level_index)] = value
                        record['quality'][(variable, level_index)] = quality
                        record['priority'][(variable, level_index)] = priority
                        if vertical_interpolation != 'exact':
                            record['pressure'][(variable, level_index)] = pressure_value
                        if use_drift_position:
                            record['level_position'][(variable, level_index)] = (level_lat, level_lon)
                        # Clear any height left over from the row this one
                        # is replacing before conditionally attaching the
                        # new row's own -- otherwise a stale height from
                        # the discarded row survives whenever the
                        # replacement itself has no eligible height,
                        # corrupting the physical-height/monotonicity QC
                        # checks downstream.
                        record['height'].pop((variable, level_index), None)
                        record['height_quality'].pop((variable, level_index), None)
                        if height_eligible:
                            record['height'][(variable, level_index)] = height_m
                            record['height_quality'][(variable, level_index)] = height_quality_value
                        if variable in level_source_error:
                            record['source_error'][(variable, level_index)] = \
                                level_source_error[variable][level_row]
                        else:
                            record['source_error'].pop((variable, level_index), None)
                        if variable in level_source_qsat:
                            record['source_qsat'][(variable, level_index)] = \
                                level_source_qsat[variable][level_row]
                        else:
                            record['source_qsat'].pop((variable, level_index), None)
                        if variable in level_default_error:
                            record['default_error'][(variable, level_index)] = \
                                level_default_error[variable][level_row]
                        else:
                            record['default_error'].pop((variable, level_index), None)
    return records


def load_raobs_parquet(
    directory,
    verification_times,
    time_tolerance_hours,
    levels_hpa,
    variables,
    prepbufr_classes=DEFAULT_RAOBS_CLASSES,
    quality_mark_max=3,
    geopotential_height_min_m=-1000.,
    geopotential_height_max_m=60000.,
    below_station_tolerance_m=100.,
    pressure_match_tolerance_hpa=0.05,
    humidity_min_pressure_hpa=None,
    station_capacity=None,
    use_source_error=False,
    use_drift_time=False,
    vertical_interpolation='exact',
    use_drift_position=False,
    q_error_rh_percent=None,
    logger=None,
):
    """Read mandatory-level radiosonde profiles into fixed per-slot arrays.

    Returns a dict: 'lat'/'lon'/'elev' (per-slot 1D float32 arrays, one
    station per entry) and, for each requested `variable` (a model base
    name in 't'/'u'/'v'/'q'), `values_<v>`/`mask_<v>` (per-slot 2D float32/
    bool arrays, shape `(len(levels_hpa[v]), n_stations_slot)`). `mask_<v>`
    is False for (level, station) pairs with no accepted observation, OR
    that failed QC -- callers must not read `values_<v>` where `mask_<v>`
    is False (it is left at 0, not NaN).

    `vertical_interpolation` ('exact' default, or 'nearest'/'log-linear'):
    see `_profile_records`'s docstring -- 'nearest'/'log-linear' drop the
    `pressure_match_tolerance_hpa` discard (every level kept, snapped to
    its nearest model level regardless of distance) and additionally
    return `pressure_<v>` (same shape as `values_<v>`, NaN where `mask_<v>`
    is False): the level's own real reported pressure, for the forward
    operator's vertical correction (ignored by 'nearest').

    When `use_source_error` is True and
    `v` is requested, also returns `source_error_<v>` (same shape, float32,
    NaN where no valid source error is available): for 't'/'u'/'v', the
    archive's own per-level PREPBUFR TOE/WOE value (the real GSI error-
    table number for that observation); for 'q', that same archive's RH-
    based QOE value converted to kg/kg via qsat at the archive's own
    static background T/p (see `_saturation_specific_humidity` and
    `_VARIABLE_ERROR_COLUMNS`'s docstring above -- NOT this run's live
    model state). Either way, also inflated x1.2 per GSI's own rule(s)
    (`_gsi_inflate_error`: quality mark 3/7 for all three; additionally
    above 100/50 hPa for t/u,v) BEFORE the NaN/missing check, matching
    what GSI itself writes out and weights with -- not the bare table
    value. NaN wherever missing/non-positive, so the caller
    (`get_raobs`) can fall back to its configured flat/table error there.
    For 'q' specifically, also returns `source_qsat_q` (same shape, NaN
    wherever `source_error_q` is NaN too): the qsat used for that
    conversion, kept separately so the caller can derive GSI's own
    RH-percent ermin/ermax gross-check bound in kg/kg WITHOUT clamping
    `source_error_q` itself (which stays the real per-level table value,
    used as the actual Jo weight -- see `_profile_records`'s comment).

    QC (physical height bounds, non-monotonic height-with-pressure,
    below-station-elevation, source PREPBUFR quality mark) is applied per
    station using whichever variable's `height` happened to be collocated
    with that level. `humidity_min_pressure_hpa`, if set, additionally masks
    'q' at any level with pressure below it (radiosonde humidity sensors are
    unreliable in the upper troposphere/stratosphere).

    `use_drift_time`: see `_profile_records`'s docstring -- assigns each
    level to its own slot via the archive's `drift_time_hours`, instead of
    every level of a profile sharing the report's single launch time.

    `use_drift_position`: see `_profile_records`'s docstring -- also
    returns `lat_<v>`/`lon_<v>` (same shape as `values_<v>`): each kept
    level's own drift-corrected position (`drift_longitude`/
    `drift_latitude`), falling back to the station's own `lat`/`lon`
    wherever no drift position was recorded. Independent of
    `use_drift_time`.

    `q_error_rh_percent`: see `_profile_records`'s docstring -- the
    DEFAULT q error (GSI's real `RH_fraction * qsat(background)`
    mechanism), returned as `default_error_q` (same shape as `values_q`)
    whenever 'q' is requested and this is given. Independent of
    `use_source_error`, which overrides it per-element wherever the
    archive's own value is available.
    """
    variables = tuple(variables)
    levels_hpa = {v: np.asarray(levels_hpa[v], dtype=np.float64) for v in variables}
    records = _profile_records(
        directory, verification_times, levels_hpa, variables, time_tolerance_hours,
        prepbufr_classes, quality_mark_max, pressure_match_tolerance_hpa,
        geopotential_height_min_m, geopotential_height_max_m, below_station_tolerance_m,
        use_source_error=use_source_error, use_drift_time=use_drift_time,
        vertical_interpolation=vertical_interpolation, use_drift_position=use_drift_position,
        q_error_rh_percent=q_error_rh_percent,
    )
    error_bases = tuple(
        v for v in variables
        if use_source_error and (v in _VARIABLE_ERROR_COLUMNS or v == 'q')
    )
    # qsat, kept alongside q's source error so the gross check (long_window_
    # 4dvar_utils.py) can convert GSI's own RH-percent ermin/ermax bounds
    # into kg/kg for the clamp -- WITHOUT touching the source error value
    # itself (see _profile_records's comment: the clamp must not flatten
    # the real per-level table that's used as the actual Jo weight).
    qsat_bases = ('q',) if use_source_error and 'q' in variables else ()
    # The default q error (RH_fraction * qsat), independent of
    # use_source_error -- see q_error_rh_percent above.
    default_error_bases = ('q',) if q_error_rh_percent and 'q' in variables else ()
    # The ob's own real pressure per kept (level, station) cell -- only
    # meaningful once the exact-match discard is dropped ('nearest'/
    # 'log-linear'); unused (and not computed) for 'exact'.
    pressure_bases = tuple(variables) if vertical_interpolation != 'exact' else ()
    # Each kept level's own drift-corrected horizontal position, falling
    # back to the station's own lat/lon where no drift position exists.
    position_bases = tuple(variables) if use_drift_position else ()

    by_slot = {}
    for (slot_index, station_key), record in records.items():
        by_slot.setdefault(slot_index, {})[station_key] = record

    n_slots = len(verification_times)
    result = {'lat': [None]*n_slots, 'lon': [None]*n_slots, 'elev': [None]*n_slots}
    for v in variables:
        result['values_'+v] = [None]*n_slots
        result['mask_'+v] = [None]*n_slots
    for v in error_bases:
        result['source_error_'+v] = [None]*n_slots
    for v in qsat_bases:
        result['source_qsat_'+v] = [None]*n_slots
    for v in default_error_bases:
        result['default_error_'+v] = [None]*n_slots
    for v in pressure_bases:
        result['pressure_'+v] = [None]*n_slots
    for v in position_bases:
        result['lat_'+v] = [None]*n_slots
        result['lon_'+v] = [None]*n_slots
    counts = {v: 0 for v in variables}

    for slot_index in range(n_slots):
        stations = by_slot.get(slot_index, {})
        station_keys = sorted(stations)
        capacity = len(station_keys) if station_capacity is None else int(station_capacity)
        if len(station_keys) > capacity:
            raise ValueError(
                'raobs slot '+str(slot_index)+' has '+str(len(station_keys))
                +' stations, exceeding fixed capacity '+str(capacity)+'.'
            )
        lat = np.zeros(capacity, dtype=np.float32)
        lon = np.zeros(capacity, dtype=np.float32)
        elev = np.full(capacity, np.nan, dtype=np.float32)
        values = {v: np.zeros((levels_hpa[v].size, capacity), dtype=np.float32) for v in variables}
        mask = {v: np.zeros((levels_hpa[v].size, capacity), dtype=bool) for v in variables}
        source_error = {v: np.full((levels_hpa[v].size, capacity), np.nan, dtype=np.float32) for v in error_bases}
        source_qsat = {v: np.full((levels_hpa[v].size, capacity), np.nan, dtype=np.float32) for v in qsat_bases}
        default_error = {v: np.full((levels_hpa[v].size, capacity), np.nan, dtype=np.float32) for v in default_error_bases}
        pressure = {v: np.full((levels_hpa[v].size, capacity), np.nan, dtype=np.float32) for v in pressure_bases}
        # Default to the station's own position (filled in below, after
        # `lat`/`lon` are populated) -- a level with no recorded drift
        # position keeps that fallback, not NaN, so callers always get a
        # usable position.
        lat_level = {v: np.zeros((levels_hpa[v].size, capacity), dtype=np.float32) for v in position_bases}
        lon_level = {v: np.zeros((levels_hpa[v].size, capacity), dtype=np.float32) for v in position_bases}

        for station_index, station_key in enumerate(station_keys):
            record = stations[station_key]
            lat[station_index] = record['lat']
            lon[station_index] = np.mod(record['lon'], 360.)
            elev[station_index] = record['elev']
            for v in position_bases:
                lat_level[v][:, station_index] = lat[station_index]
                lon_level[v][:, station_index] = lon[station_index]

            for v in variables:
                for level_index in range(levels_hpa[v].size):
                    key = (v, level_index)
                    if key not in record['values']:
                        continue
                    quality = record['quality'][key]
                    qc_flag = 0
                    if not _quality_passes(quality, quality_mark_max):
                        qc_flag |= QC_PREPBUFR_QUALITY
                    height_m = record['height'].get(key)
                    if height_m is not None:
                        if not (
                            float(geopotential_height_min_m) <= height_m
                            <= float(geopotential_height_max_m)
                        ):
                            qc_flag |= QC_PHYSICAL_HEIGHT
                        station_elevation = record['elev']
                        if (
                            np.isfinite(station_elevation)
                            and height_m < station_elevation - float(below_station_tolerance_m)
                        ):
                            qc_flag |= QC_BELOW_STATION
                    if v == 'q' and humidity_min_pressure_hpa is not None:
                        if levels_hpa['q'][level_index] < float(humidity_min_pressure_hpa):
                            continue  # not an obs at all here, not even QC-flagged
                    value = record['values'][key]
                    bounds = _RAOBS_PHYSICAL_BOUNDS.get(v)
                    if bounds is not None and not (bounds[0] <= value <= bounds[1]):
                        qc_flag |= QC_PHYSICAL_BOUNDS
                    values[v][level_index, station_index] = value
                    mask[v][level_index, station_index] = (qc_flag == 0)
                    if v in pressure_bases:
                        pressure[v][level_index, station_index] = record['pressure'][key]
                    if v in position_bases and key in record['level_position']:
                        level_lat, level_lon = record['level_position'][key]
                        lat_level[v][level_index, station_index] = level_lat
                        lon_level[v][level_index, station_index] = np.mod(level_lon, 360.)
                    if v in error_bases:
                        source = record['source_error'].get(key)
                        if source is not None and np.isfinite(source) and source > 0.:
                            source_error[v][level_index, station_index] = source
                    if v in qsat_bases:
                        qsat = record['source_qsat'].get(key)
                        if qsat is not None and np.isfinite(qsat) and qsat > 0.:
                            source_qsat[v][level_index, station_index] = qsat
                    if v in default_error_bases:
                        default = record['default_error'].get(key)
                        if default is not None and np.isfinite(default) and default > 0.:
                            default_error[v][level_index, station_index] = default

            # Non-monotonic height-with-pressure, checked per variable (each
            # variable's own level/height collocations), excluding category-4
            # rows already filtered out of `record['height']`.
            for v in variables:
                heights = {
                    level_index: record['height'][(v, level_index)]
                    for level_index in range(levels_hpa[v].size)
                    if (v, level_index) in record['height']
                }
                ordered = [
                    heights[idx] for idx in np.argsort(-levels_hpa[v])
                    if idx in heights
                ]
                if len(ordered) > 1 and np.any(np.diff(ordered) <= 0.):
                    mask[v][:, station_index] = False

        for v in variables:
            counts[v] += int(mask[v].sum())
        result['lat'][slot_index] = lat
        result['lon'][slot_index] = lon
        result['elev'][slot_index] = elev
        for v in variables:
            result['values_'+v][slot_index] = values[v]
            result['mask_'+v][slot_index] = mask[v]
        for v in error_bases:
            result['source_error_'+v][slot_index] = source_error[v]
        for v in qsat_bases:
            result['source_qsat_'+v][slot_index] = source_qsat[v]
        for v in default_error_bases:
            result['default_error_'+v][slot_index] = default_error[v]
        for v in pressure_bases:
            result['pressure_'+v][slot_index] = pressure[v]
        for v in position_bases:
            result['lat_'+v][slot_index] = lat_level[v]
            result['lon_'+v][slot_index] = lon_level[v]

    if logger is not None:
        logger.info(
            'raobs parquet: '+str(len(records))+' (slot, station) profile reports; '
            'accepted obs by variable '+str(counts)
        )
    result['levels_hpa'] = levels_hpa
    result['counts'] = counts
    return result
