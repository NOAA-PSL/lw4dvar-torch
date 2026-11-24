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

# Default physical sanity bounds on the VALUE itself (model units: K, m/s,
# kg/kg), same numbers as UFO's GFS sonde QC config (PR #29's review
# comment -- test/testinput/instrumentTests/Sonde/sonde_gfs_qc.yaml): T
# 195-327 K, q 1e-8-0.035 kg/kg, |u|,|v| <= 130 m/s. A read-time check,
# like the existing height-based ones -- independent of the background/
# model state, so it belongs here rather than in the forward operator.
# Overridable via `observations.raobs.physical_bounds` (see get_raobs/
# load_config in long_window_4dvar_utils.py); `load_raobs_parquet`'s own
# `physical_bounds=None` default falls back to this table.
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

# `use_drift_time` can shift a level's EFFECTIVE time away from its
# report's own (un-drifted) observation_time by up to this much (a real
# deep sounding: +1.62h by 5 hPa, confirmed against real data) --
# `_profile_records` widens which cycle PARTITIONS it opens by this
# margin so a level whose drift-corrected time falls inside the
# verification window, but whose report's own un-corrected time doesn't
# (e.g. a report right at a tolerance window's edge, drifting into it),
# still gets its home partition read. The per-level acceptance check
# still uses the real, un-widened `time_tolerance_hours` against the
# corrected time -- this constant only affects which files get opened.
_RAOBS_MAX_DRIFT_HOURS = 3.0

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
    use_drift_time=False,
    vertical_interpolation='exact',
    use_drift_position=False,
    q_error_rh_percent=None,
    humidity_min_pressure_hpa=None,
    physical_bounds=None,
    logger=None,
):
    """Return {(slot_index, station_key): record} before fixed-shape padding.

    `physical_bounds`: {base: (min, max)} overriding `_RAOBS_PHYSICAL_BOUNDS`
    (None keeps the default). See `load_raobs_parquet`'s docstring.

    `q_error_rh_percent`: {hPa: percent} table (sorted by pressure, any
    order) -- the q observation error: this level's own RH-error-percent
    (ln(p)-interpolated to its real reported pressure) times qsat computed
    at the archive's own static `background_temperature_c`/
    `background_pressure_mb` (see `_saturation_specific_humidity`).
    Recorded as `record['default_error']` whenever 'q' is requested and
    this table is given -- the same mechanism UFO's
    `ObsErrorModelHumidity` uses (RH-error-table-times-qsat-from-
    background), not a flat per-level kg/kg guess or a fraction of the
    observed value. None (q not requested, or no table given) skips this
    entirely. See docs/integrate_multiple_parquetobs.md.

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
    archive column -- confirmed against real data: near 0 at the
    surface, growing smoothly to +1.6h by 5 hPa for a real deep sounding)
    instead of the report's single launch time
    for every level. Falls back to the report's own time wherever
    `drift_time_hours` is missing/non-finite for a given level. This can
    split one profile's levels across multiple slots -- no special
    handling needed: the (slot_index, station_key) keying below already
    treats each slot independently. Partition discovery (which cycle
    directories get opened at all, below) is widened by
    `_RAOBS_MAX_DRIFT_HOURS` to cover this -- otherwise a report sitting
    right at (or just outside) the un-drifted tolerance window's edge
    could have its home cycle directory never opened at all, even though
    its drift-corrected time belongs inside the window.

    `humidity_min_pressure_hpa`: if set, a 'q' candidate row is dropped
    entirely -- not even a duplicate-selection candidate -- whenever its
    own REPORTED pressure (not the model level it would be matched to) is
    below this floor. Applied here, before duplicate selection, rather
    than after the fact on the final selection's nominal model-level
    pressure: in 'nearest'/'log-linear' modes a kept level's own real
    pressure can differ meaningfully from its assigned model level (e.g.
    290 hPa snapped to a 300 hPa model level), so checking the model level
    would let a floor-violating candidate win a duplicate slot over a
    candidate that would have passed, permanently losing the usable
    observation.

    `levels_hpa`: {model_base_name: 1D array of exact pressures (hPa) to
    match} -- a level row is kept only when its `pressure_mb` is within
    `pressure_match_tolerance_hpa` of one of THAT variable's own levels (so
    `u`/`v` can share one level set while `q` uses a shorter one, matching
    each variable's actual model pressure-level axis).

    Duplicate (variable, level_index) rows for the same station/slot are
    ranked by actual QC eligibility, not just the raw PREPBUFR quality
    mark: a row whose own quality fails `quality_mark_max`, OR whose
    (quality-passing, non-category-4) height fails the physical-height-
    bounds/below-station checks, OR whose own value fails the physical-
    value bounds (`_RAOBS_PHYSICAL_BOUNDS`, applied again later as QC but
    also checked HERE), ranks behind any row that passes all three --
    otherwise picking the single numerically-best-quality-mark duplicate
    (the only criterion previously used) could retain a row with a
    garbage height or an impossible value over an available row that
    would have passed every check, which then wrongly fails QC downstream
    with the usable duplicate already discarded. Within
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
    physical_bounds = _RAOBS_PHYSICAL_BOUNDS if physical_bounds is None else physical_bounds

    cycles = common.cycle_directories(directory, verification_times, time_tolerance_hours)
    if use_drift_time:
        # See the docstring above: widen which partitions get opened by
        # the maximum supported drift, tolerating missing directories in
        # just that extra margin (e.g. at the archive's own date-range
        # boundary) -- the original, unwidened `cycles` above still raise
        # as before if genuinely missing.
        padded_cycles = common.cycle_directories(
            directory, verification_times, time_tolerance_hours + _RAOBS_MAX_DRIFT_HOURS,
            allow_missing=True, logger=logger,
        )
        cycles = sorted(set(cycles) | set(padded_cycles))
    verification = np.asarray([
        np.datetime64(common.to_datetime(value), 's') for value in verification_times
    ])
    tolerance_seconds = float(time_tolerance_hours) * 3600.

    report_columns = [
        'report_id', 'observation_time', 'station_id', 'latitude',
        'longitude', 'elevation_m', 'report_type',
    ]
    base_level_columns = ['report_id', 'level_index', 'category', 'pressure_mb'] + list(_HEIGHT_COLUMNS)
    for variable in variables:
        base_level_columns.extend(_VARIABLE_COLUMNS[variable])
    # background_temperature_c/background_pressure_mb feed the qsat
    # conversion for q's error (RH_percent * qsat -- see q_error_rh_percent
    # above).
    if 'q' in variables and q_error_rh_percent:
        base_level_columns.extend(('background_temperature_c', 'background_pressure_mb'))
    base_level_columns = list(dict.fromkeys(base_level_columns))

    records = {}
    # Profile-wide (not per-slot) winner tracking for the height-order QC
    # below -- keyed by (profile identity, variable, the row's own REAL
    # reported pressure), not the matched model level, so two genuinely
    # different real levels that happen to snap to the same model-level
    # bucket (vertical_interpolation != 'exact') both stay visible to the
    # monotonicity check below instead of one hiding the other's height.
    profile_source_priority = {}
    profile_source_heights = {}
    for cycle in cycles:
        for prepbufr_class in prepbufr_classes:
            reports = common.read_partition(
                common.class_file(cycle, 'reports', prepbufr_class), report_columns
            )
            level_path = common.class_file(cycle, 'levels', prepbufr_class)
            # drift_time_hours/drift_longitude/drift_latitude are read
            # only if both requested AND actually present in THIS
            # partition's schema -- an archive (or synthetic fixture,
            # e.g. tests/data/mini_ace2/raobs) without them falls back to
            # the documented launch-time/launch-position behavior instead
            # of read_partition raising on the WHOLE read over one
            # missing optional column.
            partition_columns = common.available_columns(level_path)
            this_use_drift_time = use_drift_time and 'drift_time_hours' in partition_columns
            this_use_drift_position = (
                use_drift_position
                and 'drift_longitude' in partition_columns and 'drift_latitude' in partition_columns
            )
            level_columns = list(base_level_columns)
            if this_use_drift_time:
                level_columns.append('drift_time_hours')
            if this_use_drift_position:
                level_columns.extend(('drift_longitude', 'drift_latitude'))
            levels = common.read_partition(level_path, level_columns)
            report_ids = common.column(reports, 'report_id', np.int64)
            report_rows = {
                int(report_id): row for row, report_id in enumerate(report_ids)
            }
            report_times = common.timestamps(reports)
            # HRDR ('drift_time_hours') is cycle-relative, the SAME
            # referent as DHR -- a per-level REPLACEMENT for the report's
            # single DHR-based time, not an increment on top of it (GSI's
            # read_prepbufr.F90 and the RDASApp bufr2ioda_adpupa_prepbufr.py
            # converter both anchor it to cycle_time directly). Anchoring
            # to report_time (== cycle_time + DHR) instead would double-
            # count DHR. `cycle` is the archive's own
            # <root>/<YYYY-MM-DD>/<HH> partition directory for this cycle
            # -- parsed back into the same cycle datetime
            # cycle_directories (common.py) computed before discarding it
            # down to a bare Path, rather than adding a new archive column.
            cycle_datetime = np.datetime64(common.cycle_time_from_directory(cycle), 's')
            station_ids = reports['station_id'].to_pylist()
            latitude = common.column(reports, 'latitude')
            longitude = common.column(reports, 'longitude')
            elevation = common.column(reports, 'elevation_m')

            level_report_ids = common.column(levels, 'report_id', np.int64)
            raw_level_index = common.column(levels, 'level_index', np.int64)
            category = common.column(levels, 'category', np.int16)
            pressure = common.column(levels, 'pressure_mb')
            height = common.column(levels, 'height_m')
            height_quality = common.column(levels, 'height_quality')
            drift_time_hours = common.column(levels, 'drift_time_hours') if this_use_drift_time else None
            drift_longitude = common.column(levels, 'drift_longitude') if this_use_drift_position else None
            drift_latitude = common.column(levels, 'drift_latitude') if this_use_drift_position else None
            level_values = {
                v: common.column(levels, _VARIABLE_COLUMNS[v][0]) for v in variables
            }
            level_quality = {
                v: common.column(levels, _VARIABLE_COLUMNS[v][1]) for v in variables
            }
            level_default_error = {}
            if 'q' in variables and q_error_rh_percent:
                # qsat at the archive's own STATIC background T/p (not
                # this run's live model state -- a known, deliberate
                # simplification; see docs/integrate_multiple_parquetobs.md).
                background_t_k = common.column(levels, 'background_temperature_c') + 273.15
                background_p_mb = common.column(levels, 'background_pressure_mb')
                qsat = _saturation_specific_humidity(background_t_k, background_p_mb)
                # RH_fraction * qsat(background) -- UFO's ObsErrorModelHumidity
                # mechanism. Interpolated to each level's own REAL pressure
                # (`pressure`, not the matched model level), same ln(p)
                # convention as error_std's own per-level table.
                default_rh_percent = _interp_rh_percent_table(q_error_rh_percent, pressure)
                level_default_error['q'] = (default_rh_percent / 100.) * qsat
            if 't' in level_values:
                # current TOB is virtual temperature where moisture was
                # reported (VIRTMP); the model's t is sensible temperature
                level_values['t'] = common.sensible_temperature(
                    cycle, prepbufr_class, level_report_ids, raw_level_index, level_values['t'],
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
                    and -90. <= lat <= 90. and np.isfinite(pressure_value) and pressure_value > 0.
                ):
                    continue
                report_time = report_times[report_row]
                if np.isnat(report_time):
                    continue
                level_lat, level_lon = lat, lon
                if drift_longitude is not None and drift_latitude is not None:
                    drift_lat = drift_latitude[level_row]
                    drift_lon = drift_longitude[level_row]
                    # Same range check the launch latitude gets above --
                    # a finite-but-out-of-range drift latitude (garbage
                    # data) would otherwise reach horizontal
                    # interpolation directly.
                    if np.isfinite(drift_lat) and np.isfinite(drift_lon) and -90. <= drift_lat <= 90.:
                        level_lat, level_lon = drift_lat, drift_lon
                level_time = report_time
                if drift_time_hours is not None:
                    drift_h = drift_time_hours[level_row]
                    if np.isfinite(drift_h):
                        level_time = cycle_datetime + np.timedelta64(int(round(drift_h * 3600.)), 's')
                delta_seconds = np.abs(
                    (verification - level_time).astype('timedelta64[s]')
                    .astype(np.int64)
                )
                slot_index = int(np.argmin(delta_seconds))
                # Does NOT skip the rest of this level row when outside
                # every slot's tolerance -- only the ASSIMILATION
                # bookkeeping (the per-slot `record` below) is gated on
                # this. The profile-wide height-order QC further below
                # still sees this level's height regardless, so a level
                # that drifts just outside the window can't hide an
                # inversion that would otherwise reject an IN-WINDOW
                # sibling level of the same sounding. This gap can only
                # actually occur with use_drift_time: without it every
                # level of one report shares the same report_time, and so
                # the same accept/reject verdict, making the gap moot.
                time_accepted = delta_seconds[slot_index] <= tolerance_seconds

                station_key = common.station_key(
                    station_ids[report_row], lat, lon,
                    str(cycle)+':'+str(report_id),
                )
                # Include the report's own (un-drifted) launch time in the
                # profile identity: a bare station_key collapses two
                # DIFFERENT launches from the same physical station (e.g. a
                # special/extra sounding near a regular one) into one
                # profile, letting their levels silently replace each other
                # by quality and corrupting the height-order QC below with
                # levels from different soundings.
                # Genuinely repeated rows of the SAME launch still share one
                # report_time, so they keep merging as before.
                profile_key = (station_key, report_time)
                elevation_m = elevation[report_row]
                record = None
                if time_accepted:
                    key = (slot_index, profile_key)
                    record = records.setdefault(key, {
                        'lat': lat, 'lon': lon, 'elev': elevation_m,
                        'values': {}, 'quality': {}, 'height': {}, 'height_quality': {}, 'priority': {},
                        'default_error': {}, 'pressure': {}, 'level_position': {},
                    })

                for variable in variables:
                    if (
                        variable == 'q' and humidity_min_pressure_hpa is not None
                        and pressure_value < float(humidity_min_pressure_hpa)
                    ):
                        continue  # not an obs at all here, not even QC-flagged -- see docstring above
                    variable_levels = levels_hpa[variable]
                    # Nearest by raw hPa distance, not ln(p) -- confirmed
                    # consistent with JEDI's own nearest-neighbor method,
                    # not a deviation from it (an earlier version of this
                    # comment claimed otherwise). Read directly from the
                    # real ufo source (src/ufo/operators/vertinterp/
                    # ufo_vertinterp_mod.F90): `log()` is applied to the
                    # model profile and the observation's own coordinate
                    # ONLY on the log-linear branch (`tmp`/`tmp2 =
                    # log(...)`); the nearest-neighbor branch feeds the
                    # same `nearestneighbor_interp_index`
                    # (src/ufo/utils/vert_interp.F90) raw, un-logged
                    # values (`tmp2 = obsvcoord(iobs)`). ln(p) is a
                    # log-linear-specific step in JEDI, not a general
                    # "JEDI always works in ln(p) vertically" rule --
                    # there is no JEDI nearest-neighbor convention this
                    # mode falls short of.
                    differences = np.abs(variable_levels - pressure_value)
                    level_index = int(np.argmin(differences))
                    # 'exact': only a level within pressure_match_tolerance_hpa
                    # of a model level is eligible for ASSIMILATION (gates the
                    # time_accepted block below) -- but, like time_accepted
                    # itself, must NOT also gate the profile-wide height
                    # tracking further down: a level discarded here for being
                    # too far from any model level still has a real reported
                    # height at its own real pressure, and must still be able
                    # to reject an in-tolerance sibling for a height
                    # inversion, even though it's never itself assimilated
                    # (PR #29's review -- the same class of gap already fixed
                    # for time_accepted, found again for this gate).
                    # 'nearest'/'log-linear' have no distance cutoff at all --
                    # every level is exact_ok, so a coarse model's levels
                    # still get an observation instead of possibly none.
                    exact_ok = (
                        vertical_interpolation != 'exact'
                        or differences[level_index] <= pressure_match_tolerance_hpa
                    )
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
                    # Physical-value sanity (the same bounds load_raobs_parquet
                    # applies again as QC) checked HERE too, as part of
                    # duplicate-selection eligibility -- otherwise an
                    # impossible value with a merely better quality mark
                    # could win the slot over a value that would have
                    # passed every check, permanently losing it (see
                    # docstring above).
                    bounds = physical_bounds.get(variable)
                    physical_bad = bounds is not None and not (bounds[0] <= value <= bounds[1])
                    # A 'q' candidate whose computed default_error (RH_frac
                    # * qsat(background)) isn't usable ranks as bad too --
                    # otherwise a quality-1 row with a missing/invalid
                    # archived background could win the slot over a
                    # quality-2 row with a finite error, only to be
                    # rejected later by the forward operator's own
                    # isfinite(effective_error) guard, losing the level
                    # entirely.
                    error_bad = False
                    if variable in level_default_error:
                        default_error_value = level_default_error[variable][level_row]
                        error_bad = not (np.isfinite(default_error_value) and default_error_value > 0.)
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
                        quality_bad or height_bad or physical_bad or error_bad,
                        quality if np.isfinite(quality) else np.inf,
                        differences[level_index] if vertical_interpolation != 'exact' else 0.,
                    )

                    if time_accepted and exact_ok:
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
                            if variable in level_default_error:
                                record['default_error'][(variable, level_index)] = \
                                    level_default_error[variable][level_row]
                            else:
                                record['default_error'].pop((variable, level_index), None)

                    # Mirrors the per-slot winner selection just above, but
                    # runs UNCONDITIONALLY (not gated on time_accepted OR
                    # exact_ok) and is keyed by the whole PROFILE
                    # (profile_key, not (slot_index, profile_key)) AND the
                    # row's own REAL reported pressure (not the matched
                    # model level_index) -- feeds the height-order QC below,
                    # computed once on the COMPLETE sounding rather than
                    # per-slot or per-window, so an inversion spanning two
                    # use_drift_time slots -- or involving a level that
                    # drifted outside every slot's tolerance, or (in
                    # 'exact' mode) outside pressure_match_tolerance_hpa of
                    # any model level, and so is never itself assimilated --
                    # is still caught.
                    # Keying by real pressure instead of the
                    # matched bucket means two genuinely different real
                    # levels that snap to the SAME model level
                    # (vertical_interpolation != 'exact') both stay
                    # visible here, even though only one of them can win
                    # that bucket for assimilation above; two rows
                    # reporting the IDENTICAL real pressure -- a genuine
                    # duplicate (e.g. a retransmitted message), same
                    # convention this module's own duplicate-selection
                    # tests use -- still collapse to one winner.
                    source_key = (variable, float(pressure_value))
                    profile_key_priority = profile_source_priority.setdefault(profile_key, {})
                    existing_profile_priority = profile_key_priority.get(source_key)
                    if existing_profile_priority is None or priority < existing_profile_priority:
                        profile_key_priority[source_key] = priority
                        profile_key_heights = profile_source_heights.setdefault(profile_key, {})
                        if height_eligible:
                            profile_key_heights[source_key] = (pressure_value, height_m)
                        else:
                            profile_key_heights.pop(source_key, None)

    # Non-monotonic height-with-pressure, computed ONCE per complete
    # profile (every RAW level the archive has for this report, whether
    # or not it won its model-level bucket for assimilation, whether or
    # not its own drift-corrected time fell inside any slot's tolerance
    # window, AND whether or not it's within pressure_match_tolerance_hpa
    # of a model level in 'exact' mode -- see the winner-selection,
    # time_accepted, and exact_ok comments above), not per-slot or
    # per-matched-bucket or per-window -- an inversion straddling two
    # use_drift_time slots, hidden behind a bucket-mate that won the
    # assimilation slot, or involving a level that drifted outside every
    # slot's tolerance or sits too far from any model level to be
    # assimilated itself, must still reject the whole variable's profile,
    # matching the no-drift-time behavior where it's always checked on
    # the complete sounding. Ordered by each level's own REAL reported pressure
    # (not the matched model level) so two raw levels sharing one bucket
    # still sort correctly relative to each other. Consumed by
    # load_raobs_parquet instead of that function recomputing it from
    # its own (slot-limited, bucket-collapsed) record['height'].
    profile_height_order_bad = {}
    for profile_key, heights_by_key in profile_source_heights.items():
        bad_variables = set()
        for variable in variables:
            pairs = sorted(
                (
                    (pressure_value, height_m)
                    for (var, _), (pressure_value, height_m) in heights_by_key.items()
                    if var == variable
                ),
                key=lambda pair: -pair[0],  # descending pressure == surface-to-top, same as before
            )
            ordered = [height_m for _, height_m in pairs]
            if len(ordered) > 1 and np.any(np.diff(ordered) <= 0.):
                bad_variables.add(variable)
        if bad_variables:
            profile_height_order_bad[profile_key] = bad_variables

    # Drop records that ended up with no retained observation at all (e.g.
    # a q-only sounding with every level below humidity_min_pressure_hpa)
    # -- otherwise they still consume a station slot downstream, wasting
    # raobs_station_capacity and potentially displacing a station that
    # would have had usable observations.
    records = {
        key: record for key, record in records.items() if record['values']
    }
    for (slot_index, profile_key), record in records.items():
        record['height_order_bad'] = profile_height_order_bad.get(profile_key, set())
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
    use_drift_time=False,
    vertical_interpolation='exact',
    use_drift_position=False,
    q_error_rh_percent=None,
    physical_bounds=None,
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

    QC (physical height bounds, non-monotonic height-with-pressure,
    below-station-elevation, source PREPBUFR quality mark) is applied per
    station using whichever variable's `height` happened to be collocated
    with that level. `humidity_min_pressure_hpa`, if set, excludes any 'q'
    candidate whose own reported pressure is below it (radiosonde humidity
    sensors are unreliable in the upper troposphere/stratosphere) -- applied
    in `_profile_records`, BEFORE duplicate selection, against each
    candidate's real reported pressure (see that function's docstring for
    why, not the final selection's matched model-level pressure).

    `use_drift_time`: see `_profile_records`'s docstring -- assigns each
    level to its own slot via the archive's `drift_time_hours`, instead of
    every level of a profile sharing the report's single launch time.

    `use_drift_position`: see `_profile_records`'s docstring -- also
    returns `lat_<v>`/`lon_<v>` (same shape as `values_<v>`): each kept
    level's own drift-corrected position (`drift_longitude`/
    `drift_latitude`), falling back to the station's own `lat`/`lon`
    wherever no drift position was recorded. Independent of
    `use_drift_time`.

    `q_error_rh_percent`: see `_profile_records`'s docstring -- the q
    error (UFO's `ObsErrorModelHumidity` RH-error-table-times-qsat
    mechanism), returned as `default_error_q` (same shape as `values_q`)
    whenever 'q' is requested and this is given.

    `physical_bounds`: {base: (min, max)} overriding `_RAOBS_PHYSICAL_BOUNDS`
    (None keeps the default -- same numbers as UFO's sonde `Bounds Check`
    filters). Applied both at duplicate-selection time (`_profile_records`)
    and as the final `QC_PHYSICAL_BOUNDS` flag below.
    """
    variables = tuple(variables)
    levels_hpa = {v: np.asarray(levels_hpa[v], dtype=np.float64) for v in variables}
    physical_bounds = _RAOBS_PHYSICAL_BOUNDS if physical_bounds is None else physical_bounds
    records = _profile_records(
        directory, verification_times, levels_hpa, variables, time_tolerance_hours,
        prepbufr_classes, quality_mark_max, pressure_match_tolerance_hpa,
        geopotential_height_min_m, geopotential_height_max_m, below_station_tolerance_m,
        use_drift_time=use_drift_time,
        vertical_interpolation=vertical_interpolation, use_drift_position=use_drift_position,
        q_error_rh_percent=q_error_rh_percent, humidity_min_pressure_hpa=humidity_min_pressure_hpa,
        physical_bounds=physical_bounds, logger=logger,
    )
    # The q error (RH_fraction * qsat) -- see q_error_rh_percent above.
    default_error_bases = ('q',) if q_error_rh_percent and 'q' in variables else ()
    # The ob's own real pressure per kept (level, station) cell -- only
    # meaningful once the exact-match discard is dropped ('nearest'/
    # 'log-linear'); unused (and not computed) for 'exact'.
    pressure_bases = tuple(variables) if vertical_interpolation != 'exact' else ()
    # Each kept level's own drift-corrected horizontal position, falling
    # back to the station's own lat/lon where no drift position exists.
    position_bases = tuple(variables) if use_drift_position else ()

    by_slot = {}
    for (slot_index, profile_key), record in records.items():
        by_slot.setdefault(slot_index, {})[profile_key] = record

    n_slots = len(verification_times)
    result = {'lat': [None]*n_slots, 'lon': [None]*n_slots, 'elev': [None]*n_slots}
    for v in variables:
        result['values_'+v] = [None]*n_slots
        result['mask_'+v] = [None]*n_slots
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
        profile_keys = sorted(stations)
        capacity = len(profile_keys) if station_capacity is None else int(station_capacity)
        if len(profile_keys) > capacity:
            raise ValueError(
                'raobs slot '+str(slot_index)+' has '+str(len(profile_keys))
                +' stations, exceeding fixed capacity '+str(capacity)+'.'
            )
        lat = np.zeros(capacity, dtype=np.float32)
        lon = np.zeros(capacity, dtype=np.float32)
        elev = np.full(capacity, np.nan, dtype=np.float32)
        values = {v: np.zeros((levels_hpa[v].size, capacity), dtype=np.float32) for v in variables}
        mask = {v: np.zeros((levels_hpa[v].size, capacity), dtype=bool) for v in variables}
        default_error = {v: np.full((levels_hpa[v].size, capacity), np.nan, dtype=np.float32) for v in default_error_bases}
        pressure = {v: np.full((levels_hpa[v].size, capacity), np.nan, dtype=np.float32) for v in pressure_bases}
        # Default to the station's own position (filled in below, after
        # `lat`/`lon` are populated) -- a level with no recorded drift
        # position keeps that fallback, not NaN, so callers always get a
        # usable position.
        lat_level = {v: np.zeros((levels_hpa[v].size, capacity), dtype=np.float32) for v in position_bases}
        lon_level = {v: np.zeros((levels_hpa[v].size, capacity), dtype=np.float32) for v in position_bases}

        for station_index, profile_key in enumerate(profile_keys):
            record = stations[profile_key]
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
                    value = record['values'][key]
                    bounds = physical_bounds.get(v)
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
                    if v in default_error_bases:
                        default = record['default_error'].get(key)
                        if default is not None and np.isfinite(default) and default > 0.:
                            default_error[v][level_index, station_index] = default

            # Non-monotonic height-with-pressure: computed in
            # _profile_records across the COMPLETE source profile (every
            # level, before use_drift_time splits it across slots), not
            # just this slot's own subset -- an inversion straddling two
            # slots must reject both, matching the no-drift-time behavior.
            for v in record.get('height_order_bad', ()):
                mask[v][:, station_index] = False

        for v in variables:
            counts[v] += int(mask[v].sum())
        result['lat'][slot_index] = lat
        result['lon'][slot_index] = lon
        result['elev'][slot_index] = elev
        for v in variables:
            result['values_'+v][slot_index] = values[v]
            result['mask_'+v][slot_index] = mask[v]
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
