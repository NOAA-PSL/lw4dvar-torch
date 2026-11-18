"""Radiosonde (ADPUPA) profile observation reader for the normalized,
cycle-partitioned PREPBUFR parquet archive -- temperature, horizontal wind,
and specific humidity at mandatory pressure levels. See
docs/integrate_multiple_parquetobs.md for the design (mandatory-level
matching -- no vertical interpolation -- and why) and `psobs_parquet.py` for
the sibling surface-pressure reader this shares I/O helpers with
(`prepbufr_parquet_common.py`).

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
QC_FLAG_MASKS = np.array(
    [QC_PHYSICAL_HEIGHT, QC_NON_MONOTONIC_HEIGHT, QC_BELOW_STATION, QC_PREPBUFR_QUALITY],
    dtype=np.int16,
)
QC_FLAG_MEANINGS = (
    'physical_height non_monotonic_height below_station prepbufr_quality'
)

DEFAULT_RAOBS_CLASSES = ('ADPUPA',)

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
# `levels` table (the PREPBUFR TOE/WOE mnemonics -- the same per-type,
# per-pressure GSI error table values `psobs_parquet.py`'s
# `pressure_error_mb`/`use_source_error` already reads for ps, confirmed by
# comparing real archive values against GSI's prepobs_errtable.global: both
# land on 0.8-1.5 K for raob T and 1.4-3.2 m/s for raob wind). 'u' and 'v'
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
):
    """Return {(slot_index, station_key): record} before fixed-shape padding.

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
            level_columns.extend((
                'humidity_error_tenths_percent', 'background_temperature_c', 'background_pressure_mb',
            ))
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
            if use_source_error and 'q' in variables:
                # RH-based source error (percent/10, per
                # convert_prepbufr_to_parquet.py's own field metadata) ->
                # absolute kg/kg via qsat at the archive's own STATIC
                # background T/p (not this run's live model state -- see
                # _VARIABLE_ERROR_COLUMNS's docstring above). qsat is also
                # kept (per level) so the gross check can apply GSI's own
                # RH-percent ermin/ermax bounds later (long_window_4dvar_
                # utils.py) -- NOT inflated/clamped itself (it's a
                # physical qsat value, not an error); only the error value
                # below is. The ermin/ermax clamp is gross-check-only (see
                # GSI's setupq.f90 -- its clamped `obserror` never feeds
                # the Jo weight, confirmed by reading the source), unlike
                # this x1.2 inflation, which GSI applies directly to the
                # value it writes out and weights with.
                rh_error_percent = common.column(levels, 'humidity_error_tenths_percent') * 10.
                background_t_k = common.column(levels, 'background_temperature_c') + 273.15
                background_p_mb = common.column(levels, 'background_pressure_mb')
                qsat = _saturation_specific_humidity(background_t_k, background_p_mb)
                raw_q_error = (rh_error_percent / 100.) * qsat
                level_source_error['q'] = _gsi_inflate_error(raw_q_error, level_quality['q'])
                level_source_qsat['q'] = qsat
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
                delta_seconds = np.abs(
                    (verification - report_time).astype('timedelta64[s]')
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
                    'source_error': {}, 'source_qsat': {},
                })

                for variable in variables:
                    variable_levels = levels_hpa[variable]
                    differences = np.abs(variable_levels - pressure_value)
                    level_index = int(np.argmin(differences))
                    if differences[level_index] > pressure_match_tolerance_hpa:
                        continue
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
                    # as the pre-existing tie-break).
                    priority = (quality_bad or height_bad, quality if np.isfinite(quality) else np.inf)

                    existing_priority = record['priority'].get((variable, level_index))
                    if existing_priority is None or priority < existing_priority:
                        record['values'][(variable, level_index)] = value
                        record['quality'][(variable, level_index)] = quality
                        record['priority'][(variable, level_index)] = priority
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
    logger=None,
):
    """Read mandatory-level radiosonde profiles into fixed per-slot arrays.

    Returns a dict: 'lat'/'lon'/'elev' (per-slot 1D float32 arrays, one
    station per entry) and, for each requested `variable` (a model base
    name in 't'/'u'/'v'/'q'), `values_<v>`/`mask_<v>` (per-slot 2D float32/
    bool arrays, shape `(len(levels_hpa[v]), n_stations_slot)`). `mask_<v>`
    is False for (level, station) pairs with no accepted observation, OR
    that failed QC -- callers must not read `values_<v>` where `mask_<v>`
    is False (it is left at 0, not NaN). When `use_source_error` is True and
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
    """
    variables = tuple(variables)
    levels_hpa = {v: np.asarray(levels_hpa[v], dtype=np.float64) for v in variables}
    records = _profile_records(
        directory, verification_times, levels_hpa, variables, time_tolerance_hours,
        prepbufr_classes, quality_mark_max, pressure_match_tolerance_hpa,
        geopotential_height_min_m, geopotential_height_max_m, below_station_tolerance_m,
        use_source_error=use_source_error,
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

        for station_index, station_key in enumerate(station_keys):
            record = stations[station_key]
            lat[station_index] = record['lat']
            lon[station_index] = np.mod(record['lon'], 360.)
            elev[station_index] = record['elev']

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
                    values[v][level_index, station_index] = record['values'][key]
                    mask[v][level_index, station_index] = (qc_flag == 0)
                    if v in error_bases:
                        source = record['source_error'].get(key)
                        if source is not None and np.isfinite(source) and source > 0.:
                            source_error[v][level_index, station_index] = source
                    if v in qsat_bases:
                        qsat = record['source_qsat'].get(key)
                        if qsat is not None and np.isfinite(qsat) and qsat > 0.:
                            source_qsat[v][level_index, station_index] = qsat

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

    if logger is not None:
        logger.info(
            'raobs parquet: '+str(len(records))+' (slot, station) profile reports; '
            'accepted obs by variable '+str(counts)
        )
    result['levels_hpa'] = levels_hpa
    result['counts'] = counts
    return result
