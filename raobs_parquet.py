"""Radiosonde (ADPUPA) profile observation reader for the normalized,
cycle-partitioned PREPBUFR parquet archive -- temperature, horizontal wind,
and specific humidity at mandatory pressure levels. See
integrate_prepbufr_obs.md for the design (mandatory-level matching -- no vertical
interpolation -- and why) and `psobs_parquet.py` for the sibling
surface-pressure reader this shares I/O helpers with
(`prepbufr_parquet_common.py`).

Ported from the `long-window-4dvar` (JAX/NeuralGCM) reference repo's
`prepbufr_parquet.py` (`_profile_records`/`load_prepbufr_raob_observations`),
restricted to the ADPUPA class, with specific humidity added -- the
reference never implemented moisture for radiosondes at all.
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
):
    """Return {(slot_index, station_key): record} before fixed-shape padding.

    `levels_hpa`: {model_base_name: 1D array of exact pressures (hPa) to
    match} -- a level row is kept only when its `pressure_mb` is within
    `pressure_match_tolerance_hpa` of one of THAT variable's own levels (so
    `u`/`v` can share one level set while `q` uses a shorter one, matching
    each variable's actual model pressure-level axis).
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
    level_columns = ['report_id', 'category', 'pressure_mb'] + list(_HEIGHT_COLUMNS)
    for variable in variables:
        level_columns.extend(_VARIABLE_COLUMNS[variable])
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
                record = records.setdefault(key, {
                    'lat': lat, 'lon': lon, 'elev': elevation[report_row],
                    'values': {}, 'quality': {}, 'height': {}, 'height_quality': {},
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
                    existing_quality = record['quality'].get((variable, level_index), np.inf)
                    if (
                        (variable, level_index) not in record['values']
                        or (np.isfinite(quality) and (
                            not np.isfinite(existing_quality) or quality < existing_quality
                        ))
                    ):
                        record['values'][(variable, level_index)] = value
                        record['quality'][(variable, level_index)] = quality
                        if np.isfinite(height[level_row]) and category[level_row] != 4:
                            record['height'][(variable, level_index)] = height[level_row]
                            record['height_quality'][(variable, level_index)] = height_quality[level_row]
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

    QC (physical height bounds, non-monotonic height-with-pressure,
    below-station-elevation, source PREPBUFR quality mark) is applied per
    station using whichever variable's `height` happened to be collocated
    with that level -- ported from the reference repo's
    `load_prepbufr_raob_observations`. `humidity_min_pressure_hpa`, if set,
    additionally masks 'q' at any level with pressure below it (radiosonde
    humidity sensors are unreliable in the upper troposphere/stratosphere;
    this has no reference-repo precedent -- it never assimilated moisture).
    """
    variables = tuple(variables)
    levels_hpa = {v: np.asarray(levels_hpa[v], dtype=np.float64) for v in variables}
    records = _profile_records(
        directory, verification_times, levels_hpa, variables, time_tolerance_hours,
        prepbufr_classes, quality_mark_max, pressure_match_tolerance_hpa,
    )

    by_slot = {}
    for (slot_index, station_key), record in records.items():
        by_slot.setdefault(slot_index, {})[station_key] = record

    n_slots = len(verification_times)
    result = {'lat': [None]*n_slots, 'lon': [None]*n_slots, 'elev': [None]*n_slots}
    for v in variables:
        result['values_'+v] = [None]*n_slots
        result['mask_'+v] = [None]*n_slots
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

    if logger is not None:
        logger.info(
            'raobs parquet: '+str(len(records))+' (slot, station) profile reports; '
            'accepted obs by variable '+str(counts)
        )
    result['levels_hpa'] = levels_hpa
    result['counts'] = counts
    return result
