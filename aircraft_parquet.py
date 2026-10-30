"""Aircraft (AIRCAR/AIRCFT) observation reader for the normalized,
cycle-partitioned PREPBUFR parquet archive -- the SAME archive
`psobs_parquet.py`/`raobs_parquet.py`/`amv_parquet.py` read, just two
different classes. See integrate_aircraft_obs.md for why aircraft is read
this way (AMV-shaped: vectorized, native-pressure, GSI box-thinned) rather
than raobs-shaped (mandatory-level snap).

Ported from `amv_parquet.py`'s vectorized joined-read (report counts are
AMV-scale: ~290K AIRCAR + ~28K AIRCFT level rows in one 2015-01-01 00Z
cycle -- far too many for raobs's per-row Python loop).

Critical archive fact, checked directly against the real archive: AIRCAR/
AIRCFT report temperature and wind as SEPARATE, mutually exclusive level
rows (report_type 133 = temperature-only via a mass-report message, 233 =
wind-only via a wind-report message for the SAME physical observation --
confirmed zero rows have both `temperature_c` and `u_wind_ms`/`v_wind_ms`
finite at once). GSI's own convinfo treats these as separate thinning
types for exactly this reason. So `load_aircraft_parquet` returns TWO
independent flat point sets ('t' and 'uv'), not one combined set -- a
single shared thinning box keyed across all three variables would
otherwise silently keep only a temperature OR a wind row per box (whichever
scores better), discarding the other variable entirely for that box.
"""

import numpy as np

import prepbufr_parquet_common as common


DEFAULT_AIRCRAFT_CLASSES = ('AIRCAR', 'AIRCFT')

_T_FIELD_NAMES = (
    'lat', 'lon', 'pressure_hpa', 'temperature', 'error_std',
    'report_type', 'aircraft_id', 'time_offset_hours',
)
_UV_FIELD_NAMES = (
    'lat', 'lon', 'pressure_hpa', 'u_component_of_wind', 'v_component_of_wind',
    'error_std', 'report_type', 'aircraft_id', 'time_offset_hours',
)
_T_DTYPES = {
    'lat': np.float32, 'lon': np.float32, 'pressure_hpa': np.float32,
    'temperature': np.float32, 'error_std': np.float32,
    'report_type': np.int32, 'aircraft_id': object, 'time_offset_hours': np.float32,
}
_UV_DTYPES = {
    'lat': np.float32, 'lon': np.float32, 'pressure_hpa': np.float32,
    'u_component_of_wind': np.float32, 'v_component_of_wind': np.float32,
    'error_std': np.float32, 'report_type': np.int32, 'aircraft_id': object,
    'time_offset_hours': np.float32,
}


def load_aircraft_parquet(
    directory,
    verification_times,
    time_tolerance_hours=1.,
    prepbufr_classes=DEFAULT_AIRCRAFT_CLASSES,
    report_types=None,
    quality_mark_max=3,
    pressure_range_hpa=(1000., 50.),
    use_source_error=True,
    fallback_error_std_t_k=1.5,
    fallback_error_std_ms=3.5,
    logger=None,
):
    """Read quality-filtered aircraft temperature and wind reports into two
    ragged per-slot point sets.

    Returns `{'t': {...}, 'uv': {...}}`. Each value is a dict of
    length-`len(verification_times)` lists (one entry per slot), same shape
    convention as `amv_parquet.load_amv_parquet`:
      - 't': 'lat'/'lon' (degrees, lon in [0, 360)), 'pressure_hpa',
        'temperature' (K), 'error_std' (K), 'report_type', 'aircraft_id'
        (string, '' if missing), 'time_offset_hours'.
      - 'uv': same lat/lon/pressure_hpa/error_std/report_type/aircraft_id/
        time_offset_hours shape, plus 'u_component_of_wind'/
        'v_component_of_wind' (m/s) in place of 'temperature'.
    Neither set is thinned/superobbed here (see module docstring) --
    `get_aircraft` does the model-dependent GSI-style box thinning,
    independently for each set.

    Temperature is sensible temperature (ported from raobs_parquet.py's
    convention): the archive's `temperature_c` is the current TOB, which is
    virtual temperature wherever moisture was reported (VIRTMP) --
    corrected via `common.sensible_temperature` the same way raobs is.
    """
    prepbufr_classes = tuple(str(c).upper() for c in prepbufr_classes)
    report_types = None if report_types is None else {int(v) for v in report_types}
    if pressure_range_hpa is None or len(pressure_range_hpa) != 2:
        raise ValueError('pressure_range_hpa must contain two pressure limits.')
    pressure_min = float(min(pressure_range_hpa))
    pressure_max = float(max(pressure_range_hpa))
    if pressure_min <= 0. or pressure_max <= pressure_min:
        raise ValueError('pressure_range_hpa must define a positive, non-empty range.')
    fallback_error_std_t_k = float(fallback_error_std_t_k)
    fallback_error_std_ms = float(fallback_error_std_ms)
    if fallback_error_std_t_k <= 0. or fallback_error_std_ms <= 0.:
        raise ValueError('fallback_error_std_t_k/fallback_error_std_ms must be positive.')

    report_columns = [
        'report_id', 'observation_time', 'latitude', 'longitude',
        'report_type', 'aircraft_id',
    ]
    level_columns = [
        'report_id', 'level_index', 'pressure_mb',
        'temperature_c', 'temperature_quality', 'temperature_error_c',
        'u_wind_ms', 'v_wind_ms', 'wind_quality', 'wind_error_ms',
    ]

    cycles = common.cycle_directories(directory, verification_times, time_tolerance_hours)
    verification = np.asarray([
        np.datetime64(common.to_datetime(value), 's') for value in verification_times
    ])
    tolerance_seconds = float(time_tolerance_hours) * 3600.

    result_t = {name: [[] for _ in verification_times] for name in _T_FIELD_NAMES}
    result_uv = {name: [[] for _ in verification_times] for name in _UV_FIELD_NAMES}
    source_rows = 0
    accepted_t_rows = 0
    accepted_uv_rows = 0

    for cycle in cycles:
        for prepbufr_class in prepbufr_classes:
            reports = common.read_partition(
                common.class_file(cycle, 'reports', prepbufr_class), report_columns
            )
            levels = common.read_partition(
                common.class_file(cycle, 'levels', prepbufr_class), level_columns
            )
            report_ids = common.column(reports, 'report_id', np.int64)
            level_report_ids = common.column(levels, 'report_id', np.int64)
            if report_ids.size == 0 or level_report_ids.size == 0:
                continue

            # Vectorized report_id -> report_row join (same idiom as
            # amv_parquet.py -- AIRCAR/AIRCFT row counts are AMV-scale).
            report_order = np.argsort(report_ids)
            sorted_report_ids = report_ids[report_order]
            positions = np.searchsorted(sorted_report_ids, level_report_ids)
            valid_join = positions < sorted_report_ids.size
            joined_rows = np.zeros(level_report_ids.size, dtype=np.int64)
            joined_rows[valid_join] = report_order[positions[valid_join]]
            valid_join &= (
                sorted_report_ids[np.minimum(positions, sorted_report_ids.size - 1)]
                == level_report_ids
            )

            report_times = common.timestamps(reports)
            delta = np.abs((
                report_times[:, None] - verification[None, :]
            ).astype('timedelta64[s]').astype(np.int64))
            nearest_slot = np.argmin(delta, axis=1)
            nearest_delta = delta[np.arange(delta.shape[0]), nearest_slot]
            assigned_reports = np.where(
                nearest_delta <= tolerance_seconds, nearest_slot, -1
            ).astype(np.int32)

            assigned = assigned_reports[joined_rows]
            time_offset_hours = nearest_delta[joined_rows].astype(np.float64) / 3600.
            latitude = common.column(reports, 'latitude')[joined_rows]
            longitude = common.column(reports, 'longitude')[joined_rows]
            types = common.column(reports, 'report_type', np.float64)[joined_rows]
            aircraft_id = np.asarray(reports['aircraft_id'].to_pylist(), dtype=object)[joined_rows]
            pressure = common.column(levels, 'pressure_mb')
            temperature_c = common.column(levels, 'temperature_c')
            temperature_c = common.sensible_temperature(
                cycle, prepbufr_class,
                common.column(levels, 'report_id', np.int64),
                common.column(levels, 'level_index', np.int64), temperature_c,
            )
            temperature_quality = common.column(levels, 'temperature_quality')
            temperature_source_error = common.column(levels, 'temperature_error_c')
            u_wind = common.column(levels, 'u_wind_ms')
            v_wind = common.column(levels, 'v_wind_ms')
            wind_quality = common.column(levels, 'wind_quality')
            wind_source_error = common.column(levels, 'wind_error_ms')
            source_rows += int(level_report_ids.size)

            base_keep = (
                valid_join
                & (assigned >= 0)
                & np.isfinite(latitude) & np.isfinite(longitude)
                & (latitude >= -90.) & (latitude <= 90.)
                & np.isfinite(pressure) & (pressure >= pressure_min) & (pressure <= pressure_max)
                & np.isfinite(types)
            )
            if report_types is not None:
                finite_types = np.isfinite(types)
                type_matches = np.zeros(types.shape, dtype=bool)
                type_matches[finite_types] = np.isin(
                    types[finite_types].astype(np.int64), list(report_types)
                )
                base_keep &= type_matches

            lon_mod = np.mod(longitude, 360.)

            # --- temperature sub-stream ---
            keep_t = base_keep & np.isfinite(temperature_c)
            if quality_mark_max is not None:
                keep_t &= np.isfinite(temperature_quality) & (temperature_quality <= float(quality_mark_max))
            if use_source_error:
                keep_t &= np.isfinite(temperature_source_error) & (temperature_source_error > 0.)
                error_t = temperature_source_error
            else:
                error_t = np.full(temperature_source_error.shape, fallback_error_std_t_k, dtype=np.float64)
            accepted_t_rows += int(keep_t.sum())
            if keep_t.any():
                accepted_slots = assigned[keep_t]
                for slot_index in np.unique(accepted_slots):
                    slot_index = int(slot_index)
                    sel = keep_t & (assigned == slot_index)
                    result_t['lat'][slot_index].append(latitude[sel].astype(np.float32))
                    result_t['lon'][slot_index].append(lon_mod[sel].astype(np.float32))
                    result_t['pressure_hpa'][slot_index].append(pressure[sel].astype(np.float32))
                    result_t['temperature'][slot_index].append((temperature_c[sel] + 273.15).astype(np.float32))
                    result_t['error_std'][slot_index].append(error_t[sel].astype(np.float32))
                    result_t['report_type'][slot_index].append(types[sel].astype(np.int32))
                    result_t['aircraft_id'][slot_index].append(aircraft_id[sel])
                    result_t['time_offset_hours'][slot_index].append(time_offset_hours[sel].astype(np.float32))

            # --- wind sub-stream ---
            keep_uv = base_keep & np.isfinite(u_wind) & np.isfinite(v_wind)
            if quality_mark_max is not None:
                keep_uv &= np.isfinite(wind_quality) & (wind_quality <= float(quality_mark_max))
            if use_source_error:
                keep_uv &= np.isfinite(wind_source_error) & (wind_source_error > 0.)
                error_uv = wind_source_error
            else:
                error_uv = np.full(wind_source_error.shape, fallback_error_std_ms, dtype=np.float64)
            accepted_uv_rows += int(keep_uv.sum())
            if keep_uv.any():
                accepted_slots = assigned[keep_uv]
                for slot_index in np.unique(accepted_slots):
                    slot_index = int(slot_index)
                    sel = keep_uv & (assigned == slot_index)
                    result_uv['lat'][slot_index].append(latitude[sel].astype(np.float32))
                    result_uv['lon'][slot_index].append(lon_mod[sel].astype(np.float32))
                    result_uv['pressure_hpa'][slot_index].append(pressure[sel].astype(np.float32))
                    result_uv['u_component_of_wind'][slot_index].append(u_wind[sel].astype(np.float32))
                    result_uv['v_component_of_wind'][slot_index].append(v_wind[sel].astype(np.float32))
                    result_uv['error_std'][slot_index].append(error_uv[sel].astype(np.float32))
                    result_uv['report_type'][slot_index].append(types[sel].astype(np.int32))
                    result_uv['aircraft_id'][slot_index].append(aircraft_id[sel])
                    result_uv['time_offset_hours'][slot_index].append(time_offset_hours[sel].astype(np.float32))

    for name in result_t:
        result_t[name] = [
            np.concatenate(values) if values else np.empty(0, dtype=_T_DTYPES[name])
            for values in result_t[name]
        ]
    for name in result_uv:
        result_uv[name] = [
            np.concatenate(values) if values else np.empty(0, dtype=_UV_DTYPES[name])
            for values in result_uv[name]
        ]

    if logger is not None:
        counts_t = [arr.size for arr in result_t['lat']]
        counts_uv = [arr.size for arr in result_uv['lat']]
        logger.info(
            'aircraft (AIRCAR/AIRCFT) parquet: '+str(source_rows)+' candidate level rows, '
            +str(accepted_t_rows)+' temperature / '+str(accepted_uv_rows)+' wind accepted after '
            'class/type/QC/tolerance filtering, per-slot counts t='+str(counts_t)+' uv='+str(counts_uv)
        )
    return {'t': result_t, 'uv': result_uv}
