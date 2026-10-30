"""Satellite-derived wind (AMV) observation reader for the normalized,
cycle-partitioned PREPBUFR parquet archive's 'SATWND' Table-A class --
GOES/JMA/EUMETSAT/MODIS cloud-drift and water-vapor winds, already merged,
de-duplicated, and quality-controlled into the operational PREPBUFR feed
(the SAME archive `psobs_parquet.py`/`raobs_parquet.py` read, just a
different class). See integrate_amv_satwnd.md for why this reads the
already-decoded 'SATWND' class rather than the separate raw-BUFR GOES-16
`amv_parquet` archive (a NOAA NNJA `satwnd` archive, confusingly similarly
named -- that one is for a future GOES-16-specific stream, not this one).

Ported from the `long-window-4dvar` (JAX/NeuralGCM) reference repo's
`prepbufr_parquet.py:load_prepbufr_upper_air_points`. Returns FLAT,
per-slot-bucketed, un-superobbed points (one row per accepted raw wind
vector) -- model-grid interpolation and superobbing are deliberately left
to the caller (`get_amv` in `long_window_4dvar_utils.py`), since they
depend on the configured model's grid and pressure levels, same division
of labor as the reference's loader. Low-level parquet/cycle-directory
helpers live in `prepbufr_parquet_common.py`, shared with `psobs_parquet.py`
and `raobs_parquet.py`.
"""

import numpy as np

import prepbufr_parquet_common as common


DEFAULT_AMV_CLASSES = ('SATWND',)


def load_amv_parquet(
    directory,
    verification_times,
    time_tolerance_hours=1.,
    prepbufr_classes=DEFAULT_AMV_CLASSES,
    report_types=None,
    quality_mark_max=3,
    pressure_range_hpa=(1000., 50.),
    use_source_error=True,
    fallback_error_std_ms=7.,
    exclude_satellite_ids=(),
    logger=None,
):
    """Read quality-filtered, irregular satellite wind vectors into one
    ragged array per verification-time slot.

    Returns a dict of length-`len(verification_times)` lists (one entry per
    slot): 'lat'/'lon' (degrees, lon in [0, 360)), 'pressure_hpa',
    'u_component_of_wind'/'v_component_of_wind' (m/s), 'error_std' (m/s),
    'report_type' (PREPBUFR report_type, int), 'satellite_id' (int, -1 if
    missing), 'time_offset_hours' (abs. distance from the report's own
    observation time to its assigned verification-time slot -- used by
    `get_amv`'s GSI-style thinning score, see `read_satwnd.f90`'s `crit1`).
    Each list element is a 1-D numpy array, one row per accepted raw wind
    vector for that slot -- deliberately NOT thinned/superobbed here (see
    module docstring); `get_amv` does the model-dependent thinning.
    """
    prepbufr_classes = tuple(str(c).upper() for c in prepbufr_classes)
    report_types = None if report_types is None else {int(v) for v in report_types}
    if pressure_range_hpa is None or len(pressure_range_hpa) != 2:
        raise ValueError('pressure_range_hpa must contain two pressure limits.')
    pressure_min = float(min(pressure_range_hpa))
    pressure_max = float(max(pressure_range_hpa))
    if pressure_min <= 0. or pressure_max <= pressure_min:
        raise ValueError('pressure_range_hpa must define a positive, non-empty range.')
    fallback_error_std_ms = float(fallback_error_std_ms)
    if fallback_error_std_ms <= 0.:
        raise ValueError('fallback_error_std_ms must be positive.')
    excluded_satellites = tuple(sorted({int(v) for v in exclude_satellite_ids}))

    report_columns = [
        'report_id', 'observation_time', 'latitude', 'longitude',
        'report_type', 'satellite_id',
    ]
    level_columns = [
        'report_id', 'pressure_mb', 'u_wind_ms', 'v_wind_ms',
        'wind_quality', 'wind_error_ms',
    ]

    cycles = common.cycle_directories(directory, verification_times, time_tolerance_hours)
    verification = np.asarray([
        np.datetime64(common.to_datetime(value), 's') for value in verification_times
    ])
    tolerance_seconds = float(time_tolerance_hours) * 3600.

    field_names = (
        'lat', 'lon', 'pressure_hpa', 'u_component_of_wind',
        'v_component_of_wind', 'error_std', 'report_type', 'satellite_id',
        'time_offset_hours',
    )
    result = {name: [[] for _ in verification_times] for name in field_names}
    source_rows = 0
    accepted_rows = 0

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

            # Vectorized report_id -> report_row join (SATWND can carry
            # 1e5+ rows per cycle -- a Python dict-per-row join, fine for
            # psobs/raobs's smaller row counts, would be needlessly slow
            # here).
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
            # Nearest verification-time slot per report, vectorized across
            # every report at once (same idea as psobs/raobs's per-row
            # argmin, just batched -- SATWND's row count makes a Python
            # per-row loop undesirable).
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
            satellite = common.column(reports, 'satellite_id', np.float64)[joined_rows]
            pressure = common.column(levels, 'pressure_mb')
            u_wind = common.column(levels, 'u_wind_ms')
            v_wind = common.column(levels, 'v_wind_ms')
            quality = common.column(levels, 'wind_quality')
            source_error = common.column(levels, 'wind_error_ms')
            source_rows += int(level_report_ids.size)

            keep = (
                valid_join
                & (assigned >= 0)
                & np.isfinite(latitude) & np.isfinite(longitude)
                & (latitude >= -90.) & (latitude <= 90.)
                & np.isfinite(pressure) & (pressure >= pressure_min) & (pressure <= pressure_max)
                & np.isfinite(u_wind) & np.isfinite(v_wind)
                & np.isfinite(types)
            )
            if report_types is not None:
                finite_types = np.isfinite(types)
                type_matches = np.zeros(types.shape, dtype=bool)
                type_matches[finite_types] = np.isin(
                    types[finite_types].astype(np.int64), list(report_types)
                )
                keep &= type_matches
            if excluded_satellites:
                finite_satellite = np.isfinite(satellite)
                excluded = np.zeros(satellite.shape, dtype=bool)
                excluded[finite_satellite] = np.isin(
                    satellite[finite_satellite].astype(np.int64), excluded_satellites
                )
                keep &= ~excluded
            if quality_mark_max is not None:
                keep &= np.isfinite(quality) & (quality <= float(quality_mark_max))
            if use_source_error:
                keep &= np.isfinite(source_error) & (source_error > 0.)
                error = source_error
            else:
                error = np.full(source_error.shape, fallback_error_std_ms, dtype=np.float64)

            accepted_rows += int(keep.sum())
            if not keep.any():
                continue

            accepted_slots = assigned[keep]
            lon_mod = np.mod(longitude, 360.)
            satellite_id = np.where(np.isfinite(satellite), satellite, -1.)
            for slot_index in np.unique(accepted_slots):
                slot_index = int(slot_index)
                sel = keep & (assigned == slot_index)
                result['lat'][slot_index].append(latitude[sel].astype(np.float32))
                result['lon'][slot_index].append(lon_mod[sel].astype(np.float32))
                result['pressure_hpa'][slot_index].append(pressure[sel].astype(np.float32))
                result['u_component_of_wind'][slot_index].append(u_wind[sel].astype(np.float32))
                result['v_component_of_wind'][slot_index].append(v_wind[sel].astype(np.float32))
                result['error_std'][slot_index].append(error[sel].astype(np.float32))
                result['report_type'][slot_index].append(types[sel].astype(np.int32))
                result['satellite_id'][slot_index].append(satellite_id[sel].astype(np.int32))
                result['time_offset_hours'][slot_index].append(time_offset_hours[sel].astype(np.float32))

    dtypes = {
        'lat': np.float32, 'lon': np.float32, 'pressure_hpa': np.float32,
        'u_component_of_wind': np.float32, 'v_component_of_wind': np.float32,
        'error_std': np.float32, 'report_type': np.int32, 'satellite_id': np.int32,
        'time_offset_hours': np.float32,
    }
    for name in result:
        result[name] = [
            np.concatenate(values) if values else np.empty(0, dtype=dtypes[name])
            for values in result[name]
        ]

    if logger is not None:
        counts = [arr.size for arr in result['lat']]
        logger.info(
            'amv (SATWND) parquet: '+str(source_rows)+' candidate wind rows, '
            +str(accepted_rows)+' accepted after class/type/QC/tolerance '
            'filtering, per-slot counts '+str(counts)
        )
    return result
