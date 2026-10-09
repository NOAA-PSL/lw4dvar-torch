"""Surface-pressure (ps) observation reader for the normalized, cycle-
partitioned PREPBUFR parquet archive (see `prepbufr_to_parquet` --
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/longWin4DVar/prepbufr_to_parquet/`).

Ported from the `long-window-4dvar` (JAX/NeuralGCM) reference repo's
`prepbufr_parquet.py` (`_surface_records`/`load_prepbufr_psobs`), with one
deliberate behavioral change: that reference keeps every report within
`time_tolerance_hours` of a slot (so one station can contribute several
reports to the same slot from a wide window). This repo's historical
`.txt`-file psobs (`psobs1_<YYYYMMDDHH>.txt`, one file per `dt_obs` slot,
produced by binning each station to its single nearest hour) only ever had
one report per station per slot -- replicating that requires an extra
collapse step this module adds on top of the ported de-duplication
(see `load_psobs_parquet`'s docstring). Low-level parquet/cycle-directory
helpers live in `prepbufr_parquet_common.py`, shared with `raobs_parquet.py`.
"""

import numpy as np

import prepbufr_parquet_common as common


DEFAULT_PSOBS_CLASSES = ('ADPSFC', 'SFCSHP')
# Report types actually present in the legacy `.txt` psobs
# (/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs/psobs1_*.txt): confirmed by
# counting obtype codes across a full day (2015-01-01) of those files --
# 181/187 (ADPSFC fixed-land + mesonet) and 180 (SFCSHP ship/buoy). The
# normalized PREPBUFR archive also carries 183/281/282/284/287 for the same
# two classes (prepbufr_parquet.md: "including all surface classes/types
# ... is therefore an observation-network expansion, not a like-for-like
# repeat") -- excluded here by default to stay comparable to the legacy obs.
DEFAULT_PSOBS_REPORT_TYPES = (180, 181, 187)


def _candidate_rank(quality, report_type):
    """Lower is better, among already-QC-passing candidates -- same
    tie-break precedence as the reference repo's dedup: best (lowest)
    quality mark, then report_type < 200 (primary networks) before >= 200
    (supplementary), then report_type itself as a final deterministic
    tie-break."""
    return (
        quality if np.isfinite(quality) else np.inf,
        report_type >= 200,
        report_type,
    )


def load_psobs_parquet(
    directory,
    verification_times,
    time_tolerance_hours,
    prepbufr_classes=DEFAULT_PSOBS_CLASSES,
    report_types=DEFAULT_PSOBS_REPORT_TYPES,
    quality_mark_max=3,
    observation_error_std_hpa=1.,
    use_source_error=False,
    logger=None,
):
    """Read de-duplicated surface pressure into one ragged array per slot.

    Returns a dict of length-`len(verification_times)` lists (one entry per
    slot): 'obtype' (PREPBUFR report_type, int), 'lon'/'lat' (degrees, lon in
    [0, 360)), 'elev' (m), 'ob' (hPa), 'oberr' (hPa). Each list element is a
    1-D float/int numpy array, one row per accepted, de-duplicated station
    report for that slot -- the caller (`get_psobs`) pads these to a common
    `nobs_max` the same way it already does for the `.txt` reader.
    """
    prepbufr_classes = tuple(str(c).upper() for c in prepbufr_classes)
    report_types = None if report_types is None else {
        int(v) for v in report_types
    }
    if observation_error_std_hpa <= 0. or not np.isfinite(observation_error_std_hpa):
        raise ValueError('observation_error_std_hpa must be positive and finite.')

    cycles = common.cycle_directories(directory, verification_times, time_tolerance_hours)
    verification = np.asarray([
        np.datetime64(common.to_datetime(value), 's') for value in verification_times
    ])
    tolerance_seconds = float(time_tolerance_hours) * 3600.

    report_columns = [
        'report_id', 'observation_time', 'station_id', 'latitude',
        'longitude', 'elevation_m', 'report_type',
    ]
    level_columns = [
        'report_id', 'level_index', 'pressure_mb', 'pressure_quality',
        'pressure_error_mb',
    ]

    # best[(slot_index, station_key)] = (abs_delta_seconds, rank, record)
    best = {}
    total_rows = 0
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
            types = common.column(reports, 'report_type', np.int32)

            level_report_ids = common.column(levels, 'report_id', np.int64)
            level_indices = common.column(levels, 'level_index', np.int32)
            pressure = common.column(levels, 'pressure_mb')
            quality = common.column(levels, 'pressure_quality')
            source_error = common.column(levels, 'pressure_error_mb')

            for level_row, report_id in enumerate(level_report_ids):
                # Surface reports can carry category-0/category-6 duplicate
                # levels; only the true surface level is a ps observation.
                if level_indices[level_row] != 0:
                    continue
                report_row = report_rows[int(report_id)]
                total_rows += 1
                lat = latitude[report_row]
                lon = longitude[report_row]
                elev = elevation[report_row]
                value = pressure[level_row]
                report_type = int(types[report_row])
                if report_types is not None and report_type not in report_types:
                    continue
                if not (
                    np.isfinite(lat) and np.isfinite(lon)
                    and np.isfinite(elev) and np.isfinite(value)
                    and -90. <= lat <= 90. and value > 0.
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
                abs_delta = int(delta_seconds[slot_index])
                if abs_delta > tolerance_seconds:
                    continue
                station_key = common.station_key(
                    station_ids[report_row], lat, lon,
                    str(cycle)+':'+str(report_id),
                )
                if not common.quality_passes(quality[level_row], quality_mark_max):
                    continue
                rank = _candidate_rank(quality[level_row], report_type)
                key = (slot_index, station_key)
                error_std = (
                    float(source_error[level_row])
                    if use_source_error and np.isfinite(source_error[level_row])
                    else float(observation_error_std_hpa)
                )
                record = {
                    'obtype': report_type, 'lon': np.mod(lon, 360.), 'lat': lat,
                    'elev': elev, 'ob': value, 'oberr': error_std,
                }
                existing = best.get(key)
                # Single nearest-in-time report per station per slot -- see
                # module docstring (replicates the legacy per-hour-binned
                # `.txt` files, unlike the JAX reference's wider-window
                # multi-report-per-slot behavior).
                if existing is None or (abs_delta, rank) < (existing[0], existing[1]):
                    best[key] = (abs_delta, rank, record)

    result = {
        name: [[] for _ in verification_times]
        for name in ('obtype', 'lon', 'lat', 'elev', 'ob', 'oberr')
    }
    for (slot_index, _key), (_delta, _rank, record) in best.items():
        for name, value in record.items():
            result[name][slot_index].append(value)

    dtypes = {
        'obtype': np.int32, 'lon': np.float32, 'lat': np.float32,
        'elev': np.float32, 'ob': np.float32, 'oberr': np.float32,
    }
    for name in result:
        result[name] = [
            np.asarray(values, dtype=dtypes[name]) for values in result[name]
        ]
    if logger is not None:
        counts = [arr.size for arr in result['ob']]
        logger.info(
            'psobs parquet: '+str(total_rows)+' candidate surface-level rows, '
            +str(sum(counts))+' accepted after class/type/QC/tolerance '
            'filtering, per-slot counts '+str(counts)
        )
    return result
