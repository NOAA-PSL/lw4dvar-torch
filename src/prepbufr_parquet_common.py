"""Shared low-level readers for the normalized, cycle-partitioned PREPBUFR
parquet archive (`<root>/<YYYY-MM-DD>/<HH>/{reports,levels}/<CLASS>/
part-00000.parquet`) -- the pieces `psobs_parquet.py` and `raobs_parquet.py`
both need (cycle-directory resolution, partition reads, station-key/QC
helpers). See docs/integrate_multiple_parquetobs.md for the archive layout.

Only this module imports `pyarrow`, lazily, so nothing else in the solver
needs it installed.
"""

from datetime import datetime, timedelta
import os
from pathlib import Path

import numpy as np


def parquet():
    try:
        import pyarrow.parquet as parquet_module
    except ImportError as exc:
        raise ImportError(
            'Reading observations from the PREPBUFR parquet archive requires '
            "pyarrow (pip install pyarrow into this run's conda env)."
        ) from exc
    return parquet_module


def to_datetime(value):
    if isinstance(value, datetime):
        return value
    return datetime.strptime(str(value), '%Y-%m-%dT%H')


_CYCLE_SPAN = timedelta(hours=6)
_HALF_CYCLE = timedelta(hours=3)


def _nearest_cycle(moment, prefer):
    """Round `moment` to the 6-hourly cycle it reports into: each cycle
    directory holds reports from its own nominal hour +/- 3h (the
    standard, non-overlapping PREPBUFR synoptic window), so a report
    doesn't necessarily carry the directory's own nominal hour as its
    timestamp (e.g. a 00Z partition can hold a report timestamped 01Z).
    Exactly on the 3h midpoint between two cycles, breaks the tie toward
    `prefer` -- so a query centered exactly on a cycle hour doesn't
    spuriously pull in that cycle's neighbor."""
    floor_cycle = moment.replace(
        hour=(moment.hour // 6) * 6, minute=0, second=0, microsecond=0
    )
    offset = moment - floor_cycle
    if offset < _HALF_CYCLE:
        return floor_cycle
    if offset > _HALF_CYCLE:
        return floor_cycle + _CYCLE_SPAN
    if prefer in (floor_cycle, floor_cycle + _CYCLE_SPAN):
        return prefer
    return floor_cycle + _CYCLE_SPAN


def cycle_directories(directory, verification_times, tolerance_hours, allow_missing=False):
    """Return the six-hourly archive cycle directories that can hold a
    report within `tolerance_hours` of each verification time.

    `allow_missing`: if False (default), every computed cycle directory
    must exist on disk, or this raises -- a real guard against a
    misconfigured/incomplete archive. If True, a computed cycle directory
    that doesn't exist is silently skipped instead (used by raobs_parquet.py
    to widen partition discovery for `use_drift_time` into a margin that may
    legitimately run off the archive's own date-range boundary, without
    losing the error-on-missing guarantee for the caller's own, unwidened
    request)."""
    directory = Path(os.path.expanduser(directory)).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            'PREPBUFR parquet directory not found: '+str(directory)
        )
    tolerance = timedelta(hours=float(tolerance_hours))
    cycles = set()
    for value in verification_times:
        center = to_datetime(value)
        center_cycle = _nearest_cycle(center, None)
        cycle = _nearest_cycle(center - tolerance, center_cycle)
        last_cycle = _nearest_cycle(center + tolerance, center_cycle)
        while cycle <= last_cycle:
            cycles.add(cycle)
            cycle += _CYCLE_SPAN
    paths = []
    for cycle in sorted(cycles):
        path = directory / cycle.strftime('%Y-%m-%d') / cycle.strftime('%H')
        if not path.is_dir():
            if allow_missing:
                continue
            raise FileNotFoundError(
                'PREPBUFR parquet cycle directory not found: '+str(path)
            )
        paths.append(path)
    return paths


def cycle_time_from_directory(cycle_directory):
    """The cycle's own nominal synoptic time (00/06/12/18Z), parsed back
    from a `<root>/<YYYY-MM-DD>/<HH>` partition directory -- the same
    datetime `cycle_directories` above computed before discarding it down
    to a bare `Path`. Used by raobs_parquet.py to anchor HRDR
    (`drift_time_hours`) to the cycle it's actually relative to, without
    needing a new archive column for it (PR #29's review)."""
    cycle_directory = Path(cycle_directory)
    return datetime.strptime(
        cycle_directory.parent.name+' '+cycle_directory.name, '%Y-%m-%d %H'
    )


def class_file(cycle_directory, table, prepbufr_class):
    path = cycle_directory / table / str(prepbufr_class) / 'part-00000.parquet'
    if not path.is_file():
        raise FileNotFoundError(
            'PREPBUFR parquet table partition not found: '+str(path)
        )
    return path


def read_partition(path, columns):
    parquet_file = parquet().ParquetFile(path)
    schema_names = set(parquet_file.schema_arrow.names)
    missing = set(columns) - schema_names
    if missing:
        raise ValueError(
            'Missing required PREPBUFR parquet columns in '+str(path)+': '
            +str(sorted(missing))
        )
    return parquet_file.read(columns=list(columns))


def available_columns(path):
    """Column names present in a parquet partition's schema, without
    reading any row data -- lets a caller request an OPTIONAL column
    (e.g. raobs_parquet.py's drift_time_hours/drift_longitude/
    drift_latitude) only if it actually exists, instead of
    `read_partition` raising on the whole read because one archive
    without it is missing a column every other required one still needs
    (PR #29's review)."""
    return set(parquet().ParquetFile(path).schema_arrow.names)


def column(table, name, dtype=np.float64):
    return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=dtype)


def timestamps(table, name='observation_time'):
    return np.asarray(
        table[name].to_numpy(zero_copy_only=False)
    ).astype('datetime64[s]')


# PREPBUFR writes a handful of generic placeholder station IDs, not unique
# per physical platform: 'MASKSTID' for reports whose true identity is
# masked (restricted/proprietary mesonet feeds) and 'SHIP' for vessels with
# no assigned call sign. Treating either like a real station_id collapses
# many genuinely distinct reports into one in the per-station/per-slot
# dedup below -- confirmed against the real archive (one 2015-01-01 00Z
# cycle has 2556 of 19653 raw SFCSHP reports sharing the literal id
# 'MASKSTID', at clearly different lat/lon each) -- see
# docs/integrate_multiple_parquetobs.md's obs-count-gap investigation.
_GENERIC_STATION_IDS = frozenset({'MASKSTID', 'SHIP'})


def station_key(station_id, latitude, longitude, fallback):
    station_id = '' if station_id is None else str(station_id).strip()
    if station_id and station_id not in _GENERIC_STATION_IDS:
        return 'sid:'+station_id
    if np.isfinite(latitude) and np.isfinite(longitude):
        return 'location:'+format(latitude, '.4f')+':'+format(longitude, '.4f')
    return 'report:'+str(fallback)


def quality_passes(value, maximum):
    if maximum is None:
        return True
    return np.isfinite(value) and value <= float(maximum)


VIRTMP_PROGRAM_CODE = 8
_EVENT_COLUMNS = ['report_id', 'level_index', 'value_mnemonic', 'event_index', 'value', 'program_code']


def sensible_temperature(cycle_directory, prepbufr_class, report_ids, level_indices, temperature_c):
    """Sensible temperature (deg C) for the levels' current TOB values.

    The archive's `temperature_c` is the current (top-of-stack) TOB. Wherever
    moisture was reported, that is virtual temperature, written by PREPBUFR's
    VIRTMP step (program code 8). This returns, for each (report_id,
    level_index), the first event in the stack whose program is not VIRTMP.
    A level whose ONLY TOB event(s) are VIRTMP has no recoverable sensible
    value and comes back NaN, so it gets excluded downstream like any other
    missing value -- the input value is kept as a fallback only for a level
    with no TOB events in the archive at all (PR #29's review: falling back
    to the input there previously let a virtual temperature be assimilated
    as sensible). Values that are NaN on input stay NaN.
    """
    events = read_partition(class_file(cycle_directory, 'events', prepbufr_class), _EVENT_COLUMNS)
    mnemonic = np.asarray(events['value_mnemonic'].to_pylist())
    keep = mnemonic == 'TOB'
    rid_all = column(events, 'report_id', np.int64)[keep]
    lev_all = column(events, 'level_index', np.int64)[keep]
    idx_all = column(events, 'event_index', np.int64)[keep]
    val_all = column(events, 'value')[keep]
    prog_all = column(events, 'program_code')[keep]
    has_tob_event = set(zip(rid_all.tolist(), lev_all.tolist()))
    ok = prog_all != VIRTMP_PROGRAM_CODE
    rid, lev, idx, val = rid_all[ok], lev_all[ok], idx_all[ok], val_all[ok]
    order = np.lexsort((idx, lev, rid))
    rid, lev, val = rid[order], lev[order], val[order]
    first = np.ones(len(rid), bool)
    first[1:] = (rid[1:] != rid[:-1]) | (lev[1:] != lev[:-1])
    sensible = dict(zip(zip(rid[first].tolist(), lev[first].tolist()), val[first].tolist()))
    out = np.asarray(temperature_c, dtype=np.float64).copy()
    for i, key in enumerate(zip(np.asarray(report_ids).tolist(), np.asarray(level_indices).tolist())):
        if np.isfinite(out[i]):
            if key in sensible:
                out[i] = sensible[key]
            elif key in has_tob_event:
                out[i] = np.nan
    return out
