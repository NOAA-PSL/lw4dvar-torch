"""Shared low-level readers for the normalized, cycle-partitioned PREPBUFR
parquet archive (`<root>/<YYYY-MM-DD>/<HH>/{reports,levels}/<CLASS>/
part-00000.parquet`) -- the pieces `psobs_parquet.py` and `raobs_parquet.py`
both need (cycle-directory resolution, partition reads, station-key/QC
helpers). See integrate_prepbufr_obs.md for the archive layout and where this is
ported from (`greg-long-window-4dvar`'s `prepbufr_parquet.py`).

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


def cycle_directories(directory, verification_times, tolerance_hours):
    """Return the six-hourly archive cycle directories spanning the window."""
    directory = Path(os.path.expanduser(directory)).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            'PREPBUFR parquet directory not found: '+str(directory)
        )
    tolerance = timedelta(hours=float(tolerance_hours))
    cycles = set()
    for value in verification_times:
        center = to_datetime(value)
        start = center - tolerance
        end = center + tolerance
        cycle = start.replace(
            hour=(start.hour // 6) * 6, minute=0, second=0, microsecond=0
        )
        if cycle < start:
            cycle += timedelta(hours=6)
        while cycle <= end:
            cycles.add(cycle)
            cycle += timedelta(hours=6)
    paths = []
    for cycle in sorted(cycles):
        path = directory / cycle.strftime('%Y-%m-%d') / cycle.strftime('%H')
        if not path.is_dir():
            raise FileNotFoundError(
                'PREPBUFR parquet cycle directory not found: '+str(path)
            )
        paths.append(path)
    return paths


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


def column(table, name, dtype=np.float64):
    return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=dtype)


def timestamps(table, name='observation_time'):
    return np.asarray(
        table[name].to_numpy(zero_copy_only=False)
    ).astype('datetime64[s]')


def station_key(station_id, latitude, longitude, fallback):
    station_id = '' if station_id is None else str(station_id).strip()
    if station_id:
        return 'sid:'+station_id
    if np.isfinite(latitude) and np.isfinite(longitude):
        return 'location:'+format(latitude, '.4f')+':'+format(longitude, '.4f')
    return 'report:'+str(fallback)


def quality_passes(value, maximum):
    if maximum is None:
        return True
    return np.isfinite(value) and value <= float(maximum)
