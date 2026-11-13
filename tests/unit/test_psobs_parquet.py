"""load_psobs_parquet: the normalized-PREPBUFR-parquet ps-obs reader
(psobs_parquet.py/prepbufr_parquet_common.py). Builds a tiny, hand-crafted
archive per test (not tests/data/mini_ace2's fixture) so each check has an
independent, known-by-construction expected result."""

from datetime import datetime

import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')

from psobs_parquet import load_psobs_parquet  # noqa: E402


REPORT_COLUMNS = [
    'report_id', 'observation_time', 'station_id', 'latitude',
    'longitude', 'elevation_m', 'report_type',
]
LEVEL_COLUMNS = [
    'report_id', 'level_index', 'pressure_mb', 'pressure_quality',
    'pressure_error_mb',
]


def write_cycle(root, cycle_time, reports, levels, prepbufr_class='ADPSFC'):
    """`reports`/`levels` are lists of dicts with exactly REPORT_COLUMNS/
    LEVEL_COLUMNS keys (observation_time as a `datetime`)."""
    cycle_dir = root / f'{cycle_time:%Y-%m-%d}' / f'{cycle_time:%H}'
    for table_name, columns, rows in (
        ('reports', REPORT_COLUMNS, reports), ('levels', LEVEL_COLUMNS, levels),
    ):
        path = cycle_dir / table_name / prepbufr_class / 'part-00000.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for col in columns:
            values = [row[col] for row in rows]
            arrays[col] = pa.array(
                values, type=pa.timestamp('s') if col == 'observation_time' else None
            )
        pq.write_table(pa.table(arrays), path)


def load(root, verification_times, **kwargs):
    """load_psobs_parquet, defaulted to the single class write_cycle writes
    by default (ADPSFC) -- the real archive's SFCSHP partition isn't needed
    for these hand-crafted fixtures."""
    kwargs.setdefault('prepbufr_classes', ('ADPSFC',))
    return load_psobs_parquet(root, verification_times, **kwargs)


def report(report_id, lat=10., lon=20., elev=0., report_type=181, time=None):
    return {
        'report_id': report_id, 'observation_time': time, 'station_id': f'S{report_id}',
        'latitude': lat, 'longitude': lon, 'elevation_m': elev, 'report_type': report_type,
    }


def level(report_id, pressure=1000., quality=1., error=0.8, level_index=0):
    return {
        'report_id': report_id, 'level_index': level_index, 'pressure_mb': pressure,
        'pressure_quality': quality, 'pressure_error_mb': error,
    }


def test_accepts_matching_report_within_tolerance(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=1013.2)])
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['ob'][0].tolist() == pytest.approx([1013.2])
    assert data['lat'][0].tolist() == pytest.approx([10.])
    assert data['lon'][0].tolist() == pytest.approx([20.])
    assert data['obtype'][0].tolist() == [181]


def test_negative_longitude_wrapped_to_0_360(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, lon=-30., time=t0)], [level(1)])
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['lon'][0].tolist() == pytest.approx([330.])


def test_report_outside_time_tolerance_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    late = datetime(2015, 1, 1, 4)  # 4h from the slot, tolerance is 3h
    write_cycle(tmp_path, t0, [report(1, time=late)], [level(1)])
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['ob'][0].size == 0


def test_report_type_filter(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, report_type=120, time=t0)], [level(1)])
    data = load(tmp_path, [t0], time_tolerance_hours=3., report_types=(180, 181, 187))
    assert data['ob'][0].size == 0


def test_quality_mark_filter(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, quality=5.)])
    data = load(tmp_path, [t0], time_tolerance_hours=3., quality_mark_max=3)
    assert data['ob'][0].size == 0


def test_non_surface_level_skipped(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, level_index=1)])
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['ob'][0].size == 0


def test_single_nearest_report_kept_per_station_per_slot(tmp_path):
    # Two reports from the same station within tolerance of the same slot:
    # only the one closer in time survives (legacy-.txt-like collapse).
    t0 = datetime(2015, 1, 1, 0)
    near = datetime(2015, 1, 1, 0, 30)
    far = datetime(2015, 1, 1, 2, 0)
    write_cycle(
        tmp_path, t0,
        [report(1, time=near), report(2, time=far)],
        [level(1, pressure=1001.), level(2, pressure=1099.)],
    )
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['ob'][0].size == 2  # different station_ids (S1, S2): not deduped
    # Now force the same station id to exercise the collapse itself.
    reports = [report(1, time=near), report(2, time=far)]
    for r in reports:
        r['station_id'] = 'SAME'
    write_cycle(tmp_path, t0, reports, [level(1, pressure=1001.), level(2, pressure=1099.)])
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['ob'][0].size == 1
    assert data['ob'][0].tolist() == pytest.approx([1001.])  # the nearer-in-time report


def test_generic_placeholder_station_id_not_deduped(tmp_path):
    # 'MASKSTID'/'SHIP' are PREPBUFR's generic non-unique placeholder ids
    # (see prepbufr_parquet_common.py's station_key) -- two reports sharing
    # one of these at clearly different locations are genuinely distinct
    # stations and must both survive the per-slot dedup, unlike a real
    # shared station_id (test_single_nearest_report_kept_per_station_per_slot).
    t0 = datetime(2015, 1, 1, 0)
    near = datetime(2015, 1, 1, 0, 30)
    far = datetime(2015, 1, 1, 2, 0)
    reports = [
        report(1, lat=10., lon=20., time=near),
        report(2, lat=-40., lon=200., time=far),
    ]
    for r in reports:
        r['station_id'] = 'MASKSTID'
    write_cycle(
        tmp_path, t0, reports,
        [level(1, pressure=1001.), level(2, pressure=1099.)],
    )
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['ob'][0].size == 2
    assert sorted(data['ob'][0].tolist()) == pytest.approx([1001., 1099.])


def test_flat_vs_source_error(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, error=0.37)])
    flat = load(tmp_path, [t0], time_tolerance_hours=3., observation_error_std_hpa=2.5)
    assert flat['oberr'][0].tolist() == pytest.approx([2.5])
    source = load(
        tmp_path, [t0], time_tolerance_hours=3., observation_error_std_hpa=2.5, use_source_error=True,
    )
    assert source['oberr'][0].tolist() == pytest.approx([0.37])


def test_missing_cycle_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path, [datetime(2015, 1, 1, 0)], time_tolerance_hours=3.)


def test_invalid_error_std_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1)])
    with pytest.raises(ValueError, match='observation_error_std_hpa'):
        load(tmp_path, [t0], time_tolerance_hours=3., observation_error_std_hpa=0.)


def test_multiple_slots_bin_independently(tmp_path):
    t0, t1 = datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 6)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=1000.)])
    write_cycle(tmp_path, t1, [report(1, time=t1)], [level(1, pressure=1020.)])
    data = load(tmp_path, [t0, t1], time_tolerance_hours=3.)
    assert [arr.size for arr in data['ob']] == [1, 1]
    assert data['ob'][0].tolist() == pytest.approx([1000.])
    assert data['ob'][1].tolist() == pytest.approx([1020.])


def test_candidate_rank_prefers_lower_quality_mark(tmp_path):
    # Two reports from the same station, tied on time-to-slot: the
    # dedup tie-break picks the better (lower) PREPBUFR quality mark.
    t0 = datetime(2015, 1, 1, 0)
    same_time = [report(1, time=t0), report(2, time=t0)]
    for r in same_time:
        r['station_id'] = 'DUP'
    write_cycle(
        tmp_path, t0, same_time,
        [level(1, pressure=1001., quality=3.), level(2, pressure=1002., quality=1.)],
    )
    data = load(tmp_path, [t0], time_tolerance_hours=3.)
    assert data['ob'][0].size == 1
    assert data['ob'][0].tolist() == pytest.approx([1002.])  # better (lower) quality mark wins the tie
