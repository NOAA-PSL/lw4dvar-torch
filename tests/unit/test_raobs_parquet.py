"""load_raobs_parquet: the normalized-PREPBUFR-parquet radiosonde reader
(raobs_parquet.py/prepbufr_parquet_common.py). Builds a tiny, hand-crafted
archive per test (not tests/data/mini_ace2's fixture) so each check has an
independent, known-by-construction expected result."""

from datetime import datetime

import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')

from raobs_parquet import load_raobs_parquet  # noqa: E402


REPORT_COLUMNS = [
    'report_id', 'observation_time', 'station_id', 'latitude',
    'longitude', 'elevation_m', 'report_type',
]
LEVEL_COLUMNS = [
    'report_id', 'level_index', 'category', 'pressure_mb', 'height_m',
    'height_quality', 'temperature_c', 'temperature_quality', 'u_wind_ms',
    'wind_quality', 'v_wind_ms', 'specific_humidity_mg_kg', 'humidity_quality',
]
EVENT_COLUMNS = ['report_id', 'level_index', 'value_mnemonic', 'event_index', 'value', 'program_code']


def write_cycle(root, cycle_time, reports, levels, events=(), prepbufr_class='ADPUPA'):
    """`reports`/`levels` are lists of dicts with exactly REPORT_COLUMNS/
    LEVEL_COLUMNS keys (observation_time as a `datetime`). `events` (TOB
    event-stack rows, for the sensible-temperature lookup) defaults to
    empty -- common.sensible_temperature falls back to the input
    temperature_c when no event matches, so most tests don't need any."""
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

    events_path = cycle_dir / 'events' / prepbufr_class / 'part-00000.parquet'
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events = list(events)
    event_arrays = {
        'report_id': pa.array([e['report_id'] for e in events], type=pa.int64()),
        'level_index': pa.array([e['level_index'] for e in events], type=pa.int64()),
        'value_mnemonic': pa.array([e['value_mnemonic'] for e in events], type=pa.string()),
        'event_index': pa.array([e['event_index'] for e in events], type=pa.int64()),
        'value': pa.array([e['value'] for e in events], type=pa.float64()),
        'program_code': pa.array([e['program_code'] for e in events], type=pa.int64()),
    }
    pq.write_table(pa.table(event_arrays), events_path)


def load(root, verification_times, levels_hpa, variables, **kwargs):
    kwargs.setdefault('prepbufr_classes', ('ADPUPA',))
    return load_raobs_parquet(root, verification_times, 3., levels_hpa, variables, **kwargs)


def report(report_id, lat=10., lon=20., elev=0., report_type=120, time=None, station_id=None):
    return {
        'report_id': report_id, 'observation_time': time,
        'station_id': station_id if station_id is not None else f'S{report_id}',
        'latitude': lat, 'longitude': lon, 'elevation_m': elev, 'report_type': report_type,
    }


def level(
    report_id, pressure=850., height=1500., height_quality=1., category=1,
    temperature=300., temperature_quality=1., u=5., v=-5., wind_quality=1.,
    q=5000., humidity_quality=1., level_index=0,
):
    return {
        'report_id': report_id, 'level_index': level_index, 'category': category,
        'pressure_mb': pressure, 'height_m': height, 'height_quality': height_quality,
        'temperature_c': temperature, 'temperature_quality': temperature_quality,
        'u_wind_ms': u, 'wind_quality': wind_quality, 'v_wind_ms': v,
        'specific_humidity_mg_kg': q, 'humidity_quality': humidity_quality,
    }


def event(report_id, value, level_index=0, event_index=0, program_code=0, mnemonic='TOB'):
    return {
        'report_id': report_id, 'level_index': level_index, 'value_mnemonic': mnemonic,
        'event_index': event_index, 'value': value, 'program_code': program_code,
    }


def test_mandatory_level_matched_within_tolerance(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850.02, temperature=26.85)])
    data = load(tmp_path, [t0], {'t': [1000., 850., 500.]}, ['t'])
    assert data['mask_t'][0][1, 0]  # 850 hPa is index 1
    assert data['values_t'][0][1, 0] == pytest.approx(300.0)  # 26.85 C -> K


def test_level_outside_pressure_match_tolerance_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=849.9)])
    data = load(tmp_path, [t0], {'t': [1000., 850., 500.]}, ['t'], pressure_match_tolerance_hpa=0.05)
    assert not data['mask_t'][0].any()


def test_unphysical_height_qc_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., height=70000.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert not data['mask_t'][0][0, 0]
    assert data['values_t'][0][0, 0] != 0.  # value is still recorded, just flagged unused


def test_below_station_elevation_qc_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0, elev=1000.)],
        [level(1, pressure=850., height=800.)],  # 200 m below station, tolerance is 100 m
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert not data['mask_t'][0][0, 0]


def test_non_monotonic_height_qc_rejects_whole_profile(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)],
        [
            level(1, pressure=1000., height=500., level_index=0),
            level(1, pressure=850., height=400., level_index=1),  # height must INCREASE as pressure drops
        ],
    )
    data = load(tmp_path, [t0], {'t': [1000., 850.]}, ['t'])
    assert not data['mask_t'][0].any()


def test_quality_mark_qc_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature_quality=5.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], quality_mark_max=3)
    assert not data['mask_t'][0][0, 0]


def test_sensible_temperature_used_instead_of_virtual(tmp_path):
    # The archive's temperature_c is the (virtual) TOB when moisture was
    # reported; the first non-VIRTMP (program_code != 8) event stack entry
    # is the sensible value common.sensible_temperature must return instead.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=27.0)],
        events=[
            event(1, value=26.0, program_code=0),   # sensible (pre-VIRTMP)
            event(1, value=27.0, program_code=8),    # VIRTMP-corrected (matches levels' temperature_c)
        ],
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['values_t'][0][0, 0] == pytest.approx(26.0 + 273.15)


def test_sensible_temperature_falls_back_to_input_when_no_event(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=10.0)], events=[])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['values_t'][0][0, 0] == pytest.approx(10.0 + 273.15)


def test_humidity_masked_above_pressure_floor(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)],
        [
            # heights must increase with altitude (decreasing pressure) or
            # the non-monotonic-height QC rejects the whole profile.
            level(1, pressure=500., height=5500., q=4000., level_index=0),
            level(1, pressure=200., height=11800., q=3000., level_index=1),
        ],
    )
    data = load(tmp_path, [t0], {'q': [500., 200.]}, ['q'], humidity_min_pressure_hpa=300.)
    assert data['mask_q'][0][0, 0]  # 500 hPa: below the floor (lower altitude), kept
    assert not data['mask_q'][0][1, 0]  # 200 hPa: above the floor, floor-masked
    assert data['values_q'][0][1, 0] == 0.  # floor-masked: never assigned, stays at the zero fill
    assert data['values_q'][0][0, 0] == pytest.approx(4000. / 1.e6)


def test_multiple_variables_and_slots(tmp_path):
    t0, t1 = datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 6)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=0., u=10., v=-10.)])
    write_cycle(tmp_path, t1, [report(1, time=t1)], [level(1, pressure=850., temperature=20., u=1., v=-1.)])
    data = load(tmp_path, [t0, t1], {'t': [850.], 'u': [850.], 'v': [850.]}, ['t', 'u', 'v'])
    assert data['values_t'][0][0, 0] == pytest.approx(273.15)
    assert data['values_u'][0][0, 0] == pytest.approx(10.)
    assert data['values_v'][0][0, 0] == pytest.approx(-10.)
    assert data['values_t'][1][0, 0] == pytest.approx(293.15)


def test_unsupported_variable_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1)])
    with pytest.raises(ValueError, match='Unsupported raobs variables'):
        load(tmp_path, [t0], {'bogus': [850.]}, ['bogus'])


def test_station_capacity_overflow_raises(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0,
        [report(1, time=t0, station_id='A'), report(2, time=t0, station_id='B')],
        [level(1, pressure=850.), level(2, pressure=850., level_index=0)],
    )
    with pytest.raises(ValueError, match='exceeding fixed capacity'):
        load(tmp_path, [t0], {'t': [850.]}, ['t'], station_capacity=1)
