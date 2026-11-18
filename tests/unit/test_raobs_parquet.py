"""load_raobs_parquet: the normalized-PREPBUFR-parquet radiosonde reader
(raobs_parquet.py/prepbufr_parquet_common.py). Builds a tiny, hand-crafted
archive per test (not tests/data/mini_ace2's fixture) so each check has an
independent, known-by-construction expected result."""

from datetime import datetime

import numpy as np
import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')

from raobs_parquet import load_raobs_parquet, _saturation_specific_humidity  # noqa: E402


REPORT_COLUMNS = [
    'report_id', 'observation_time', 'station_id', 'latitude',
    'longitude', 'elevation_m', 'report_type',
]
LEVEL_COLUMNS = [
    'report_id', 'level_index', 'category', 'pressure_mb', 'height_m',
    'height_quality', 'temperature_c', 'temperature_quality', 'u_wind_ms',
    'wind_quality', 'v_wind_ms', 'specific_humidity_mg_kg', 'humidity_quality',
    'temperature_error_c', 'wind_error_ms', 'humidity_error_tenths_percent',
    'background_temperature_c', 'background_pressure_mb',
]
# Nullable-float columns: archive rows legitimately have no source error --
# force float64 (not pyarrow's type inference, which would call an
# all-None column `null`-typed and break downstream float conversion).
_FLOAT_COLUMNS = {
    'temperature_error_c', 'wind_error_ms', 'humidity_error_tenths_percent',
    'background_temperature_c', 'background_pressure_mb',
}
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
            col_type = (
                pa.timestamp('s') if col == 'observation_time'
                else pa.float64() if col in _FLOAT_COLUMNS else None
            )
            arrays[col] = pa.array(values, type=col_type)
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
    temperature_error=None, wind_error=None,
    humidity_error_tenths_percent=None, background_temperature=None, background_pressure=None,
):
    return {
        'report_id': report_id, 'level_index': level_index, 'category': category,
        'pressure_mb': pressure, 'height_m': height, 'height_quality': height_quality,
        'temperature_c': temperature, 'temperature_quality': temperature_quality,
        'u_wind_ms': u, 'wind_quality': wind_quality, 'v_wind_ms': v,
        'specific_humidity_mg_kg': q, 'humidity_quality': humidity_quality,
        'temperature_error_c': temperature_error, 'wind_error_ms': wind_error,
        'humidity_error_tenths_percent': humidity_error_tenths_percent,
        'background_temperature_c': background_temperature, 'background_pressure_mb': background_pressure,
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


def test_use_source_error_disabled_omits_source_error_key(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, temperature_error=0.8, wind_error=1.4)])
    data = load(tmp_path, [t0], {'t': [850.], 'u': [850.]}, ['t', 'u'])
    assert 'source_error_t' not in data
    assert 'source_error_u' not in data


def test_use_source_error_read_and_fallback_when_missing(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0,
        [report(1, time=t0, station_id='A'), report(2, time=t0, station_id='B')],
        [
            level(1, pressure=850., temperature_error=0.8, wind_error=1.4),
            level(2, pressure=850., temperature_error=None, wind_error=None, level_index=0),
        ],
    )
    data = load(tmp_path, [t0], {'t': [850.], 'u': [850.]}, ['t', 'u'], use_source_error=True)
    # Both stations share lat/lon/elev defaults, so station order follows
    # station_key (sid-based): 'sid:S1' < 'sid:S2' -> index 0 is report 1.
    assert data['source_error_t'][0][0, 0] == pytest.approx(0.8)
    assert data['source_error_u'][0][0, 0] == pytest.approx(1.4)
    assert np.isnan(data['source_error_t'][0][0, 1])
    assert np.isnan(data['source_error_u'][0][0, 1])


def test_use_source_error_nonpositive_treated_as_missing(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature_error=0.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], use_source_error=True)
    assert np.isnan(data['source_error_t'][0][0, 0])


def test_use_source_error_wind_shared_between_u_and_v(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., wind_error=1.6)])
    data = load(tmp_path, [t0], {'u': [850.], 'v': [850.]}, ['u', 'v'], use_source_error=True)
    assert data['source_error_u'][0][0, 0] == pytest.approx(1.6)
    assert data['source_error_v'][0][0, 0] == pytest.approx(1.6)


def test_use_source_error_q_converted_via_static_background_qsat(tmp_path):
    # humidity_error_tenths_percent is percent/10 (confirmed against
    # convert_prepbufr_to_parquet.py's own field metadata): 2.0 -> 20% RH.
    # background_temperature is Celsius (the column's own name/convention,
    # matching background_temperature_c) -- 20 C == 293.15 K.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(
        1, pressure=850., humidity_error_tenths_percent=2.0,
        background_temperature=20., background_pressure=1000.,
    )])
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'], use_source_error=True)
    expected = 0.20 * _saturation_specific_humidity(293.15, 1000.)
    assert data['source_error_q'][0][0, 0] == pytest.approx(expected, rel=1e-6)
    assert expected > 0.  # sanity: a real kg/kg magnitude, not degenerate


def test_use_source_error_q_missing_background_is_nan(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850.)])  # no background_*/QOE
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'], use_source_error=True)
    assert np.isnan(data['source_error_q'][0][0, 0])


def test_use_source_error_t_not_clamped_to_gsi_bounds(tmp_path):
    # GSI's own ermin/ermax (1.3-5.6 K for T) only clamp the gross-check
    # comparison (long_window_4dvar_utils.py), never source_error_t itself
    # -- confirmed by reading GSI's setupq.f90 directly: the ermin/ermax-
    # clamped `obserror` there never feeds the actual Jo weight, only a
    # later diagnostic. A raw archive value outside that range (e.g. 0.8 K
    # at 850 hPa, the real archive's own mid-troposphere value) must come
    # through unmodified -- that's the whole point of reading the real
    # per-level table instead of a flat constant.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature_error=0.8)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], use_source_error=True)
    assert data['source_error_t'][0][0, 0] == pytest.approx(0.8)


def test_use_source_error_q_qsat_exposed_alongside_source_error(tmp_path):
    # source_qsat_q lets the gross check (long_window_4dvar_utils.py)
    # derive GSI's RH-percent ermin/ermax bounds in kg/kg later, without
    # the read-time conversion itself being clamped.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(
        1, pressure=850., humidity_error_tenths_percent=2.0,
        background_temperature=20., background_pressure=1000.,
    )])
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'], use_source_error=True)
    expected_qsat = _saturation_specific_humidity(293.15, 1000.)
    assert data['source_qsat_q'][0][0, 0] == pytest.approx(expected_qsat, rel=1e-6)


def test_use_source_error_q_qsat_nan_when_background_missing(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850.)])  # no background_*/QOE
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'], use_source_error=True)
    assert np.isnan(data['source_qsat_q'][0][0, 0])


def test_use_source_error_t_inflated_for_suspect_quality_mark(tmp_path):
    # GSI's x1.2 inflation for quality mark 3 or 7 (suspect, not
    # rejected) -- quality_mark_max defaults to 3, so this ob still
    # passes QC (mask_t True) while its error is inflated.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=500., temperature_error=1.0, temperature_quality=3)])
    data = load(tmp_path, [t0], {'t': [500.]}, ['t'], use_source_error=True)
    assert data['source_error_t'][0][0, 0] == pytest.approx(1.2)
    assert data['mask_t'][0][0, 0]


def test_use_source_error_t_not_inflated_for_good_quality_mark(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=500., temperature_error=1.0, temperature_quality=1)])
    data = load(tmp_path, [t0], {'t': [500.]}, ['t'], use_source_error=True)
    assert data['source_error_t'][0][0, 0] == pytest.approx(1.0)


def test_use_source_error_t_inflated_above_100hpa(tmp_path):
    # GSI's x1.2 inflation for T above 100 hPa (ppb < r100 in
    # read_prepbufr.F90) -- independent of quality mark.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=50., temperature_error=1.0, temperature_quality=1)])
    data = load(tmp_path, [t0], {'t': [50.]}, ['t'], use_source_error=True)
    assert data['source_error_t'][0][0, 0] == pytest.approx(1.2)


def test_use_source_error_t_inflation_factors_multiply(tmp_path):
    # Both triggers at once (qm=7, above 100 hPa) -> x1.2 x1.2 = x1.44.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=50., temperature_error=1.0, temperature_quality=7)])
    data = load(tmp_path, [t0], {'t': [50.]}, ['t'], use_source_error=True)
    assert data['source_error_t'][0][0, 0] == pytest.approx(1.0 * 1.2 * 1.2)


def test_use_source_error_wind_inflated_above_50hpa_not_t_threshold(tmp_path):
    # Wind's pressure threshold is 50 hPa, not T's 100 hPa -- 70 hPa is
    # below T's threshold but NOT below wind's, so wind must stay
    # uninflated there.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=70., wind_error=2.0, wind_quality=1)])
    data = load(tmp_path, [t0], {'u': [70.]}, ['u'], use_source_error=True)
    assert data['source_error_u'][0][0, 0] == pytest.approx(2.0)

    write_cycle(
        tmp_path / 'below50', datetime(2015, 1, 1, 0),
        [report(1, time=t0)], [level(1, pressure=30., wind_error=2.0, wind_quality=1)],
    )
    data2 = load(tmp_path / 'below50', [t0], {'u': [30.]}, ['u'], use_source_error=True)
    assert data2['source_error_u'][0][0, 0] == pytest.approx(2.0 * 1.2)


def test_use_source_error_q_inflated_for_suspect_quality_mark_no_pressure_trigger(tmp_path):
    # q gets the same quality-mark inflation as t/u/v, but GSI has no
    # pressure-based inflation for q -- even at a very low pressure
    # (here 10 hPa), only the quality-mark factor applies.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(
        1, pressure=10., humidity_error_tenths_percent=2.0, humidity_quality=3,
        background_temperature=20., background_pressure=1000.,
    )])
    data = load(tmp_path, [t0], {'q': [10.]}, ['q'], use_source_error=True)
    expected = 1.2 * 0.20 * _saturation_specific_humidity(293.15, 1000.)
    assert data['source_error_q'][0][0, 0] == pytest.approx(expected, rel=1e-6)
