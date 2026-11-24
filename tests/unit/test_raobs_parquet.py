"""load_raobs_parquet: the normalized-PREPBUFR-parquet radiosonde reader
(raobs_parquet.py/prepbufr_parquet_common.py). Builds a tiny, hand-crafted
archive per test (not tests/data/mini_ace2's fixture) so each check has an
independent, known-by-construction expected result."""

from datetime import datetime, timedelta

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
    'background_temperature_c', 'background_pressure_mb', 'drift_time_hours',
    'drift_longitude', 'drift_latitude',
]
# Nullable-float columns: archive rows legitimately have no source error --
# force float64 (not pyarrow's type inference, which would call an
# all-None column `null`-typed and break downstream float conversion).
_FLOAT_COLUMNS = {
    'temperature_error_c', 'wind_error_ms', 'humidity_error_tenths_percent',
    'background_temperature_c', 'background_pressure_mb', 'drift_time_hours',
    'drift_longitude', 'drift_latitude',
}
EVENT_COLUMNS = ['report_id', 'level_index', 'value_mnemonic', 'event_index', 'value', 'program_code']


def write_cycle(root, cycle_time, reports, levels, events=(), prepbufr_class='ADPUPA', omit_level_columns=()):
    """`reports`/`levels` are lists of dicts with exactly REPORT_COLUMNS/
    LEVEL_COLUMNS keys (observation_time as a `datetime`). `events` (TOB
    event-stack rows, for the sensible-temperature lookup) defaults to
    empty -- common.sensible_temperature falls back to the input
    temperature_c when no event matches, so most tests don't need any.
    `omit_level_columns`: column names to leave entirely out of the
    written levels schema (not just null-valued) -- simulates an archive
    that never had them at all, e.g. one predating drift_time_hours/
    drift_longitude/drift_latitude (PR #29's review)."""
    cycle_dir = root / f'{cycle_time:%Y-%m-%d}' / f'{cycle_time:%H}'
    for table_name, columns, rows in (
        ('reports', REPORT_COLUMNS, reports), ('levels', LEVEL_COLUMNS, levels),
    ):
        path = cycle_dir / table_name / prepbufr_class / 'part-00000.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = columns if table_name != 'levels' else [c for c in columns if c not in omit_level_columns]
        arrays = {}
        for col in cols:
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
    # 15 C (288.15 K): physically VALID by default (within the 195-327 K
    # physical_bounds check), so a test that doesn't care about
    # temperature and never overrides it isn't also, silently, triggering
    # a physical_bounds rejection of its own -- the old 300. default
    # (written straight to temperature_c, so 573.15 K) failed that check
    # unconditionally, meaning every height/below-station/quality-mark/
    # cross-slot-height-order rejection test using this helper without an
    # explicit temperature= would still "pass" even if ITS intended QC
    # mechanism were broken, since physical_bounds would reject it
    # regardless (PR #29's review). Tests that specifically want an
    # impossible temperature (the physical_bounds tests themselves) set
    # it explicitly.
    temperature=15., temperature_quality=1., u=5., v=-5., wind_quality=1.,
    q=5000., humidity_quality=1., level_index=0,
    temperature_error=None, wind_error=None,
    humidity_error_tenths_percent=None, background_temperature=None, background_pressure=None,
    drift_time_hours=None, drift_longitude=None, drift_latitude=None,
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
        'drift_time_hours': drift_time_hours,
        'drift_longitude': drift_longitude, 'drift_latitude': drift_latitude,
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


def test_non_positive_pressure_rejected(tmp_path):
    # A non-positive reported pressure is physically meaningless and
    # would reach np.log in 'log-linear' mode with an invalid argument
    # (PR #29's review) -- rejected up front, same as a non-finite one.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=0., temperature=10.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], vertical_interpolation='nearest')
    assert data['mask_t'][0].shape[-1] == 0  # no station: its only level was rejected outright


def test_level_outside_pressure_match_tolerance_rejected(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=849.9)])
    data = load(tmp_path, [t0], {'t': [1000., 850., 500.]}, ['t'], pressure_match_tolerance_hpa=0.05)
    assert not data['mask_t'][0].any()
    assert 'pressure_t' not in data  # 'exact' mode never returns the real-pressure field


@pytest.mark.parametrize('vertical_interpolation', ['nearest', 'log-linear'])
def test_relaxed_modes_keep_level_outside_exact_tolerance(tmp_path, vertical_interpolation):
    t0 = datetime(2015, 1, 1, 0)
    # temperature=26.85 (-> 300 K): level()'s own default (300., read as
    # Celsius -> 573.15 K) fails the physical-bounds QC (195-327 K),
    # unrelated to this test.
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=849.9, temperature=26.85)])
    data = load(
        tmp_path, [t0], {'t': [1000., 850., 500.]}, ['t'],
        pressure_match_tolerance_hpa=0.05, vertical_interpolation=vertical_interpolation,
    )
    assert data['mask_t'][0][1, 0]  # snapped to 850 hPa (index 1) despite being outside tolerance
    assert data['pressure_t'][0][1, 0] == pytest.approx(849.9)  # real pressure retained


def test_exact_mode_does_not_populate_pressure_field(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850.0, temperature=26.85)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], vertical_interpolation='exact')
    assert data['mask_t'][0][0, 0]
    assert 'pressure_t' not in data


@pytest.mark.parametrize('vertical_interpolation', ['nearest', 'log-linear'])
def test_relaxed_modes_distance_tiebreak_prefers_closer_duplicate(tmp_path, vertical_interpolation):
    """Two distinct raw levels both nearest to the same model level (850 hPa,
    between 1000/500) -- the physically closer one should win when both are
    otherwise QC-equal (irrelevant for 'exact', where the tight tolerance
    already makes this moot)."""
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)],
        [
            # heights must increase as pressure drops (830 < 860) or the
            # profile-wide non-monotonic-height QC rejects both -- these
            # are two genuinely different real levels now visible to that
            # check (PR #29's review), not just competing for one bucket.
            level(1, pressure=830., height=1600., temperature=10., level_index=0),  # 20 hPa from 850
            level(1, pressure=860., height=1400., temperature=20., level_index=1),  # 10 hPa from 850 -- closer
        ],
    )
    data = load(
        tmp_path, [t0], {'t': [1000., 850., 500.]}, ['t'],
        vertical_interpolation=vertical_interpolation,
    )
    assert data['mask_t'][0][1, 0]
    assert data['values_t'][0][1, 0] == pytest.approx(293.15)  # 20 C -> K: the 860 hPa duplicate won
    assert data['pressure_t'][0][1, 0] == pytest.approx(860.)


def test_duplicate_selection_prefers_physically_valid_value(tmp_path):
    # Two raw rows compete for the same matched model level (same
    # pressure, 850 hPa): one has the BETTER quality mark but an
    # impossible value (300 C -> 573 K), the other a worse-but-still-
    # passing quality mark with a realistic value. Ranking by quality
    # mark alone (the old behavior) would select the impossible one,
    # which then fails physical-bounds QC downstream -- losing the
    # usable duplicate entirely. The fix folds physical-value eligibility
    # into duplicate selection itself (PR #29's review).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)],
        [
            level(1, pressure=850., temperature=300., temperature_quality=1, level_index=0),
            level(1, pressure=850., temperature=15., temperature_quality=2, level_index=1),
        ],
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['mask_t'][0][0, 0]
    assert data['values_t'][0][0, 0] == pytest.approx(15. + 273.15)


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
    # reported; the LOWEST-event_index event ABOVE the VIRTMP event's own
    # position is the sensible value common.sensible_temperature must
    # return instead. Multiple non-VIRTMP candidates, given in shuffled
    # input order with distinct event_index values, confirm this picks by
    # stack order -- not input-list order and not value magnitude.
    # VIRTMP sits at the LOWEST index (0, i.e. most recent) here --
    # confirmed the realistic/common arrangement against real archive
    # data (PR #29's review); see
    # test_sensible_temperature_skips_event_above_virtmp_too below for the
    # real-world case where a later, non-VIRTMP event sits even more
    # recently than VIRTMP itself.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=27.0)],
        events=[
            event(1, value=99.0, event_index=3, program_code=0),  # non-VIRTMP, but not first above VIRTMP
            event(1, value=50.0, event_index=2, program_code=0),  # non-VIRTMP, but not first above VIRTMP
            event(1, value=26.0, event_index=1, program_code=0),  # non-VIRTMP, FIRST above VIRTMP -- the answer
            event(1, value=27.0, event_index=0, program_code=8),  # VIRTMP-corrected (matches levels' temperature_c)
        ],
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['values_t'][0][0, 0] == pytest.approx(26.0 + 273.15)


def test_sensible_temperature_skips_event_above_virtmp_too(tmp_path):
    # Real example from the archive (2015-01-01 00Z, ADPUPA, report_id=621,
    # level_index=14 -- one of 239 such cases found in that one cycle):
    # VIRTMP is NOT always at event_index 0. A later, non-VIRTMP event
    # (program_code 10 here) can sit ABOVE VIRTMP in the stack, carrying
    # VIRTMP's own (still virtual) value forward unchanged. A filter that
    # only excludes `program_code == VIRTMP_PROGRAM_CODE` would incorrectly
    # pick that event up as "sensible" since it isn't itself tagged VIRTMP
    # -- common.sensible_temperature must instead exclude everything at or
    # above VIRTMP's own event_index, regardless of program_code, landing
    # on the real sensible value one level further up the stack.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=14.7)],
        events=[
            event(1, value=14.7, event_index=0, program_code=10),  # NOT VIRTMP, but still virtual-valued
            event(1, value=14.7, event_index=1, program_code=8),   # VIRTMP-corrected (matches levels' temperature_c)
            event(1, value=14.6, event_index=2, program_code=1),   # the real sensible value -- the answer
        ],
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['values_t'][0][0, 0] == pytest.approx(14.6 + 273.15)


def test_sensible_temperature_falls_back_to_input_when_no_event(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=10.0)], events=[])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['values_t'][0][0, 0] == pytest.approx(10.0 + 273.15)


def test_sensible_temperature_nan_when_only_virtmp_event(tmp_path):
    # Every TOB event for this level is VIRTMP (program_code 8) -- there
    # is no recoverable sensible value, so sensible_temperature must
    # return NaN (not silently keep the virtual temperature as if it
    # were sensible) and the level must be excluded like any other
    # missing value (PR #29's review).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=27.0)],
        events=[
            event(1, value=27.0, program_code=8),  # VIRTMP-corrected -- the ONLY event
        ],
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['mask_t'][0].shape[-1] == 0  # no station at all: its only level was NaN'd out


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


def test_humidity_pressure_floor_checked_against_real_pressure_not_nominal_level(tmp_path):
    # 'nearest' mode snaps a 290 hPa report to the single 300 hPa model
    # level -- checking the NOMINAL level (300, not itself below the
    # 300 hPa floor) would wrongly keep this observation; the candidate's
    # own REAL reported pressure (290, below the floor) must exclude it
    # (PR #29's review). With its only candidate excluded, the record has
    # no retained value at all and is dropped entirely (PR #29's review,
    # second round) rather than consuming a station slot with an all-
    # False mask.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=290., q=4000.)])
    data = load(
        tmp_path, [t0], {'q': [300.]}, ['q'],
        humidity_min_pressure_hpa=300., vertical_interpolation='nearest',
    )
    assert data['mask_q'][0].shape[-1] == 0  # no station at all landed in this slot


def test_humidity_pressure_floor_excluded_candidate_cannot_displace_a_valid_one(tmp_path):
    # Both rows snap to the same 300 hPa model level under 'nearest'. The
    # below-floor row has the BETTER quality mark -- if the floor isn't
    # applied until after duplicate selection (using the nominal level,
    # which never excludes anything here), it wins the slot and the
    # above-floor row that should have been kept is lost (PR #29's review).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)],
        [
            level(1, pressure=290., q=4000., humidity_quality=1, level_index=0),
            level(1, pressure=310., q=3000., humidity_quality=2, level_index=1),
        ],
    )
    data = load(
        tmp_path, [t0], {'q': [300.]}, ['q'],
        humidity_min_pressure_hpa=300., vertical_interpolation='nearest',
    )
    assert data['mask_q'][0][0, 0]
    assert data['values_q'][0][0, 0] == pytest.approx(3000. / 1.e6)


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


def test_two_launches_from_same_station_stay_separate_profiles(tmp_path):
    # Two DIFFERENT launches (report_id 1 and 2, one hour apart) from the
    # SAME physical station landing in the same slot: a bare station_key
    # would merge them into one profile, letting their levels compete/
    # replace each other and corrupting the per-profile height-order QC
    # with levels from different soundings. Including each report's own
    # launch time in the profile identity keeps them as two separate
    # stations (PR #29's review).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0,
        [
            report(1, time=t0, station_id='SAME', lat=10., lon=20.),
            report(2, time=t0 + timedelta(hours=1), station_id='SAME', lat=10., lon=20.),
        ],
        [level(1, pressure=850., temperature=10.), level(2, pressure=850., temperature=20.)],
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['lat'][0].size == 2


def test_default_q_error_rh_percent_times_static_qsat(tmp_path):
    # GSI/UFO's real q-error mechanism (RH_fraction * qsat(background),
    # per jswhit's PR #29 review comment) -- the table is interpolated
    # (ln p) to the level's own real pressure.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(
        1, pressure=850., background_temperature=20., background_pressure=1000.,
    )])
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'], q_error_rh_percent={1000.: 20., 500.: 10.})
    import math  # 850 hPa sits between the table's two entries, interpolated in ln(p)
    tp = [math.log(500.), math.log(1000.)]
    tv = [10., 20.]
    expected_rh_fraction = np.interp(math.log(850.), tp, tv) / 100.
    expected = expected_rh_fraction * _saturation_specific_humidity(293.15, 1000.)
    assert data['default_error_q'][0][0, 0] == pytest.approx(expected, rel=1e-6)
    assert expected > 0.


def test_default_q_error_absent_without_table(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850.)])
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'])
    assert 'default_error_q' not in data


def test_duplicate_q_selection_prefers_usable_error_over_quality_mark(tmp_path):
    # Two rows compete for the same matched model level (850 hPa): one
    # has the BETTER quality mark but a missing archived background (so
    # its computed default_error_q is NaN -- unusable), the other a
    # worse-but-passing quality mark with a valid background. Ranking by
    # quality mark alone (the old behavior) would select the unusable
    # one, which the forward operator then rejects outright via its own
    # isfinite(effective_error) guard -- losing the usable duplicate
    # entirely (PR #29's review).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0)],
        [
            level(1, pressure=850., q=4000., humidity_quality=1, level_index=0,
                  background_temperature=None, background_pressure=None),
            level(1, pressure=850., q=3000., humidity_quality=2, level_index=1,
                  background_temperature=20., background_pressure=1000.),
        ],
    )
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'], q_error_rh_percent={1000.: 20., 500.: 10.})
    assert data['mask_q'][0][0, 0]
    assert data['values_q'][0][0, 0] == pytest.approx(3000. / 1.e6)
    assert np.isfinite(data['default_error_q'][0][0, 0])


def test_physical_bounds_qc_rejects_impossible_temperature(tmp_path):
    # UFO/GSI's sonde bounds check (PR #29's review comment): T 195-327 K.
    # temperature is Celsius (the archive column's own convention) -- 300
    # C -> 573 K, far outside any physically real atmospheric value.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=300.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert not data['mask_t'][0][0, 0]
    assert data['values_t'][0][0, 0] != 0.  # value still recorded, just flagged unused


def test_physical_bounds_qc_accepts_realistic_temperature(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=15.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert data['mask_t'][0][0, 0]


def test_physical_bounds_qc_rejects_impossible_humidity(tmp_path):
    # q 1e-8-0.035 kg/kg; q is specific_humidity_mg_kg (archive units,
    # 1e6x) -- 50000 mg/kg -> 0.05 kg/kg, above the upper bound.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., q=50000.)])
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'])
    assert not data['mask_q'][0][0, 0]


def test_physical_bounds_qc_rejects_impossible_wind(tmp_path):
    # |u|, |v| <= 130 m/s.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., u=200., v=-5.)])
    data = load(tmp_path, [t0], {'u': [850.], 'v': [850.]}, ['u', 'v'])
    assert not data['mask_u'][0][0, 0]
    assert data['mask_v'][0][0, 0]  # v (-5 m/s) is unaffected by u's own rejection


def test_physical_bounds_override_widens_a_variable(tmp_path):
    # 573 K (300 C) fails the default T bound (195-327 K) -- test_physical_
    # bounds_qc_rejects_impossible_temperature above -- but an explicit,
    # wider override for 't' alone must rescue it.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=300.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], physical_bounds={'t': (100., 600.)})
    assert data['mask_t'][0][0, 0]


def test_physical_bounds_override_replaces_wholesale_at_this_layer(tmp_path):
    # load_raobs_parquet itself does no per-variable merging with the
    # default table (that happens one layer up, in get_raobs -- see
    # test_config.py's test_get_raobs_physical_bounds_merged_per_variable)
    # -- a variable not present in an explicitly-given dict gets NO bound
    # check at all here, same as q_error_rh_percent's own "replace
    # wholesale, don't merge" contract.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., q=50000.)])  # 0.05 kg/kg
    data = load(tmp_path, [t0], {'q': [850.]}, ['q'], physical_bounds={'t': (100., 600.)})
    assert data['mask_q'][0][0, 0]  # no 'q' entry in the given dict -> unchecked here


def test_physical_bounds_override_narrows_a_variable(tmp_path):
    # A narrower override must reject a value the default would accept.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [level(1, pressure=850., temperature=15.)])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], physical_bounds={'t': (300., 327.)})
    assert not data['mask_t'][0][0, 0]


def test_use_drift_time_assigns_each_level_to_its_own_slot(tmp_path):
    # One report (observation_time=t0), two levels: level 0 at drift 0h
    # (stays at t0's own slot), level 1 at drift +6h (the balloon reaching
    # 500 hPa six hours after launch -- a real, if deep, sounding) -- with
    # use_drift_time, level 1 moves to the t0+6h slot instead of staying
    # at t0's, confirming one profile's levels can split across slots.
    t0, t1 = datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 6)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=850., height=1500., temperature=-10., level_index=0, drift_time_hours=0.0),
        level(1, pressure=500., height=5500., temperature=-10., level_index=1, drift_time_hours=6.0),
    ])
    write_cycle(tmp_path, t1, [], [])  # empty -- just needs the cycle directory to exist
    data = load(tmp_path, [t0, t1], {'t': [850., 500.]}, ['t'], use_drift_time=True)
    # slot 0 (t0): only the 850 hPa level (level_index 0) is present.
    assert data['mask_t'][0][0, 0]
    assert not data['mask_t'][0][1, 0]
    # slot 1 (t0+6h): only the 500 hPa level (level_index 1) is present.
    assert not data['mask_t'][1][0, 0]
    assert data['mask_t'][1][1, 0]


def test_use_drift_time_false_keeps_whole_profile_in_one_slot(tmp_path):
    # Same archive as above, but use_drift_time defaults to False --
    # every level uses the report's own single observation_time (t0) for
    # slot assignment, ignoring drift_time_hours entirely, so both levels
    # land in slot 0; nothing lands in slot 1 at all.
    t0, t1 = datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 6)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=850., height=1500., temperature=-10., level_index=0, drift_time_hours=0.0),
        level(1, pressure=500., height=5500., temperature=-10., level_index=1, drift_time_hours=6.0),
    ])
    write_cycle(tmp_path, t1, [], [])  # empty -- just needs the cycle directory to exist
    data = load(tmp_path, [t0, t1], {'t': [850., 500.]}, ['t'])
    assert data['mask_t'][0][0, 0]
    assert data['mask_t'][0][1, 0]
    assert data['mask_t'][1].shape[-1] == 0  # no station at all landed in slot 1


def test_non_monotonic_height_spanning_use_drift_time_slots_rejects_both(tmp_path):
    # One profile, split across two use_drift_time slots: level 0 (1000
    # hPa, height 500) stays in slot 0; level 1 (850 hPa, height 400 --
    # LOWER despite being at higher altitude, an inversion) drifts into
    # slot 1. Checked per-slot (the old behavior), neither slot has a
    # second level to compare against, so the inversion is invisible and
    # both wrongly pass. Checked on the COMPLETE profile (PR #29's
    # review), the inversion is caught and BOTH slots are rejected.
    t0, t1 = datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 6)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=1000., height=500., level_index=0, drift_time_hours=0.0),
        level(1, pressure=850., height=400., level_index=1, drift_time_hours=6.0),
    ])
    write_cycle(tmp_path, t1, [], [])  # empty -- just needs the cycle directory to exist
    data = load(tmp_path, [t0, t1], {'t': [1000., 850.]}, ['t'], use_drift_time=True)
    assert not data['mask_t'][0].any()  # slot 0 (1000 hPa level) rejected too
    assert not data['mask_t'][1].any()  # slot 1 (850 hPa level) rejected too


def test_use_drift_time_falls_back_to_report_time_when_missing(tmp_path):
    # drift_time_hours missing (None/NaN) for this level -- even with
    # use_drift_time=True, it must fall back to the report's own time,
    # not be dropped or mis-slotted.
    t0, t1 = datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 6)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=850., temperature=-10., drift_time_hours=None)])
    write_cycle(tmp_path, t1, [], [])  # empty -- just needs the cycle directory to exist
    data = load(tmp_path, [t0, t1], {'t': [850.]}, ['t'], use_drift_time=True)
    assert data['mask_t'][0][0, 0]
    assert data['mask_t'][1].shape[-1] == 0  # no station at all landed in slot 1


def test_use_drift_time_widens_partition_discovery_for_a_boundary_report(tmp_path):
    # A single verification time at 00Z with the default 3h tolerance:
    # the UN-drifted window [21:00, 03:00] ties to 00Z alone at the exact
    # 3h midpoint (_nearest_cycle's own tie-break favors the query's own
    # cycle) -- so the 18Z partition, which actually holds this report,
    # would never be opened without widening discovery for
    # use_drift_time. The report itself launched ON its own 18Z cycle
    # (DHR 0); +3.5h of drift_time_hours (cycle-relative HRDR -- anchored
    # to the 18Z cycle, not the report's own time, PR #29's review) puts
    # its one level at 21:30Z, inside the 3h window around 00Z.
    t0 = datetime(2015, 1, 1, 0)
    cycle = datetime(2014, 12, 31, 18)
    write_cycle(
        tmp_path, cycle, [report(1, time=cycle)],
        [level(1, pressure=850., drift_time_hours=3.5)],  # level()'s own default temperature is valid
    )
    write_cycle(tmp_path, t0, [], [])  # the 00Z cycle directory must exist too
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], use_drift_time=True)
    assert data['mask_t'][0][0, 0]


def test_use_drift_time_anchors_to_cycle_time_not_report_time(tmp_path):
    # HRDR ('drift_time_hours') is cycle-relative, the SAME referent as
    # the report's own DHR -- a per-level REPLACEMENT for the report's
    # DHR-based time, not an increment on top of it (PR #29's review:
    # anchoring to report_time instead double-counts DHR). A report
    # launched well off its own cycle hour (DHR +2.9h) with a small drift
    # (+0.2h) must land at cycle_time + drift (00:12, well inside the 3h
    # tolerance around the 00Z verification time), not report_time +
    # drift (03:06, just outside it) -- this test would fail under the
    # old (report_time-anchored) formula.
    cycle = datetime(2015, 1, 1, 0)
    launch = datetime(2015, 1, 1, 2, 54)  # DHR +2.9h
    write_cycle(
        tmp_path, cycle, [report(1, time=launch)],
        [level(1, pressure=850., temperature=-10., drift_time_hours=0.2)],
    )
    data = load(tmp_path, [cycle], {'t': [850.]}, ['t'], use_drift_time=True)
    assert data['mask_t'][0][0, 0]  # cycle_time + drift (00:12) is inside the 3h window


def test_use_drift_time_falls_back_when_archive_has_no_drift_columns(tmp_path):
    # use_drift_time/use_drift_position default to True in get_raobs
    # (long_window_4dvar_utils.py), so an archive that simply predates
    # drift_time_hours/drift_longitude/drift_latitude -- not just missing
    # a VALUE, missing the COLUMN entirely, e.g. tests/data/mini_ace2/
    # raobs -- must still load via the documented launch-time/position
    # fallback instead of common.read_partition raising on the whole
    # levels read over one missing optional column (PR #29's review).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(
        tmp_path, t0, [report(1, time=t0, lat=10., lon=20.)],
        [level(1, pressure=850., temperature=-10.)],
        omit_level_columns=('drift_time_hours', 'drift_longitude', 'drift_latitude'),
    )
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], use_drift_time=True, use_drift_position=True)
    assert data['mask_t'][0][0, 0]
    assert data['lat_t'][0][0, 0] == pytest.approx(10.)  # launch position, not drift (column absent)
    assert data['lon_t'][0][0, 0] == pytest.approx(20.)


def test_use_drift_position_rejects_out_of_range_drift_latitude(tmp_path):
    # A finite but out-of-range drift latitude (garbage/corrupted data)
    # must not overwrite the already-validated launch coordinates --
    # same range check the launch latitude itself gets (PR #29's review).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0, lat=10., lon=20.)], [
        level(1, pressure=850., temperature=-10., drift_latitude=190., drift_longitude=20.05),
    ])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], use_drift_position=True)
    assert data['mask_t'][0][0, 0]
    assert data['lat_t'][0][0, 0] == pytest.approx(10.)  # launch fallback, not the bad drift value
    assert data['lon_t'][0][0, 0] == pytest.approx(20.)


def test_non_monotonic_height_visible_across_bucket_sharing_duplicates(tmp_path):
    # Two genuinely different real levels (830.5/829 hPa) both snap to
    # the single model level (830 hPa) in 'nearest' mode -- the 830.5 hPa
    # row wins that bucket for assimilation (0.5 hPa away vs. 1 hPa), but
    # the 829 hPa row's height must still be visible to the profile-wide
    # non-monotonic-height check: it's inverted (50 m, LOWER) relative to
    # a third, higher-pressure level (900 hPa, 100 m) that IS kept.
    # Before this fix, only the bucket's WINNING height ever reached the
    # check, hiding the losing candidate's inversion entirely (PR #29's
    # review, second round).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=900., height=100., level_index=0),
        level(1, pressure=830.5, height=2500., level_index=1),  # wins the 830 hPa bucket (closer)
        level(1, pressure=829., height=50., level_index=2),     # loses the bucket, height=50 < 100
    ])
    data = load(
        tmp_path, [t0], {'t': [900., 830.]}, ['t'],
        vertical_interpolation='nearest',
    )
    assert not data['mask_t'][0].any()  # the whole profile is rejected, not just the losing candidate


def test_non_monotonic_height_visible_when_sibling_drifts_outside_every_slots_window(tmp_path):
    # One profile: level A (850 hPa) drifts negligibly and lands in the
    # only slot; level B (500 hPa) drifts +5h, outside every slot's 3h
    # tolerance, so it is never itself assimilated. B's height (1000 m)
    # is LOWER than A's (1500 m) despite being at a much higher altitude
    # (lower pressure) -- a real inversion. Before this fix, B's height
    # never reached the profile-wide completeness check at all (the time
    # cutoff discarded the whole level before any QC bookkeeping ran),
    # so A sailed through even though the complete sounding it came from
    # has a genuine inversion (PR #29's review, third round on this
    # mechanism).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=850., height=1500., temperature=-10., level_index=0, drift_time_hours=0.0),
        level(1, pressure=500., height=1000., temperature=-10., level_index=1, drift_time_hours=5.0),
    ])
    data = load(tmp_path, [t0], {'t': [850., 500.]}, ['t'], use_drift_time=True)
    # A (850 hPa) is the only level ever assimilated (B drifted outside
    # the window) -- its whole station must still be rejected.
    assert not data['mask_t'][0].any()


def test_non_monotonic_height_visible_when_sibling_fails_exact_mode_tolerance(tmp_path):
    # Copilot's review example: the model has only one 't' level (850
    # hPa). Level A (850 hPa, 1500 m) matches it exactly and is kept for
    # assimilation under the default 'exact' mode. Level B (830 hPa,
    # 1000 m) is 20 hPa from the model's only level -- way outside
    # pressure_match_tolerance_hpa, so 'exact' mode discards it for
    # ASSIMILATION -- but B is at LOWER pressure than A, so B's height
    # must be HIGHER than A's, not lower: a real inversion. Before this
    # fix, the 'exact'-mode tolerance discard skipped the profile-wide
    # height-order bookkeeping entirely (the same class of gap already
    # fixed for time_accepted), so A sailed through even though the
    # complete sounding it came from has a genuine inversion.
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0)], [
        level(1, pressure=850., height=1500., temperature=27.0, level_index=0),
        level(1, pressure=830., height=1000., temperature=26.0, level_index=1),
    ])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    # A (850 hPa) is the only level ever assimilated (B fails the 'exact'
    # tolerance) -- its whole station must still be rejected.
    assert not data['mask_t'][0].any()


def test_use_drift_position_gives_each_level_its_own_position(tmp_path):
    # One report launched at (20, 10); two levels with distinct drift
    # positions -- 850 hPa barely drifted, 50 hPa drifted substantially
    # (a real deep sounding; see docs/integrate_multiple_parquetobs.md).
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0, lat=10., lon=20.)], [
        level(1, pressure=850., height=1500., temperature=26.85, level_index=0,
              drift_longitude=20.05, drift_latitude=10.02),
        level(1, pressure=50., height=20000., temperature=26.85, level_index=1,
              drift_longitude=19.60, drift_latitude=10.45),
    ])
    data = load(tmp_path, [t0], {'t': [850., 50.]}, ['t'], use_drift_position=True)
    assert data['mask_t'][0][0, 0] and data['mask_t'][0][1, 0]
    assert data['lat_t'][0][0, 0] == pytest.approx(10.02)
    assert data['lon_t'][0][0, 0] == pytest.approx(20.05)
    assert data['lat_t'][0][1, 0] == pytest.approx(10.45)
    assert data['lon_t'][0][1, 0] == pytest.approx(19.60)


def test_use_drift_position_false_uses_launch_position_only(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0, lat=10., lon=20.)], [
        level(1, pressure=850., temperature=26.85, drift_longitude=19.60, drift_latitude=10.45),
    ])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'])
    assert 'lat_t' not in data
    assert 'lon_t' not in data


def test_use_drift_position_falls_back_to_launch_position_when_missing(tmp_path):
    t0 = datetime(2015, 1, 1, 0)
    write_cycle(tmp_path, t0, [report(1, time=t0, lat=10., lon=20.)], [
        level(1, pressure=850., temperature=26.85, drift_longitude=None, drift_latitude=None),
    ])
    data = load(tmp_path, [t0], {'t': [850.]}, ['t'], use_drift_position=True)
    assert data['mask_t'][0][0, 0]
    assert data['lat_t'][0][0, 0] == pytest.approx(10.)
    assert data['lon_t'][0][0, 0] == pytest.approx(20.)
