"""Small solver helpers: obs-density Jo weights, the control-variable mask,
and the date utilities."""

import math
from datetime import timezone

import pytest
import torch

import long_window_4dvar_utils as utils


def density_weights(lats, lons, used, box_deg=10.0):
    f = lambda v: torch.tensor(v, dtype=torch.float32)  # noqa: E731
    return utils._obs_density_weights(torch.tensor(used), f(lats), f(lons), box_deg)


def test_density_weights_share_a_box():
    # two obs in one 10-deg box, one alone in another at the same latitude,
    # one QC-rejected, one padding slot (lat/lon = 1e10)
    w = density_weights([5.0, 6.0, 5.0, 5.0, 1e10], [10.0, 12.0, 100.0, 200.0, 1e10],
                        [True, True, True, False, False])
    torch.testing.assert_close(w, torch.tensor([0.75, 0.75, 1.5, 0.0, 0.0]))


def test_density_weights_area_factor():
    # isolated obs: weights follow the cosine of their box-centre latitude
    w = density_weights([5.0, 75.0], [0.0, 0.0], [True, True])
    assert float(w.sum()) == pytest.approx(2.0, rel=1e-6)
    ratio = math.cos(math.radians(75.0)) / math.cos(math.radians(5.0))
    assert float(w[1] / w[0]) == pytest.approx(ratio, rel=1e-5)


def test_density_weights_wrap_longitude():
    # -170 and 190 deg are the same place
    w = density_weights([5.0, 5.0], [-170.0, 190.0], [True, True])
    torch.testing.assert_close(w, torch.tensor([1.0, 1.0]))


def test_control_mask_channels_first():
    state = torch.ones(1, 3, 2, 2)
    ref = torch.zeros(1, 3, 2, 2)
    keep = torch.tensor([True, False, True])
    out = utils._apply_control_mask(state, keep, ref, 'channels_first')
    assert out[0, :, 0, 0].tolist() == [1.0, 0.0, 1.0]


def test_control_mask_channels_last():
    state = torch.ones(1, 1, 4, 3)
    ref = torch.zeros(1, 1, 4, 3)
    keep = torch.tensor([False, True, True])
    out = utils._apply_control_mask(state, keep, ref, 'channels_last')
    assert out[0, 0, 0].tolist() == [0.0, 1.0, 1.0]


def test_control_mask_unknown_layout():
    with pytest.raises(ValueError, match="state_layout"):
        utils._apply_control_mask(torch.ones(1), torch.tensor([True]), torch.zeros(1), 'nhwc')


@pytest.mark.parametrize("date, hours, expected", [
    ('2015-01-01T00', 6, '2015-01-01T06'),
    ('2015-12-31T18', 12, '2016-01-01T06'),
    ('2016-03-01T00', -24, '2016-02-29T00'),
])
def test_add_hours(date, hours, expected):
    assert utils.add_hours(date, hours) == expected


def test_get_yyyymmddhh():
    assert utils.get_YYYYMMDDHH('2015-01-02T18') == '2015010218'


@pytest.mark.parametrize("backend, aware", [('aifs', False), ('ace2', False), ('fcn3', True), ('sfno', True)])
def test_to_utc_datetime(backend, aware):
    dt = utils._to_utc_datetime('2015-01-02T18', backend)
    assert (dt.tzinfo is timezone.utc) == aware
    assert (dt.year, dt.month, dt.day, dt.hour) == (2015, 1, 2, 18)


def _jcz_case(seed=0, nlev=3, npts=50):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(nlev, npts, generator=g, dtype=torch.float64) * torch.tensor([[1.], [10.], [100.]], dtype=torch.float64)
    z1 = z0 + 0.1 * torch.randn(nlev, npts, generator=g, dtype=torch.float64)
    w = torch.rand(npts, generator=g, dtype=torch.float64)
    return z0, z1, w / w.sum()


def test_jc_z_penalty_matches_hand_computation():
    z0, z1, w = _jcz_case()
    expected = 0.0
    for k in range(z0.shape[0]):
        m = float((w * z0[k]).sum())
        sd = math.sqrt(float((w * (z0[k] - m) ** 2).sum()))
        expected += float((w * ((z1[k] - z0[k]) / sd) ** 2).sum()) / z0.shape[0]
    assert utils._jc_z_penalty(z0, z1, w, 7.0).item() == pytest.approx(7.0 * expected, rel=1e-12)


def test_jc_z_penalty_scale_invariant_and_zero_without_tendency():
    z0, z1, w = _jcz_case()
    j = utils._jc_z_penalty(z0, z1, w, 1.0)
    assert utils._jc_z_penalty(3.0 * z0, 3.0 * z1, w, 1.0).item() == pytest.approx(j.item(), rel=1e-12)
    assert utils._jc_z_penalty(z0, z0.clone(), w, 1.0).item() == 0.0


def test_jc_z_penalty_gradcheck():
    z0, z1, w = _jcz_case(npts=8)
    z0.requires_grad_(True)
    z1.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: utils._jc_z_penalty(a, b, w, 2.0), (z0, z1))


class _FakeRaobsGridInterp:
    """`grid_interp.interp(idx, wts, field)` fake for
    `_compute_raobs_observation_diagnostics_at_time`: returns a fixed
    background regardless of `idx`/`wts`, so a test can set up an exact
    O-B without needing real interpolation weights."""

    def __init__(self, background):
        self.background = background

    def interp(self, idx, wts, field):
        return self.background


def _raobs_traj_case(observation, error, tuv_bg_check):
    # one obs slot, one level, two stations -- (n_obs, n_levels, capacity)
    background = torch.zeros(1, 2)
    return {
        'variables': ['t'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_t': torch.as_tensor([[observation]], dtype=torch.float32),
        'mask_t': torch.as_tensor([[[True, True]]]),
        'error_t': error,
        'tuv_bg_check': tuv_bg_check,
    }, background


def test_raobs_tuv_bg_check_rejects_outlier_not_the_good_ob():
    # error_t=1.0, tuv_bg_check=4.0 -> reject |O-B| > 4.0; 3.0 passes, 5.0 doesn't
    raobs_traj, background = _raobs_traj_case([3.0, 5.0], error=1.0, tuv_bg_check=4.0)
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['t']
    assert used.tolist() == [[True, False]]


def test_raobs_tuv_bg_check_disabled_keeps_everything_mask_allows():
    raobs_traj, background = _raobs_traj_case([3.0, 5.0], error=1.0, tuv_bg_check=None)
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['t']
    assert used.tolist() == [[True, True]]


def test_raobs_tuv_bg_check_does_not_override_read_time_qc():
    # mask_t already False for the second station (read-time QC rejected
    # it) -- the bg check must not resurrect it even though its own |O-B|
    # would pass.
    raobs_traj, background = _raobs_traj_case([3.0, 3.0], error=1.0, tuv_bg_check=4.0)
    raobs_traj['mask_t'] = torch.as_tensor([[[True, False]]])
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['t']
    assert used.tolist() == [[True, False]]


def test_raobs_source_error_tensor_used_per_station():
    # error_t as a per-(obs, level, station) tensor (use_source_error: True
    # in get_raobs) instead of the flat float every other test here uses --
    # same |O-B|=3.0 at both stations, but station 0's tight source error
    # (0.5 -> 4-sigma threshold 2.0) fails the check while station 1's
    # looser one (2.0 -> threshold 8.0) passes it, proving effective_error
    # is actually read per-station, not broadcast from a single scalar.
    raobs_traj, background = _raobs_traj_case([3.0, 3.0], error=1.0, tuv_bg_check=4.0)
    raobs_traj['error_t'] = torch.as_tensor([[[0.5, 2.0]]])
    # Wide enough to not affect this test's own numbers -- real get_raobs
    # output always sets these whenever error_t is a tensor; see the GSI
    # ermin/ermax clamp tests below for their actual effect.
    raobs_traj['ermin_t'], raobs_traj['ermax_t'] = 0.01, 100.0
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, effective_error, used = results['t']
    assert effective_error.tolist() == [[0.5, 2.0]]
    assert used.tolist() == [[False, True]]


def test_raobs_source_error_q_overrides_fraction_error_when_finite():
    # q has no single flat fallback (unlike t/u/v) -- source_error_q
    # overrides the fraction-of-observed-value error per element, falling
    # back to it wherever the archive had no valid source value (NaN).
    background = torch.zeros(1, 2)
    raobs_traj = {
        'variables': ['q'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_q': torch.as_tensor([[[0.01, 0.01]]], dtype=torch.float32),
        'mask_q': torch.as_tensor([[[True, True]]]),
        'error_q_fraction': 0.2, 'error_q_floor': 1e-5,
        'source_error_q': torch.as_tensor([[[0.005, float('nan')]]]),
        'q_bg_check': None,
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'q': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, effective_error, _ = results['q']
    assert effective_error[0, 0].item() == pytest.approx(0.005)       # overridden
    assert effective_error[0, 1].item() == pytest.approx(0.2 * 0.01)  # fraction fallback


def test_raobs_tuv_gsi_ermin_ermax_clamps_gross_check_not_the_weight():
    # Confirmed by reading GSI's setupq.f90 directly: the ermin/ermax
    # clamp there never feeds the actual Jo weight, only the gross-check
    # decision. error_t raw = 0.5 (below the test ermin=1.0): unclamped,
    # bg_check=4 * 0.5 = 2.0 < |O-B|=3.0 would reject; GSI's clamp raises
    # the COMPARISON's denominator to ermin=1.0 (4*1.0=4.0 >= 3.0, passes)
    # while effective_error (the weight) stays the real, unclamped 0.5.
    raobs_traj, background = _raobs_traj_case([3.0, 3.0], error=1.0, tuv_bg_check=4.0)
    raobs_traj['error_t'] = torch.as_tensor([[[0.5, 0.5]]])
    raobs_traj['ermin_t'] = 1.0
    raobs_traj['ermax_t'] = 2.0
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, effective_error, used = results['t']
    assert effective_error.tolist() == [[0.5, 0.5]]
    assert used.tolist() == [[True, True]]


def test_raobs_tuv_clamp_not_applied_to_flat_error_std():
    # error_t a flat float (use_source_error: False) -- GSI's clamp must
    # not touch this repo's own deliberately-chosen flat error_std.
    raobs_traj, background = _raobs_traj_case([3.0, 3.0], error=0.5, tuv_bg_check=4.0)
    raobs_traj['ermin_t'] = 1.0
    raobs_traj['ermax_t'] = 2.0
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['t']
    # Unclamped: bg_check=4 * 0.5 = 2.0 < |O-B|=3.0 -> rejected at both.
    assert used.tolist() == [[False, False]]


def test_raobs_q_gsi_ermin_ermax_clamps_gross_check_not_the_weight():
    # station 0 has a real source value and qsat; station 1 has neither
    # (falls back to the fraction-of-ob error, which has nothing GSI-
    # sourced to clamp against).
    background = torch.zeros(1, 2)
    raobs_traj = {
        'variables': ['q'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_q': torch.as_tensor([[[0.003, 0.003]]], dtype=torch.float32),
        'mask_q': torch.as_tensor([[[True, True]]]),
        'error_q_fraction': 0.2, 'error_q_floor': 1e-5,
        'source_error_q': torch.as_tensor([[[0.0005, float('nan')]]]),
        'source_qsat_q': torch.as_tensor([[[0.01, float('nan')]]]),
        'q_bg_check': 4.0,
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'q': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, effective_error, used = results['q']
    # effective_error (the Jo weight) is unclamped: 0.0005 (overridden by
    # the source value) and 0.0006 (0.2 * 0.003, the fraction fallback).
    assert effective_error[0, 0].item() == pytest.approx(0.0005)
    assert effective_error[0, 1].item() == pytest.approx(0.2 * 0.003)
    # station 0: unclamped, bg_check=4*0.0005=0.002 < |O-B|=0.003 would
    # reject; GSI's clamp raises the comparison to ermin=10%*qsat=0.001
    # (4*0.001=0.004 >= 0.003, passes). station 1: no qsat -> unclamped
    # fraction error used, 4*0.0006=0.0024 < 0.003 -> rejected.
    assert used.tolist() == [[True, False]]
