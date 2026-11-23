"""Small solver helpers: obs-density Jo weights, the control-variable mask,
and the date utilities."""

import math
from datetime import timezone

import numpy as np
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


def test_raobs_term_cost_gradient_matches_finite_difference():
    # Independent finite-difference check on the raobs cost term itself
    # (mirrors _raobs_term's exact Jt = sum((innov/err)**2) formula,
    # long_window_4dvar_utils.py) -- the math-gate toy-model tests
    # (tests/math/) never pass raobs_traj at all, so this cost's gradient
    # had no coverage independent of the real backends (PR #29's review).
    background = torch.tensor([[280.0, 282.0, 284.0]], dtype=torch.float64, requires_grad=True)
    raobs_traj = {
        'variables': ['t'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_t': torch.tensor([[[279.0, 283.5, 290.0]]], dtype=torch.float64),
        'mask_t': torch.tensor([[[True, True, False]]]),  # third station masked out at read time
        'error_t': 1.5,
        'tuv_bg_check': None,  # no background check: keep every masked-in ob
    }

    def cost(b):
        results = utils._compute_raobs_observation_diagnostics_at_time(
            None, {'t': b}, raobs_traj, _FakeRaobsGridInterp(b), 0)
        innovation, _, effective_error, used = results['t']
        innov = torch.where(used, innovation, torch.zeros_like(innovation))
        err = torch.where(used, effective_error, torch.full_like(effective_error, 1.e10))
        return ((innov / err) ** 2).sum()

    assert torch.autograd.gradcheck(cost, (background,))
    # The third (masked-out) station must not influence the cost or its
    # gradient at all.
    cost(background).backward()
    assert background.grad[0, 2].item() == 0.0


def test_raobs_frozen_qc_prevents_gaming_the_gross_check():
    # Copilot's adversarial example (PR #29's review): two unit-error
    # wind observations, 0 and 12, 8-sigma threshold, both compared
    # against one shared model value. At the background (model value 6),
    # BOTH pass (|0-6|=6, |12-6|=6, both <= 8) -- recomputing the gross
    # check from the LIVE state instead lets the optimizer move the
    # state to 12 and get a FREE reduction in cost by dropping the first
    # observation outright (|0-12|=12 > 8) rather than fitting it: cost
    # 72 at state 6 (both fit, badly) falls to 0 at state 12 (one
    # observation dropped, the other fits exactly) -- lower cost from
    # moving AWAY from a better fit. With the background check frozen,
    # state 12 must still account for the first observation's now-much-
    # worse residual: cost rises to 144, not 0.
    raobs_traj, _ = _raobs_traj_case([0.0, 12.0], error=1.0, tuv_bg_check=8.0)

    def jo(decoded, frozen_qc):
        results = utils._compute_raobs_observation_diagnostics_at_time(
            None, {'t': decoded}, raobs_traj, _FakeRaobsGridInterp(decoded), 0, frozen_qc=frozen_qc)
        innovation, _, effective_error, used = results['t']
        innov = torch.where(used, innovation, torch.zeros_like(innovation))
        err = torch.where(used, effective_error, torch.full_like(effective_error, 1.e10))
        return ((innov / err) ** 2).sum().item(), used.tolist()

    state6 = torch.full((1, 2), 6.0)
    state12 = torch.full((1, 2), 12.0)

    # Live recompute (today's code path when no background pass has
    # run, frozen_qc=False): reproduces the pathology directly.
    cost6_live, used6_live = jo(state6, frozen_qc=False)
    cost12_live, used12_live = jo(state12, frozen_qc=False)
    assert cost6_live == pytest.approx(72.0)
    assert used6_live == [[True, True]]
    assert cost12_live == pytest.approx(0.0)
    assert used12_live == [[False, True]]  # the gaming: observation 0 dropped

    # Freeze the gross check at the background's own verdict (both pass
    # at state 6, as compute_optimal would resolve it before
    # optimization starts) -- state 12 can no longer get a free ride.
    raobs_traj['bg_used'] = {'t': torch.tensor([[[True, True]]])}
    cost6_frozen, used6_frozen = jo(state6, frozen_qc=True)
    cost12_frozen, used12_frozen = jo(state12, frozen_qc=True)
    assert cost6_frozen == pytest.approx(72.0)
    assert used6_frozen == [[True, True]]
    assert cost12_frozen == pytest.approx(144.0)
    assert used12_frozen == [[True, True]]


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


def test_raobs_vertical_interp_weights_exact_match_gives_zero_alpha():
    levels = np.array([1000., 850., 500.])
    pressure = np.array([[1000.], [850.], [np.nan]])
    _, alpha = utils._raobs_vertical_interp_weights(levels, pressure)
    assert alpha[0, 0] == pytest.approx(0.0)
    assert alpha[1, 0] == pytest.approx(0.0)


def test_raobs_vertical_interp_weights_blends_between_levels():
    levels = np.array([1000., 500.])
    mid = math.exp((math.log(1000.) + math.log(500.)) / 2.)  # halfway in ln(p)
    pressure = np.array([[mid], [np.nan]])
    neighbor_idx, alpha = utils._raobs_vertical_interp_weights(levels, pressure)
    assert neighbor_idx[0, 0] == 1
    assert alpha[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_raobs_vertical_interp_weights_out_of_range_clamps_not_rejects():
    # 1200 hPa is beyond the top (highest-pressure) edge level (1000) --
    # JEDI's own vert_interp_weights rule: clamp to the edge, never reject.
    levels = np.array([1000., 850., 500.])
    pressure = np.array([[1200.], [np.nan], [np.nan]])
    neighbor_idx, alpha = utils._raobs_vertical_interp_weights(levels, pressure)
    assert neighbor_idx[0, 0] == 0  # clamped to self
    assert alpha[0, 0] == pytest.approx(0.0)


def test_raobs_log_linear_blends_adjacent_model_level():
    # 2 levels, 2 stations. Background: level0=10 everywhere, level1=20
    # everywhere. Station 0's observation sits 25% of the way (in ln p)
    # toward level1 -- station 1's sits exactly on level0 (alpha=0, must
    # reproduce the unmodified/'exact'-equivalent value).
    background = torch.tensor([[10., 10.], [20., 20.]])
    raobs_traj = {
        'variables': ['t'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_t': torch.as_tensor([[[12.5, 10.0], [0., 0.]]], dtype=torch.float32),
        'mask_t': torch.as_tensor([[[True, True], [False, False]]]),
        'error_t': 1.0,
        'tuv_bg_check': None,
        'vertical_interpolation': 'log-linear',
        'vert_neighbor_idx_t': torch.as_tensor([[[1, 1], [0, 0]]], dtype=torch.int64),
        'vert_alpha_t': torch.as_tensor([[[0.25, 0.0], [0.0, 0.0]]], dtype=torch.float32),
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    innovation, model_equivalent, _, used = results['t']
    assert model_equivalent[0].tolist() == pytest.approx([12.5, 10.0])  # 0.75*10 + 0.25*20, and unchanged
    assert innovation[0].tolist() == pytest.approx([0.0, 0.0])  # observation chosen to match exactly
    assert used[0].tolist() == [True, True]


def test_raobs_exact_and_nearest_modes_ignore_vertical_interp_fields():
    # Without vertical_interpolation set at all (every pre-existing
    # hand-built raobs_traj in this file), the forward operator must take
    # the old, uncorrected path -- regression guard for the new branch.
    raobs_traj, background = _raobs_traj_case([3.0, 5.0], error=1.0, tuv_bg_check=None)
    assert 'vertical_interpolation' not in raobs_traj
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, model_equivalent, _, _ = results['t']
    assert model_equivalent.tolist() == background.tolist()  # no vertical correction applied


def test_raobs_gather_per_level_matches_shared_weights_when_constant_across_levels():
    field = torch.tensor([[10., 20., 30.], [40., 50., 60.]])  # (2 levels, 3 points)
    idx = torch.tensor([[[0, 1]], [[0, 1]]])                  # same idx at both levels
    wts = torch.tensor([[[0.25, 0.75]], [[0.25, 0.75]]])
    result = utils._raobs_gather_per_level(field, idx, wts)
    expected = (field[:, [0, 1]] * torch.tensor([0.25, 0.75])).sum(-1, keepdim=True)
    assert torch.allclose(result, expected)


def test_raobs_gather_per_level_uses_different_neighbors_per_level():
    field = torch.tensor([[10., 20., 30.], [40., 50., 60.]])
    idx = torch.tensor([[[0]], [[2]]])   # level0 picks point0, level1 picks point2
    wts = torch.tensor([[[1.0]], [[1.0]]])
    result = utils._raobs_gather_per_level(field, idx, wts)
    assert torch.allclose(result, torch.tensor([[10.], [60.]]))


def test_raobs_gather_per_level_field_level_idx_overrides_own_level():
    # field_level_idx (log-linear's neighbor index) selects which row of
    # `field` is sampled, independently of which row of `idx`/`wts` (the
    # horizontal weights) is used -- level0's OWN weights (point0) applied
    # to level1's FIELD row, and vice versa.
    field = torch.tensor([[10., 20., 30.], [40., 50., 60.]])
    idx = torch.tensor([[[0]], [[2]]])   # level0's own weights pick point0, level1's pick point2
    wts = torch.tensor([[[1.0]], [[1.0]]])
    field_level_idx = torch.tensor([[1], [0]])  # level0 samples field row1, level1 samples field row0
    result = utils._raobs_gather_per_level(field, idx, wts, field_level_idx=field_level_idx)
    # level0: field[1, point0]=40. level1: field[0, point2]=30.
    assert torch.allclose(result, torch.tensor([[40.], [30.]]))


def test_raobs_log_linear_with_drift_position_uses_current_levels_horizontal_weights():
    # Regression for PR #29's review: with BOTH use_drift_position and
    # vertical_interpolation='log-linear', the neighbor level's value for
    # the vertical blend must be sampled at the CURRENT level's own
    # drift-corrected horizontal position, not reused from the already
    # per-level-interpolated model_obspace (whose neighbor row used the
    # NEIGHBOR's own, different position) -- that would mix horizontal
    # locations into the vertical correction.
    decoded = {'t': torch.tensor([[1000., 2000., 3000.], [4000., 5000., 6000.]])}
    raobs_traj = {
        'variables': ['t'],
        'interp_idx': [None], 'interp_wts': [None],
        # level0's own drift position picks point0; level1's picks point2.
        'interp_idx_t': torch.tensor([[[[0]], [[2]]]], dtype=torch.int64),   # (1, 2, 1, 1)
        'interp_wts_t': torch.tensor([[[[1.0]], [[1.0]]]]),
        'values_t': torch.as_tensor([[[0.], [0.]]], dtype=torch.float32),
        'mask_t': torch.as_tensor([[[True], [True]]]),
        'error_t': 1.0,
        'tuv_bg_check': None,
        'vertical_interpolation': 'log-linear',
        # level0's vertical neighbor is level1, level1's is level0; alpha=1
        # so the blend is fully the neighbor value, making a wrong
        # (mis-positioned) answer unmistakable.
        'vert_neighbor_idx_t': torch.tensor([[[1], [0]]], dtype=torch.int64),
        'vert_alpha_t': torch.tensor([[[1.0], [1.0]]]),
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, decoded, raobs_traj, _FakeRaobsGridInterp(None), 0)
    _, model_equivalent, _, _ = results['t']
    # Correct: neighbor level's field sampled at THIS level's own position
    # (level0 -> decoded['t'][1, point0]=4000; level1 -> decoded['t'][0, point2]=3000).
    # The bug would instead give [[6000.], [1000.]] (the neighbor's OWN position).
    assert torch.allclose(model_equivalent, torch.tensor([[4000.], [3000.]]))


def test_raobs_use_drift_position_applies_per_level_horizontal_weights():
    # 2 levels, 1 station, 3 grid points with distinct values. Each level
    # picks a DIFFERENT grid point via its own interp_idx_t/interp_wts_t --
    # the forward operator must use that, not a single shared weight set.
    decoded = {'t': torch.tensor([[1000., 2000., 3000.], [4000., 5000., 6000.]])}
    raobs_traj = {
        'variables': ['t'],
        'interp_idx': [None], 'interp_wts': [None],
        'interp_idx_t': torch.tensor([[[[0]], [[2]]]], dtype=torch.int64),   # (1, 2, 1, 1)
        'interp_wts_t': torch.tensor([[[[1.0]], [[1.0]]]]),
        'values_t': torch.as_tensor([[[1000.], [6000.]]], dtype=torch.float32),
        'mask_t': torch.as_tensor([[[True], [True]]]),
        'error_t': 1.0,
        'tuv_bg_check': None,
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, decoded, raobs_traj, _FakeRaobsGridInterp(None), 0)
    innovation, model_equivalent, _, used = results['t']
    assert torch.allclose(model_equivalent, torch.tensor([[1000.], [6000.]]))
    assert torch.allclose(innovation, torch.zeros_like(innovation))
    assert used.tolist() == [[True], [True]]


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


def test_raobs_effective_error_nonfinite_rejected_even_with_bg_check_disabled():
    # A missing archived background can leave q's default AND source
    # error both NaN at read time -- with no bg_check configured to
    # screen it, `used` must still reject it rather than let
    # _raobs_term divide by a non-finite weight later.
    background = torch.zeros(1, 2)
    raobs_traj = {
        'variables': ['q'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_q': torch.as_tensor([[[0.01, 0.01]]], dtype=torch.float32),
        'mask_q': torch.as_tensor([[[True, True]]]),
        'default_error_q': torch.as_tensor([[[float('nan'), 0.002]]]),
        'q_bg_check': None,
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'q': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['q']
    assert used.tolist() == [[False, True]]


def test_raobs_effective_error_nonpositive_rejected():
    raobs_traj, background = _raobs_traj_case([3.0, 3.0], error=0.0, tuv_bg_check=None)
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['t']
    assert used.tolist() == [[False, False]]


def test_interp_level_table_matches_entries_and_holds_beyond_range():
    # Same ln(p)-interpolation convention raobs uses throughout
    # (error_std tables): exact at table
    # entries, held constant (nearest entry) beyond the table's range.
    table = {1000.: 1.2, 500.: 0.8, 250.: 1.2}
    out = utils._interp_level_table(table, [1000., 500., 250., 2000., 100.])
    assert out[0] == pytest.approx(1.2)
    assert out[1] == pytest.approx(0.8)
    assert out[2] == pytest.approx(1.2)
    assert out[3] == pytest.approx(1.2)   # above the table's own range
    assert out[4] == pytest.approx(1.2)   # below the table's own range


def test_interp_level_table_broadcasts_over_per_cell_pressure_array():
    # 'nearest'/'log-linear' raobs error tables are evaluated at each
    # observation's own real pressure (get_raobs, PR #29's review) --
    # an (n_obs, n_levels, capacity)-shaped array, not the 1D levels_hpa
    # vector _interp_level_table was previously only ever called with.
    # NaN (an unused padding cell) must propagate through as NaN, not
    # raise or silently become some other value.
    table = {1000.: 1.2, 500.: 0.8}
    pressure_cells = np.array([[[1000., 500.], [750., np.nan]]])
    out = utils._interp_level_table(table, pressure_cells)
    assert out.shape == pressure_cells.shape
    assert out[0, 0, 0] == pytest.approx(1.2)
    assert out[0, 0, 1] == pytest.approx(0.8)
    assert 0.8 < out[0, 1, 0] < 1.2  # 750 hPa: strictly between the two table entries
    assert np.isnan(out[0, 1, 1])


def test_raobs_error_std_table_tensor_used_per_level():
    # error_t a per-level TABLE tensor (a user-configured error_std
    # table, e.g. the UFO/GSI sonde table from PR #29's review comment),
    # broadcast across every obs/station -- confirms effective_error is
    # read correctly from a tensor, not just a flat float.
    raobs_traj, background = _raobs_traj_case([3.0, 3.0], error=1.0, tuv_bg_check=4.0)
    raobs_traj['error_t'] = torch.as_tensor([[[0.5, 0.5]]])
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, effective_error, used = results['t']
    assert effective_error.tolist() == [[0.5, 0.5]]
    assert used.tolist() == [[False, False]]  # bg_check=4*0.5=2.0 < |O-B|=3.0 -> rejected


def test_raobs_gross_check_threshold_resolves_override_then_shared():
    raobs_traj = {'tuv_bg_check': 8.0}
    assert utils._raobs_gross_check_threshold(raobs_traj, 't', 'tuv_bg_check') == (8.0, None)
    raobs_traj['bg_check_threshold_t'] = 7.0
    raobs_traj['bg_check_absolute_t'] = 9.0
    assert utils._raobs_gross_check_threshold(raobs_traj, 't', 'tuv_bg_check') == (7.0, 9.0)
    # a variable with no override still falls back to the shared scalar
    assert utils._raobs_gross_check_threshold(raobs_traj, 'u', 'tuv_bg_check') == (8.0, None)


def test_raobs_tuv_bg_check_per_variable_override_and_absolute_limit():
    # UFO/GSI's per-variable gross check (PR #29's review comment):
    # bg_check_threshold_t/bg_check_absolute_t override the shared
    # tuv_bg_check (deliberately set loose, 100.0, so it would pass
    # everything below if it were used instead of the override).
    background = torch.zeros(1, 3)
    raobs_traj = {
        'variables': ['t'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_t': torch.as_tensor([[[3.0, 1.0, 3.0]]], dtype=torch.float32),
        'mask_t': torch.as_tensor([[[True, True, True]]]),
        'error_t': torch.as_tensor([[[1.0, 1.0, 10.0]]]),
        'tuv_bg_check': 100.0,
        'bg_check_threshold_t': 2.0,
        'bg_check_absolute_t': 1.5,
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'t': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['t']
    # station 0: ratio 3/1=3 > 2.0 -> rejected by the ratio alone.
    # station 1: ratio 1/1=1 <= 2.0, absolute 1 <= 1.5 -> accepted.
    # station 2: ratio 3/10=0.3 <= 2.0 (would pass the ratio check alone),
    # but absolute 3 > 1.5 -> rejected -- the absolute limit catching a
    # case the ratio check misses, same as GSI/UFO's design intent.
    assert used.tolist() == [[False, True, False]]


def test_raobs_q_bg_check_per_variable_override_and_absolute_limit():
    background = torch.zeros(1, 2)
    raobs_traj = {
        'variables': ['q'],
        'interp_idx': [None], 'interp_wts': [None],
        'values_q': torch.as_tensor([[[0.01, 0.01]]], dtype=torch.float32),
        'mask_q': torch.as_tensor([[[True, True]]]),
        'default_error_q': torch.as_tensor([[[0.01, 0.01]]]),  # effective_error == observation here
        'q_bg_check': 100.0,                  # deliberately loose shared default
        'bg_check_threshold_q': 2.0,
        'bg_check_absolute_q': 0.005,
    }
    results = utils._compute_raobs_observation_diagnostics_at_time(
        None, {'q': background}, raobs_traj, _FakeRaobsGridInterp(background), 0)
    _, _, _, used = results['q']
    # Both stations: effective_error = 1.0 * 0.01 = 0.01, |O-B| = 0.01.
    # Ratio: 0.01 <= 2.0*0.01=0.02 -> passes. Absolute: 0.01 > 0.005 ->
    # rejected -- the absolute limit firing even though the ratio (and
    # the loose shared q_bg_check, were it used) would both pass.
    assert used.tolist() == [[False, False]]
