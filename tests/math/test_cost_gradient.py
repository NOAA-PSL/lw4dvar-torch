"""Math gates on the full 4D-Var cost (`compute_loss_4dvar`) with the toy
model: the gradient matches finite differences, the Taylor remainder is
second order, and gradient checkpointing does not change the gradient --
for every option that changes the cost's code path. Each gate also has a
planted bug it must catch."""

import pytest
import torch

import long_window_4dvar_utils as utils
from gates import gradient_error, taylor_order
from problem import Problem
from toy_model import ToyModel, ToyModelAtStart

GRAD_TOL = 1e-6

VARIANTS = {
    'plain': {},
    'no_checkpointing': {'checkpoint_stride': 0},
    'checkpoint_every_2nd': {'checkpoint_stride': 2},
    'control_variables': {'control_variables': ['z', 'sp']},
    'jc_ps_weight': {'jc_ps_weight': 30.0},
    'obs_density': {'obs_density_box_deg': 10.0},
    'jc_z_weight': {'jc_z_weight': 20.0},
    'all': {'control_variables': ['z', 't', 'sp'], 'jc_ps_weight': 30.0,
            'obs_density_box_deg': 10.0, 'jc_z_weight': 20.0, 'checkpoint_stride': 2},
    # A toy t/q radiosonde trajectory (Problem._build_raobs_traj), sharing
    # psobs's own obs-slot grid so it includes a between-step slot for
    # free -- independent finite-difference coverage of the raobs cost
    # term through the real compute_loss_4dvar path, not just the
    # isolated unit tests of _compute_raobs_observation_diagnostics_
    # at_time in tests/unit/test_solver_helpers.py. 'raobs' has no
    # background check at all (bg_check=None -- frozen vs. live QC is
    # moot, there's no state-dependent threshold either way). The other
    # two both DO have a real, state-dependent 3-sigma threshold, so
    # frozen vs. live actually matters for them:
    # 'raobs_bg_check' (frozen -- matches this module's own already-
    # validated behavior, Problem's own default, NOT production's own
    # default) attaches 'bg_used' from the SAME compute_ps_observation_hx
    # call compute_optimal itself makes for observations.raobs.
    # frozen_qc: True.
    # 'raobs_bg_check_live' (install_raobs_bg_used=False) instead leaves
    # 'bg_used' unset, exercising the LIVE-recompute QC branch --
    # production's own default (observations.raobs.frozen_qc: False,
    # PR #29's review) -- which nothing here covered before this
    # variant existed.
    'raobs': {'raobs': True},
    'raobs_bg_check': {'raobs': True, 'raobs_bg_check': 3.0},
    'raobs_bg_check_live': {'raobs': True, 'raobs_bg_check': 3.0, 'install_raobs_bg_used': False},
}


@pytest.fixture(scope='module')
def x0():
    return Problem().random_increment(seed=7)


@pytest.mark.parametrize('variant', VARIANTS)
def test_gradient_matches_finite_differences(variant, x0):
    p = Problem(**VARIANTS[variant])
    assert gradient_error(p.cost_and_grad, p.cost, x0) < GRAD_TOL


# increment_at_start (NeuralGCM's path): the increment and the control mask
# apply to the step-0 state, and the Jc/Jz penalties span steps 0 -> 1
AT_START_VARIANTS = ['plain', 'control_variables', 'jc_ps_weight', 'jc_z_weight', 'all']


@pytest.mark.parametrize('variant', AT_START_VARIANTS)
def test_gradient_matches_finite_differences_increment_at_start(variant, x0):
    p = Problem(model=ToyModelAtStart(), **VARIANTS[variant])
    assert gradient_error(p.cost_and_grad, p.cost, x0) < GRAD_TOL


def test_jc_z_penalty_spans_steps_0_to_1_increment_at_start(x0):
    m = ToyModelAtStart()
    p = Problem(model=m, **VARIANTS['jc_z_weight'])
    Jz = float(p.cost(x0)) - float(Problem(model=m).cost(x0))
    with torch.no_grad():
        s0 = m.apply_increment(p.state, x0)
        s1 = m.advance(s0, steps=1, use_checkpoint=False)
        expected = utils._jc_z_penalty(m.decode_state(s0)['z'], m.decode_state(s1)['z'], *p.jcz_ref)
    assert Jz > 0
    assert Jz == pytest.approx(float(expected), rel=1e-9)


def test_taylor_remainder_is_second_order(x0):
    p = Problem(**VARIANTS['all'])
    assert taylor_order(p.cost_and_grad, p.cost, x0) > 1.9


@pytest.mark.parametrize('stride', [1, 2, 3])
def test_checkpointing_does_not_change_the_gradient(stride, x0):
    J0, g0 = Problem(checkpoint_stride=0).cost_and_grad(x0)
    J, g = Problem(checkpoint_stride=stride).cost_and_grad(x0)
    assert float(J) == float(J0)
    torch.testing.assert_close(g, g0, rtol=1e-12, atol=0)


@pytest.mark.parametrize('frozen_qc', [True, False])
def test_raobs_taylor_remainder_is_second_order(frozen_qc, x0):
    # Both the cached (production opt-in) and live-recompute (production
    # default) background-check masks get their own Taylor check -- see
    # VARIANTS['raobs_bg_check']/['raobs_bg_check_live']'s own comment for
    # why this only matters with a real bg_check threshold (PR #29's
    # review).
    p = Problem(raobs=True, raobs_bg_check=3.0, install_raobs_bg_used=frozen_qc, checkpoint_stride=2)
    assert taylor_order(p.cost_and_grad, p.cost, x0) > 1.9


@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('frozen_qc', [True, False])
def test_raobs_checkpointing_does_not_change_the_gradient(frozen_qc, stride, x0):
    # A real bg_check threshold (3.0), unlike the plain 'raobs' VARIANTS
    # entry -- otherwise this gate would never exercise any gross-check/
    # checkpointing interaction at all, with either mask (PR #29's
    # review).
    J0, g0 = Problem(
        raobs=True, raobs_bg_check=3.0, install_raobs_bg_used=frozen_qc, checkpoint_stride=0).cost_and_grad(x0)
    J, g = Problem(
        raobs=True, raobs_bg_check=3.0, install_raobs_bg_used=frozen_qc, checkpoint_stride=stride).cost_and_grad(x0)
    assert float(J) == float(J0)
    torch.testing.assert_close(g, g0, rtol=1e-12, atol=0)


def test_raobs_adds_a_nonzero_cost(x0):
    # Confirms the raobs trajectory actually contributes to J (not just
    # that its gradient is correct if it did) -- same sanity check
    # test_cost_is_nonlinear makes for the psobs-only cost.
    with_raobs = float(Problem(raobs=True).cost(x0))
    without = float(Problem().cost(x0))
    assert with_raobs > without + 1.0  # well above float64 noise


def test_cost_is_nonlinear(x0):
    # the gates would pass trivially for a quadratic cost of a linear model
    p = Problem()
    _, g0 = p.cost_and_grad(torch.zeros_like(x0))
    _, g1 = p.cost_and_grad(x0)
    J0 = p.cost(torch.zeros_like(x0))
    quadratic_prediction = 0.5 * float(((g0 + g1) * x0).sum())
    assert abs(float(p.cost(x0) - J0) - quadratic_prediction) > 1e-3 * abs(quadratic_prediction)


# --- planted bugs -----------------------------------------------------------

def _surface_pressure_detached_weight(pressure_levels, geopotential, orography):
    """`get_surface_pressure` with one term of its interpolation weight
    detached: the same values, but part of the gradient missing."""
    pressure_levels = torch.flip(pressure_levels, dims=(0,))
    geopotential = torch.flip(geopotential, dims=(0,))
    xp = (orography[None, :] - geopotential).transpose(0, 1).contiguous()
    log_p = -torch.log(pressure_levels)
    u = torch.clamp(torch.searchsorted(xp, torch.zeros((xp.shape[0], 1), dtype=xp.dtype), right=True).squeeze(-1),
                    1, xp.shape[1] - 1)
    xp_lo = torch.gather(xp, 1, (u - 1)[:, None])[:, 0]
    xp_hi = torch.gather(xp, 1, u[:, None])[:, 0]
    w = (0.0 - xp_lo.detach()) / (xp_hi - xp_lo)
    return torch.exp(-((1 - w) * log_p[u - 1] + w * log_p[u]))


def test_planted_bug_detached_operator_term(x0, monkeypatch):
    p = Problem()
    monkeypatch.setattr(utils, 'get_surface_pressure', _surface_pressure_detached_weight)
    J = p.cost(x0)
    monkeypatch.undo()
    torch.testing.assert_close(J, p.cost(x0))          # forward unchanged...
    monkeypatch.setattr(utils, 'get_surface_pressure', _surface_pressure_detached_weight)
    assert gradient_error(p.cost_and_grad, p.cost, x0) > 1e-2   # ...gradient caught


class _CheckpointDropsGradient(ToyModel):
    """A checkpoint wrapper that detaches its input: right values, but the
    gradient stops at every checkpointed step."""

    def advance(self, state, steps=1, use_checkpoint=True, latent_increment=None):
        if use_checkpoint:
            state = self.wrap_state(state.state.detach(), state.date)
        return super().advance(state, steps, use_checkpoint, latent_increment)


def test_planted_bug_checkpoint_drops_gradient(x0):
    _, g0 = Problem(model=_CheckpointDropsGradient(), checkpoint_stride=0).cost_and_grad(x0)
    _, g = Problem(model=_CheckpointDropsGradient(), checkpoint_stride=2).cost_and_grad(x0)
    assert not torch.allclose(g, g0, rtol=1e-6)
    p = Problem(model=_CheckpointDropsGradient(), checkpoint_stride=2)
    assert gradient_error(p.cost_and_grad, p.cost, x0) > 1e-2


def test_planted_bug_gradient_off_by_a_factor(x0):
    p = Problem()
    _, g = p.cost_and_grad(x0)
    assert taylor_order(p.cost_and_grad, p.cost, x0, grad=1.01 * g) < 1.5


def test_jc_z_penalty_value():
    # z1 - z0 = c_k * (level k's w-weighted spatial std of z0) everywhere:
    # the penalty is factor * mean_k c_k^2, whatever z0 and w are
    g = torch.Generator().manual_seed(3)
    z0 = 5000.0 + 300.0 * torch.randn(3, 40, generator=g, dtype=torch.float64)
    w = torch.rand(40, generator=g, dtype=torch.float64)
    w = w / w.sum()
    std = torch.sqrt((w * (z0 - (w * z0).sum(1, keepdim=True)) ** 2).sum(1, keepdim=True))
    c = torch.tensor([[0.1], [0.2], [0.4]], dtype=torch.float64)
    Jz = utils._jc_z_penalty(z0, z0 + c * std, w, 7.0)
    assert float(Jz) == pytest.approx(7.0 * float((c ** 2).mean()), rel=1e-12)


def test_planted_bug_jc_z_detached_normalization(x0, monkeypatch):
    # the penalty's normalization (z0's spatial std) depends on the
    # increment too; detaching it leaves the value unchanged
    def detached_std(z0, z1, w, factor):
        z0r = z0.reshape(-1, w.shape[0])
        zstd = torch.sqrt((w * (z0r - (w * z0r).sum(1, keepdim=True)) ** 2).sum(1, keepdim=True)).detach()
        return factor * (w * (((z1.reshape(-1, w.shape[0]) - z0r) / zstd) ** 2).mean(dim=0)).sum()

    p = Problem(**VARIANTS['jc_z_weight'])
    J = float(p.cost(x0))
    monkeypatch.setattr(utils, '_jc_z_penalty', detached_std)
    assert float(p.cost(x0)) == pytest.approx(J, rel=1e-12)
    monkeypatch.setattr(utils, '_jc_z_penalty', detached_std)
    assert gradient_error(p.cost_and_grad, p.cost, x0) > 1e-3


@pytest.mark.parametrize('at_start, n_steps, ok', [(False, 1, False), (False, 2, True),
                                                   (True, 0, False), (True, 1, True)])
def test_first_step_penalties_reject_too_short_windows(at_start, n_steps, ok):
    for key in ('jc_ps_weight', 'jc_z_weight'):
        if ok:
            utils._check_first_step_penalty_window({key: 1.0}, at_start, n_steps)
        else:
            with pytest.raises(ValueError, match='at least'):
                utils._check_first_step_penalty_window({key: 1.0}, at_start, n_steps)
    utils._check_first_step_penalty_window({}, at_start, n_steps)   # penalties off: anything goes


def test_jc_z_diagnostic_with_no_obs(x0, capsys):
    # an empty window makes the penalty's factor 0: the printed rms must not divide by it
    p = Problem(**VARIANTS['jc_z_weight'])
    J, _, _ = utils.compute_loss_4dvar(
        p.model, x0, 1.0, p.state, p.verif_ic, p.psobs_traj, 1, p.exp, p.grid_interp, 1,
        p.interp_specs, jcz_ref=(p.jcz_ref[0], 0.0))
    assert 'Jz = 0.0' in capsys.readouterr().out
