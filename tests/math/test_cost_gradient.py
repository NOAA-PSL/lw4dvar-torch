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
from toy_model import ToyModel

GRAD_TOL = 1e-6

VARIANTS = {
    'plain': {},
    'no_checkpointing': {'checkpoint_stride': 0},
    'checkpoint_every_2nd': {'checkpoint_stride': 2},
    'control_variables': {'control_variables': ['z', 'sp']},
    'jc_ps_weight': {'jc_ps_weight': 30.0},
    'obs_density': {'obs_density_box_deg': 10.0},
    'all': {'control_variables': ['z', 't', 'sp'], 'jc_ps_weight': 30.0,
            'obs_density_box_deg': 10.0, 'checkpoint_stride': 2},
}


@pytest.fixture(scope='module')
def x0():
    return Problem().random_increment(seed=7)


@pytest.mark.parametrize('variant', VARIANTS)
def test_gradient_matches_finite_differences(variant, x0):
    p = Problem(**VARIANTS[variant])
    assert gradient_error(p.cost_and_grad, p.cost, x0) < GRAD_TOL


def test_taylor_remainder_is_second_order(x0):
    p = Problem(**VARIANTS['all'])
    assert taylor_order(p.cost_and_grad, p.cost, x0) > 1.9


@pytest.mark.parametrize('stride', [1, 2, 3])
def test_checkpointing_does_not_change_the_gradient(stride, x0):
    J0, g0 = Problem(checkpoint_stride=0).cost_and_grad(x0)
    J, g = Problem(checkpoint_stride=stride).cost_and_grad(x0)
    assert float(J) == float(J0)
    torch.testing.assert_close(g, g0, rtol=1e-12, atol=0)


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
