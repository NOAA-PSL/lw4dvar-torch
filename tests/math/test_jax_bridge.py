"""Dot-product gate for backends/jax_bridge.py's custom VJP (rule 5, issue
#20): the torch backward that `torch_function` builds from `jax.vjp` must be
the adjoint of the JAX rollout's tangent-linear, <A v, w> = <v, A^T w>.

A (the tangent-linear) comes from `jax.jvp` on the plain JAX function, A^T
from torch autograd through the bridge, so the test checks the bridge's
backward against an independent linearization. The rollout is a small
scanned, checkpointed, nonlinear step like the JAX backends' `advance()`:
two outputs (final state and trajectory), three differentiable inputs and a
non-differentiated constant. Float64 on CPU.

Each case has a planted bug it must catch: a backward that drops the second
output's cotangent, and one that linearizes about the wrong point.
"""

import types

import numpy as np
import pytest
import torch

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

from backends import jax_bridge  # noqa: E402

N, STEPS = 6, 4
CONSTANTS = {'forcing': np.full((N,), 0.01)}


def rollout(constants, x, increment, w):
    @jax.checkpoint
    def step(carry, i):
        y = carry + jnp.where(i == 0, increment, 0.0)
        y = y + 0.1 * jnp.tanh(w @ y) + constants['forcing']
        return y, y

    return jax.lax.scan(step, x, jnp.arange(STEPS))


def _inputs():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(*s, generator=g, dtype=torch.float64) for s in ((N,), (N,), (N, N))]


def _directions(shapes, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*s, generator=g, dtype=torch.float64) for s in shapes]


def dot_product_error(bridged, wrt=(0, 1, 2), outputs=(0, 1), seed=1):
    """|<A v, w> - <v, A^T w>| / max(...) for the rollout's inputs `wrt` and
    `outputs`; A from jax.jvp, A^T from torch autograd through `bridged`."""
    xs = _inputs()
    out_shapes = [o.shape for o in rollout(CONSTANTS, *(jnp.asarray(x.numpy()) for x in xs))]
    v = _directions([xs[i].shape for i in wrt], seed)
    w = _directions([out_shapes[k] for k in outputs], seed + 1)

    tangents = [jnp.zeros(x.shape) for x in xs]
    for i, vi in zip(wrt, v):
        tangents[i] = jnp.asarray(vi.numpy())
    _, Av = jax.jvp(lambda *a: rollout(CONSTANTS, *a), tuple(jnp.asarray(x.numpy()) for x in xs),
                    tuple(tangents))
    lhs = sum(float(jnp.vdot(Av[k], jnp.asarray(wk.numpy()))) for k, wk in zip(outputs, w))

    args = [x.clone().requires_grad_(i in wrt) for i, x in enumerate(xs)]
    outs = bridged(*args, constants=CONSTANTS)
    loss = sum((outs[k] * wk).sum() for k, wk in zip(outputs, w))
    ATw = torch.autograd.grad(loss, [args[i] for i in wrt])
    rhs = sum(float((vi * gi).sum()) for vi, gi in zip(v, ATw))
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)


CASES = {
    'all inputs, both outputs': dict(wrt=(0, 1, 2), outputs=(0, 1)),
    # torch passes None for the unused output's cotangent; the bridge fills zeros
    'final state only': dict(wrt=(0, 1, 2), outputs=(0,)),
    # x and w constant (requires_grad False): their cotangents are dropped
    'increment only': dict(wrt=(1,), outputs=(0, 1)),
}


@pytest.mark.parametrize('case', CASES)
def test_jax_bridge_adjoint(case):
    assert dot_product_error(jax_bridge.torch_function(rollout), **CASES[case]) < 1e-12


def _broken_bridge(monkeypatch, bad_vjp):
    """torch_function with jax.vjp replaced, as seen by jax_bridge (its
    backward is traced on first use, inside the monkeypatch)."""
    shim = types.SimpleNamespace(**{k: getattr(jax, k) for k in ('jit', 'dlpack')}, vjp=bad_vjp)
    monkeypatch.setattr(jax_bridge, 'jax', shim)
    return jax_bridge.torch_function(rollout)


def test_planted_bug_dropped_output_cotangent(monkeypatch):
    def bad_vjp(f, *primals):
        out, vjp = jax.vjp(f, *primals)
        return out, lambda ct: vjp((ct[0], jnp.zeros_like(ct[1])))

    assert dot_product_error(_broken_bridge(monkeypatch, bad_vjp)) > 1e-2


def test_planted_bug_wrong_linearization_point(monkeypatch):
    def bad_vjp(f, *primals):
        return jax.vjp(f, *(1.1 * p for p in primals))

    assert dot_product_error(_broken_bridge(monkeypatch, bad_vjp)) > 1e-3
