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
import gates  # noqa: E402

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


def _split(flat, shapes):
    sizes = [int(np.prod(s)) for s in shapes]
    return [p.reshape(s) for p, s in zip(torch.split(flat, sizes), shapes)]


def bridge_tangent_and_adjoint(bridged, wrt=(0, 1, 2), outputs=(0, 1)):
    """The rollout's tangent-linear A (from jax.jvp on the plain JAX
    function) and adjoint A^T (torch autograd through `bridged`), as maps
    between flat vectors of the inputs `wrt` and the `outputs`, linearized
    about _inputs(). Returns (forward, adjoint, x_shape, y_shape) for
    gates.dot_product_error."""
    xs = _inputs()
    primals = tuple(jnp.asarray(x.numpy()) for x in xs)
    out_shapes = [o.shape for o in rollout(CONSTANTS, *primals)]
    in_shapes = [xs[i].shape for i in wrt]
    y_shapes = [out_shapes[k] for k in outputs]

    def forward(v):
        tangents = [jnp.zeros(x.shape) for x in xs]
        for i, vi in zip(wrt, _split(v, in_shapes)):
            tangents[i] = jnp.asarray(vi.numpy())
        _, Av = jax.jvp(lambda *a: rollout(CONSTANTS, *a), primals, tuple(tangents))
        return torch.cat([torch.from_numpy(np.array(Av[k])).reshape(-1) for k in outputs])

    def adjoint(w):
        args = [x.clone().requires_grad_(i in wrt) for i, x in enumerate(xs)]
        outs = bridged(*args, constants=CONSTANTS)
        loss = sum((outs[k] * wk).sum() for k, wk in zip(outputs, _split(w, y_shapes)))
        ATw = torch.autograd.grad(loss, [args[i] for i in wrt])
        return torch.cat([g.reshape(-1) for g in ATw])

    size = lambda shapes: (sum(int(np.prod(s)) for s in shapes),)
    return forward, adjoint, size(in_shapes), size(y_shapes)


def dot_product_error(bridged, wrt=(0, 1, 2), outputs=(0, 1), seed=1):
    return gates.dot_product_error(*bridge_tangent_and_adjoint(bridged, wrt, outputs), seed=seed)


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
