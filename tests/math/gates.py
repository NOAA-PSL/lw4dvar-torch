"""The math gates, each a function that returns its error measure, so a test
can assert both that the real code passes and that a deliberately broken
version fails (rule 5 of the merge rules, issue #20).

All of them are meant for float64 inputs.
"""

import numpy as np
import torch


def _directions(shape, n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g, dtype=torch.float64) for _ in range(n)]


def gradient_error(cost_and_grad, cost, x, n_dirs=3, eps=1e-4, seed=0):
    """Largest relative mismatch, over random directions d, between the
    directional derivative from the gradient, <grad J(x), d>, and the central
    finite difference (J(x + eps d) - J(x - eps d)) / (2 eps). The mismatch
    is relative to |grad J|, the typical size of <grad J, d> for d with N(0, 1)
    entries, so a direction that happens to be nearly orthogonal to the
    gradient does not inflate it. The default eps balances truncation error
    (~1e-9 relative for the toy cost) against the cost's float64 rounding
    (~1e-13 relative, amplified by 1/eps)."""
    _, g = cost_and_grad(x)
    scale = float(torch.linalg.vector_norm(g))
    worst = 0.0
    for d in _directions(x.shape, n_dirs, seed):
        with torch.no_grad():
            fd = (cost(x + eps * d) - cost(x - eps * d)) / (2 * eps)
        ad = (g * d).sum()
        worst = max(worst, float(abs(ad - fd)) / max(abs(float(fd)), abs(float(ad)), scale, 1e-30))
    return worst


def taylor_order(cost_and_grad, cost, x, epsilons=(1e-1, 1e-2, 1e-3, 1e-4), seed=0, grad=None):
    """Convergence order of the first-order Taylor remainder
    r(eps) = |J(x + eps d) - J(x) - eps <grad J(x), d>|, from the slope of
    log r against log eps between successive eps; 2 if the gradient is
    right, 1 if it is wrong. Returns the smallest slope. `grad` overrides the
    gradient (to plant a wrong one)."""
    J0, g = cost_and_grad(x)
    if grad is not None:
        g = grad
    (d,) = _directions(x.shape, 1, seed)
    rem = []
    with torch.no_grad():
        for e in epsilons:
            rem.append(float(abs(cost(x + e * d) - J0 - e * (g * d).sum())))
    slopes = np.diff(np.log(rem)) / np.diff(np.log(epsilons))
    return float(slopes.min())


def dot_product_error(forward, adjoint, x_shape, y_shape, seed=0):
    """Relative mismatch |<A v, w> - <v, A^T w>| / max(|<A v, w>|, |<v, A^T w>|)
    for random v, w: zero (to rounding) iff `adjoint` is the adjoint of the
    linear map `forward`."""
    v, = _directions(x_shape, 1, seed)
    w, = _directions(y_shape, 1, seed + 1)
    lhs = float((forward(v).to(torch.float64) * w).sum())
    rhs = float((v * adjoint(w).to(torch.float64)).sum())
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)


def model_tangent_and_adjoint(model, state, increment, steps):
    """The tangent-linear and adjoint of increment -> decoded packed state
    after `steps` steps (no checkpointing: forward-mode AD does not go
    through torch.utils.checkpoint), linearized about `increment`.
    Returns (forward, adjoint, output_shape)."""

    def f(inc):
        return model.advance(state, steps=steps, use_checkpoint=False, latent_increment=inc).state

    x0 = increment.detach()
    y_shape = f(x0).shape

    def forward(v):
        return torch.func.jvp(f, (x0,), (v,))[1]

    def adjoint(w):
        x = x0.clone().requires_grad_(True)
        (gx,) = torch.autograd.grad(f(x), x, grad_outputs=w)
        return gx

    return forward, adjoint, y_shape
