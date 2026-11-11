"""Checks of backends/jax_bridge.py: torch gradients through a JAX rollout.

Runs on CPU (login node) or GPU (the same checks, plus zero-copy and
stream-sync exercise on the device):
  python backends/test_jax_bridge.py
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # gradcheck needs float64

import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backends import jax_bridge  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, STEPS = 6, 4


def rollout(constants, x, increment, w):
    """Toy `advance()`: a scanned nonlinear step, increment on step 0 only,
    plus a per-call forcing passed as a non-differentiated constant."""

    @jax.checkpoint
    def step(carry, i):
        y = carry + jnp.where(i == 0, increment, 0.0)
        y = y + 0.1 * jnp.tanh(w @ y) + constants['forcing']
        return y, y

    final, trajectory = jax.lax.scan(step, x, jnp.arange(STEPS))
    return final, trajectory


def main():
    print(f"device: {DEVICE}, jax: {jax.devices()}")
    g = torch.Generator().manual_seed(0)
    make = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64).to(DEVICE)
    x, inc, w = make(N), make(N), make(N, N)

    forcing = {'forcing': jnp.full((N,), 0.01, jnp.float64)}
    bridged = jax_bridge.torch_function(rollout)
    fn = lambda *t: bridged(*t, constants=forcing)

    # 1. values match a direct JAX evaluation.
    final, trajectory = fn(x, inc, w)
    ref_final, _ = rollout(forcing, *(jnp.asarray(t.cpu().numpy()) for t in (x, inc, w)))
    assert torch.allclose(final.cpu(), torch.from_numpy(jax.device_get(ref_final).copy())), "forward mismatch"
    print("forward matches JAX: ok")

    # 2. gradients vs finite differences, through both outputs, all inputs.
    for t in (x, inc, w):
        t.requires_grad_(True)
    ok = torch.autograd.gradcheck(lambda a, b, c: fn(a, b, c), (x, inc, w), eps=1e-6, atol=1e-6)
    print(f"gradcheck (both outputs, all inputs): {ok}")

    # 3. one output unused (torch passes None for its cotangent), torch ops
    #    on both sides, and a constant (no-grad) input.
    w_const = w.detach()
    loss_fn = lambda a, b: (fn(2 * a, b, w_const)[0] ** 2).sum()
    ok = torch.autograd.gradcheck(loss_fn, (x, inc), eps=1e-6, atol=1e-6)
    print(f"gradcheck (one output unused, torch ops around, constant input): {ok}")
    a, b = x.detach().clone().requires_grad_(), inc.detach().clone().requires_grad_()
    loss_fn(a, b).backward()
    assert b.grad is not None and torch.isfinite(b.grad).all()

    # 4. a new constant value reuses the compiled function (no retrace).
    traces = []
    counted = jax_bridge.torch_function(lambda c, *a: (traces.append(1), rollout(c, *a))[1])
    for v in (0.01, 0.02, 0.03):
        counted(x.detach(), inc.detach(), w.detach(), constants={'forcing': jnp.full((N,), v)})
    print(f"constants change without retracing: {len(traces) == 1} ({len(traces)} trace(s))")

    # 5. JAX -> torch is zero-copy (torch -> JAX copies, see jax_bridge).
    y = jax.numpy.ones(N, jax.numpy.float64)
    if DEVICE.type == "cuda":
        same = jax_bridge.to_torch(y).data_ptr() == y.unsafe_buffer_pointer()
        print(f"jax -> torch zero-copy on GPU: {same}")
    print("all checks passed")


if __name__ == "__main__":
    main()
