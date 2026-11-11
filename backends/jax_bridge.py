"""Bridge for running JAX forecast models behind the torch solver core.

The solver core only needs `LatentForecastModel.advance()` to be
differentiable in torch (`loss.backward()` in compute_optimal). A JAX
backend gets that by wrapping its pure rollout function with
`torch_function(fn)`: the forward pass hands the torch inputs to JAX through
DLPack, runs `fn` jitted, and hands the outputs back (zero-copy); the backward pass recomputes `fn` under `jax.vjp` with the
incoming torch cotangents. Recomputing in backward rather than keeping the
VJP residuals is the JAX-side analogue of the other backends'
`torch.utils.checkpoint`; for long rollouts `fn` should additionally
`jax.checkpoint` its per-step function inside its `jax.lax.scan`.

Cross the boundary once per `advance()` call (the whole multi-step rollout
in one `fn`), not once per model step, so XLA sees the full scan.

Both frameworks allocate on the same GPU: set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (the launcher does) so JAX does not
reserve 75 % of the device up front and starve torch's caching allocator.

torch -> JAX copies into a JAX-owned buffer (a device-to-device copy, MBs
per call for a TL63 state): XLA's CUDA-graph command buffers fail on
torch-allocated buffers ("Failed to set memcpy d2d node params:
CUDA_ERROR_INVALID_VALUE" in a full model step, though not in small test
functions).

Streams: torch and JAX run on different CUDA streams, so the bridge
synchronizes torch's current stream before JAX reads a torch buffer and
blocks on JAX results before torch reads them. The cost is one sync per
`advance()` call.
"""

import functools
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import torch


def to_jax(x: torch.Tensor) -> jax.Array:
    """torch -> JAX via DLPack, copied into a JAX-owned buffer (see above)."""
    x = x.detach()
    if not x.is_contiguous():
        x = x.contiguous()
    if x.is_cuda:
        torch.cuda.current_stream(x.device).synchronize()
    # from_dlpack(copy=True) is UNIMPLEMENTED through the GPU PJRT C API, so
    # import zero-copy and copy on device; block so torch can't reuse the
    # source buffer before the copy has read it.
    return jnp.copy(jax.dlpack.from_dlpack(x)).block_until_ready()


def to_torch(x: jax.Array) -> torch.Tensor:
    """JAX -> torch via DLPack (zero-copy)."""
    x.block_until_ready()
    return torch.from_dlpack(x)


def torch_function(fn: Callable[..., jax.Array | Sequence[jax.Array]]) -> Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]:
    """A torch-differentiable version of a pure JAX function of arrays.

    `fn(constants, *arrays)` must return one array or a tuple of arrays.
    `constants` is any JAX pytree (forcings, the valid time, ...) passed to
    the jitted function as a traced, non-differentiated argument, so
    per-call values do not trigger recompilation (closing over them would).
    The returned function is called as `apply(*tensors, constants=...)`.
    Gradients flow to every torch input that requires grad; inputs that do
    not are treated as constants (their JAX cotangents are computed but
    dropped). Tensor inputs must be floating point.
    """
    forward = jax.jit(fn)

    @jax.jit
    def backward(constants, args, cotangents):
        outputs, vjp = jax.vjp(lambda *a: fn(constants, *a), *args)
        if not isinstance(outputs, (tuple, list)):
            cotangents = cotangents[0]
        return vjp(cotangents)

    class _Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, constants, *tensors):
            ctx.save_for_backward(*tensors)
            ctx.constants = constants
            outputs = forward(constants, *(to_jax(t) for t in tensors))
            ctx.single = not isinstance(outputs, (tuple, list))
            outputs = (to_torch(outputs),) if ctx.single else tuple(to_torch(o) for o in outputs)
            ctx.templates = [(o.shape, o.dtype, o.device) for o in outputs]
            return outputs[0] if ctx.single else outputs

        @staticmethod
        def backward(ctx, *grads):
            tensors = ctx.saved_tensors
            # torch passes None for outputs that did not reach the loss.
            grads = [g if g is not None else torch.zeros(s, dtype=d, device=v)
                     for g, (s, d, v) in zip(grads, ctx.templates)]
            arg_grads = backward(ctx.constants, tuple(to_jax(t) for t in tensors),
                                 tuple(to_jax(g) for g in grads))
            return (None,) + tuple(to_torch(g) if t.requires_grad else None
                                   for g, t in zip(arg_grads, tensors))

    @functools.wraps(fn)
    def apply(*tensors: torch.Tensor, constants=None):
        return _Function.apply(constants, *tensors)

    return apply
