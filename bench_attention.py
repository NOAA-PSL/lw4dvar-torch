"""Benchmark sliding-window attention variants for the 1-degree AIFS processor.

One attention call, forward + backward, at the processor's shape: batch 1,
16 heads, 10944 hidden-mesh nodes, head_dim 64, window 512. Each variant is
checked against full masked attention in fp32 and timed in fp16 (the
rollout's autocast dtype) and fp32.

Usage: python bench_attention.py [--seq 10944] [--iters 20]
"""

import argparse
import sys
import time
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "backends/aifs")
import aifs_attention  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seq", type=int, default=10944)
p.add_argument("--heads", type=int, default=16)
p.add_argument("--head-dim", type=int, default=64)
p.add_argument("--window", type=int, default=512)
p.add_argument("--iters", type=int, default=20)
args = p.parse_args()

dev = "cuda"
warnings.filterwarnings("ignore")
S, H, D, W = args.seq, args.heads, args.head_dim, args.window
print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.version.cuda,
      torch.cuda.get_device_name(0), flush=True)
print(f"shape b=1 h={H} s={S} d={D} window={W}", flush=True)

torch.manual_seed(0)
q0, k0, v0, g0 = (torch.randn(1, H, S, D, device=dev) for _ in range(4))

idx = torch.arange(S, device=dev)
band = (idx[:, None] - idx[None, :]).abs() <= W


def full_masked(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=band)


def flash_fn():
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return None
    if getattr(__import__("flash_attn"), "__version__", "").endswith("stub"):
        return None
    return lambda q, k, v: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           window_size=(W, W)).transpose(1, 2)


def wrap(module):
    return lambda q, k, v: module(q, k, v, batch_size=1, window_size=W)


def with_backend(fn, backend):
    def run(q, k, v):
        with sdpa_kernel([backend]):
            return fn(q, k, v)
    return run


variants = {
    "full_masked_sdpa": full_masked,
    "block5d (current)": wrap(aifs_attention.BlockSlidingWindowAttention(fold_blocks=False)),
    "block4d": wrap(aifs_attention.BlockSlidingWindowAttention(fold_blocks=True)),
    "block4d+efficient": with_backend(wrap(aifs_attention.BlockSlidingWindowAttention(fold_blocks=True)),
                                      SDPBackend.EFFICIENT_ATTENTION),
    "block5d+efficient": with_backend(wrap(aifs_attention.BlockSlidingWindowAttention(fold_blocks=False)),
                                      SDPBackend.EFFICIENT_ATTENTION),
    "flex": wrap(aifs_attention.FlexSlidingWindowAttention()),
}
if flash_fn() is not None:
    variants["flash_attn"] = flash_fn()


def fwd_bwd(fn, dtype):
    q, k, v = (x.detach().clone().to(dtype).requires_grad_() for x in (q0, k0, v0))
    out = fn(q, k, v)
    out.backward(g0.to(dtype))
    return out, q.grad, k.grad, v.grad


# fp32 reference
with sdpa_kernel([SDPBackend.MATH]):
    ref = [t.detach() for t in fwd_bwd(full_masked, torch.float32)]


def rel_err(xs):
    return max(((x.float() - r).abs().max() / r.abs().max()).item() for x, r in zip(xs, ref))


print(f"\n{'variant':22s} {'dtype':8s} {'ms/iter':>9s} {'peak GB':>8s} {'rel err':>9s}", flush=True)
for dtype in (torch.float16, torch.float32):
    for name, fn in variants.items():
        if name == "flash_attn" and dtype == torch.float32:
            continue
        try:
            t_c = time.time()
            res = [t.detach() for t in fwd_bwd(fn, dtype)]  # warm-up / compile
            t_c = time.time() - t_c
            err = rel_err(res)
            del res
            for _ in range(2):
                fwd_bwd(fn, dtype)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fwd_bwd(fn, dtype)
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - base) / 1e9
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(args.iters):
                fwd_bwd(fn, dtype)
            t1.record()
            torch.cuda.synchronize()
            ms = t0.elapsed_time(t1) / args.iters
            print(f"{name:22s} {str(dtype)[6:]:8s} {ms:9.2f} {peak:8.2f} {err:9.1e}   (first call {t_c:.1f}s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:22s} {str(dtype)[6:]:8s} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:120]}",
                  flush=True)
        torch.cuda.empty_cache()
