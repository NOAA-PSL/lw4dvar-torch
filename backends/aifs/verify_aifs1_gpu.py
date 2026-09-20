import torch
from flash_attn import flash_attn_func

# Minimal real flash-attn CUDA kernel exercise, not just import
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, causal=False)
assert torch.isfinite(out).all(), "flash_attn output has NaN/Inf!"
print(f"flash_attn CUDA kernel OK: output shape={tuple(out.shape)}, finite={torch.isfinite(out).all().item()}")

from anemoi.inference.runners.simple import SimpleRunner
import os
ckpt_path = "backends/aifs/aifs-single-1.1/aifs-single-mse-1.1.ckpt"
runner = SimpleRunner(ckpt_path, device="cuda")
runner.model.eval()
for p in runner.model.parameters():
    p.requires_grad_(False)
n_params = sum(p.numel() for p in runner.model.parameters())
print(f"checkpoint loaded on CUDA OK: {n_params} params, timestep={runner.checkpoint.timestep}")
print("ALL GPU CHECKS PASSED")
