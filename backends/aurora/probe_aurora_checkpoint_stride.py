"""
One-shot memory/timing probe for `checkpoint_stride` values other than the
two extremes already tested (1 = every step checkpointed, the default;
0 = fully disabled, see probe_aurora_checkpoint.py). Replicates
long_window_4dvar_utils.py's compute_loss_4dvar EXACT per-step logic
(`use_ckpt = (checkpoint_stride > 0) and (s % checkpoint_stride == 0)`,
one `model.advance(state, steps=1, use_checkpoint=use_ckpt, ...)` call per
6h step) rather than a single whole-rollout flag, since AuroraModel.advance
itself only takes one use_checkpoint flag applied to every step of however
many `steps` are requested in one call -- the actual per-step alternation
this option relies on lives in the driver's loop, not the model wrapper.

Deliberately a SEPARATE process per (checkpoint_stride, compile) trial --
same rationale as probe_aurora_checkpoint.py: after a CUDA OOM, letting
the process exit is a more reliable cleanup than chasing every reference
in a long-lived process.

Run from the repo root so backends/aurora/'s flat imports resolve.
"""

import argparse
import datetime
import os
import sys
import time

import torch

# aurora_model.py itself does `import forecast_model`, a repo-root module
# -- needs the repo root on sys.path (aurora_ic/aurora_model themselves
# resolve fine via sys.path[0], this script's own directory).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import aurora_ic
import aurora_model

PACKAGE_ROOT = "backends/aurora/hf_cache/models--microsoft--aurora/snapshots/a96afd7ee6d65e3bd2d476f3be798a25a56f2296/"
DATE0 = datetime.datetime(2014, 12, 30, 0, tzinfo=datetime.timezone.utc)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--checkpoint_stride", type=int, required=True,
                     help="0 disables checkpointing entirely, matching "
                          "compute_loss_4dvar's own convention")
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

print(f"loading model (compile={args.compile})...", file=sys.stderr)
model = aurora_model.AuroraModel(PACKAGE_ROOT, device="cuda", compile_wrapper=args.compile)

input_state = aurora_ic.build_input_state(DATE0, "ic_cache/")
input_encoded = model.prepare_initial_state(input_state, DATE0)

torch.cuda.reset_peak_memory_stats()
increment = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)

t0 = time.time()
try:
    state = input_encoded
    n_checkpointed = 0
    for s in range(1, args.steps + 1):
        inj = increment if s == 1 else None
        use_ckpt = (args.checkpoint_stride > 0) and (s % args.checkpoint_stride == 0)
        n_checkpointed += int(use_ckpt)
        state = model.advance(state, steps=1, use_checkpoint=use_ckpt, latent_increment=inj)
    loss = state.state.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"RESULT steps={args.steps} checkpoint_stride={args.checkpoint_stride} "
        f"compile={args.compile} ({n_checkpointed}/{args.steps} steps checkpointed): "
        f"OK peak={peak_gib:.2f}GiB time={dt:.2f}s "
        f"grad_norm={increment.grad.norm().item():.4g}"
    )
except torch.OutOfMemoryError:
    dt = time.time() - t0
    print(
        f"RESULT steps={args.steps} checkpoint_stride={args.checkpoint_stride} "
        f"compile={args.compile}: OOM after {dt:.2f}s"
    )
