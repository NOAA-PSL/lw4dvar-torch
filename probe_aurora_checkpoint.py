"""
One-shot memory/timing probe for whether Aurora's own internal
`configure_activation_checkpointing()` (per-Swin3D-block) alone is
sufficient for memory, without this wrapper's ADDITIONAL outer per-step
`torch.utils.checkpoint` around `_advance_one_step` -- i.e. whether the
outer layer is redundant double-checkpointing. See CLAUDE.md "Aurora
per-epoch runtime optimization" for the motivating question.

Before the bfloat16 fix (see CLAUDE.md "Aurora 16-step divergence"), a
`checkpoint_stride: 0` test at 16 steps was confounded by the NaN
collapse -- the exact same collapse signature appeared whether or not the
outer checkpoint was disabled, so its real memory/timing effect was never
measured. This probe re-tests it in isolation, now that the fix removes
that confound.

Deliberately a SEPARATE process per (outer_checkpoint, steps) trial (see
run_probe_aurora_checkpoint.sh) -- same rationale as FCN3's
probe_decode_memory.py: after a CUDA OOM, letting the process exit is a
more reliable cleanup than chasing every reference in a long-lived process.

Run from the repo root so backends/aurora/'s flat imports resolve.
"""

import argparse
import datetime
import sys
import time

import torch

sys.path.insert(0, "backends/aurora")
import aurora_ic
import aurora_model

PACKAGE_ROOT = "backends/aurora/hf_cache/models--microsoft--aurora/snapshots/a96afd7ee6d65e3bd2d476f3be798a25a56f2296/"
DATE0 = datetime.datetime(2014, 12, 30, 0, tzinfo=datetime.timezone.utc)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--no_outer_checkpoint", action="store_true")
args = parser.parse_args()

print(f"loading model...", file=sys.stderr)
model = aurora_model.AuroraModel(PACKAGE_ROOT, device="cuda")

input_state = aurora_ic.build_input_state(DATE0, "ic_cache/")
input_encoded = model.prepare_initial_state(input_state, DATE0)

torch.cuda.reset_peak_memory_stats()
increment = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)

t0 = time.time()
try:
    out = model.advance(
        input_encoded, steps=args.steps,
        use_checkpoint=not args.no_outer_checkpoint,
        latent_increment=increment,
    )
    loss = out.state.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"RESULT steps={args.steps} outer_checkpoint={not args.no_outer_checkpoint}: "
        f"OK peak={peak_gib:.2f}GiB time={dt:.2f}s "
        f"grad_norm={increment.grad.norm().item():.4g}"
    )
except torch.OutOfMemoryError:
    dt = time.time() - t0
    print(
        f"RESULT steps={args.steps} outer_checkpoint={not args.no_outer_checkpoint}: "
        f"OOM after {dt:.2f}s"
    )
