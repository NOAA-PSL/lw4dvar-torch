"""
Staged torch.compile experiment for the Aurora backend, following the
profiling finding (see CLAUDE.md "Aurora per-epoch runtime optimization")
that ~31.6% of CPU time goes to GPU-command-queue backpressure ("Command
Buffer Full") plus heavy copy/roll/reshape bookkeeping -- kernel-launch-
count overhead, which torch.compile's kernel fusion directly targets.

Genuinely risky given the confirmed fused-SDPA-kernel crash on this exact
PyTorch/CUDA stack (use_fp16_safe_attention=False -- see CLAUDE.md "Aurora
16-step divergence" era work) and the dynamic forward_pre_hook this
wrapper registers/removes on model.wrapper.decoder on EVERY call (a
pattern torch.compile's guards may not handle cleanly). Staged so a
failure at one stage doesn't block learning from earlier ones:

  Stage 0: baseline (uncompiled) timing, forward+backward, no outer
           checkpoint -- steps=1, repeated 3x for steady-state timing.
  Stage 1: same, but model.wrapper replaced with torch.compile(model.wrapper)
           -- isolates compile's effect without checkpoint interactions.
  Stage 2: same as stage 1, but with the outer per-step
           torch.utils.checkpoint enabled (the real production setting)
           -- tests the compile+checkpoint interaction specifically.
  Stage 3: same as stage 2, but steps=4 (rules out a stage-2 pass being a
           steps=1 special case, e.g. no cross-step recompilation).

Each stage is independently try/excepted; a crash at stage N is reported
and the script moves on to see if later increments behave differently
(no point risking the whole probe on one early failure when the point is
to map out exactly where it breaks, if it does).
"""
import datetime
import sys
import time
import traceback

import torch

sys.path.insert(0, "backends/aurora")
import aurora_ic
import aurora_model

PACKAGE_ROOT = "backends/aurora/hf_cache/models--microsoft--aurora/snapshots/a96afd7ee6d65e3bd2d476f3be798a25a56f2296/"
DATE0 = datetime.datetime(2014, 12, 30, 0, tzinfo=datetime.timezone.utc)


def timed_calls(model, steps, use_checkpoint, n_calls, label):
    """Run n_calls independent forward+backward passes, print per-call
    time, return the list of times (so steady-state -- calls after the
    first -- can be examined separately from one-time compile overhead)."""
    input_state = aurora_ic.build_input_state(DATE0, "ic_cache/")
    input_encoded = model.prepare_initial_state(input_state, DATE0)
    times = []
    for i in range(n_calls):
        increment = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
        torch.cuda.synchronize()
        t0 = time.time()
        out = model.advance(input_encoded, steps=steps, use_checkpoint=use_checkpoint, latent_increment=increment)
        loss = out.state.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        print(f"  [{label}] call {i+1}/{n_calls}: {dt:.2f}s peak={peak_gib:.2f}GiB "
              f"grad_norm={increment.grad.norm().item():.4g}", flush=True)
    return times


print("=== Stage 0: baseline (uncompiled), steps=1, no outer checkpoint ===", flush=True)
print("loading model...", flush=True)
model = aurora_model.AuroraModel(PACKAGE_ROOT, device="cuda")
try:
    t0 = timed_calls(model, steps=1, use_checkpoint=False, n_calls=3, label="stage0")
    print(f"stage0 steady-state (calls 2-3 mean): {sum(t0[1:]) / len(t0[1:]):.2f}s", flush=True)
except Exception:
    print("stage0 FAILED:", flush=True)
    traceback.print_exc()

print("\n=== Stage 1: torch.compile(model.wrapper), steps=1, no outer checkpoint ===", flush=True)
try:
    model.wrapper = torch.compile(model.wrapper)
    t1 = timed_calls(model, steps=1, use_checkpoint=False, n_calls=3, label="stage1")
    print(f"stage1 steady-state (calls 2-3 mean): {sum(t1[1:]) / len(t1[1:]):.2f}s "
          f"(call 1 includes compilation: {t1[0]:.2f}s)", flush=True)
except Exception:
    print("stage1 FAILED:", flush=True)
    traceback.print_exc()

print("\n=== Stage 2: compiled wrapper + outer checkpoint (production setting), steps=1 ===", flush=True)
try:
    t2 = timed_calls(model, steps=1, use_checkpoint=True, n_calls=3, label="stage2")
    print(f"stage2 steady-state (calls 2-3 mean): {sum(t2[1:]) / len(t2[1:]):.2f}s", flush=True)
except Exception:
    print("stage2 FAILED:", flush=True)
    traceback.print_exc()

print("\n=== Stage 3: compiled wrapper + outer checkpoint, steps=4 (rules out cross-step recompilation) ===", flush=True)
try:
    t3 = timed_calls(model, steps=4, use_checkpoint=True, n_calls=3, label="stage3")
    print(f"stage3 steady-state (calls 2-3 mean): {sum(t3[1:]) / len(t3[1:]):.2f}s", flush=True)
except Exception:
    print("stage3 FAILED:", flush=True)
    traceback.print_exc()

print("\nDone.", flush=True)
