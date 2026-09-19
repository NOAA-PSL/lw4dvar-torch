"""
Clean apples-to-apples confirmation of probe_aurora_compile.py's stage3
finding: within the SAME script/loaded-model instance, compare
uncompiled-vs-compiled steady-state timing at the exact production
setting (steps=4, outer checkpoint on), rather than cross-referencing
against probe_aurora_checkpoint.py's single-cold-call number (58.32s),
which might include one-time overhead not present in true steady state.
"""
import datetime
import sys
import time

import torch

sys.path.insert(0, "backends/aurora")
import aurora_ic
import aurora_model

PACKAGE_ROOT = "backends/aurora/hf_cache/models--microsoft--aurora/snapshots/a96afd7ee6d65e3bd2d476f3be798a25a56f2296/"
DATE0 = datetime.datetime(2014, 12, 30, 0, tzinfo=datetime.timezone.utc)


def timed_calls(model, steps, use_checkpoint, n_calls, label):
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
        print(f"  [{label}] call {i+1}/{n_calls}: {dt:.2f}s", flush=True)
    return times


print("loading model...", flush=True)
model = aurora_model.AuroraModel(PACKAGE_ROOT, device="cuda")

print("=== UNCOMPILED, steps=4, outer checkpoint ON (matches production) ===", flush=True)
t_uncompiled = timed_calls(model, steps=4, use_checkpoint=True, n_calls=3, label="uncompiled")
print(f"uncompiled steady-state (calls 2-3 mean): {sum(t_uncompiled[1:]) / 2:.2f}s", flush=True)

print("\n=== COMPILED (same model instance), steps=4, outer checkpoint ON ===", flush=True)
model.wrapper = torch.compile(model.wrapper)
t_compiled = timed_calls(model, steps=4, use_checkpoint=True, n_calls=3, label="compiled")
print(f"compiled steady-state (calls 2-3 mean): {sum(t_compiled[1:]) / 2:.2f}s "
      f"(call 1 includes recompilation for the new module identity: {t_compiled[0]:.2f}s)", flush=True)

speedup = (sum(t_uncompiled[1:]) / 2) / (sum(t_compiled[1:]) / 2)
print(f"\nsteady-state speedup: {speedup:.2f}x", flush=True)
print("Done.", flush=True)
