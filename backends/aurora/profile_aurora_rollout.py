"""
Operator-level profiling of a real differentiable Aurora rollout (per-step
torch.utils.checkpoint, matching the actual compute_loss_4dvar/
compute_optimal code path) -- to find out WHERE Aurora's ~100s/epoch cost
(vs. AIFS's ~18-25s/epoch for a matched window) actually goes, before
trying to speed it up blindly. Same methodology as
long-window-4dvar-fcstnetv3's profile_fcn3_rollout.py.

Runs on GPU, in the `aurora` conda env. Uses a real ERA5-derived IC
(ic_cache/, no fresh fetch needed -- 2014-12-30T00 and its lagged
2014-12-29T18 predecessor are both already cached) for a realistic
(non-degenerate) input. steps=4 (enough to see steady-state per-step cost
without paying for a full 20-step run).
"""
import datetime
import os
import sys

import torch

# aurora_model.py itself does `import forecast_model`, a repo-root module
# -- needs the repo root on sys.path (aurora_ic/aurora_model themselves
# resolve fine via sys.path[0], this script's own directory).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import aurora_ic
import aurora_model

torch.manual_seed(0)

print("loading model...")
model = aurora_model.AuroraModel(
    package_root="backends/aurora/hf_cache/models--microsoft--aurora/snapshots/a96afd7ee6d65e3bd2d476f3be798a25a56f2296/",
    device="cuda",
)

date0 = datetime.datetime(2014, 12, 30, 0, tzinfo=datetime.timezone.utc)
input_state = aurora_ic.build_input_state(date0, "ic_cache/")
input_encoded = model.prepare_initial_state(input_state, date0)

increment = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)

STEPS = 4

# warmup (excludes cuDNN/kernel autotuning and CUDA context setup from the
# profiled region)
state = model.advance(input_encoded, steps=STEPS, latent_increment=increment, use_checkpoint=True)
loss = state.state.float().pow(2).sum()
loss.backward()
increment.grad = None
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    with torch.profiler.record_function("full_rollout_fwd"):
        state = model.advance(input_encoded, steps=STEPS, latent_increment=increment, use_checkpoint=True)
    with torch.profiler.record_function("loss"):
        loss = state.state.float().pow(2).sum()
    with torch.profiler.record_function("backward"):
        loss.backward()
    torch.cuda.synchronize()

peak_gib = torch.cuda.max_memory_allocated() / 2**30
print(f"\npeak memory during profiled region: {peak_gib:.2f} GiB\n")

print("=== top 25 ops by CUDA time total ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

print("\n=== top-level regions (self CUDA time) ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

# Group by a coarse category based on op name substrings, to answer
# "attention vs conv (patch embed/reconstruction) vs matmul/linear vs
# checkpoint overhead" without needing to add record_function calls inside
# aurora_model.py or the aurora package itself.
totals = {}
for evt in prof.key_averages():
    name = evt.key
    cuda_t = evt.self_device_time_total
    if cuda_t == 0:
        continue
    lname = name.lower()
    if "attention" in lname or "sdpa" in lname or "flash" in lname:
        cat = "attention (scaled_dot_product_attention)"
    elif "conv" in lname:
        cat = "conv (patch embed/reconstruction)"
    elif "checkpoint" in lname:
        cat = "checkpoint bookkeeping"
    elif "norm" in lname:
        cat = "layer/group norm"
    elif "aten::mm" in lname or "aten::bmm" in lname or "gemm" in lname or "aten::linear" in lname or "addmm" in lname:
        cat = "matmul/gemm/linear"
    elif "copy" in lname or "clone" in lname or "index" in lname or "cat" in lname or "reshape" in lname or "permute" in lname or "view" in lname:
        cat = "copy/reshape/indexing bookkeeping"
    else:
        cat = "other"
    totals[cat] = totals.get(cat, 0) + cuda_t

print("\n=== coarse category breakdown (self CUDA time, us) ===")
total_all = sum(totals.values())
for cat, t in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f"{cat:45s} {t/1000:10.1f} ms  ({100*t/total_all:5.1f}%)")

print("\nDone.")
