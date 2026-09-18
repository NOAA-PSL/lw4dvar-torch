"""
Exploratory probe (2026-09-18): before writing aurora_model.py, determine
(a) the actual shape of the backbone's output ("latent_shape", the tensor
a 4D-Var control increment would be added to) and (b) whether gradients
really do flow end-to-end with no torch.no_grad() blocking anywhere, using
a forward_pre_hook on model.decoder rather than reimplementing forward()'s
(fairly involved) surrounding logic -- a hook lets us inject/inspect the
backbone->decoder tensor using the REAL, unmodified forward() path, so
this result isn't an approximation of the real behavior.

Uses synthetic surf/atmos data at the real static-field grid (721x1440),
held at each variable's own checkpoint-derived mean (aurora.normalisation
.locations) PLUS small per-pixel Gaussian noise scaled by that variable's
own std (aurora.normalisation.scales) -- a first attempt using raw zeros
produced a loss of 1.8e18 and a NaN increment gradient (a wildly
out-of-distribution input for most variables post-normalisation, e.g.
msl's real mean is ~101325 Pa). A SECOND attempt at the correct mean but
still perfectly spatially UNIFORM (zero variance across the whole globe)
also produced a NaN gradient -- log_transform (used for `scaled_*` vars)
was checked and ruled out as the cause (log(x+1e-3), finite and smooth at
x=0). A perfectly constant field is the same class of degenerate input
that caused a spurious result when first testing FCN3 elsewhere in this
project (e.g. some layer/normalisation computing a real, un-guarded
variance across space that's exactly 0 for a spatially uniform field) --
small noise breaks that degeneracy cheaply, without needing a full
ERA5-based IC pipeline just to test differentiability.

Run from the repo root: python backends/aurora/probe_aurora_latent.py
"""
import pickle
from datetime import datetime

import torch

from aurora import AuroraV1p5, Batch, Metadata
from aurora.normalisation import locations as NORM_LOCATIONS
from aurora.normalisation import scales as NORM_SCALES

torch.manual_seed(0)

CKPT_DIR = "backends/aurora/hf_cache/models--microsoft--aurora/snapshots/a96afd7ee6d65e3bd2d476f3be798a25a56f2296"

print("loading static fields...")
with open(f"{CKPT_DIR}/aurora-0.25-v1.5-static.pickle", "rb") as f:
    static_raw = pickle.load(f)
static_vars = {k: torch.from_numpy(v) for k, v in static_raw.items()}

print("constructing model...")
model = AuroraV1p5()
model.load_checkpoint_local(f"{CKPT_DIR}/aurora-0.25-v1.5.ckpt")
model = model.to("cuda")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

H, W = 721, 1440
B, T = 1, 2  # batch=1, 2 lagged time levels (max_history_size=2)

lat = torch.linspace(90, -90, H)
lon = torch.linspace(0, 360, W + 1)[:-1]

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def _surf_field(name):
    loc = NORM_LOCATIONS.get(name, 0.0)
    scale = NORM_SCALES.get(name, 1.0)
    return loc + 0.05 * scale * torch.randn(B, T, H, W)


def _atmos_field(name, level):
    loc = NORM_LOCATIONS.get(f"{name}_{level}", 0.0)
    scale = NORM_SCALES.get(f"{name}_{level}", 1.0)
    return loc + 0.05 * scale * torch.randn(B, T, H, W)


surf_vars = {
    k: _surf_field(k) for k in model.surf_vars if k not in model.output_only_surf_vars
}
atmos_vars = {
    k: torch.stack([_atmos_field(k, lev) for lev in LEVELS], dim=2) for k in model.atmos_vars
}

batch = Batch(
    surf_vars=surf_vars,
    static_vars=static_vars,
    atmos_vars=atmos_vars,
    metadata=Metadata(
        lat=lat,
        lon=lon,
        time=(datetime(2015, 1, 1),),
        atmos_levels=LEVELS,
    ),
).to("cuda")

captured = {}
increment_holder = {"increment": None}


def _hook(module, args, kwargs):
    x = args[0]
    captured["shape"] = x.shape
    captured["dtype"] = x.dtype
    inc = increment_holder["increment"]
    if inc is not None:
        x = x + inc
    return (x,) + args[1:], kwargs


handle = model.decoder.register_forward_pre_hook(_hook, with_kwargs=True)

lead_times = torch.full((B,), model.timestep.total_seconds() / 3600, device="cuda")

print("running forward pass (no increment) to determine latent shape...")
with torch.no_grad():
    pred = model(batch, lead_times=lead_times)
print(f"latent shape (backbone output, == decoder input): {captured['shape']}, dtype={captured['dtype']}")
print(f"pred surf_vars keys: {sorted(pred.surf_vars.keys())}")
print(f"pred atmos_vars keys: {sorted(pred.atmos_vars.keys())}")

print("\nchecking differentiability: injecting a real increment via the hook...")
increment = torch.zeros(captured["shape"], device="cuda", dtype=captured["dtype"], requires_grad=True)
increment_holder["increment"] = increment

pred2 = model(batch, lead_times=lead_times)
# NOTE: pred2's surf_vars/atmos_vars are already UNNORMALISED (raw physical
# units, e.g. msl ~1e5 Pa) -- forward() calls pred.unnormalise(...) at the
# very end. A raw squared-sum loss on those huge magnitudes produced a
# finite loss (8.9e17) but a NaN increment.grad on the first two attempts
# (ruled out degenerate-input as the cause -- both a uniform field and a
# spatially-varying one gave the identical NaN). Since increment is added
# via an identity connection right at the decoder's input, d(loss)/d(increment)
# is exactly d(loss)/d(decoder_input) -- so the NaN has to originate INSIDE
# the decoder's own backward pass, independent of what's fed in upstream.
# encoder/backbone/decoder all run under torch.autocast(dtype=torch.float16)
# (AuroraV1p5's default) -- fp16's dynamic range is only ~+-65504, and an
# unnormalised-scale loss produces gradients many orders of magnitude
# larger than what fp16-autocast internals are designed to carry through
# backward without overflowing to inf (then nan). Re-normalising pred2
# before computing the loss (matching the scale the network itself
# operates in) tests that hypothesis directly instead of asserting it.
pred2_norm = pred2.normalise(surf_stats=model.surf_stats)
loss = sum(v.float().pow(2).mean() for v in pred2_norm.surf_vars.values()) + sum(
    v.float().pow(2).mean() for v in pred2_norm.atmos_vars.values()
)
loss.backward()
print(f"[normalised-scale loss] loss={loss.item():.4g}, increment.grad is None: {increment.grad is None}")
if increment.grad is not None:
    print(f"increment.grad norm: {increment.grad.norm().item():.4g}, "
          f"finite: {torch.isfinite(increment.grad).all().item()}")

handle.remove()
print("\nDone.")
