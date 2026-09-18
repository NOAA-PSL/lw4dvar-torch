"""
Model-backend validation smoke test for AuroraModel, mirroring the bar
AIFSModel/FCN3Model's own smoke tests use: (0) decode_state round-trips a
synthetic initial state correctly, (1) a real differentiable advance()+
backward() produces a finite, non-zero gradient w.r.t. a latent increment,
(2) AdamW actually reduces a real (properly scaled -- see aurora_model.py's
module docstring for the fp16-autocast pitfall this avoids) loss through
that gradient over several epochs.

Uses synthetic surf/atmos fields at each variable's own checkpoint-derived
mean plus small per-pixel noise (aurora.normalisation.locations/scales) --
NOT a physically real ERA5 state (aurora_ic.py, the real IC fetcher, is
not written yet) -- same convention probe_aurora_latent.py validated works
(a perfectly uniform field caused a spurious-looking issue there; small
noise avoids that class of degenerate input).

Run from the repo root: python backends/aurora/smoke_test_aurora.py
"""
import datetime
import sys

import numpy as np
import torch

sys.path.insert(0, ".")  # forecast_model.py (repo root) -- not on sys.path when
                          # running `python backends/aurora/smoke_test_aurora.py`
                          # directly (sys.path[0] is the script's own directory,
                          # not cwd); the real driver runs from the repo root so
                          # this isn't needed there.
sys.path.insert(0, "backends/aurora")
from aurora_model import AuroraModel, _ATMOS_VARS, _LEVELS, _SURF_VARS, _OUTPUT_ONLY_SURF_VARS  # noqa: E402

from aurora.normalisation import locations as NORM_LOCATIONS  # noqa: E402
from aurora.normalisation import scales as NORM_SCALES  # noqa: E402

torch.manual_seed(0)
PACKAGE_ROOT = "backends/aurora/hf_cache/models--microsoft--aurora/snapshots/a96afd7ee6d65e3bd2d476f3be798a25a56f2296"


# 720 rows, not 721 -- Aurora's own Batch.crop() drops the South Pole row
# (see aurora_model.py's _NLAT comment); this backend's native grid is
# 720x1440 to match exactly what the checkpoint actually computes.
_NLAT, _NLON = 720, 1440


def _synthetic_fields():
    fields = {}
    for name in _SURF_VARS:
        loc = NORM_LOCATIONS.get(name, 0.0)
        scale = NORM_SCALES.get(name, 1.0)
        fields[name] = (loc + 0.05 * scale * np.random.randn(2, _NLAT, _NLON)).astype(np.float32)
    for base in _ATMOS_VARS:
        arr = np.zeros((2, len(_LEVELS), _NLAT, _NLON), dtype=np.float32)
        for k, lev in enumerate(_LEVELS):
            loc = NORM_LOCATIONS[f"{base}_{lev}"]
            scale = NORM_SCALES[f"{base}_{lev}"]
            arr[:, k] = loc + 0.05 * scale * np.random.randn(2, _NLAT, _NLON)
        fields[base] = arr
    return fields


print("loading model...")
model = AuroraModel(PACKAGE_ROOT, device="cuda")
print(f"latent_shape={model.latent_shape}, n_points={len(model.lats)}, timestep={model.timestep}")

date0 = datetime.datetime(2015, 1, 1)
input_state = {"fields": _synthetic_fields()}
state0 = model.prepare_initial_state(input_state, date0)

# --- check 0: decode_state round-trips correctly ---
decoded = model.decode_state(state0)
z500 = decoded["z"][_LEVELS.index(500)]
expected_z500_mean = NORM_LOCATIONS["z_500"]
print(f"\n[check 0] decoded z500 mean={z500.mean().item():.3f} (expected ~{expected_z500_mean:.3f})")
assert abs(z500.mean().item() - expected_z500_mean) < 0.1 * abs(expected_z500_mean), "decode_state round-trip mismatch"
assert "geopotential" in decoded and torch.equal(decoded["geopotential"], decoded["z"])
assert "surface_pressure" in decoded and torch.equal(decoded["surface_pressure"], decoded["sp"])
assert "geopotential_at_surface" in decoded
print("[check 0] PASSED")

# --- check 1: real differentiable advance + backward, finite nonzero grad ---
print("\n[check 1] differentiable advance (steps=1) + backward...")
increment = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
state1 = model.advance(state0, steps=1, use_checkpoint=True, latent_increment=increment)
decoded1 = model.decode_state(state1)

# Scale-aware loss (see aurora_model.py docstring's fp16-autocast pitfall):
# normalise each field by its own (location, scale) before squaring, rather
# than summing raw physical-unit values directly.
def _normalised_sq(name, tensor, level=None):
    key = name if level is None else f"{name}_{level}"
    scale = NORM_SCALES.get(key, 1.0)
    loc = NORM_LOCATIONS.get(key, 0.0)
    return ((tensor - loc) / scale).pow(2).mean()


loss_terms = [_normalised_sq(name, decoded1[name]) for name in model._single_by_base]
for base in _ATMOS_VARS:
    for k, lev in enumerate(_LEVELS):
        loss_terms.append(_normalised_sq(base, decoded1[base][k], level=lev))
loss1 = sum(loss_terms)
loss1.backward()
print(f"loss={loss1.item():.4g}, grad norm={increment.grad.norm().item():.4g}, "
      f"finite={torch.isfinite(increment.grad).all().item()}")
assert torch.isfinite(increment.grad).all(), "non-finite gradient"
assert increment.grad.norm().item() > 0, "zero gradient"
print("[check 1] PASSED")

# --- check 2: AdamW reduces a real loss ---
print("\n[check 2] AdamW loss reduction (single step, target = background + shift)...")
increment2 = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
optimizer = torch.optim.AdamW([increment2], lr=0.05)

# Target: push decoded t2m at step 1 to be 5K warmer than the background's
# own t2m (same convention FCN3's own smoke test used).
target_2t = decoded["2t"] + 5.0

losses = []
for epoch in range(15):
    optimizer.zero_grad()
    state1 = model.advance(state0, steps=1, use_checkpoint=True, latent_increment=increment2)
    decoded1 = model.decode_state(state1, only=["2t"])
    scale = NORM_SCALES["2t"]
    loss = ((decoded1["2t"] - target_2t) / scale).pow(2).mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"epoch {epoch}: loss={loss.item():.6f}")

print(f"\nloss[0]={losses[0]:.6f} -> loss[-1]={losses[-1]:.6f}")
assert losses[-1] < 0.5 * losses[0], "AdamW did not meaningfully reduce the loss"
print("[check 2] PASSED")

print("\nALL CHECKS PASSED")
