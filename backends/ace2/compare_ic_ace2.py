"""
Validate ace2_ic.py's ARCO-ERA5-derived initial condition against Ai2's own
processed IC for the same time (HF initial_conditions/ic_YYYY.nc, which came
from the dataset ACE2-ERA5 was trained on). Per field: area-weighted mean
difference, rms difference, and rms difference relative to the field's own
spatial std.

Run from the repo root (any env with xarray):
    python backends/ace2/compare_ic_ace2.py [ic_cache_ace2/ace2_ic_2020010100.nc]
"""
import sys

import numpy as np
import xarray as xr

SNAP = "backends/ace2/ACE2-ERA5"
ours = xr.open_dataset(sys.argv[1] if len(sys.argv) > 1 else "ic_cache_ace2/ace2_ic_2020010100.nc")
ref = xr.open_dataset(f"{SNAP}/initial_conditions/ic_2020.nc").sel(time=ours.attrs["valid_time"])

assert np.allclose(ours.latitude, ref.latitude) and np.allclose(ours.longitude, ref.longitude), "grid mismatch"
w = np.cos(np.deg2rad(ref.latitude.values))[:, None] * np.ones((1, ref.longitude.size))
w /= w.sum()
print(f"{'field':26s} {'mean diff':>11s} {'rms diff':>11s} {'rms/std':>9s} {'max|diff|':>11s}")
worst = 0.0
for n in ref.data_vars:
    if n not in ours:
        print(f"{n:26s} MISSING from ours")
        continue
    a, b = ours[n].values.astype(np.float64), ref[n].values.astype(np.float64)
    d = a - b
    std = np.sqrt((w * (b - (w * b).sum()) ** 2).sum())
    rel = np.sqrt((w * d**2).sum()) / std
    worst = max(worst, rel)
    print(f"{n:26s} {(w * d).sum():11.4g} {np.sqrt((w * d**2).sum()):11.4g} {rel:9.2e} {np.abs(d).max():11.4g}")
print(f"extra fields in ours: {sorted(set(ours.data_vars) - set(ref.data_vars))}")
print(f"worst rms/std = {worst:.2e}")
