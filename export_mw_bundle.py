#!/usr/bin/env python
"""Export a radianced two-head run of any microwave sensor as an operator bundle ("twohead/2") for
obs_atms_operator.py, and check it against radianced's own model on one cycle of ERA5 pairs:
forward (z = 0 and a random z) to <= 1e-3 K on every channel with a target, finite gradients.

Run with radianced's venv, from the radianced repo root (load_run reads the run's config path):
    cd /scratch4/BMC/gsienkf/Sergey.Frolov/atms
    .venv/bin/python /scratch4/BMC/gsienkf/Sergey.Frolov/lwatms/export_mw_bundle.py --sensor amsua \
        --run_dir outputs/amsua_twohead_s1 --out outputs/amsua_twohead_s1/amsua_twohead_s1.pt \
        --check data/amsua_era5_pre/2015010100.parquet
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from obs_atms_operator import BUNDLE_VERSION, AtmsOperator  # noqa: E402
from twohead import load_run  # noqa: E402

T_LEVELS = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
Q_LEVELS = T_LEVELS[1:]
# radianced docs/handoff_lw4dvar_sensors.md, "Channel use" (1-based channels)
CHANNEL_SETS = {
    "amsua": {"assimilate": [4, 5, 6, 7, 8], "emissivity": [1, 2, 3, 15], "monitor": [9, 10],
              "exclude": [11, 12, 13], "off": [14]},
}
SURFACE = {"skt": "skt", "2t": "t2m", "2d": "d2m", "sp": "sp", "msl": "msl", "10u": "u10", "10v": "v10",
           "tcw": "tcw", "lsm": "lsm", "z": "zsfc", "sdor": "sdor", "slor": "slor"}
OBS = ("land_frac", "water_frac", "ice_frac", "snow_frac", "Sat_Zenith_Angle", "Scan_Angle", "Sat_Azimuth_Angle",
       "Obs_Time", "sat_id")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check", required=True, help="one ERA5-pairs parquet of this sensor")
    a = p.parse_args()

    model, meta, ms = load_run(a.run_dir, "cpu")
    nch = int(meta.get("nch", 22))
    fin = meta["input_features"]
    ycol = [f"ObsBC-{c}" for c in range(nch)]
    hidden = [m.out_features for m in model.P.layers if isinstance(m, torch.nn.Linear)][:-1]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    bundle = {
        "version": BUNDLE_VERSION, "sensor": a.sensor, "nch": nch,
        "source": f"radianced {git} {a.run_dir}",
        "input_features": fin, "t_levels_hpa": T_LEVELS, "q_levels_hpa": Q_LEVELS, "hidden": hidden, "K": model.K,
        "P": model.P.state_dict(), "D": model.D.state_dict(),
        "x_mean": torch.tensor(ms.loc[fin, "mean"].to_numpy(np.float32)),
        "x_std": torch.tensor(ms.loc[fin, "stdev"].to_numpy(np.float32)),
        "y_mean": torch.tensor(ms.loc[ycol, "mean"].to_numpy(np.float32)),
        "y_std": torch.tensor(ms.loc[ycol, "stdev"].to_numpy(np.float32)),
        "e_mu": model.e_mu.clone(), "e_sd": model.e_sd.clone(), "U": model.U.clone(),
        "sigma_o": torch.tensor(meta["sigma_o"], dtype=torch.float32),
        "channel_sets": CHANNEL_SETS[a.sensor],
    }
    torch.save(bundle, a.out)
    print(f"wrote {a.out}: {a.sensor}, {nch} channels, {len(fin)} inputs, hidden {hidden}, K {model.K}, {bundle['source']}")

    op = AtmsOperator.load(a.out)
    d = pd.read_parquet(a.check)
    col = lambda c: torch.tensor(d[c].to_numpy(np.float32))  # noqa: E731
    fields = {"t": torch.stack([col(f"tmp-{k}") for k in range(len(T_LEVELS))]),
              "q": torch.stack([col(f"spfh-{k}") for k in range(1, len(T_LEVELS))])}
    fields.update({k: col(c) for k, c in SURFACE.items()})
    x = op.assemble(fields, {c: col(c) for c in OBS})
    raw = torch.tensor(d[fin].to_numpy(np.float32))
    print(f"{a.check}: {len(d)} obs; assemble vs file inputs max |diff| {float((x - raw).abs().max()):.3g}")
    has_target = torch.isfinite(bundle["y_mean"])
    with torch.no_grad():
        for z in (None, 0.5 * torch.randn(len(d), model.K)):
            tb, _, _ = op(x, z)
            ref = reference(model, ms, fin, ycol, raw, z)
            diff = (tb - ref)[:, has_target].abs()
            print(f"z {'0' if z is None else 'random'}: operator vs radianced max |diff| {float(diff.max()):.2e} K")
    xg = x.clone().requires_grad_(True)
    tb, _, _ = op(xg)
    tb[:, has_target].sum().backward()
    print("gradient finite:", bool(torch.isfinite(xg.grad).all()))


def reference(model, ms, fin, ycol, raw, z):
    """radianced's forward on the file inputs, in K (TwoHead: normalized in, normalized out)."""
    xn = (raw - torch.tensor(ms.loc[fin, "mean"].to_numpy(np.float32))) / torch.tensor(ms.loc[fin, "stdev"].to_numpy(np.float32))
    e_hat, logs = model.surface(xn)
    e = e_hat if z is None else e_hat + z @ model.U.T
    out = model.radiance(xn, e)
    return out * torch.tensor(ms.loc[ycol, "stdev"].to_numpy(np.float32)) + torch.tensor(ms.loc[ycol, "mean"].to_numpy(np.float32))


if __name__ == "__main__":
    main()
