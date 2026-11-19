"""Stage-2 gate (b) for the A2 fine-tunes (notes/adjoint_finetune_plan.md): 2022 forecast skill vs ERA5 (held-out
year), the blur check, and robustness to realistic initial errors.

For N init dates of 2022 (00 UTC, every --every days): a 120-h forecast from ERA5 (clean) and one from ERA5 + a fixed
real analysis-error draw (finetune/analysis_noise.py resample mode, 'end' pool, both input levels, lag corr 0.27; the
same draws for every model). Scores (area-weighted, model.area_weights) vs ERA5 at every 6 h:
  RMSE z500 (m), t850 (K), ws250 (vector wind, m/s), q850 (g/kg)
  small-scale kinetic energy at 250 hPa (residual after 3 passes of a 24-neighbour mean, ~400 km) / ERA5's
Usage (A2 env, GPU): python finetune/eval_skill.py --label NAME [--weights CKPT] --out output/finetune/eval/NAME.json
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'finetune'))
import long_window_4dvar_utils as utils  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--weights', default=None)
ap.add_argument('--config', default='a37tmp/adjoint/config_pita22_ot_s_0104.yml')
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=30)
ap.add_argument('--every', type=int, default=12)
ap.add_argument('--steps', type=int, default=20)
ap.add_argument('--noise_stats', default='output/finetune/analysis_noise_stats.npz')
a = ap.parse_args()

exp, windows = utils.load_config(a.config)
exp['date'] = exp['sdate']
exp = utils.get_window(exp, list(windows)[0], windows)
if a.weights:
    exp['model_weights'] = a.weights
model = utils.get_model(exp)
from aifs_model import AIFSState  # noqa: E402  (on the path after get_model)
dev = model.device
V = model.var_to_idx
aw = torch.as_tensor(np.asarray(model.area_weights), device=dev, dtype=torch.float64)
aw = aw / aw.sum()
from scipy.spatial import cKDTree  # noqa: E402
la, lo = np.radians(model.lats), np.radians(model.lons)
xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
nb = torch.as_tensor(cKDTree(xyz).query(xyz, k=24)[1], device=dev)


def smooth(f):
    for _ in range(3):
        f = f[nb].mean(1)
    return f


from analysis_noise import AnalysisNoise  # noqa: E402
from model_correction import input_normaliser  # noqa: E402
mul, _ = input_normaliser(model)
AN = AnalysisNoise(a.noise_stats, model.lats, model.lons, device=dev, mode='resample', which='end', seed=2022)
dates = [dt.datetime(2022, 1, 3) + dt.timedelta(days=a.every * i) for i in range(a.n)]
dates = [d for d in dates if d + dt.timedelta(hours=6 * a.steps) <= dt.datetime(2022, 12, 31, 18)]
noise = [AN.sample() for _ in dates]
noise_old = [AN.sample() for _ in dates]


def scores(x, t):
    st = x[0, -1]
    out = {}
    for nm, key, f in (('z500', 'z_500', 1 / 9.80665), ('t850', 't_850', 1.0), ('q850', 'q_850', 1e3)):
        d = (st[:, V[key]].double() - torch.as_tensor(t[key], device=dev, dtype=torch.float64)) * f
        out[nm] = float((d * d * aw).sum().sqrt())
    u, v = st[:, V['u_250']].double(), st[:, V['v_250']].double()
    ut = torch.as_tensor(t['u_250'], device=dev, dtype=torch.float64)
    vt = torch.as_tensor(t['v_250'], device=dev, dtype=torch.float64)
    out['ws250'] = float((((u - ut) ** 2 + (v - vt) ** 2) * aw).sum().sqrt())
    ke = lambda uu, vv: float((((uu - smooth(uu)) ** 2 + (vv - smooth(vv)) ** 2) * aw).sum())  # noqa: E731
    out['ke_small_ratio'] = ke(u, v) / ke(ut, vt)
    return out


R = {'label': a.label, 'weights': a.weights, 'dates': [d.isoformat() for d in dates], 'lead_h': [],
     'clean': [], 'noisy': []}
lag = 0.27
with torch.no_grad():
    for i, d0 in enumerate(dates):
        x0 = model.prepare_initial_state(model.build_input_state(d0), d0)
        xn = AN.apply(x0.state, model, d=noise[i], both_levels=False)
        dold = lag * noise[i] + (1 - lag ** 2) ** 0.5 * noise_old[i]
        xn[0, 0][:, model.pmask_in] = xn[0, 0][:, model.pmask_in] + (dold / mul[model.pmask_in]).to(xn.dtype)
        qi = [j for n, j in V.items() if n.startswith('q_')]
        xn[..., qi] = xn[..., qi].clamp_min(0.0)
        runs = {'clean': x0, 'noisy': AIFSState(xn, d0)}
        rows = {k: [] for k in runs}
        for k, x in runs.items():
            for s in range(1, a.steps + 1):
                x = model.advance(x, steps=1, use_checkpoint=False)
                if s in (1, 2, 4, 8, 12, 16, 20):
                    rows[k].append(scores(x.state, model.era5.fields(d0 + dt.timedelta(hours=6 * s))))
        for k in runs:
            R[k].append(rows[k])
        print(d0, 'clean 120h', rows['clean'][-1], flush=True)
R['lead_h'] = [6, 12, 24, 48, 72, 96, 120]
summ = {}
for k in ('clean', 'noisy'):
    arr = R[k]
    summ[k] = {m: np.mean([[r[m] for r in run] for run in arr], 0).tolist() for m in arr[0][0]}
R['mean'] = summ
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(R, open(a.out, 'w'), indent=1)
print('saved', a.out, json.dumps(summ)[:800])
