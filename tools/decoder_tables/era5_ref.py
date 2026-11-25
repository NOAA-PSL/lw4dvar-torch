"""Strat drift, hypothesis 1, reference: the cycling decoder (a1q70 two-head bundle) on ERA5 itself.

d = (y_o - BC) - H(x_ERA5) on the ERA5-paired ATMS files (atms/data/era5_pre37_1516, built from the same 37-level
ERA5 zarr T1 was trained on), every 6-h file in [START, END]. This is the obs-space signature of "no model drift":
the decoder's own bias (and its seasonal / latitudinal change), which the cycling O-B carries too.
Over land, snow and sea ice the emissivity control z is fitted per ob to window channels 1-4, 16, 17
(as lwatms/atms_decoder_error.py); over ocean z = 0. Also the replay GSI O-F (adjusted) on the same obs.
Per file, lat band and channel: n, sum, sumsq of d and of the GSI O-F, |d| < 15 K.

Usage (CPU, lwaifs2-rocm venv): python era5_ref.py OUT.npz [START=2015010100] [END=2015061518]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))  # the lwatms repo
from obs_atms_operator import AtmsOperator  # noqa: E402

OUT = sys.argv[1]
START = sys.argv[2] if len(sys.argv) > 2 else '2015010100'
END = sys.argv[3] if len(sys.argv) > 3 else '2015061518'
PRE = os.environ.get('PRE', '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/era5_pre37_1516')
BUNDLE = os.environ.get('BUNDLE', '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/outputs/aifs37_a1q70_twohead_s1/atms_twohead_a1q70_s1.pt')
CH = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
WIN = [1, 2, 3, 4, 16, 17]
BANDS = [(-90, -60), (-60, -20), (-20, 20), (20, 60), (60, 90), (-90, 90)]
torch.set_num_threads(int(os.environ.get('NTHREADS', 4)))


def fit_z(op, x, y, sig_w, iters=4):
    iw = [c - 1 for c in WIN]
    with torch.no_grad():
        _, _, log_sig = op(x)
    pinv = torch.exp(-2 * log_sig)
    z = torch.zeros(x.shape[0], op.K)
    for _ in range(iters):
        zz = z.clone().requires_grad_(True)
        tb = op(x, zz)[0][:, iw]
        J = torch.stack([torch.autograd.grad(tb[:, w].sum(), zz, retain_graph=True)[0] for w in range(len(iw))], 1)
        r = (y[:, iw] - tb.detach()) / sig_w
        Jn = J / sig_w[None, :, None]
        A = Jn.transpose(1, 2) @ Jn + torch.diag_embed(pinv)
        g = (Jn.transpose(1, 2) @ r[..., None])[..., 0] - pinv * z
        z = z + torch.linalg.solve(A, g[..., None])[..., 0]
    return z.detach()


op = AtmsOperator.load(BUNDLE)
it0 = op.input_features.index('Obs_Time')
sig_w = op.sigma_o[[c - 1 for c in WIN]].clone()
files = [f for f in sorted(glob.glob(f'{PRE}/2015*.parquet')) if START <= os.path.basename(f)[:10] <= END]
ic = [c - 1 for c in CH]
dates, S = [], []
for f in files:
    pre = pd.read_parquet(f)
    x = torch.tensor(pre[op.input_features].to_numpy(np.float32))
    x[:, it0] = 0.0
    y = torch.tensor(pre[[f'ObsBC-{i}' for i in range(22)]].to_numpy(np.float32))
    wf, ice = pre.water_frac.to_numpy(), pre.ice_frac.to_numpy()
    zmask = ~((wf >= 0.99) & (ice <= 0))
    with torch.no_grad():
        hx = op(x)[0]
    if zmask.any():
        zi = torch.as_tensor(np.nonzero(zmask)[0])
        z = fit_z(op, x[zi], y[zi], sig_w)
        with torch.no_grad():
            hx[zi] = op(x[zi], z)[0]
    d = (y - hx).numpy()[:, ic].astype(np.float64)
    g = pre[[f'Obs_Minus_Forecast_adjusted-{i - 1}' for i in CH]].to_numpy(np.float64)
    lat = pre.lat.to_numpy()
    s = np.zeros((5, len(BANDS), len(CH)))  # n, sum d, sumsq d, sum gsi, sumsq gsi
    ok = np.isfinite(d) & np.isfinite(g) & (np.abs(d) < 15)
    for b, (a, c) in enumerate(BANDS):
        m = ok & ((lat >= a) & ((lat < c) if c < 90 else (lat <= c)))[:, None]
        s[0, b] = m.sum(0)
        s[1, b] = np.where(m, d, 0).sum(0)
        s[2, b] = np.where(m, d ** 2, 0).sum(0)
        s[3, b] = np.where(m, g, 0).sum(0)
        s[4, b] = np.where(m, g ** 2, 0).sum(0)
    dates.append(os.path.basename(f)[:10]); S.append(s)
    print(dates[-1], len(pre), flush=True)
np.savez(OUT, dates=np.array(dates), S=np.stack(S), channels=CH, bands=np.array(BANDS))
print('wrote', OUT)
