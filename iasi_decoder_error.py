"""IASI observation error from the decoder on ERA5 pairs (the IASI counterpart of atms_decoder_error.py).

Runs the two-head IASI decoder on the radianced ERA5-paired 2015 files as the 4D-Var does (Obs_Time = 0, no
emissivity control: open ocean only). d = ObsBC - H(x_ERA5) per channel, where ObsBC is finite only where GSI's
cloud QC kept the channel (the GSI clear-sky convention), on open ocean (lsm <= 0.01, siconc <= 0). Gross values are
trimmed per channel at 5 robust std. The inter-channel covariance is pairwise-complete (each pair over the obs that
use both channels), converted to a correlation and made positive definite by eigenvalue clipping.

Usage: PRE=... BUNDLE=... python iasi_decoder_error.py [N_FILES] [OUT_PREFIX]
       (every 4th 6-hourly file of Jan-Apr 2015, default 120). Writes OUT_PREFIX.json in the format
       _setup_atms_corr_r reads (channels = 1-based decoder channel ids; scenes.ocean_clear.{n, std, mean, corr}).
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from obs_atms_operator import AtmsOperator

PRE = os.environ.get('PRE', '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/iasi_era5_pre37')
BUNDLE = os.environ['BUNDLE']
NF = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figures/obs_error/iasi_decoder_era5'
EIG_MIN = float(os.environ.get('EIG_MIN', '1e-3'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)

op = AtmsOperator.load(BUNDLE)
nch = int(op.sigma_o.shape[0])
it0 = op.input_features.index('Obs_Time')
files = [f for f in sorted(glob.glob(f'{PRE}/2015*.parquet')) if os.path.basename(f) < '20150501'][::4][:NF]
ds = []
for f in files:
    pre = pd.read_parquet(f)
    ocean = (pre.lsm.to_numpy() <= 0.01) & (pre.siconc.fillna(0).to_numpy() <= 0)
    pre = pre[ocean]
    x = torch.tensor(pre[op.input_features].to_numpy(np.float32))
    x[:, it0] = 0.0
    y = pre[[f'ObsBC-{i}' for i in range(nch)]].to_numpy(np.float64)
    with torch.no_grad():
        hx = op(x)[0].double().numpy()
    ds.append(y - hx)
    print(os.path.basename(f), int(ocean.sum()), 'ocean obs,', int(np.isfinite(y).any(1).sum()), 'with a clear channel', flush=True)
d = np.concatenate(ds)
n_all = np.isfinite(d).sum(0)
med = np.nanmedian(d, 0)
mad = 1.4826 * np.nanmedian(np.abs(d - med), 0)
d[np.abs(d - med) > 5 * mad] = np.nan
m = np.isfinite(d)
mean = np.nanmean(d, 0)
z = np.where(m, d - mean, 0.0)
M = m.astype(np.float64)
npair = M.T @ M
C = (z.T @ z) / np.maximum(npair - 1, 1)
sd = np.sqrt(np.diag(C))
with np.errstate(invalid='ignore', divide='ignore'):
    R = C / np.outer(sd, sd)
ok = (np.diag(npair) > 100) & np.isfinite(sd) & (sd > 0)
idx = np.nonzero(ok)[0]
Rk = R[np.ix_(idx, idx)]
Rk = np.where(npair[np.ix_(idx, idx)] >= 30, Rk, 0.0)  # too few joint obs: no correlation
np.fill_diagonal(Rk, 1.0)
lam, V = np.linalg.eigh(0.5 * (Rk + Rk.T))
print(f'pairwise corr: min eigenvalue {lam[0]:.3g}, {int((lam < EIG_MIN).sum())} clipped to {EIG_MIN}')
Rp = (V * np.maximum(lam, EIG_MIN)) @ V.T
s = np.sqrt(np.diag(Rp))
Rp = Rp / np.outer(s, s)
chans = [int(i) + 1 for i in idx]
res = {'channels': chans, 'bundle_sigma_o': [float(op.sigma_o[i]) for i in idx], 'files': len(files),
       'scenes': {'ocean_clear': dict(n=[int(np.diag(npair)[i]) for i in idx], n_all=[int(n_all[i]) for i in idx],
                                      std=[float(sd[i]) for i in idx], mean=[float(mean[i]) for i in idx],
                                      corr=Rp.tolist(), cond=float(np.linalg.cond(Rp)))}}
with open(OUT + '.json', 'w') as fh:
    json.dump(res, fh)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print('wrote', OUT + '.json', len(chans), 'channels, cond', f'{np.linalg.cond(Rp):.0f}')
print('std(y - H(ERA5))', np.array(res['scenes']['ocean_clear']['std']))
print('bundle sigma_o  ', np.array(res['bundle_sigma_o']))
