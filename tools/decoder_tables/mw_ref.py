"""Stratospheric channels of AMSU-A / IASI for T1 cycling: the a1q70 decoder on ERA5 itself (as era5_ref.py for ATMS).

d = ObsBC - H(x_ERA5) on the ERA5-paired 2015 files (atms/data/{sensor}_era5_pre37, the same 37-level ERA5 T1 was
trained on), z = 0 (no emissivity control: these channels do not see the surface), GSI QC passed (qc == 0, ObsBC and
O-F finite), |d| < 15 K. Also the replay GSI O-F (adjusted) on the same obs: the operational operator's misfit to its
own background, the yardstick for "is the decoder good enough here". Per file, lat band (5 + global), satellite and
channel: n, sum, sumsq of d and of O-F. Every STRIDE-th 6-h file in [START, END].

Usage (anemoi-dev, from lwatms): python tools/decoder_tables/mw_ref.py SENSOR OUT.npz [START=2015010100] [END=2015061518] [STRIDE=2]"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))  # the lwatms repo
from obs_atms_operator import AtmsOperator  # noqa: E402

SENSOR, OUT = sys.argv[1], sys.argv[2]
START = sys.argv[3] if len(sys.argv) > 3 else '2015010100'
END = sys.argv[4] if len(sys.argv) > 4 else '2015061518'
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 2
A = '/scratch4/BMC/gsienkf/Sergey.Frolov/atms'
CFG = {'amsua': (f'{A}/data/amsua_era5_pre37', f'{A}/outputs/amsua_a1q70_twohead_s1/amsua_twohead_a1q70_s1.pt',
                 list(range(4, 15))),
       'iasi': (f'{A}/data/iasi_era5_pre37', f'{A}/outputs/iasi37_a1q70_twohead_s1_jac10/iasi_twohead_a1q70_jac10_s1.pt',
                list(range(1, 61)))}
PRE, BUNDLE, CH = CFG[SENSOR]
PRE, BUNDLE = os.environ.get('PRE', PRE), os.environ.get('BUNDLE', BUNDLE)
if os.environ.get('CHANNELS'):   # e.g. CHANNELS=1-175 (IASI 77-channel bias table, 2026-11-23)
    lo_, hi_ = os.environ['CHANNELS'].split('-'); CH = list(range(int(lo_), int(hi_) + 1))   # L3 v2: layer decoders on ml37 pre
BANDS = [(-90, -60), (-60, -20), (-20, 20), (20, 60), (60, 90), (-90, 90)]
NSAT = 6
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

op = AtmsOperator.load(BUNDLE)
it0 = op.input_features.index('Obs_Time')
files = [f for f in sorted(glob.glob(f'{PRE}/2015*.parquet')) if START <= os.path.basename(f)[:10] <= END][::STRIDE]
ic = [c - 1 for c in CH]
dates, S = [], []
for f in files:
    pre = pd.read_parquet(f)
    x = torch.tensor(pre[op.input_features].to_numpy(np.float32))
    x[:, it0] = 0.0
    with torch.no_grad():
        hx = op(x)[0].numpy()[:, ic].astype(np.float64)
    y = pre[[f'ObsBC-{i}' for i in ic]].to_numpy(np.float64)
    g = pre[[f'Obs_Minus_Forecast_adjusted-{i}' for i in ic]].to_numpy(np.float64)
    qc = pre[[f'qc-{i}' for i in ic]].to_numpy(np.float64) if f'qc-{ic[0]}' in pre else np.zeros_like(y)  # IASI: ObsBC NaN = not clear
    d = y - hx
    ok = np.isfinite(d) & np.isfinite(g) & (qc == 0) & (np.abs(d) < 15)
    lat = pre.lat.to_numpy(); sat = pre.sat_id.to_numpy().astype(int)
    s = np.zeros((5, len(BANDS), NSAT + 1, len(CH)))   # n, sum d, sumsq d, sum gsi, sumsq gsi; sat NSAT = all
    for b, (lo, hi) in enumerate(BANDS):
        mb = (lat >= lo) & ((lat < hi) if hi < 90 else (lat <= hi))
        for k in range(NSAT + 1):
            m = ok & (mb & ((sat == k) if k < NSAT else True))[:, None]
            s[0, b, k] = m.sum(0)
            s[1, b, k] = np.where(m, d, 0).sum(0); s[2, b, k] = np.where(m, d ** 2, 0).sum(0)
            s[3, b, k] = np.where(m, g, 0).sum(0); s[4, b, k] = np.where(m, g ** 2, 0).sum(0)
    dates.append(os.path.basename(f)[:10]); S.append(s)
    print(dates[-1], len(pre), int(ok.sum()), flush=True)
np.savez(OUT, dates=np.array(dates), S=np.stack(S), channels=CH, bands=np.array(BANDS))
print('wrote', OUT)
