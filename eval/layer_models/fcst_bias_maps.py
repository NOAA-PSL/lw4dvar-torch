"""Test 2 of the drift diagnosis: mean error growth of 5-day forecasts from ERA5 (no data assimilation).

For init dates START..END every EVERY h: IC = ERA5 (the repo's own get_input, nhrs_back=0), forecast LEADS 6 h steps,
decode to pressure levels, and accumulate forecast - ERA5 at every 24 h lead for z500, t850, t500, t250 (sum and sum of
squares per grid point). Truth is the same for every model: the 37-level ERA5 training zarr, nearest O96 point to each
model grid point (identity for O96 models). Run from the model's own repo (lwatms for T1/L2, lwatms_ace2 for ACE2).
Writes OUT (npz): sum, sumsq (nlead, nvar, npoints), n, leads, vars, lats, lons.
Usage: python fcst_bias_maps.py REPO CONFIG OUT START END [--every 48] [--leads 20] [--amd]"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
import zarr
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('repo'); ap.add_argument('config'); ap.add_argument('out'); ap.add_argument('start'); ap.add_argument('end')
ap.add_argument('--every', type=int, default=48); ap.add_argument('--leads', type=int, default=20)
ap.add_argument('--amd', action='store_true', help="aifs_attention: flex_sliding_window (as chain_job.sh a37-amd)")
args = ap.parse_args()
os.chdir(args.repo); sys.path.insert(0, args.repo)
import long_window_4dvar_utils as utils  # noqa: E402

Z37 = '/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr'
VARS = [('z', 500), ('t', 850), ('t', 500), ('t', 250)]
G = 9.80665
exp, windows = utils.load_config(args.config)
if args.amd:
    exp['aifs_attention'] = 'flex_sliding_window'
exp['date'] = exp['sdate']
exp = utils.get_window(exp, list(windows)[0], windows)
logger = utils.get_logger()
model = utils.get_model(exp)
lats, lons = np.asarray(model.lats, dtype=np.float64), np.asarray(model.lons, dtype=np.float64)
z = zarr.open(Z37, 'r'); zvar = list(z.attrs['variables'])
zd = np.asarray(z['dates'][:]).astype('datetime64[s]')


def xyz(la, lo):
    la, lo = np.deg2rad(la), np.deg2rad(lo)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


NN = cKDTree(xyz(np.asarray(z['latitudes'][:]), np.asarray(z['longitudes'][:]))).query(xyz(lats, lons))[1]
zi = [zvar.index(f'{v}_{p}') for v, p in VARS]
lev = {b: [float(p) for p in model.pressure_levels(b)] for b in ('z', 't')}
LEADS = list(range(0, args.leads + 1, 4))   # every 24 h
S = np.zeros((len(LEADS), len(VARS), len(lats))); S2 = np.zeros_like(S); n = 0


def truth(dt):
    row = np.asarray(z['data'][int(np.where(zd == np.datetime64(dt))[0][0]), :, 0, :])
    return np.stack([row[i][NN] for i in zi]) / np.array([G if v == 'z' else 1. for v, _ in VARS])[:, None]


T = datetime.strptime(args.start, '%Y-%m-%dT%H'); END = datetime.strptime(args.end, '%Y-%m-%dT%H')
while T <= END:
    ts = T.strftime('%Y-%m-%dT%H')
    with torch.no_grad():
        state, _ = utils.get_input(dict(exp, date=ts, sdate=ts, nhrs_back=0, restart_window=False), model, logger)
        for k in range(args.leads + 1):
            if k > 0:
                state = model.advance(state, steps=1, use_checkpoint=False)
            if k not in LEADS:
                continue
            d = model.decode_state(state)
            f = np.stack([d[v][lev[v].index(float(p))].detach().cpu().numpy().astype(np.float64) for v, p in VARS])
            f[0] /= G
            e = f - truth(T + timedelta(hours=6 * k))
            j = LEADS.index(k); S[j] += e; S2[j] += e * e
    n += 1
    logger.info(f'{ts}: z500 mean err at 120 h so far {np.mean(S[-1, 0]) / n:+.2f} m (n={n})')
    np.savez(args.out, sum=S, sumsq=S2, n=n, leads=np.array(LEADS) * 6, vars=np.array([f'{v}{p}' for v, p in VARS]),
             lats=lats, lons=lons)
    T += timedelta(hours=args.every)
print('done', n)
