"""Radiosonde humidity bias: is it the sondes or the model?

At the same raob q obs (Bo's ps + raobs run, cycles 2-21), by level:
  O - B        obs minus the 4D-Var background (raobs run)
  O - ERA5     sondes vs ERA5 (4-NN inverse-distance on O96, as the solver)
  B - ERA5     model background vs ERA5
  O - B_ps     obs minus Bo's ps-only background (never saw raobs; saved at
               1000/850/500 hPa only)
O - ERA5 uses each cycle's first 12 h of slots (each sounding once); O - B is
also split by lead day in the window, and by 00Z/12Z and region.
Mean and RMS in g/kg. Usage: python raob_q_bias.py [RAOB_RUN_DIR] [N_CYCLES]  (default Bo's raobs run, 21)
"""

import glob
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, 'diagnostics')
sys.path.insert(0, 'backends/aifs')
import aifs_ic  # noqa: E402
import z500err_window as zw  # noqa: E402

CACHE = '/scratch4/BMC/gsienkf/Sergey.Frolov/ic_cache/ic_cache_aifs2-1.0deg-v1.0/'
B = '/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/add-raobs-parquet/lw4dvar-torch-test1/output/smoke_test_aifs2-1.0deg-v1.0-pretraining-'
RAOB, PS = (sys.argv[1].rstrip('/') + '/' if len(sys.argv) > 1 and sys.argv[1] else B + 'ps-raobs-parquet-new/'), B + 'ps-parquet-new/'
CYCLES = slice(1, int(sys.argv[2]) if len(sys.argv) > 2 else 21)

files = sorted(glob.glob(RAOB + '*_raobs_diagnostics_120h_100it.nc'))[CYCLES]
grid = xr.open_dataset(sorted(glob.glob(PS + '*_control_forecast_120h_100it.nc'))[0])
glat, glon = grid.latitude.values, grid.longitude.values % 360
xyz = lambda la, lo: np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),  # noqa: E731
                               np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))], -1)
tree = cKDTree(xyz(glat, glon))
era = {}


def era5_q(date):
    if date not in era:
        era[date] = aifs_ic.read_single_date_fields(None, datetime.strptime(date, '%Y-%m-%dT%H'), CACHE)
    return era[date]


rows = []
for f in files:
    d = xr.open_dataset(f)
    cyc = f.split('/')[-1][:13]
    lev = d.level_q.values
    bg_ps = xr.open_dataset(PS + f'{cyc}_control_forecast_120h_100it.nc')
    ps_lev = list(bg_ps.level_q.values)
    for j, t in enumerate(d.time.values):
        used = d.q_used.values[j].astype(bool)  # (level, station); stored as int8
        if not used.any():
            continue
        st = used.any(0)
        lat, lon = d.station_lat.values[j][st], d.station_lon.values[j][st] % 360
        dist, nn = tree.query(xyz(lat, lon), k=4)
        w = 1.0 / np.maximum(dist, 1e-9)
        w /= w.sum(1, keepdims=True)
        date = pd.Timestamp(t).strftime('%Y-%m-%dT%H')
        e = era5_q(date)
        for k, p in enumerate(lev):
            m = used[k][st]
            if not m.any() or f'q_{int(p)}' not in e:
                continue
            o = d.q_observation.values[j, k][st][m] * 1e3
            b = d.q_background.values[j, k][st][m] * 1e3
            er = (e[f'q_{int(p)}'][nn[m]] * w[m]).sum(1) * 1e3
            bp = np.full(m.sum(), np.nan)
            if int(p) in ps_lev and j < bg_ps.sizes['time']:
                bp = (bg_ps.q.isel(time=j).sel(level_q=int(p)).values[nn[m]] * w[m]).sum(1) * 1e3
            la = lat[m]
            rows.append(pd.DataFrame({'p': int(p), 'lead_h': int(d.lead_time_hours.values[j]), 'hour': pd.Timestamp(t).hour,
                                      'region': np.where(la > 20, 'NH', np.where(la < -20, 'SH', 'Tropics')),
                                      'O': o, 'B': b, 'ERA5': er, 'Bps': bp}))
    print(cyc, flush=True)
df = pd.concat(rows, ignore_index=True)
df['OmB'], df['OmE'], df['BmE'], df['OmBps'] = df.O - df.B, df.O - df.ERA5, df.B - df.ERA5, df.O - df.Bps


def stat(g, col):
    x = g[col].dropna()
    return f'{x.mean():+.3f}/{np.sqrt((x ** 2).mean()):.3f}' if len(x) else '      --     '


first = df[df.lead_h < 12]
print('\nby level, mean/rms (g/kg). O-B and B-ERA5 over all leads; O-ERA5 over each cycle\'s first 12 h (each sounding once)')
print(' p     n_first   O mean   O-ERA5          O-B             B-ERA5          O-B_ps(ps-only bg)')
for p, g in df.groupby('p', sort=False):
    g1 = first[first.p == p]
    print(f'{p:5d} {len(g1):8d}  {g1.O.mean():6.2f}   {stat(g1, "OmE"):14s}  {stat(g, "OmB"):14s}  {stat(g, "BmE"):14s}  {stat(g, "OmBps")}')
print('\nO-B mean (g/kg) by lead day in the window (raobs run | ps-only background)')
df['day'] = df.lead_h // 24
for p in (1000, 850, 700, 500):
    g = df[df.p == p]
    print(f'{p:5d}  ' + '  '.join(f'd{dd}: {gg.OmB.mean():+.3f} | {gg.OmBps.mean():+.3f}' for dd, gg in g.groupby('day')))
print('\nO-ERA5 mean (g/kg), first 12 h, by synoptic hour and region (daytime sonde dry bias would show as 12Z vs 00Z by region)')
for p in (925, 850, 700, 500):
    g = first[first.p == p]
    print(f'{p:5d}  ' + '  '.join(f'{r} {h:02d}Z {gg.OmE.mean():+.3f} (n={len(gg)})' for (r, h), gg in g[g.hour.isin([0, 12])].groupby(['region', 'hour'])))
