"""ps vs ps + raobs vs ps + ATMS: RMSE vs ERA5 across the 5-day window, mean
over the first 10 cycles (2015-01-01T00 .. 01-05T12), analysis (solid) and
background (dashed); rows = regions, columns = z500, q850, 250 hPa wind speed.
ps only and ps + raobs: Bo's 324-cycle runs (H100, flash); ps + ATMS: ours (MI300X, flex).

Writes figures/compare_10cyc/rmse_window.png and rmse_window.csv.
Usage: python compare_obs_10cyc.py [OUT_DIR] [label=dir ...]  (extra runs)
"""

import csv
import glob
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, 'diagnostics')
sys.path.insert(0, 'backends/aifs')
import aifs_ic  # noqa: E402
import z500err_window as zw  # noqa: E402

CACHE = '/scratch4/BMC/gsienkf/Sergey.Frolov/ic_cache/ic_cache_aifs2-1.0deg-v1.0/'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figures/compare_10cyc'
NCYC = int(os.environ.get('NCYC', 10))  # cycles scored from 2015-01-01T00
BASE = os.environ.get('BASE', '1') != '0'  # BASE=0: only the runs given on the command line
BG = os.environ.get('BG', '1') != '0'      # BG=0: analysis trajectories only, no background
# FLOOR=csv (region,var,rmse_12h): dashed reference line per panel, e.g. 12 h forecasts from ERA5
FLOOR = {}
if os.environ.get('FLOOR'):
    with open(os.environ['FLOOR']) as fh:
        FLOOR = {(r['region'], r['var']): float(r['rmse_12h']) for r in csv.DictReader(fh)}
os.makedirs(OUT, exist_ok=True)
BO = '/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/add-raobs-parquet/lw4dvar-torch-test1/output/'
RUNS = [('ps only (Bo)', BO + 'smoke_test_aifs2-1.0deg-v1.0-pretraining-ps-parquet-new', 'k'),
        ('ps + raobs (Bo)', BO + 'smoke_test_aifs2-1.0deg-v1.0-pretraining-ps-raobs-parquet-new', 'tab:blue'),
        ('ps + ATMS ch 5-7', 'output/lwatms_e1_ps_atms', 'tab:red')][:None if NCYC <= 10 else 2]  # E1 has 10 cycles
RUNS = RUNS if BASE else []
RUNS += [(*a.rsplit('=', 1), c) for a, c in zip(sys.argv[2:], ['tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:gray', 'tab:cyan', 'tab:olive', 'tab:pink'])]
VARS = [('z500', 'z500 (m)'), ('t500', 't500 (K)'), ('q850', 'q850 (g/kg)'), ('ws250', '250 hPa wind speed (m/s)')]
REGIONS = [('Global', 'Global'), ('NH', 'NH (> 20N)'), ('Tropics', 'Tropics (20S-20N)'), ('SH', 'SH (< 20S)')]
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
_era5 = {}


Z37 = '/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr'


def run_truth(run):
    """each run is scored against its own model's ERA5 (as lw4dvar's get_verif does): the 37-level zarr for
    aifs37/r2 runs, Bo's ic_cache (= the v8 zarr) otherwise. They differ by ~2.5 m z500 rms (2026-11-18 audit)."""
    cfgs = sorted(glob.glob(os.path.join(run, 'config.yml*')))
    return 'zarr' if cfgs and 'aifs37: True' in open(cfgs[-1]).read() else 'cache'


def _zarr37(date, keys=('z_500', 't_500', 'q_850', 'u_250', 'v_250')):
    import zarr
    z = zarr.open(Z37, 'r')
    var = list(z.attrs['variables'])
    i = int(np.where(np.asarray(z['dates'][:]).astype('datetime64[s]') == np.datetime64(date + ':00'))[0][0])
    row = np.asarray(z['data'][i, :, 0, :])
    return {k: row[var.index(k)] for k in keys}


def truth(date, src='cache'):
    if (src, date) not in _era5:
        e = _zarr37(date) if src == 'zarr' else aifs_ic.read_single_date_fields(None, datetime.strptime(date, '%Y-%m-%dT%H'), CACHE)
        _era5[src, date] = {'z500': e['z_500'] / zw.GRAV, 't500': e['t_500'], 'q850': e['q_850'] * 1e3,
                       'ws250': np.hypot(e['u_250'], e['v_250'])}
    return _era5[src, date]


def curve(nc, t0, src='cache'):
    """lead hours from window start t0, {var: {region: rmse(lead)}}"""
    ds = xr.open_dataset(nc)
    start = zw._parse_nc_date(ds.attrs['trajectory_start_date'])
    shift = int((datetime.strptime(start, '%Y-%m-%dT%H') - datetime.strptime(t0, '%Y-%m-%dT%H')).total_seconds() // 3600)
    masks = zw.region_masks(ds.latitude.values)
    f = {'z500': ds.z.sel(level_z=500).values / zw.GRAV, 't500': ds.t.sel(level_t=500).values,
         'q850': ds.q.sel(level_q=850).values * 1e3,
         'ws250': np.hypot(ds.u.sel(level_u=250).values, ds.v.sel(level_v=250).values)}
    nt = f['z500'].shape[0]
    out = {v: {r: np.full(nt, np.nan) for r in masks} for v, _ in VARS}
    for t in range(nt):
        tr = truth(zw.add_hours(start, 6 * t), src)
        for v, _ in VARS:
            for r, m in masks.items():
                out[v][r][t] = zw.getrms(f[v][t] - tr[v], m)
    return shift + 6 * np.arange(nt), out


mean = {}
for _, run, _ in RUNS:
    for kind in ('control', 'optimal') if BG else ('optimal',):
        fs = sorted(glob.glob(f'{run}/*_{kind}_forecast_120h_100it.nc'))[:NCYC]
        assert len(fs) == NCYC, (run, kind, len(fs))
        cs = [curve(f, os.path.basename(f)[:13], run_truth(run)) for f in fs]
        mean[run, kind] = (cs[0][0], {v: {r: np.nanmean([c[1][v][r] for c in cs], 0) for r, _ in REGIONS} for v, _ in VARS})
    print(run, 'scored', flush=True)

fig, axes = plt.subplots(len(REGIONS), len(VARS), figsize=(4.7 * len(VARS), 14), sharex=True)
for i, (reg, rtitle) in enumerate(REGIONS):
    for j, (v, vlabel) in enumerate(VARS):
        ax = axes[i, j]
        for label, run, color in RUNS:
            lo, o = mean[run, 'optimal']
            ax.plot(lo, o[v][reg], color=color, label=label, marker='.' if not BG else None)
            if BG:
                lc, c = mean[run, 'control']
                ax.plot(lc, c[v][reg], color=color, ls='--', lw=1)
        if (reg, v) in FLOOR:
            ax.axhline(FLOOR[reg, v], color='k', ls='--', lw=1.2, label='12 h forecast from ERA5')
        ax.set_title(f'{rtitle}: {vlabel}')
        if i == len(REGIONS) - 1:
            ax.set_xlabel('hours into the 4D-Var window')
        if j == 0:
            ax.set_ylabel('RMSE vs ERA5')
axes[0, 0].legend(fontsize=9)
fig.suptitle(f'RMSE vs ERA5 across the 5-day window, mean over the first {NCYC} cycles from 2015-01-01T00\n'
             + ('solid: analysis trajectory, dashed: background' if BG else 'analysis trajectory'), y=1.0)
fig.tight_layout()
fig.savefig(f'{OUT}/rmse_window.png', dpi=150, bbox_inches='tight')
print('wrote', f'{OUT}/rmse_window.png')

with open(f'{OUT}/rmse_window.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['run', 'region', 'var', 'analysis_mean', 'analysis_24h', 'analysis_end', 'background_mean'])
    for label, run, _ in RUNS:
        lo, o = mean[run, 'optimal']
        c = mean[run, 'control'][1] if BG else None
        for reg, _ in REGIONS:
            for v, _ in VARS:
                w.writerow([label, reg, v, f'{np.nanmean(o[v][reg]):.4f}', f'{o[v][reg][lo == 24][0]:.4f}',
                            f'{o[v][reg][-1]:.4f}', f'{np.nanmean(c[v][reg]):.4f}' if BG else ''])
print('wrote', f'{OUT}/rmse_window.csv')
