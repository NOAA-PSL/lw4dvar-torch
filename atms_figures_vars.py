"""Companions to atms_figures.py's fig1_z500_window.png for q850 and 250 hPa
wind speed: RMSE vs ERA5 across the 5-day window, mean over cycles 2-10,
analysis (solid) and background (dashed), ps only vs ps + ATMS; NH /
tropics / SH x Jan / Feb. Writes fig1_q850_window.png, fig1_ws250_window.png.

Usage: python atms_figures_vars.py [OUT_DIR]
"""

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
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figures/atms_e1'
RUNS = {'Jan 2015': [('ps only', 'lwatms_e1_ps', 'k'), ('ps + ATMS', 'lwatms_e1_ps_atms', 'tab:red')],
        'Feb 2015': [('ps only', 'lwatms_e1feb_ps', 'k'), ('ps + ATMS', 'lwatms_e1feb_ps_atms', 'tab:red')]}
VARS = {'q850': ('q850 RMSE vs ERA5 (g/kg)', 'q850'), 'ws250': ('250 hPa wind speed RMSE vs ERA5 (m/s)', '250 hPa wind speed')}
REGIONS = [('NH', 'NH (> 20N)'), ('Tropics', 'Tropics (20S-20N)'), ('SH', 'SH (< 20S)')]
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
_era5 = {}


def truth(date):
    if date not in _era5:
        e = aifs_ic.read_single_date_fields(None, datetime.strptime(date, '%Y-%m-%dT%H'), CACHE)
        _era5[date] = {'q850': e['q_850'] * 1e3, 'ws250': np.hypot(e['u_250'], e['v_250'])}
    return _era5[date]


def curve(nc, t0):
    """lead hours from window start t0, {var: {region: rmse}}"""
    ds = xr.open_dataset(nc)
    start = zw._parse_nc_date(ds.attrs['trajectory_start_date'])
    shift = int((datetime.strptime(start, '%Y-%m-%dT%H') - datetime.strptime(t0, '%Y-%m-%dT%H')).total_seconds() // 3600)
    masks = zw.region_masks(ds.latitude.values)
    q = ds.q.sel(level_q=850).values * 1e3
    ws = np.hypot(ds.u.sel(level_u=250).values, ds.v.sel(level_v=250).values)
    nt = q.shape[0]
    out = {v: {r: np.full(nt, np.nan) for r in masks} for v in VARS}
    for t in range(nt):
        tr = truth(zw.add_hours(start, 6 * t))
        for v, f in (('q850', q[t]), ('ws250', ws[t])):
            for r, m in masks.items():
                out[v][r][t] = zw.getrms(f - tr[v], m)
    return shift + 6 * np.arange(nt), out


mean = {}  # (run, kind) -> (lead, {var: {region: mean over cycles 2-10}})
for runs in RUNS.values():
    for _, run, _ in runs:
        for kind in ('control', 'optimal'):
            fs = sorted(glob.glob(f'output/{run}/*_{kind}_forecast_120h_100it.nc'))[1:]
            cs = [curve(f, os.path.basename(f)[:13]) for f in fs]
            lead = cs[0][0]
            mean[run, kind] = (lead, {v: {r: np.nanmean([c[1][v][r] for c in cs], 0) for r, _ in REGIONS} for v in VARS})
        print(run, flush=True)

for v, (ylabel, name) in VARS.items():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, (period, runs) in enumerate(RUNS.items()):
        for col, (reg, title) in enumerate(REGIONS):
            ax = axes[row, col]
            for label, run, color in runs:
                lo, o = mean[run, 'optimal']
                lc, c = mean[run, 'control']
                ax.plot(lo, o[v][reg], color=color, label=label)
                ax.plot(lc, c[v][reg], color=color, ls='--')
            ax.set_title(f'{period}, {title}')
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel('hours into the 4D-Var window')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(f'{name} error across the 5-day window, mean over cycles 2-10 '
                 '(solid: analysis trajectory, dashed: background)', y=1.0)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_{v}_window.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', f'{OUT}/fig1_{v}_window.png')
    # numbers for the slide text: mean over the window, analysis
    for period, runs in RUNS.items():
        for reg, _ in REGIONS:
            vals = [np.nanmean(mean[run, 'optimal'][1][v][reg]) for _, run, _ in runs]
            print(f'  {v} {period} {reg}: ps {vals[0]:.3f} -> ATMS {vals[1]:.3f} ({100 * (vals[1] - vals[0]) / vals[0]:+.1f}%)')
