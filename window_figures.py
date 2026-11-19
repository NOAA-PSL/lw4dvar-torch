"""Window-length test (notes/integrate_atms_obs.md): first cycle 2015-01-01T00,
4/5/7/10-day windows, 300 epochs, ps only vs ps + ATMS.

  winfig1_<region>.png  RMSE vs ERA5 across the window for z500 (m), q850
                        (g/kg) and 250 hPa wind speed (m/s); solid analysis,
                        dashed background, colour = window length
                        (Hakim et al. 2026, Fig. 1)
  winfig2_loss.png      Jtot / Jtot(epoch 1) vs epoch, and the drop over the
                        last 50 epochs -- is the loss still falling?
  window_scores.csv     the numbers behind winfig1, plus loss decrease

Usage: python window_figures.py [OUT_DIR] [MAX_EPOCH]
"""

import csv
import glob
import json
import os
import sys
from datetime import datetime, timedelta

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
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figures/window_length'
NIT = sys.argv[2] if len(sys.argv) > 2 else '300'
os.makedirs(OUT, exist_ok=True)
DAYS = [4, 5, 7, 10]
ARMS = [('ps', 'ps only'), ('ps_atms', 'ps + ATMS')]
COLORS = {4: 'tab:blue', 5: 'tab:green', 7: 'tab:orange', 10: 'tab:red'}
VARS = [('z500', 'z500 (m)'), ('q850', 'q850 (g/kg)'), ('ws250', '250 hPa wind speed (m/s)')]
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
_era5 = {}


def era5(date):
    if date not in _era5:
        _era5[date] = aifs_ic.read_single_date_fields(None, datetime.strptime(date, '%Y-%m-%dT%H'), CACHE)
    return _era5[date]


def fields(ds, t):
    return {'z500': ds.z.isel(time=t).sel(level_z=500).values / zw.GRAV,
            'q850': ds.q.isel(time=t).sel(level_q=850).values * 1e3,
            'ws250': np.hypot(ds.u.isel(time=t).sel(level_u=250).values, ds.v.isel(time=t).sel(level_v=250).values)}


def truth(date):
    e = era5(date)
    return {'z500': e['z_500'] / zw.GRAV, 'q850': e['q_850'] * 1e3, 'ws250': np.hypot(e['u_250'], e['v_250'])}


def score(nc, t0):
    """{region: {var: rmse(lead)}}, lead hours from the window start t0."""
    ds = xr.open_dataset(nc)
    start = zw._parse_nc_date(ds.attrs['trajectory_start_date'])
    shift = int((datetime.strptime(start, '%Y-%m-%dT%H') - datetime.strptime(t0, '%Y-%m-%dT%H')).total_seconds() // 3600)
    masks = zw.region_masks(ds.latitude.values)
    nt = ds.sizes['time']
    lead = shift + 6 * np.arange(nt)
    out = {r: {v: np.full(nt, np.nan) for v, _ in VARS} for r in masks}
    for t in range(nt):
        f, tr = fields(ds, t), truth(zw.add_hours(start, 6 * t))
        for r, m in masks.items():
            for v, _ in VARS:
                out[r][v][t] = zw.getrms(f[v] - tr[v], m)
    return lead, out


res, loss = {}, {}
for days in DAYS:
    for arm, _ in ARMS:
        run = f'lwatms_win{days}d_{arm}'
        cf = glob.glob(f'output/{run}/2015-01-01T00_control_forecast_{24 * days}h_{NIT}it.nc')
        if not cf:
            print('missing', run)
            continue
        res[days, arm] = {k: score(cf[0].replace('_control_forecast_', f'_{k}_forecast_'), '2015-01-01T00')
                          for k in ('control', 'optimal')}
        h = json.load(open(f'output/{run}/2015-01-01T00_loss_latent_{24 * days}h_{NIT}it.json'))
        loss[days, arm] = (np.array([e['epoch'] for e in h]), np.array([e['loss'] for e in h]),
                           np.array([sum(e['all_loss']) for e in h]))
        print(run, 'scored', flush=True)

for region in ['Global', 'NH', 'Tropics', 'SH']:
    fig, axes = plt.subplots(len(ARMS), len(VARS), figsize=(14, 7.5), sharex=True, sharey='col')
    for i, (arm, alabel) in enumerate(ARMS):
        for j, (v, vlabel) in enumerate(VARS):
            ax = axes[i, j]
            for days in DAYS:
                if (days, arm) not in res:
                    continue
                for k, ls in (('optimal', '-'), ('control', '--')):
                    lead, sc = res[days, arm][k]
                    ax.plot(lead / 24, sc[region][v], ls=ls, color=COLORS[days],
                            label=f'{days}-day window' if k == 'optimal' else None)
            ax.set_title(f'{alabel}: {vlabel}')
            if i == len(ARMS) - 1:
                ax.set_xlabel('days into the window')
            if j == 0:
                ax.set_ylabel(f'RMSE vs ERA5, {region}')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'First cycle (2015-01-01T00), {NIT} epochs: analysis trajectory (solid) and background (dashed), {region}', y=1.0)
    fig.tight_layout()
    fig.savefig(f'{OUT}/winfig1_{region}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for (days, arm), (ep, J, jps) in loss.items():
    ls = '-' if arm == 'ps_atms' else '--'
    axes[0].semilogy(ep, J / J[0], ls=ls, color=COLORS[days], label=f'{days} d, {dict(ARMS)[arm]}')
    axes[1].semilogy(ep, jps / jps[0], ls=ls, color=COLORS[days])
    if arm == 'ps_atms':
        ja = J - jps
        axes[2].semilogy(ep, ja / ja[0], ls=ls, color=COLORS[days], label=f'{days} d')
axes[0].set_title('Jtot / Jtot(epoch 1)')
axes[1].set_title('Jo surface pressure / initial')
axes[2].set_title('Jo ATMS (Jtot - Jo ps) / initial, ps + ATMS runs')
for ax in axes:
    ax.set_xlabel('epoch')
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle('Is the loss still falling? (solid: ps + ATMS, dashed: ps only)', y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/winfig2_loss.png', dpi=150, bbox_inches='tight')
plt.close(fig)

with open(f'{OUT}/window_scores.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['days', 'arm', 'region', 'var', 'bg_mean', 'an_mean', 'an_day1', 'an_end',
                'J_epoch1', 'J_final', 'J_drop_last50_pct'])
    for (days, arm), r in res.items():
        ep, J, _ = loss[days, arm]
        drop50 = 100 * (J[-51] - J[-1]) / J[-51] if len(J) > 50 else np.nan
        for region in ['Global', 'NH', 'Tropics', 'SH']:
            for v, _ in VARS:
                lc, c = r['control']
                lo, o = r['optimal']
                w.writerow([days, arm, region, v, f'{np.nanmean(c[region][v]):.4f}', f'{np.nanmean(o[region][v]):.4f}',
                            f'{o[region][v][lo == 24][0]:.4f}', f'{o[region][v][-1]:.4f}',
                            f'{J[0]:.6g}', f'{J[-1]:.6g}', f'{drop50:.2f}'])
print('wrote', OUT)
