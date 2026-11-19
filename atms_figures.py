"""Shareable figures for the ATMS E1 experiments (notes/integrate_atms_obs.md),
in the style of Hakim et al. 2026 (arXiv 2608.11515):

  fig1_z500_window.png   z500 RMSE vs ERA5 across the 5-day window, mean over
                         cycles, background (dashed) and analysis (solid);
                         NH / tropics / SH x Jan / Feb          (paper Fig. 1)
  fig2_z500_cycles.png   NH and SH z500 RMSE per cycle, 24 h into the window
                         and at its end                          (paper Fig. 2)
  fig3_u_zonal.png       zonal-mean u RMSE of the analyses (lat x pressure),
                         ps + ATMS and its difference from ps only (paper Fig. 3)
  fig4_tcw_maps.png      total column water at the last Feb analysis: ps + ATMS,
                         ERA5, ps only, and the errors / ATMS impact (paper Fig. 4)
  figS1_obs_map.png      ps and ATMS obs in one 6-h slot          (paper Fig. S1)
  figS2_atms_fits.png    ATMS O-B per channel and cycle, used and withheld obs

"Analysis" = the `*_optimal_inputs_*.nc` state (window start + 6 h, all 14
levels); ERA5 from the cached GRIBs. Usage: python atms_figures.py [OUT_DIR]
"""

import glob
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
from forecast_model import area_weights_from_lats  # noqa: E402

CACHE = '/scratch4/BMC/gsienkf/Sergey.Frolov/ic_cache/ic_cache_aifs2-1.0deg-v1.0/'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figures/atms_e1'
os.makedirs(OUT, exist_ok=True)

# period -> [(label, run dir, style)]
STYLE = {'ps': dict(color='k'), 'ps2': dict(color='0.6', lw=1),
         'atms': dict(color='tab:red'), 'dz': dict(color='tab:orange', lw=1)}
RUNS = {
    'Jan 2015': [('ps only', 'lwatms_e1_ps', 'ps'),
                 ('ps only (2nd run)', 'lwatms_e1_ps_atmsmon', 'ps2'),
                 ('ps + ATMS', 'lwatms_e1_ps_atms', 'atms'),
                 ('ps + ATMS, Desroziers $\\sigma_o$', 'lwatms_e1b_ps_atms_dz', 'dz')],
    'Feb 2015': [('ps only', 'lwatms_e1feb_ps', 'ps'),
                 ('ps + ATMS', 'lwatms_e1feb_ps_atms', 'atms'),
                 ('ps + ATMS, Desroziers $\\sigma_o$', 'lwatms_e1feb_ps_atms_dz', 'dz')],
}
MAIN = lambda runs: [r for r in runs if r[2] in ('ps', 'atms')]  # noqa: E731  figs 1-2: ps only vs ps + ATMS
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
_era5 = {}


def era5(date):
    if date not in _era5:
        _era5[date] = aifs_ic.read_single_date_fields(None, datetime.strptime(date, '%Y-%m-%dT%H'), CACHE)
    return _era5[date]


def files(run, kind):
    return sorted(glob.glob(f'output/{run}/*_{kind}_120h_100it.nc'))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(OUT, name), flush=True)


# ---- per-cycle z500 error curves, shared by figs 1 and 2
curves = {}  # run -> list of (control lead, control errs, optimal lead (window axis), optimal errs, date)
for period, runs in RUNS.items():
    for _, run, _ in runs:
        out = []
        for cf in files(run, 'control_forecast'):
            of = cf.replace('_control_forecast_', '_optimal_forecast_')
            sc, lc, ec, _ = zw.z500_error_curve(cf, CACHE)
            so, lo, eo, _ = zw.z500_error_curve(of, CACHE)
            shift = (datetime.strptime(so, '%Y-%m-%dT%H') - datetime.strptime(sc, '%Y-%m-%dT%H')).total_seconds() / 3600
            out.append((lc, ec, lo + int(shift), eo, sc))
        curves[run] = out
        print(run, len(out), 'cycles', flush=True)


def mean_curve(run, kind, region, skip_first=False):
    rows = curves[run][1 if skip_first else 0:]
    i = 0 if kind == 'control' else 2
    lead = rows[0][i]
    return lead, np.nanmean([r[i + 1][region] for r in rows], 0)


# ---- fig 1: z500 RMSE across the window
regions = [('NH', 'NH (> 20N)'), ('Tropics', 'Tropics (20S-20N)'), ('SH', 'SH (< 20S)')]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for row, (period, runs) in enumerate(RUNS.items()):
    for col, (reg, title) in enumerate(regions):
        ax = axes[row, col]
        for label, run, st in MAIN(runs):
            lc, ec = mean_curve(run, 'control', reg, skip_first=True)
            lo, eo = mean_curve(run, 'optimal', reg, skip_first=True)
            ax.plot(lo, eo, **STYLE[st], label=label)
            ax.plot(lc, ec, ls='--', **STYLE[st])
        ax.set_title(f'{period}, {title}')
        if col == 0:
            ax.set_ylabel('z500 RMSE vs ERA5 (m)')
        if row == 1:
            ax.set_xlabel('hours into the 4D-Var window')
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc='upper left')
fig.suptitle('z500 error across the 5-day window, mean over cycles 2-10 '
             '(solid: analysis trajectory, dashed: background)', y=1.0)
fig.tight_layout()
save(fig, 'fig1_z500_window.png')

# ---- fig 2: z500 RMSE per cycle at 24 h and end of window
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey='row')
for col, (period, runs) in enumerate(RUNS.items()):
    for row, reg in enumerate(['NH', 'SH']):
        ax = axes[row, col]
        for label, run, st in MAIN(runs):
            d = [datetime.strptime(r[4], '%Y-%m-%dT%H') for r in curves[run]]
            e24 = [r[3][reg][r[2] == 24][0] for r in curves[run]]
            eend = [r[3][reg][-1] for r in curves[run]]
            ax.plot(d, e24, marker='o', ms=3, **STYLE[st], label=f'{label}, 24 h')
            ax.plot(d, eend, marker='s', ms=3, ls=':', **STYLE[st])
        ax.set_title(f'{period}, {reg}: z500 RMSE of the analysis trajectory')
        ax.tick_params(axis='x', labelrotation=30)
        if col == 0:
            ax.set_ylabel('m')
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle('Per cycle (window start on x): 24 h into the window (solid, o) and at its end (dotted, s)', y=1.0)
fig.tight_layout()
save(fig, 'fig2_z500_cycles.png')


# ---- fig 3: zonal-mean u RMSE of the analyses
def u_err_sq(run):
    """(levels, lat rings, mean over time and longitude of (u_a - u_ERA5)^2), cycles 2-10."""
    fs = files(run, 'optimal_inputs')[1:]
    acc = None
    for f in fs:
        d = xr.open_dataset(f)
        date = datetime.fromisoformat(d.attrs['valid_date']).strftime('%Y-%m-%dT%H')
        levs = d.level_u.values
        e = era5(date)
        diff2 = np.stack([(d.u.sel(level_u=p).values - e[f'u_{p}']) ** 2 for p in levs])
        acc = diff2 if acc is None else acc + diff2
        lat = d.latitude.values
    rings, inv = np.unique(lat, return_inverse=True)
    zm = np.stack([np.bincount(inv, weights=row) / np.bincount(inv) for row in acc / len(fs)])
    return levs, rings, zm


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for row, (period, runs) in enumerate(RUNS.items()):
    r = {st: run for _, run, st in runs}
    levs, lat, zp = u_err_sq(r['ps'])
    _, _, za = u_err_sq(r['atms'])
    ra, rp = np.sqrt(za), np.sqrt(zp)
    c = axes[row, 0].contourf(lat, levs, ra, levels=np.arange(0, 6.1, 0.5), cmap='viridis', extend='max')
    fig.colorbar(c, ax=axes[row, 0], label='m/s')
    axes[row, 0].set_title(f'{period}: u RMSE, ps + ATMS analyses')
    c = axes[row, 1].contourf(lat, levs, ra - rp, levels=np.arange(-1.5, 1.51, 0.25), cmap='RdBu_r', extend='both')
    fig.colorbar(c, ax=axes[row, 1], label='m/s')
    axes[row, 1].set_title(f'{period}: (ps + ATMS) - (ps only); blue = ATMS better')
    for ax in axes[row]:
        ax.set_yscale('log')
        ax.set_ylim(1000, 10)
        ax.set_yticks([1000, 850, 700, 500, 300, 200, 100, 50, 10])
        ax.set_yticklabels([1000, 850, 700, 500, 300, 200, 100, 50, 10])
        ax.grid(False)
    axes[row, 0].set_ylabel('pressure (hPa)')
for ax in axes[1]:
    ax.set_xlabel('latitude')
fig.suptitle('Time-mean, zonal-mean zonal-wind RMS error vs ERA5 of the analyses, cycles 2-10', y=1.0)
fig.tight_layout()
save(fig, 'fig3_u_zonal.png')


# ---- fig 4: tcw maps at the last Feb analysis
def to_latlon(lat, lon, val, res=1.0):
    from scipy.spatial import cKDTree
    xyz = lambda la, lo: np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),  # noqa: E731
                                   np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))], -1)
    glat, glon = np.arange(-90 + res / 2, 90, res), np.arange(0, 360, res)
    LO, LA = np.meshgrid(glon, glat)
    _, idx = cKDTree(xyz(lat, lon)).query(xyz(LA.ravel(), LO.ravel()))
    return glon, glat, np.asarray(val)[idx].reshape(LA.shape)


import cartopy.crs as ccrs  # noqa: E402

feb = {st: run for _, run, st in RUNS['Feb 2015']}
fa = xr.open_dataset(files(feb['atms'], 'optimal_inputs')[-1])
fp = xr.open_dataset(files(feb['ps'], 'optimal_inputs')[-1])
date = datetime.fromisoformat(fa.attrs['valid_date']).strftime('%Y-%m-%dT%H')
assert date == datetime.fromisoformat(fp.attrs['valid_date']).strftime('%Y-%m-%dT%H')
lat, lon = fa.latitude.values, fa.longitude.values % 360
w = area_weights_from_lats(lat)
ta, tp, te = fa.tcw.values, fp.tcw.values, era5(date)['tcw']
rms = lambda x: np.sqrt((w * x ** 2).sum() / w.sum())  # noqa: E731
panels = [(ta, 'ps + ATMS analysis', 'tcw'), (te, 'ERA5', 'tcw'), (tp, 'ps-only analysis', 'tcw'),
          (ta - te, f'ps + ATMS - ERA5 (RMS {rms(ta - te):.2f})', 'err'),
          (ta - tp, 'ATMS impact: (ps + ATMS) - (ps only)', 'err'),
          (tp - te, f'ps only - ERA5 (RMS {rms(tp - te):.2f})', 'err')]
fig, axes = plt.subplots(2, 3, figsize=(16, 6.5), subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180)))
for ax, (v, title, kind) in zip(axes.ravel(), panels):
    glon, glat, g = to_latlon(lat, lon, v)
    if kind == 'tcw':
        m = ax.pcolormesh(glon, glat, g, transform=ccrs.PlateCarree(), cmap='YlGnBu', vmin=0, vmax=60, shading='auto')
    else:
        m = ax.pcolormesh(glon, glat, g, transform=ccrs.PlateCarree(), cmap='BrBG', vmin=-8, vmax=8, shading='auto')
    ax.coastlines(lw=0.5)
    ax.set_title(title, fontsize=10)
    fig.colorbar(m, ax=ax, shrink=0.7, label='kg m$^{-2}$')
fig.suptitle(f'Total column water at {date} (analysis after 10 cycles from 2015-02-01T00)', y=1.0)
fig.tight_layout()
save(fig, 'fig4_tcw_maps.png')

# ---- fig S1: obs in one slot
jan = {st: run for _, run, st in RUNS['Jan 2015']}
f0 = sorted(glob.glob(f"output/{jan['atms']}/*_atms_diagnostics_120h_100it.nc"))[0]
a = xr.open_dataset(f0)
p = xr.open_dataset(f0.replace('_atms_diagnostics_', '_observation_diagnostics_'))
slot = 1
t = str(a.time.values[slot])[:13]
fig, ax = plt.subplots(figsize=(11, 5.5), subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180)))
ax.coastlines(lw=0.5)
scr = a.screened.values[slot].astype(bool)
wh = a.withheld.values[slot].astype(bool)
used = a.used.values[slot, :, 0].astype(bool)
pu = p.surface_pressure_used.values[slot].astype(bool)
kw = dict(transform=ccrs.PlateCarree(), s=2, lw=0)
ax.scatter(a.longitude.values[slot][scr & wh], a.latitude.values[slot][scr & wh], c='tab:orange', **kw,
           label=f'ATMS withheld ({int((scr & wh).sum())})')
ax.scatter(a.longitude.values[slot][used], a.latitude.values[slot][used], c='tab:red', **kw,
           label=f'ATMS assimilated, ch 5 ({int(used.sum())})')
ax.scatter(p.observation_longitude.values[slot][pu], p.observation_latitude.values[slot][pu], c='k', **kw,
           label=f'surface pressure ({int(pu.sum())})')
ax.legend(loc='lower left', markerscale=4, fontsize=8)
ax.set_global()
ax.set_title(f'Observations in the 6-h slot at {t} (S-NPP ATMS: clear-sky ocean only)')
save(fig, 'figS1_obs_map.png')

# ---- fig S2: ATMS fits per channel and cycle (Jan: ATMS assimilated vs monitored only)
chans = [5, 6, 7, 8, 9, 10, 11]
fig, axes = plt.subplots(1, len(chans), figsize=(18, 3.6), sharex=True)
for label, run, st in RUNS['Jan 2015'][1:] + RUNS['Feb 2015'][1:]:
    period = 'Jan' if run in [r for _, r, _ in RUNS['Jan 2015']] else 'Feb'
    rows = []
    for f in sorted(glob.glob(f'output/{run}/*_atms_diagnostics_120h_100it.nc')):
        d = xr.open_dataset(f)
        wh = (d.withheld.values.astype(bool) & d.screened.values.astype(bool))
        ok = np.abs(d.omb.values) <= 4 * d.sigma_o.values
        rows.append([np.sqrt(np.nanmean(d.omb.values[..., i][wh & ok[..., i]] ** 2)) for i in range(len(chans))])
    rows = np.array(rows)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(1, len(rows) + 1), rows[:, i], ls='-' if period == 'Jan' else '--', marker='o', ms=3,
                **STYLE[st], label=f'{period}: {label}' if label != 'ps only (2nd run)' else 'Jan: ps only (ATMS monitored)')
for ax, c in zip(axes, chans):
    ax.set_title(f'ch {c}' + (' (assimilated)' if c <= 7 else ' (monitored)'), fontsize=9)
    ax.set_xlabel('cycle')
axes[0].set_ylabel('RMS O-B, withheld obs (K)')
axes[0].legend(fontsize=7)
fig.suptitle('Background fit to withheld (never assimilated) ATMS obs', y=1.02)
fig.tight_layout()
save(fig, 'figS2_atms_fits.png')
