"""Aggregate forecast_skill.py output: RMSE vs lead, mean over the init dates common to every
label, against ERA5 (area-weighted fields) and against obs (O - F pooled over dates).

Writes <OUT>/fcst_era5.png, <OUT>/fcst_obs.png and <OUT>/fcst_skill.csv.
Usage: python fcst_skill_plot.py OUT LABEL[=legend] ... [--start T] [--end T]
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('labels', nargs='+', help='LABEL or LABEL=legend')
ap.add_argument('--start', default='')
ap.add_argument('--end', default='9999')
a = ap.parse_args()
OUT, start, end = a.out, a.start, a.end
labels = [(lab.split('=', 1)[0], lab.split('=', 1)[-1]) for lab in a.labels]
os.makedirs(OUT, exist_ok=True)
REGIONS = ['Global', 'NH', 'Tropics', 'SH']
ERA_VARS = [('z500', 'z500 (m)'), ('t850', 't850 (K)'), ('q850', 'q850 (g/kg)'), ('vw250', '250 hPa vector wind (m/s)'),
            ('msl', 'msl (hPa)')]
OBS_VARS = [('ps', None, 'surface pressure (hPa)', 1.), ('t', 850., 'raob t850 (K)', 1.), ('t', 500., 'raob t500 (K)', 1.),
            ('q', 850., 'raob q850 (g/kg)', 1e3), ('q', 700., 'raob q700 (g/kg)', 1e3), ('uv', 250., 'raob 250 hPa vector wind (m/s)', 1.)]
COLORS = ['k', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']

files = {lab: {os.path.basename(f)[:13]: f for f in glob.glob(f'output/fcst_skill/{lab}/*.nc')} for lab, _ in labels}
dates = sorted(d for d in set.intersection(*[set(v) for v in files.values()]) if start <= d <= end)
assert dates, 'no common init dates'
print(f'{len(dates)} common init dates, {dates[0]} .. {dates[-1]}')


def load(lab):
    dss = [xr.open_dataset(files[lab][d]) for d in dates]
    era = {v: np.mean([ds[f'rmse_{v}'].values for ds in dss], 0) for v, _ in ERA_VARS}
    obs = {}
    for v, p, _, scale in OBS_VARS:
        parts = ['u', 'v'] if v == 'uv' else [v]
        n = s2 = 0
        for b in parts:
            a = {s: sum(ds[f'obs_{b}_{s}'] for ds in dss) for s in ('n', 's2')}
            if p is not None:
                a = {s: x.sel({f'level_{b}': p}) for s, x in a.items()}
            n, s2 = n + a['n'].values, s2 + a['s2'].values
        nn = n / len(parts)
        obs[v, p] = (np.sqrt(s2 / nn) * scale, nn)
    return dss[0].lead_hours.values, era, obs


res = {lab: load(lab) for lab, _ in labels}
lead = res[labels[0][0]][0]


def panel(kind, vars_, fname, title):
    fig, axes = plt.subplots(len(vars_), len(REGIONS), figsize=(16, 3.2 * len(vars_)), sharex=True, squeeze=False)
    for i, var in enumerate(vars_):
        for j, r in enumerate(REGIONS):
            ax = axes[i, j]
            for (lab, legend), c in zip(labels, COLORS):
                _, era, obs = res[lab]
                y = era[var[0]][:, j] if kind == 'era5' else obs[var[0], var[1]][0][:, j]
                ax.plot(lead / 24, y, color=c, label=legend)
            ax.set_title(f'{r}: {var[1] if kind == "era5" else var[2]}', fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(vars_) - 1:
                ax.set_xlabel('forecast lead (days)')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f'{title}, mean over {len(dates)} daily 00Z forecasts {dates[0][:10]} .. {dates[-1][:10]}', y=1.0)
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}', dpi=130, bbox_inches='tight')
    print('wrote', f'{OUT}/{fname}')


panel('era5', ERA_VARS, 'fcst_era5.png', 'Forecast RMSE vs ERA5')
panel('obs', OBS_VARS, 'fcst_obs.png', 'Forecast RMS(O - F) vs obs')
with open(f'{OUT}/fcst_skill.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['label', 'kind', 'var', 'region'] + [f'{int(h)}h' for h in lead])
    for lab, legend in labels:
        _, era, obs = res[lab]
        for v, _ in ERA_VARS:
            for j, r in enumerate(REGIONS):
                w.writerow([legend, 'era5', v, r] + [f'{x:.4f}' for x in era[v][:, j]])
        for v, p, name, _ in OBS_VARS:
            for j, r in enumerate(REGIONS):
                w.writerow([legend, 'obs', f'{v}{"" if p is None else int(p)}', r] + [f'{x:.4f}' for x in obs[v, p][0][:, j]])
print('wrote', f'{OUT}/fcst_skill.csv')
