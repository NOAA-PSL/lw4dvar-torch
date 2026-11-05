"""Analysis RMSE vs ERA5 per DA cycle (window start on x), to see whether a
cycled run drifts or degrades over time.

For each run and cycle, reads `<date>_optimal_forecast_<suffix>.nc` (the
analysis trajectory) and scores z500, q850 and 250 hPa wind speed against
ERA5 at chosen hours into the window, in four regions (area-weighted, via
z500err_window.region_masks). Thin lines: per cycle; thick lines: running
mean over `--smooth` cycles. Per-cycle scores are cached per run in
`--cache` (JSON), so rerunning on a growing run only scores new cycles.

Usage (repo root):
  python diagnostics/analysis_error_ts.py OUT LABEL=RUN_DIR[@IC_CACHE] [...]
      [--leads 24 114] [--start 2015-01-01T00] [--end ...] [--smooth 14]
      [--cache output/analysis_error_ts] [--suffix SUFFIX] [--ic-cache DIR]
Each run is scored on its own grid against the ERA5 GRIBs in its IC_CACHE
(default --ic-cache, the 1-degree O96 cache), so runs at other resolutions
(e.g. Jeff Whitaker's 0.25-degree AIFS-single 2.0 runs, with
/scratch4/BMC/gsienkf/Jeffrey.Whitaker/long-window-4dvar-aifsv2/ic_cache)
can be compared. --suffix (e.g. 120h_100it) restricts the files read; by
default any `<date>_optimal_forecast_*.nc` is used (one per cycle).
Writes OUT/analysis_error_ts_<lead>h.png and OUT/analysis_error_ts.csv.
"""

import argparse
import collections
import csv
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backends', 'aifs'))
import aifs_ic  # noqa: E402
import z500err_window as zw  # noqa: E402

VARS = [('z500', 'z500 (m)'), ('q850', 'q850 (g/kg)'), ('ws250', '250 hPa wind speed (m/s)')]
REGIONS = [('Global', 'Global'), ('NH', 'NH (> 20N)'), ('Tropics', 'Tropics (20S-20N)'), ('SH', 'SH (< 20S)')]
COLORS = ['k', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:cyan', 'tab:olive',
          'tab:pink', 'gold', 'navy', 'magenta', 'lime']
_era5 = collections.OrderedDict()  # (cache, date) -> fields, least recently used first
_ERA5_KEEP = 64  # dates kept: a cycle's window plus the next cycles' overlap (0.25-degree fields are large)


def truth(date, cache_dir):
    """ERA5 at date from an ic_cache directory of GRIBs, or, with cache_dir = 'zarr:v8' / 'zarr:era5_37', from a
    training zarr (physical_scorecard.TRUTHS). Score each model against its own training data: the ic_cache GRIBs
    equal v8 (A0's); A1/A2 need 'zarr:era5_37' (the two differ by ~2.4 m in z500)."""
    key = (cache_dir, date)
    if key not in _era5 and cache_dir.startswith('zarr:'):
        import physical_scorecard as ps
        want = {'z': {500}, 'q': {850}, 'u': {250}, 'v': {250}, 't': set()}
        f = ps.read_era5(datetime.strptime(date, '%Y-%m-%dT%H'), want, cache_dir[5:])  # z in m, q in g/kg
        _era5[key] = {'z500': f[('z', 500)], 'q850': f[('q', 850)], 'ws250': np.hypot(f[('u', 250)], f[('v', 250)])}
    if key not in _era5:
        e = aifs_ic.read_single_date_fields(None, datetime.strptime(date, '%Y-%m-%dT%H'), cache_dir)
        _era5[key] = {'z500': e['z_500'] / zw.GRAV, 'q850': e['q_850'] * 1e3,
                      'ws250': np.hypot(e['u_250'], e['v_250'])}
        while len(_era5) > _ERA5_KEEP:
            _era5.popitem(last=False)
    _era5.move_to_end(key)
    return _era5[key]


def score_cycle(nc, t0, cache_dir):
    """{lead hours from window start t0 (str): {var: {region: rmse}}}"""
    with xr.open_dataset(nc) as ds:
        start = zw._parse_nc_date(ds.attrs['trajectory_start_date'])
        shift = int((datetime.strptime(start, '%Y-%m-%dT%H') - datetime.strptime(t0, '%Y-%m-%dT%H')).total_seconds()
                    // 3600)
        masks = zw.region_masks(ds.latitude.values)
        f = {'z500': ds.z.sel(level_z=500).values / zw.GRAV, 'q850': ds.q.sel(level_q=850).values * 1e3,
             'ws250': np.hypot(ds.u.sel(level_u=250).values, ds.v.sel(level_v=250).values)}
    out = {}
    for t in range(f['z500'].shape[0]):
        tr = truth(zw.add_hours(start, 6 * t), cache_dir)
        out[str(shift + 6 * t)] = {v: {r: float(zw.getrms(f[v][t] - tr[v], m)) for r, m in masks.items()}
                                   for v, _ in VARS}
    return out


def score_run(run, ic_cache, args):
    """{cycle date: scores}, from the JSON cache plus any newly scored cycles."""
    os.makedirs(args.cache, exist_ok=True)
    cfile = os.path.join(args.cache, os.path.abspath(run).strip('/').replace('/', '__') + '.json')
    scores = {}
    if os.path.exists(cfile):
        with open(cfile) as fh:
            scores = json.load(fh)
    new = 0
    for nc in sorted(glob.glob(f'{run}/*_optimal_forecast_{args.suffix or "*"}.nc')):
        date = os.path.basename(nc)[:13]
        if date in scores or date < args.start or (args.end and date > args.end):
            continue
        try:
            scores[date] = score_cycle(nc, date, ic_cache)
            new += 1
        except (OSError, ValueError, KeyError) as err:  # e.g. a file still being written
            print(f'skip {nc}: {err}', flush=True)
    if new:
        with open(cfile + '.tmp', 'w') as fh:
            json.dump(scores, fh)
        os.replace(cfile + '.tmp', cfile)
    print(f'{run}: {len(scores)} cycles ({new} new)', flush=True)
    return {d: s for d, s in scores.items() if d >= args.start and (not args.end or d <= args.end)}


def running_mean(y, n):
    if n <= 1 or len(y) < n:
        return y
    k = np.ones(n)
    good = np.isfinite(y)
    num = np.convolve(np.where(good, y, 0.), k, 'same')
    den = np.convolve(good.astype(float), k, 'same')
    return np.where(den >= n // 2, num / np.maximum(den, 1), np.nan)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out')
    p.add_argument('runs', nargs='+', help='LABEL=RUN_DIR[@IC_CACHE]')
    p.add_argument('--leads', type=int, nargs='+', default=[24, 114], help='hours into the window')
    p.add_argument('--start', default='2015-01-01T00')
    p.add_argument('--end', default=None)
    p.add_argument('--smooth', type=int, default=14, help='running-mean width in cycles')
    p.add_argument('--cache', default='output/analysis_error_ts')
    p.add_argument('--suffix', default=None, help='file suffix, e.g. 120h_100it (default: any)')
    p.add_argument('--ic-cache', default='/scratch4/BMC/gsienkf/Sergey.Frolov/ic_cache/ic_cache_aifs2-1.0deg-v1.0/')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

    runs, scores = [], {}
    for a, color in zip(args.runs, COLORS):
        label, run = a.split('=', 1)
        run, _, ic_cache = run.partition('@')
        runs.append((label, run, color))
        scores[run] = score_run(run, ic_cache or args.ic_cache, args)

    rows = []
    for lead in args.leads:
        fig, axes = plt.subplots(len(VARS), len(REGIONS), figsize=(18, 3.6 * len(VARS)), sharex=True)
        for label, run, color in runs:
            dates = sorted(scores[run])
            x = [datetime.strptime(d, '%Y-%m-%dT%H') for d in dates]
            for i, (v, vlabel) in enumerate(VARS):
                for j, (reg, rtitle) in enumerate(REGIONS):
                    y = np.array([scores[run][d].get(str(lead), {}).get(v, {}).get(reg, np.nan) for d in dates])
                    ax = axes[i, j]
                    ax.plot(x, y, color=color, lw=0.6, alpha=0.35)
                    ax.plot(x, running_mean(y, args.smooth), color=color, lw=2,
                            label=f'{label} ({len(dates)})' if i == j == 0 else None)
                    if i == 0:
                        ax.set_title(rtitle)
                    if j == 0:
                        ax.set_ylabel(vlabel)
                    rows += [[label, lead, v, reg, d, f'{val:.4f}'] for d, val in zip(dates, y)]
        for ax in axes[-1]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.tick_params(axis='x', rotation=30)
        axes[0, 0].legend(fontsize=8, loc='upper left')
        fig.suptitle(f'Analysis RMSE vs ERA5, {lead} h into the 5-day window, per cycle (window start on x); '
                     f'thick: {args.smooth}-cycle running mean', y=1.0)
        fig.tight_layout()
        fname = os.path.join(args.out, f'analysis_error_ts_{lead}h.png')
        fig.savefig(fname, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print('wrote', fname)

    with open(os.path.join(args.out, 'analysis_error_ts.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['label', 'lead_hours', 'var', 'region', 'cycle', 'rmse'])
        w.writerows(rows)
    print('wrote', os.path.join(args.out, 'analysis_error_ts.csv'))


if __name__ == '__main__':
    main()
