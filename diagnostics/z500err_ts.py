"""
Z500 error time series ACROSS DA cycles, at one chosen lead time WITHIN the
window -- complements z500err_window.py, which averages across cycles to
show the error-growth curve across lead time for one experiment.

Previously (see git history) this parsed long_window_4dvar.py's stdout log
for the single `z500err before/after <date> ...` line it prints per cycle,
which is fixed at `window_start + dt_verif` -- there was no way to see the
time series at any other lead time. Rewritten to instead read the saved
`<date>_control_forecast_*.nc` / `_optimal_forecast_*.nc` trajectories
directly (reusing z500err_window.py's netCDF/ERA5-GRIB machinery, including
its NaN-safe RMS and its in-process ERA5 verification cache), so an
arbitrary lead time -- any multiple of AIFS's native 6h step, from 0 up to
the window length -- can be picked with one CLI argument.

Backend-aware via z500err_window.py: ACE2 output (an `h500` variable in the
saved forecasts) is scored with ACE2's own h500 against the F90-regridded
ERA5 z500 in ./ic_cache_ace2/ (see z500err_window.py's docstring); AIFS
output as before against ./ic_cache/.
"""

import glob
import os
import sys
from datetime import datetime

import numpy as np

import z500err_window as zw


def cycle_values_at_lead(control_path, optimal_path, cache_dir, lead_hours, backend='aifs'):
    """(cycle_date, before, after) for one cycle at absolute lead time
    `lead_hours` from window start. `before`/`after` are {region: rms_err}
    dicts, or None if that lead time isn't available in the respective
    trajectory (beyond the saved window length for `before`; before the
    latent `dt_verif` shift, or beyond the window, for `after`).
    """
    start_c, lead_c, errs_c, _ = zw.z500_error_curve(control_path, cache_dir, backend)
    start_o, lead_o, errs_o, _ = zw.z500_error_curve(optimal_path, cache_dir, backend)
    shift_hours = round(
        (datetime.strptime(start_o, '%Y-%m-%dT%H') - datetime.strptime(start_c, '%Y-%m-%dT%H')).total_seconds()
        / 3600.0
    )

    idx_c = np.where(lead_c == lead_hours)[0]
    before = {r: errs_c[r][idx_c[0]] for r in errs_c} if idx_c.size else None

    idx_o = np.where(lead_o == lead_hours - shift_hours)[0]
    after = {r: errs_o[r][idx_o[0]] for r in errs_o} if idx_o.size else None

    return start_c, before, after


def get_z500err_ts(expt_dir, cache_dir, lead_hours, suffix=None):
    """{region: (dates, before_err, after_err)} time series across every
    cycle found in expt_dir, at the given absolute lead time. `cache_dir`
    None -> the detected backend's default; `suffix` (e.g. '120h_100it')
    keeps one run's files when several runs share a directory."""
    regions = ['NH', 'Tropics', 'SH', 'Global']
    dates, before_by_region, after_by_region = [], {r: [] for r in regions}, {r: [] for r in regions}
    diverged = []
    control_files = sorted(glob.glob(os.path.join(expt_dir, '*_control_forecast_*.nc')))
    if suffix is not None:
        control_files = [f for f in control_files if f.endswith(f'_control_forecast_{suffix}.nc')]
    backend = zw.detect_backend(control_files[0]) if control_files else 'aifs'
    if cache_dir is None:
        cache_dir = zw.DEFAULT_CACHE_DIR[backend]
    print(f'  backend={backend}, truth from {cache_dir}')
    all_dates = [m.group(1) for m in map(zw._DATE_RE.match, map(os.path.basename, control_files)) if m]
    dups = sorted({d for d in all_dates if all_dates.count(d) > 1})
    if dups:
        print(f'  ** WARNING: {len(dups)} date(s) have files from more than one run (e.g. {dups[0]}) -- '
              f'each is plotted as a separate point; pass --suffix=... to pick one run **')
    for cf in control_files:
        m = zw._DATE_RE.match(os.path.basename(cf))
        if not m:
            continue
        date, suffix = m.groups()
        of = os.path.join(expt_dir, f'{date}_optimal_forecast_{suffix}.nc')
        if not os.path.exists(of):
            continue
        cycle_date, before, after = cycle_values_at_lead(cf, of, cache_dir, lead_hours, backend)
        if before is None and after is None:
            continue  # this lead time isn't covered by this cycle's window at all
        dates.append(cycle_date)
        for r in regions:
            b = before[r] if before is not None else np.nan
            a = after[r] if after is not None else np.nan
            before_by_region[r].append(b)
            after_by_region[r].append(a)
            # only a value that EXISTS but is NaN signals divergence -- an
            # absent one (e.g. no analysis yet at lead 0h, it starts at
            # +dt_verif) is expected, not a diverged cycle
            if (before is not None and np.isnan(b)) or (after is not None and np.isnan(a)):
                diverged.append(date)
    if diverged:
        print(f'  ** possibly-diverged cycles at lead={lead_hours}h: {sorted(set(diverged))} **')
    dates = np.array(dates, dtype='datetime64[h]')
    return {r: (dates, np.array(before_by_region[r]), np.array(after_by_region[r])) for r in regions}, backend


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # python diagnostics/z500err_ts.py [lead_hours] [label=dir ...]  (run
    # from the repo root -- relative paths like ic_cache/ below assume it).
    # lead_hours: absolute lead time within the window, any multiple of 6
    # (AIFS's native step), default 6 (== dt_verif for every config used so
    # far -- matches what the old log-based version was limited to).
    # Optional `--cache_dir=DIR` (default: per backend, ./ic_cache/ for AIFS,
    # ./ic_cache_ace2/ for ACE2) and `--suffix=120h_100it` (one run's files).
    cache_dir = None
    suffix = None
    args = []
    for a in sys.argv[1:]:
        if a.startswith('--cache_dir='):
            cache_dir = a.split('=', 1)[1]
        elif a.startswith('--suffix='):
            suffix = a.split('=', 1)[1]
        else:
            args.append(a)
    lead_hours = 24
    if args and '=' not in args[0]:
        lead_hours = int(args[0])
        args = args[1:]
    if args:
        # rsplit (not split) on the LAST '=' -- a label like 'lr=1e-3' is a
        # very natural thing to want, and directory paths never contain '='.
        expts = dict(a.rsplit('=', 1) for a in args)
    else:
        # Default: the mainline reset_skt_ocean cycling experiment (see
        # CLAUDE.md). Pass experiments explicitly to compare others, e.g.:
        #   python diagnostics/z500err_ts.py 6 \
        #       'lr=1e-3=output/test_aifs_ctlvars_n_init20_50it' \
        #       'lr=2e-3=output/test_aifs_ctlvars_n_init20_50it_lr2e-3'
        # (that lr=2e-3 run diverged and was killed -- see CLAUDE.md's
        # "learn_rate sweep" section -- so its curve will mostly be gaps.)
        expts = {'lr=1.5e-3': 'output/test_aifs_latent_reset_skt_12h_100it'}

    regions = ['NH', 'Tropics', 'SH', 'Global']
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    any_data = False
    backends_seen = set()
    for label, expt_dir in expts.items():
        if not os.path.isdir(expt_dir):
            print(f'warning: {expt_dir} not found, skipping {label!r}')
            continue
        print(f'{label} ({expt_dir}), lead={lead_hours}h:')
        series, backend = get_z500err_ts(expt_dir, cache_dir, lead_hours, suffix)
        backends_seen.add(backend)
        n = series['Global'][0].size
        if n == 0:
            print(f'  no cycles cover lead={lead_hours}h yet (job may still be running, or window too short)')
            continue
        any_data = True
        print(f'  {n} cycles found')
        for ax, region in zip(axes, regions):
            dates, before, after = series[region]
            ax.plot(dates, before, '--', label=f'{label} before (mean={np.nanmean(before):.2f})')
            ax.plot(dates, after, '-', label=f'{label} after (mean={np.nanmean(after):.2f})')

    for ax, region in zip(axes, regions):
        ax.set_ylabel('%s Z500\nrms err (m)' % region)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('cycle date (window start)')
    names = {'aifs': 'AIFS', 'ace2': 'ACE2-ERA5 (own h500)'}
    fig.suptitle(f'Z500 error time series at lead={lead_hours}h (long-window 4D-Var, '
                 + ', '.join(names[b] for b in sorted(backends_seen) or ['aifs']) + ')')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig('z500err_ts.png')
    print('wrote z500err_ts.png' if any_data else 'wrote z500err_ts.png (empty -- no data found)')
