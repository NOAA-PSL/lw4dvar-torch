"""
Z500 error as a function of lead time WITHIN the 4D-Var window (0, 6, ...,
96h), averaged across every DA cycle of an experiment -- complements
z500err_ts.py, which only has the single lead time (window_start + dt_verif)
long_window_4dvar.py's own printz500err logs.

Recomputes directly from two sources per cycle, instead of the driver's log:
  - the model side: `<date>_control_forecast_*.nc` / `_optimal_forecast_*.nc`
    (saved by save_xr_trajectory -- the full-window decoded trajectory on
    AIFS's native unstructured grid, 'z' variable at level_z=500).
  - the truth side: ERA5 geopotential at 500 hPa, read directly from the
    cached GRIB files in ic_cache/ (aifs_ic.read_single_date_fields) at each
    lead time's valid date -- already regridded to the model's own grid by
    the CDS fetch (same grid `read_single_date_fields` is used for
    everywhere else in this codebase, e.g. get_verif()), so no extra
    regridding step is needed here either.

No AIFSModel/checkpoint/GPU load needed: lat/lon come straight from the
saved netCDF's own 'latitude'/'longitude' coordinates (written from
model.lats/model.lons when the file was saved), and
aifs_ic.read_single_date_fields's `runner` argument is only ever touched on
a cache MISS (see aifs_ic.fetch_era5_grib) -- passing `runner=None` is safe
whenever ic_cache/ is already warm for the dates needed, which is the
common case for an experiment whose driver job already ran to completion.
A cache miss raises a clear error naming the missing date instead of
silently guessing.

ACE2 output (detected from an `h500` variable in the saved forecast file)
is scored differently: model z500 is ACE2's OWN `h500` decoder output (m),
not the hydrostatically derived pressure-level 'z' (which runs ~19 m low vs
ERA5 -- see CLAUDE.md's ACE2 section), and truth is the ERA5 z500 regridded
to ACE2's F90 grid by `backends/ace2/ace2_ic.py --verif` (`load_verif`,
default cache `./ic_cache_ace2/`). Same row-major (lat, lon) flattening as
the saved file's `values` dimension, so no regridding here either.
"""

import glob
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

# Backend IC/verif readers live in backends/<name>/ -- imported lazily (only
# the one an experiment needs), since each backend's env lacks the others'
# dependencies. Relative paths used below (e.g. './ic_cache/') still assume
# the repo root as the CWD.
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _backend_module(backend, name):
    d = os.path.join(_REPO_ROOT, 'backends', backend)
    if d not in sys.path:
        sys.path.insert(0, d)
    return __import__(name)


DEFAULT_CACHE_DIR = {'aifs': './ic_cache/', 'ace2': './ic_cache_ace2/'}

GRAV = 9.80665
_DATE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2})_control_forecast_(.+)\.nc$')


def add_hours(date_str, nhours):
    d = datetime.strptime(date_str, '%Y-%m-%dT%H')
    return (d + timedelta(hours=int(nhours))).strftime('%Y-%m-%dT%H')


def _parse_nc_date(value):
    """`trajectory_start_date` attr ('2015-01-01 00:00:00', from
    str(datetime...) in save_xr_trajectory) -> '%Y-%m-%dT%H' string."""
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%dT%H')


def getrms(diff, weights):
    """Region-masked RMS, NaN-safe against a diverged/missing `diff` (not
    just an out-of-region `weights`): `nansum` of an all-NaN slice silently
    returns 0.0, not NaN, so masking only via `weights` (which never has NaN
    where `diff` does) would misreport a fully-NaN forecast as a perfect
    0.0 RMS error instead of "no data" -- discovered from a real diverged
    cycle, see CLAUDE.md. Points where `diff` is NaN are excluded from BOTH
    the numerator and the weight-sum denominator, not just zeroed.
    """
    weight = np.where(np.isnan(diff), np.nan, weights)
    denom = np.nansum(weight)
    if not denom > 0:
        return np.nan
    return np.sqrt(np.nansum(weight * diff ** 2) / denom)


def region_masks(lats):
    # exact grid-cell areas (forecast_model.area_weights_from_lats) -- the same
    # weights the driver's printz500err uses; cos(lat) per point was wrong
    # for AIFS's reduced octahedral grid
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from forecast_model import area_weights_from_lats
    w = area_weights_from_lats(lats)
    return {
        'NH': np.where(lats > 20., w, np.nan),
        'Tropics': np.where((lats >= -20) & (lats <= 20), w, np.nan),
        'SH': np.where(lats < -20., w, np.nan),
        'Global': w,
    }


_verif_cache = {}


def get_z500_truth(date_str, cache_dir, backend='aifs'):
    """ERA5 z (geopotential, m^2/s^2) at 500 hPa, raw (n_points,) array
    already on the model's native grid (see module docstring). Cached
    in-process since the same valid date is shared by many cycles/lead-times."""
    key = (backend, date_str)
    if key not in _verif_cache:
        date_dt = datetime.strptime(date_str, '%Y-%m-%dT%H')
        try:
            if backend == 'ace2':
                truth = _backend_module('ace2', 'ace2_ic').load_verif(date_dt, cache_dir)
                _verif_cache[key] = GRAV * np.asarray(truth['h500'], dtype=np.float64).reshape(-1)
            else:
                fields = _backend_module('aifs', 'aifs_ic').read_single_date_fields(None, date_dt, cache_dir)
                _verif_cache[key] = fields['z_500']
        except Exception as e:
            hint = ('backends/ace2/ace2_ic.py --verif' if backend == 'ace2' else 'aifs_prefetch_ic.py')
            raise RuntimeError(
                f"no cached ERA5 z500 for {date_str} in {cache_dir} ({e}) -- "
                f"prefetch it from a login node first ({hint}), "
                f"this script never fetches over the network itself."
            )
    return _verif_cache[key]


def detect_backend(nc_path):
    """'ace2' if the saved forecast carries ACE2's own h500 output, else 'aifs'."""
    with xr.open_dataset(nc_path) as ds:
        return 'ace2' if 'h500' in ds.data_vars else 'aifs'


def _model_z500(ds, backend):
    """(time, values) model z500 geopotential (m^2/s^2): ACE2's own h500 * g
    for ACE2, else the saved pressure-level 'z' at 500 hPa."""
    if backend == 'ace2':
        return GRAV * ds['h500'].values.astype(np.float64)
    return ds['z'].sel(level_z=500).values


def z500_error_curve(nc_path, cache_dir, backend='aifs'):
    """(start_date, lead_hours, {region: rms_err}, n_diverged) for one saved
    *_forecast_*.nc file. `lead_hours` is relative to THIS file's own
    `trajectory_start_date` -- for an `_optimal_forecast_` file that's
    `window_start + dt_verif` (the latent shift), NOT the window start, so
    callers comparing control vs. optimal on one absolute-lead-time axis
    must add that shift themselves (see experiment_mean_curves) rather than
    treating both files' lead_hours as directly comparable.
    """
    ds = xr.open_dataset(nc_path)
    start_date = _parse_nc_date(ds.attrs['trajectory_start_date'])
    lats = ds['latitude'].values
    masks = region_masks(lats)
    z500 = _model_z500(ds, backend)  # (time, values)
    ntime = z500.shape[0]
    lead_hours = np.arange(ntime) * 6  # every backend's native 6h timestep
    errs = {r: np.full(ntime, np.nan) for r in masks}
    n_diverged = 0
    for t in range(ntime):
        vdate = add_hours(start_date, lead_hours[t])
        truth = get_z500_truth(vdate, cache_dir, backend)
        diff = (z500[t] - truth) / GRAV
        if np.isnan(diff).all():
            n_diverged += 1
        for r, mask in masks.items():
            errs[r][t] = getrms(diff, mask)
    ds.close()
    return start_date, lead_hours, errs, n_diverged


def experiment_mean_curves(expt_dir, cache_dir=None, suffix=None):
    """Average control/optimal z500 error curves across every cycle found in
    expt_dir. Returns (lead_hours, {'control': {region: mean_err},
    'optimal': {region: mean_err}}, n_cycles) -- 'optimal' curves are
    shorter (see save_trajectory_diagnostics: the latent analysis trajectory
    starts dt_verif later than the background one) and averaged only where
    at least one cycle has data at that lead time.
    """
    control_files = sorted(glob.glob(os.path.join(expt_dir, '*_control_forecast_*.nc')))
    # `suffix` (e.g. '120h_100it') keeps only one run's files when several
    # runs (different max_epoch/window) wrote into the same directory.
    if suffix is not None:
        control_files = [f for f in control_files if f.endswith(f'_control_forecast_{suffix}.nc')]
    dates = [m.group(1) for m in map(_DATE_RE.match, map(os.path.basename, control_files)) if m]
    dups = sorted({d for d in dates if dates.count(d) > 1})
    if dups:
        print(f'  ** WARNING: {len(dups)} date(s) have files from more than one run (e.g. {dups[0]}) -- '
              f'each counts as a separate cycle; pass --suffix=... to pick one run **')
    backend = detect_backend(control_files[0]) if control_files else 'aifs'
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR[backend]
    print(f'  backend={backend}, truth from {cache_dir}')
    regions = ['NH', 'Tropics', 'SH', 'Global']
    accum = {'control': {r: [] for r in regions}, 'optimal': {r: [] for r in regions}}
    n_cycles = 0
    diverged_cycles = []
    for cf in control_files:
        m = _DATE_RE.match(os.path.basename(cf))
        if not m:
            continue
        date, suffix = m.groups()
        of = os.path.join(expt_dir, f'{date}_optimal_forecast_{suffix}.nc')
        if not os.path.exists(of):
            print(f'  warning: no matching optimal_forecast for {cf}, skipping cycle')
            continue
        start_c, lead_c, errs_c, ndiv_c = z500_error_curve(cf, cache_dir, backend)
        start_o, lead_o, errs_o, ndiv_o = z500_error_curve(of, cache_dir, backend)
        if ndiv_c or ndiv_o:
            diverged_cycles.append(date)
            print(f'  cycle {date}: DIVERGED (NaN geopotential at {ndiv_c} background / '
                  f'{ndiv_o} analysis lead times) -- excluded from the mean at those lead times')
        else:
            print(f'  cycle {date}...')
        # `lead_o` is relative to the optimal file's own (shifted) start
        # date -- shift it onto the control file's (window-start) absolute
        # lead-time axis before the two are compared/averaged together.
        shift_hours = (
            datetime.strptime(start_o, '%Y-%m-%dT%H') - datetime.strptime(start_c, '%Y-%m-%dT%H')
        ).total_seconds() / 3600.0
        lead_o_abs = lead_o + round(shift_hours)
        for r in regions:
            accum['control'][r].append((lead_c, errs_c[r]))
            accum['optimal'][r].append((lead_o_abs, errs_o[r]))
        n_cycles += 1
    if diverged_cycles:
        print(f'  ** {len(diverged_cycles)}/{n_cycles} cycles diverged to NaN: {diverged_cycles} **')

    def _mean_over_cycles(curves, all_lead_hours):
        # curves: list of (lead_hours, err) pairs, possibly different lengths
        # (optimal trajectories are shorter/shifted) -- average by lead hour
        # value, not by array index.
        out = np.full(all_lead_hours.shape, np.nan)
        for i, h in enumerate(all_lead_hours):
            vals = [v for lead, err in curves if h in lead for v in [err[lead == h][0]] if not np.isnan(v)]
            if vals:
                out[i] = np.mean(vals)
        return out

    max_lead = max((lead.max() for r in regions for lead, _ in accum['control'][r]), default=0)
    all_lead_hours = np.arange(0, max_lead + 1, 6)
    mean_curves = {kind: {r: _mean_over_cycles(accum[kind][r], all_lead_hours) for r in regions}
                   for kind in ('control', 'optimal')}
    return all_lead_hours, mean_curves, n_cycles, backend


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # python diagnostics/z500err_window.py [label=dir ...]  (run from the
    # repo root -- relative paths like ic_cache/ below assume it).
    # {label -> output dir}, ic_cache assumed at ./ic_cache/ for each.
    # Defaults to the mainline reset_skt_ocean cycling experiment (see
    # CLAUDE.md). Override with `label=dir` args, e.g.:
    #   python diagnostics/z500err_window.py \
    #       'lr=1e-3=output/test_aifs_ctlvars_n_init20_50it' \
    #       'lr=2e-3=output/test_aifs_ctlvars_n_init20_50it_lr2e-3'
    # (that lr=2e-3 run diverged and was killed -- see CLAUDE.md's
    # "learn_rate sweep" section -- so its curve will mostly be gaps.)
    # Optional `--cache_dir=DIR` overrides the per-backend default
    # (./ic_cache/ for AIFS, ./ic_cache_ace2/ for ACE2).
    # Optional `--suffix=120h_100it` keeps only that run's files.
    cache_dir = None
    suffix = None
    args = []
    for arg in sys.argv[1:]:
        if arg.startswith('--cache_dir='):
            cache_dir = arg.split('=', 1)[1]
        elif arg.startswith('--suffix='):
            suffix = arg.split('=', 1)[1]
        else:
            args.append(arg)
    if args:
        expts = {}
        for arg in args:
            # rsplit (not split) on the LAST '=' -- a label like 'lr=1e-3'
            # is a very natural thing to want, and directory paths never
            # contain '='.
            label, path = arg.rsplit('=', 1)
            expts[label] = path
    else:
        expts = {
            'lr=1.5e-3': 'output/test_aifs_latent_reset_skt_12h_100it'
#            'lr=2.e-3 (current)': 'output/test_aifs_ctlvars_n_init20_50it_lr2e-3',
        }
    regions = ['NH', 'Tropics', 'SH', 'Global']
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    any_data = False
    backends_seen = set()
    for (label, expt_dir), color in zip(expts.items(), colors):
        if not os.path.isdir(expt_dir):
            print(f'warning: {expt_dir} not found, skipping {label!r}')
            continue
        print(f'{label} ({expt_dir}):')
        lead_hours, curves, n_cycles, backend = experiment_mean_curves(expt_dir, cache_dir, suffix)
        backends_seen.add(backend)
        if n_cycles == 0:
            print(f'  no complete cycles found yet (job may still be running)')
            continue
        any_data = True
        print(f'  averaged over {n_cycles} cycles')
        for ax, region in zip(axes, regions):
            ax.plot(lead_hours, curves['control'][region], '--', color=color,
                     label=f'{label} background (n={n_cycles})')
            ax.plot(lead_hours, curves['optimal'][region], '-', color=color,
                     label=f'{label} analysis (n={n_cycles})')

    for ax, region in zip(axes, regions):
        ax.set_ylabel('%s Z500\nrms err (m)' % region)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('lead time within window (h)')
    names = {'aifs': 'AIFS', 'ace2': 'ACE2-ERA5 (own h500)'}
    fig.suptitle('Z500 error growth within window, averaged across DA cycles ('
                 + ', '.join(names[b] for b in sorted(backends_seen) or ['aifs']) + ')')
    fig.tight_layout()
    fig.savefig('z500err_window.png')
    print('wrote z500err_window.png' if any_data else 'wrote z500err_window.png (empty -- no complete cycles yet)')
