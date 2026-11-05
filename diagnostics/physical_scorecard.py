"""Physical-space scorecard: where (maps), at what height (zonal means) and how much (global
scorecard) run A differs from run B, both scored against ERA5, paired by date.

For each window start T0 and each lead L (hours after T0), the state of A and of B valid at T0+L is
compared with ERA5 at T0+L from the compared model's own training zarr (--truth; see TRUTHS).
Per grid point and per pressure level it accumulates over the dates:
  bias  = mean(X - ERA5)                          rmse = sqrt(mean((X - ERA5)^2))
  d     = (A - ERA5)^2 - (B - ERA5)^2 per date    -> paired t statistic per point / per zonal ring
Significance is a paired t test per point (or per ring and level) with the Benjamini-Hochberg false
discovery rate at alpha_FDR = 0.1 (Wilks 2016, for spatially correlated fields); stippled = significant.
Dates of a cycled run are not independent (consecutive analyses share observations), so the test is
optimistic there; for PITA's scattered independent windows it is about right.

Sources (LABEL=SPEC):
  LABEL=DIR[#TRAJ]     an lw4dvar output directory: <T0>_<TRAJ>_forecast_*.nc (TRAJ default optimal;
                       control = first-guess forecast). One file per T0; --suffix picks one if several.
  LABEL=healda:DIR     HealDA analyses DIR/healda_<YYYYMMDDHH>.nc (0.25 deg, bilinear to the O96 points),
                       valid at T0+L.
Fields: z (m), t (K), u (m/s), q (g/kg) and the vector wind error |dV| (rmse only), on the pressure
levels common to A, B and ERA5.

Outputs in OUT/:
  scorecard.png / scorecard.json   global area-weighted rmse of A and B per variable, level and lead,
                                   paired % change 100 (rmse_A / rmse_B - 1) per date, mean +- 95% CI
  zonal_<var>_<L>h.png             latitude x pressure: rmse A, rmse B, % change (stippled); bias A, B, A - B
  map_<var><lev>_<L>h.png          the same as maps at one level
  stats_<L>h.npz                   the accumulated per-point and zonal statistics

Usage (repo root, CPU; any env with xarray, zarr, cartopy and torch, e.g. aifs2):
  python diagnostics/physical_scorecard.py OUT A=output/a37_a2_m3_jac B=output/a37_a2_m3 --truth era5_37 \\
      --leads 6 66 120 --hours 0 --start 2015-01-01T00 --end 2015-03-01T00 --workers 16
  python diagnostics/physical_scorecard.py OUT window_end=output/pita22_a0_*_era5 \\
      HealDA=healda:/scratch4/BMC/gsienkf/Sergey.Frolov/healda/out --truth v8 --leads 120
A DIR may be a glob over several run directories (PITA: one directory per window).
"""

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Truth = the model's own training data (decided 2026-11-03): the two O96 ERA5 sets differ by ~2.4 m in z500,
# 0.23 K in t850, 0.6 m/s in u250 (different regridding chains), as much as half an analysis error.
#   v8      A0's training zarr (MARS O96; identical to the ic_cache GRIBs that score_cycle reads), 13 levels 50-1000
#   era5_37 A1/A2's 37-level zarr (from the GCS 1-degree ERA5), 1-1000 hPa
TRUTHS = {'era5_37': '/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/'
                     'era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr',
          'v8': '/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/data/training/'
                'era5-o96-1979-2023-6h-v8.zarr'}
GRAV = 9.80665
VARS = ['z', 't', 'u', 'q', 'wind']        # 'wind' = vector wind; error |dV|, no bias
UNITS = {'z': 'm', 't': 'K', 'u': 'm/s', 'q': 'g/kg', 'wind': 'm/s'}
SCALE = {'z': 1 / GRAV, 't': 1.0, 'u': 1.0, 'v': 1.0, 'q': 1e3}
MAPS = ['z500', 't850', 'q850', 'wind250', 't250', 'u50']
FDR_ALPHA = 0.1
TRUTH = ''


# ----------------------------------------------------------------------------------------- sources
def parse_source(spec):
    label, s = spec.split('=', 1)
    if s.startswith('healda:'):
        return {'label': label, 'kind': 'healda', 'dir': s[len('healda:'):]}
    d, _, traj = s.partition('#')
    return {'label': label, 'kind': 'run', 'dirs': sorted(glob.glob(d)), 'traj': traj or 'optimal'}


def run_files(src, suffix):
    """{T0 'YYYY-MM-DDTHH': path} of a run source."""
    out = {}
    for d in src['dirs']:
        for f in sorted(glob.glob(os.path.join(d, f"*_{src['traj']}_forecast_*.nc"))):
            b = os.path.basename(f)
            if suffix and not b.endswith(f'_{suffix}.nc'):
                continue
            out.setdefault(b[:13], f)
    return out


def healda_path(src, valid):
    return os.path.join(src['dir'], f'healda_{valid:%Y%m%d%H}.nc')


def run_levels(path):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return {v: set(int(x) for x in ds[f'level_{v}'].values) for v in 'ztuvq'}


def healda_levels(src):
    import xarray as xr
    f = sorted(glob.glob(os.path.join(src['dir'], 'healda_*.nc')))[0]
    with xr.open_dataset(f) as ds:
        names = list(ds.data_vars)
    return {v: {int(m.group(1)) for n in names for m in [re.fullmatch(v + r'(\d+)', n)] if m} for v in 'ztuvq'}


def read_run(path, valid, want):
    """{(var, lev): (npts,)} at the valid time, or None if outside the trajectory."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        start = datetime.strptime(str(ds.attrs['trajectory_start_date'])[:13].replace(' ', 'T'), '%Y-%m-%dT%H')
        k, r = divmod((valid - start).total_seconds(), 6 * 3600)
        if r or k < 0 or k >= ds.sizes['time']:
            return None
        k = int(k)
        out = {}
        for v in 'ztuvq':
            levs = sorted(want[v])
            a = ds[v].isel(time=k).sel({f'level_{v}': levs}).values * SCALE[v]
            out.update({(v, lv): a[i] for i, lv in enumerate(levs)})
    return out


_HG = {}


def read_healda(src, valid, want, lat, lon):
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    f = healda_path(src, valid)
    if not os.path.exists(f):
        return None
    pts = np.c_[lat, lon % 360.0]
    out = {}
    with xr.open_dataset(f) as ds:
        glat, glon = ds['lat'].values, ds['lon'].values
        flip = glat[0] > glat[-1]
        glat = glat[::-1] if flip else glat
        glon = np.r_[glon, 360.0]
        for v in 'ztuvq':
            for lv in want[v]:
                a = ds[f'{v}{lv}'].values[0]
                a = a[::-1] if flip else a
                a = np.concatenate([a, a[:, :1]], axis=1)
                out[(v, lv)] = RegularGridInterpolator((glat, glon), a)(pts) * SCALE[v]
    return out


_ZARR = {}


def read_era5(valid, want, truth):
    import zarr
    if _ZARR.get('truth') != truth:
        _ZARR.clear()
        z = zarr.open(TRUTHS[truth], 'r')
        names = list(z.attrs['variables'])
        _ZARR.update(truth=truth, z=z['data'], idx={n: i for i, n in enumerate(names)},
                     start=datetime.fromisoformat(str(z['dates'][0])[:19]))
    i, r = divmod((valid - _ZARR['start']).total_seconds(), 6 * 3600)
    if r:
        raise ValueError(f'{valid} is not 6-hourly')
    block = np.asarray(_ZARR['z'][int(i), :, 0, :])
    return {(v, lv): block[_ZARR['idx'][f'{v}_{lv}']] * SCALE[v] for v in 'ztuvq' for lv in want[v]}


# ---------------------------------------------------------------------------------------- per date
def one(job):
    """Errors of A and B against ERA5 at one valid time: {field: (eA, eB)} float32, or None."""
    t0, lead, A, B, fa, fb, want, lat, lon, truth = job
    valid = datetime.strptime(t0, '%Y-%m-%dT%H') + timedelta(hours=lead)
    xs = []
    for src, f in ((A, fa), (B, fb)):
        x = read_run(f, valid, want) if src['kind'] == 'run' else read_healda(src, valid, want, lat, lon)
        if x is None:
            return t0, None
        xs.append(x)
    e = read_era5(valid, want, truth)
    out = {}
    for v in 'ztuq':
        for lv in want[v]:
            out[f'{v}{lv}'] = tuple((x[(v, lv)] - e[(v, lv)]).astype(np.float32) for x in xs)
    for lv in sorted(want['u'] & want['v']):
        out[f'wind{lv}'] = tuple(np.hypot(x[('u', lv)] - e[('u', lv)], x[('v', lv)] - e[('v', lv)]).astype(np.float32)
                                 for x in xs)
    return t0, out


# ------------------------------------------------------------------------------------------- stats
def fdr_mask(t, n, alpha=FDR_ALPHA):
    """Benjamini-Hochberg on two-sided paired-t p values; True = significant."""
    from scipy import stats
    p = 2 * stats.t.sf(np.abs(np.nan_to_num(t)), max(n - 1, 1))
    flat = np.sort(p.ravel())
    m = flat.size
    ok = flat <= alpha * np.arange(1, m + 1) / m
    thr = flat[ok].max() if ok.any() else -1.0
    return p <= thr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('A')
    ap.add_argument('B')
    ap.add_argument('--leads', type=int, nargs='+', default=[120])
    ap.add_argument('--hours', type=int, nargs='*', help='keep T0 at these UTC hours (default all)')
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--suffix', help='e.g. 120h_200it, if a directory has several forecasts per T0')
    ap.add_argument('--maps', nargs='*', default=MAPS)
    ap.add_argument('--truth', choices=list(TRUTHS), required=True,
                    help="the compared model's training data: v8 for A0, era5_37 for A1/A2")
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global TRUTH
    TRUTH = a.truth

    A, B = parse_source(a.A), parse_source(a.B)
    if A['kind'] != 'run':
        sys.exit('A must be an lw4dvar run (it defines the dates)')
    fa = run_files(A, a.suffix)
    fb = run_files(B, a.suffix) if B['kind'] == 'run' else None
    t0s = sorted(t for t in fa if (fb is None or t in fb)
                 and (not a.hours or int(t[11:13]) in a.hours)
                 and (not a.start or t >= a.start) and (not a.end or t <= a.end))
    if not t0s:
        sys.exit('no common dates')
    la = run_levels(fa[t0s[0]])
    lb = run_levels(fb[t0s[0]]) if fb else healda_levels(B)
    import zarr
    names = set(zarr.open(TRUTHS[a.truth], 'r').attrs['variables'])
    want = {v: {lv for lv in la[v] & lb[v] if f'{v}_{lv}' in names} for v in 'ztuvq'}
    import xarray as xr
    with xr.open_dataset(fa[t0s[0]]) as ds:
        lat, lon = ds.latitude.values, ds.longitude.values
    from forecast_model import area_weights_from_lats
    w = area_weights_from_lats(lat)
    rings, ring = np.unique(np.round(lat, 6), return_inverse=True)
    nring = len(rings)
    rcount = np.bincount(ring, minlength=nring)
    print(f'{A["label"]} vs {B["label"]}: {len(t0s)} dates {t0s[0]}..{t0s[-1]}, leads {a.leads}; levels',
          {v: sorted(want[v]) for v in want}, 'truth', a.truth, flush=True)

    card = {}
    for lead in a.leads:
        jobs = [(t, lead, A, B, fa[t], fb[t] if fb else None, want, lat, lon, a.truth) for t in t0s]
        acc, zon, glob_rmse, used = {}, {}, {}, []
        with ProcessPoolExecutor(a.workers) as ex:
            for t0, err in ex.map(one, jobs, chunksize=2):
                if err is None:
                    continue
                used.append(t0)
                for k, (ea, eb) in err.items():
                    ea, eb = ea.astype(np.float64), eb.astype(np.float64)
                    d = ea ** 2 - eb ** 2
                    s = acc.setdefault(k, np.zeros((6, ea.size)))
                    s += np.stack([ea, ea ** 2, eb, eb ** 2, d, d ** 2])
                    zr = lambda x: np.bincount(ring, x, nring) / rcount  # noqa: E731
                    zon.setdefault(k, []).append(np.stack([zr(ea), zr(ea ** 2), zr(eb), zr(eb ** 2)]))
                    glob_rmse.setdefault(k, []).append((np.sqrt(np.sum(w * ea ** 2) / w.sum()),
                                                        np.sqrt(np.sum(w * eb ** 2) / w.sum())))
        n = len(used)
        print(f'lead {lead} h: {n} dates', flush=True)
        if n < 3:
            continue
        st = {}
        for k, s in acc.items():
            m = s / n
            sd = np.sqrt(np.maximum(m[5] - m[4] ** 2, 0) * n / (n - 1))
            st[k] = dict(bias_a=m[0], rmse_a=np.sqrt(m[1]), bias_b=m[2], rmse_b=np.sqrt(m[3]),
                         t=m[4] / np.maximum(sd / np.sqrt(n), 1e-30))
            z = np.array(zon[k])                                # (n, 4, nring)
            dz = z[:, 1] - z[:, 3]
            st[k].update(zbias_a=z[:, 0].mean(0), zrmse_a=np.sqrt(z[:, 1].mean(0)), zbias_b=z[:, 2].mean(0),
                         zrmse_b=np.sqrt(z[:, 3].mean(0)),
                         zt=dz.mean(0) / np.maximum(dz.std(0, ddof=1) / np.sqrt(n), 1e-30))
            g = np.array(glob_rmse[k])
            pct = 100 * (g[:, 0] / g[:, 1] - 1)
            card.setdefault(k, {})[str(lead)] = dict(
                rmse_a=float(np.sqrt(np.mean(g[:, 0] ** 2))), rmse_b=float(np.sqrt(np.mean(g[:, 1] ** 2))),
                pct=float(pct.mean()), ci95=float(1.96 * pct.std(ddof=1) / np.sqrt(n)),
                better=int((pct < 0).sum()), n=n)
        np.savez_compressed(os.path.join(a.out, f'stats_{lead}h.npz'), lat=lat, lon=lon, rings=rings, dates=used,
                            **{f'{k}__{q}': v for k, d in st.items() for q, v in d.items()})
        plot_zonal(a.out, st, rings, lead, A['label'], B['label'], n)
        plot_maps(a.out, st, lat, lon, lead, A['label'], B['label'], n, a.maps)
    meta = dict(A=a.A, B=a.B, truth=a.truth, dates=t0s, leads=a.leads, fdr_alpha=FDR_ALPHA,
                pct='mean over dates of 100 (rmse_A / rmse_B - 1), global area-weighted; ci95 = 1.96 sd / sqrt(n)')
    json.dump(dict(meta=meta, card=card), open(os.path.join(a.out, 'scorecard.json'), 'w'), indent=1)
    plot_card(a.out, card, A['label'], B['label'])


# ------------------------------------------------------------------------------------------- plots
def _fields_by_var(st):
    by = {}
    for k in st:
        v, lv = re.fullmatch(r'([a-z]+)(\d+)', k).groups()
        by.setdefault(v, []).append(int(lv))
    return {v: sorted(l) for v, l in by.items()}


def _pct(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 100 * (a / b - 1)


def plot_zonal(out, st, rings, lead, la, lb, n):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for v, levs in _fields_by_var(st).items():
        if len(levs) < 2:
            continue
        P = np.array(levs, float)
        get = lambda q: np.array([st[f'{v}{lv}'][q] for lv in levs])  # noqa: E731  (nlev, nring)
        ra, rb, ba, bb, zt = get('zrmse_a'), get('zrmse_b'), get('zbias_a'), get('zbias_b'), get('zt')
        pc = _pct(ra, rb)
        sig = fdr_mask(zt, n)
        fig, ax = plt.subplots(2 if v != 'wind' else 1, 3, figsize=(17, 9 if v != 'wind' else 4.8),
                               sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        vmax = np.nanpercentile(np.r_[ra.ravel(), rb.ravel()], 98)
        for j, (f, ttl) in enumerate(((ra, f'rmse {la}'), (rb, f'rmse {lb}'))):
            m = ax[0, j].contourf(rings, P, f, levels=np.linspace(0, vmax, 13), cmap='viridis', extend='max')
            ax[0, j].set_title(ttl)
        fig.colorbar(m, ax=ax[0, :2], label=f'{v} rmse ({UNITS[v]})', shrink=0.9)
        pmax = min(np.nanpercentile(np.abs(pc), 95), 50) or 1
        m = ax[0, 2].contourf(rings, P, np.clip(pc, -pmax, pmax), levels=np.linspace(-pmax, pmax, 17),
                              cmap='RdBu_r')
        ax[0, 2].contourf(rings, P, sig.astype(float), levels=[0.5, 1.5], colors='none', hatches=['..'])
        ax[0, 2].set_title(f'rmse change {la} vs {lb} (%), dots: FDR-significant')
        fig.colorbar(m, ax=ax[0, 2], label='%', shrink=0.9)
        if v != 'wind':
            bmax = np.nanpercentile(np.abs(np.r_[ba.ravel(), bb.ravel()]), 98) or 1
            for j, (f, ttl) in enumerate(((ba, f'bias {la}'), (bb, f'bias {lb}'), (ba - bb, f'bias {la} - {lb}'))):
                m = ax[1, j].contourf(rings, P, np.clip(f, -bmax, bmax), levels=np.linspace(-bmax, bmax, 17),
                                      cmap='RdBu_r')
                ax[1, j].set_title(ttl)
            fig.colorbar(m, ax=ax[1, :], label=f'{v} bias vs ERA5 ({UNITS[v]})', shrink=0.9)
        for x in ax.ravel():
            x.set_yscale('log')
            x.set_ylim(P.max(), P.min())
            x.set_yticks(P)
            x.set_yticklabels([str(int(p)) for p in P], fontsize=7)
            x.minorticks_off()
        for x in ax[-1]:
            x.set_xlabel('latitude')
        for x in ax[:, 0]:
            x.set_ylabel('hPa')
        fig.suptitle(f'{v}: zonal mean vs ERA5 ({TRUTH}) at {lead} h, {n} dates')
        fig.savefig(os.path.join(out, f'zonal_{v}_{lead}h.png'), dpi=110)
        plt.close(fig)


def _regrid(lat, lon):
    """Nearest O96 point for each cell of a 1-degree lat-lon grid (for plotting only)."""
    from scipy.spatial import cKDTree
    xyz = lambda la, lo: np.c_[np.cos(np.radians(la)) * np.cos(np.radians(lo)),  # noqa: E731
                               np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))]
    glat, glon = np.arange(-89.5, 90, 1.0), np.arange(0.5, 360, 1.0)
    LO, LA = np.meshgrid(glon, glat)
    _, idx = cKDTree(xyz(lat, lon)).query(xyz(LA.ravel(), LO.ravel()))
    return glat, glon, idx.reshape(LA.shape)


def plot_maps(out, st, lat, lon, lead, la, lb, n, which):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    glat, glon, idx = _regrid(lat, lon)
    pc_ = ccrs.PlateCarree()
    for k in which:
        if k not in st:
            continue
        s = st[k]
        v = re.match('[a-z]+', k).group()
        ra, rb = s['rmse_a'][idx], s['rmse_b'][idx]
        pc = _pct(s['rmse_a'], s['rmse_b'])[idx]
        sig = fdr_mask(s['t'], n)[idx]
        rows = 1 if v == 'wind' else 2
        fig, ax = plt.subplots(rows, 3, figsize=(19, 4.3 * rows + 0.6), squeeze=False, constrained_layout=True,
                               subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
        vmax = np.nanpercentile(np.r_[ra.ravel(), rb.ravel()], 99)
        for j, (f, ttl) in enumerate(((ra, f'rmse {la}'), (rb, f'rmse {lb}'))):
            m = ax[0, j].pcolormesh(glon, glat, f, vmin=0, vmax=vmax, cmap='viridis', transform=pc_, shading='auto')
            ax[0, j].set_title(ttl)
        fig.colorbar(m, ax=ax[0, :2], label=f'{k} rmse ({UNITS[v]})', shrink=0.8)
        pmax = min(np.nanpercentile(np.abs(pc), 95), 60) or 1
        m = ax[0, 2].pcolormesh(glon, glat, pc, vmin=-pmax, vmax=pmax, cmap='RdBu_r', transform=pc_, shading='auto')
        ax[0, 2].contourf(glon, glat, sig.astype(float), levels=[0.5, 1.5], colors='none', hatches=['...'],
                          transform=pc_)
        ax[0, 2].set_title(f'rmse change {la} vs {lb} (%), dots: FDR-significant')
        fig.colorbar(m, ax=ax[0, 2], label='%', shrink=0.8)
        if rows == 2:
            ba, bb = s['bias_a'][idx], s['bias_b'][idx]
            bmax = np.nanpercentile(np.abs(np.r_[ba.ravel(), bb.ravel()]), 99) or 1
            for j, (f, ttl) in enumerate(((ba, f'bias {la}'), (bb, f'bias {lb}'), (ba - bb, f'bias {la} - {lb}'))):
                m = ax[1, j].pcolormesh(glon, glat, f, vmin=-bmax, vmax=bmax, cmap='RdBu_r', transform=pc_,
                                        shading='auto')
                ax[1, j].set_title(ttl)
            fig.colorbar(m, ax=ax[1, :], label=f'{k} bias vs ERA5 ({UNITS[v]})', shrink=0.8)
        for x in ax.ravel():
            x.coastlines(linewidth=0.4, color='0.3')
            x.set_global()
        fig.suptitle(f'{k} vs ERA5 ({TRUTH}) at {lead} h, {n} dates')
        fig.savefig(os.path.join(out, f'map_{k}_{lead}h.png'), dpi=100)
        plt.close(fig)


def plot_card(out, card, la, lb):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    keys = sorted(card, key=lambda k: (VARS.index(re.match('[a-z]+', k).group()), -int(re.search(r'\d+', k).group())))
    leads = sorted({int(L) for k in keys for L in card[k]})
    P = np.full((len(keys), len(leads)), np.nan)
    C = np.full_like(P, np.nan)
    for i, k in enumerate(keys):
        for j, L in enumerate(leads):
            c = card[k].get(str(L))
            if c:
                P[i, j], C[i, j] = c['pct'], c['ci95']
    lim = min(np.nanmax(np.abs(P)), 30) if np.isfinite(P).any() else 1
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(leads), 0.8 + 0.24 * len(keys)), constrained_layout=True)
    m = ax.imshow(np.clip(P, -lim, lim), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(len(keys)):
        for j in range(len(leads)):
            if np.isfinite(P[i, j]):
                star = '' if abs(P[i, j]) > C[i, j] else ' ·'
                ax.text(j, i, f'{P[i, j]:+.1f}{star}', ha='center', va='center', fontsize=7,
                        color='k' if abs(P[i, j]) < 0.6 * lim else 'w')
    ax.set_xticks(range(len(leads)), [f'{L} h' for L in leads])
    ax.set_yticks(range(len(keys)), keys, fontsize=7)
    fig.colorbar(m, label=f'rmse {la} vs {lb} (%)', shrink=0.6)
    ax.set_title(f'{la} vs {lb}: global rmse change vs ERA5 ({TRUTH})\n(blue = {la} better; "·" = within 95% CI)', fontsize=9)
    fig.savefig(os.path.join(out, 'scorecard.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
