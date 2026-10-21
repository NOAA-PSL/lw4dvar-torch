"""Forecast skill of daily 5-day forecasts from the end of the 4D-Var window,
against ERA5 and against observations (see forecast_skill.md).

For each 00Z date T in [START, END] (daily):
  era5  IC = ERA5 at T (the checkpoint's lagged pair T-6h, T; no spin-up)
  run   IC = the analysis of the window ending at T,
        <RUN_DIR>/<T - window>_optimal_inputs_<window>h_*it.pkl (valid at window start + 6 h),
        advanced to T, then skt over ocean reset from ERA5 at T, as the cycling does
then a forecast of LEADS 6-h steps, scored at every step (lead 0 .. 6*LEADS h).

vs ERA5 (area-weighted, regions Global / NH (> 20N) / Tropics / SH (< 20S)): RMSE and mean
error (forecast - ERA5) of z500 m, t850 K, q850 g/kg, ws250 m/s (speed), vw250 m/s (vector,
RMSE only), msl hPa, ws10, vw10 m/s, t2m K, q2m g/kg (from 2 m dewpoint and surface pressure).

vs obs (regions by station latitude): n, sum and sum of squares of O - F per lead and region
  ps         PREPBUFR surface pressure through the solver's operator (hPa)
  raob_<b>   radiosonde t K, q g/kg, u, v m/s on the model levels; raob_ws speed; raob_uv vector
  sfc_<net>_<v>  land stations / marine (ship, buoy) / scat (ASCAT) from sfcobs_parquet.py:
             t (2 m, K), q (2 m, g/kg), ws (10 m speed), uv (10 m vector)
  QC, identical for every forecast: the readers' read-time QC (PREPBUFR quality marks,
  bounds); the solver's ps orography screen; for land and marine stations
  |station elevation - model orography| <= --zthresh; GSI convinfo use flag (used or
  monitored) and gross check, valid at T (gsi_convinfo.py, NOAA-PSL/build_gsinfo), with ERA5
  at the valid time as the background, so each obs is in or out for every run and lead alike:
  reject when |O - H(ERA5)| > gross * clip(err, ermin, ermax). err: the solver's ps effective
  error and raob t/u/v errors; surface t 2.5 K, wind 2.5 m/s (stations) / 3.5 m/s (ASCAT); q as
  relative humidity, |dq| / qsat > gross * clip(0.2, ermin, ermax) / 100.
  2 m temperature: the model value is moved to the station elevation with a 6.5 K/km lapse rate.
  conv_<v>   upper-air conventional profiles (convobs_parquet.py) -- radiosondes at every level,
             aircraft, wind profilers / VAD winds (satellite winds left out): t K, q g/kg, uv
             (vector) at the reported pressure, the model interpolated linearly in ln p between
             its levels (obs outside the model's levels or below its surface are dropped; q only
             at p >= --q-min-pressure); binned into pressure layers (PBIN_EDGES) per platform
             group; within --conv-tolerance-hours of the valid time; QC: quality mark <= 3,
             convinfo use flag and gross check vs ERA5 with the archive's obs errors

Lead 0 of a run's forecast is the window's last analysis time: its obs were assimilated.
Writes <OUT>/<LABEL>/<T>.nc (dates already written by this SKILL_VERSION are skipped; older
versions are recomputed and replaced).

Usage: python diagnostics/forecast_skill.py CONFIG LABEL era5|RUN_DIR START END [options]
  CONFIG: an lwatms config supplying the model, ic_cache and observations (raobs enabled)
"""

import argparse
import glob
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsi_convinfo  # noqa: E402
import long_window_4dvar_utils as utils  # noqa: E402
import convobs_parquet  # noqa: E402
import sfcobs_parquet  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('config')
ap.add_argument('label')
ap.add_argument('source', help="'era5' or a run's output directory")
ap.add_argument('start', help='first init date, YYYY-MM-DDTHH')
ap.add_argument('end', help='last init date, YYYY-MM-DDTHH')
ap.add_argument('--out', default='output/forecast_skill')
ap.add_argument('--leads', type=int, default=20, help='number of 6-h steps (default 20 = 5 days)')
ap.add_argument('--every', type=int, default=24, help='hours between init dates (default 24)')
ap.add_argument('--window-hours', type=int, default=120, help="the run's window length (default 120)")
ap.add_argument('--gsinfo', default=gsi_convinfo.DEFAULT_GSINFO, help='build_gsinfo checkout')
ap.add_argument('--zthresh', type=float, default=200., help='surface station elevation screen, m (default 200)')
ap.add_argument('--sfc-tolerance-hours', type=float, default=1., help='surface obs time window, +/- h (default 1)')
ap.add_argument('--conv-tolerance-hours', type=float, default=1.5, help='upper-air obs time window, +/- h (default 1.5)')
ap.add_argument('--q-min-pressure', type=float, default=300., help='upper-air q only at p >= this, hPa (default 300)')
args = ap.parse_args()

LEADS = args.leads
REGIONS = ('Global', 'NH', 'Tropics', 'SH')
ERA_FIELDS = ('z500', 't850', 'q850', 'ws250', 'vw250', 'msl', 'ws10', 'vw10', 't2m', 'q2m')
LAPSE = 0.0065  # K/m
SFC_ERR = {'t': 2.5, 'q': 0.2, 'uv': 2.5}
SCAT_ERR_UV = 3.5
SKILL_VERSION = 2  # 2: conv_* profiles
CONV_GROUPS = ('raob', 'aircraft', 'profiler')  # satellite winds left out
CONV_ERR = {'t': 1.5, 'q': 0.2, 'uv': 2.5}  # where the archive has no obs error
PBIN_EDGES = np.array([1050., 950., 850., 750., 650., 550., 450., 350., 275., 225., 175., 125., 75., 30.])
PBIN_MID = np.array([1000., 900., 800., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.])
OUT = os.path.join(args.out, args.label)
os.makedirs(OUT, exist_ok=True)

exp, windows = utils.load_config(args.config)
exp['date'] = exp['sdate']
exp = utils.get_window(exp, list(windows)[0], windows)
exp['n_verif'], exp['dt_verif'], exp['dt_obs'] = LEADS, 6, 6
exp.pop('raobs_q_bg_check', None)  # the convinfo gross check below replaces it
logger = utils.get_logger()
model = utils.get_model(exp)
grid_interp = utils.get_grid_interpolator(model, exp)
device = model.device
area = np.asarray(model.area_weights, dtype=np.float64)
lats = np.asarray(model.lats)
lev = {b: [float(p) for p in model.pressure_levels(b)] for b in ('z', 't', 'q', 'u', 'v')}


def region_masks(lat):
    return (np.ones_like(lat, bool), lat > 20, np.abs(lat) <= 20, lat < -20)


GRID_REGIONS = region_masks(lats)


def qsat(t_k, p_hpa):
    """saturation specific humidity (kg/kg), Bolton (1980) over water"""
    es = 6.112 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))
    return 0.622 * es / (p_hpa - 0.378 * es)


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float64)


def fields(d):
    g = lambda b, p: to_np(d[b][lev[b].index(p)])  # noqa: E731
    sp = to_np(d['sp']) / 100.
    return {'z500': g('z', 500.) / utils.GRAV, 't850': g('t', 850.), 'q850': g('q', 850.) * 1e3,
            'u250': g('u', 250.), 'v250': g('v', 250.), 'msl': to_np(d['msl']) / 100.,
            'u10': to_np(d['10u']), 'v10': to_np(d['10v']), 't2m': to_np(d['2t']),
            'q2m': qsat(to_np(d['2d']), sp) * 1e3, 'sp': sp}


def era5_scores(f, e):
    out = {}
    for name in ERA_FIELDS:
        if name in ('ws250', 'ws10'):
            lv = name[2:]
            diff = np.hypot(f['u' + lv], f['v' + lv]) - np.hypot(e['u' + lv], e['v' + lv])
        elif name in ('vw250', 'vw10'):
            lv = name[2:]
            diff, sq = None, (f['u' + lv] - e['u' + lv]) ** 2 + (f['v' + lv] - e['v' + lv]) ** 2
        else:
            diff = f[name] - e[name]
        for i, m in enumerate(GRID_REGIONS):
            w = area * m
            if diff is None:
                out[name, i] = (np.sqrt((w * sq).sum() / w.sum()), np.nan)
            else:
                out[name, i] = (np.sqrt((w * diff ** 2).sum() / w.sum()), (w * diff).sum() / w.sum())
    return out


class Acc:
    """n, sum and sum of squares of O - F per (lead, region, level)"""

    def __init__(self):
        self.data, self.nlev, self.nrej = {}, {}, {}

    def add(self, name, k, lat, d, rejected=0):
        """d: (nlev, nobs) or (nobs,) departures, NaN where not used; `rejected`: obs that
        failed the gross check"""
        d = np.atleast_2d(d)
        if name not in self.data:
            self.data[name] = np.zeros((3, LEADS + 1, len(REGIONS), d.shape[0]))
            self.nrej[name] = np.zeros(LEADS + 1)
        for i, m in enumerate(region_masks(lat)):
            x = np.where(m[None, :], d, np.nan)
            ok = np.isfinite(x)
            self.data[name][0, k, i] += ok.sum(1)
            self.data[name][1, k, i] += np.where(ok, x, 0).sum(1)
            self.data[name][2, k, i] += np.where(ok, x ** 2, 0).sum(1)
        self.nrej[name][k] += rejected

    def add_vector(self, name, k, lat, du, dv, rejected=0):
        """vector difference: s2 = sum(du^2 + dv^2), s1 = sum |dV|"""
        self.add(name, k, lat, np.hypot(du, dv), rejected=rejected)


def gross_ok(table, otype, types, subtypes, dep, err, scale=1.):
    """GSI gross check per obs: the convinfo row for (otype, type, subtype) must have iuse 1 or
    -1 (a type not in convinfo fails), and |dep| <= gross * clip(err, ermin, ermax) / scale.
    NaN departures fail."""
    pairs, inv = np.unique(np.stack([np.asarray(types), np.asarray(subtypes)], 1).astype(np.int64), axis=0,
                           return_inverse=True)
    par = np.full((len(pairs), 3), np.nan)
    for i, (t, s) in enumerate(pairs):
        row = gsi_convinfo.lookup(table, otype, t, s)
        if row is not None and row['iuse'] in (1, -1):
            par[i] = row['gross'], row['ermin'] / scale, row['ermax'] / scale
    g, lo, hi = par[inv.ravel()].T
    with np.errstate(invalid='ignore'):
        return np.isfinite(g) & (np.abs(dep) <= g * np.clip(err, lo, hi))


def score_ps(acc, k, d_f, d_e, verif0, psobs, e, table):
    inn_f, _, err, _, _, qc = utils._compute_ps_observation_diagnostics_at_time(model, d_f, verif0, psobs, e, k, grid_interp)
    inn_e = utils._compute_ps_observation_diagnostics_at_time(model, d_e, verif0, psobs, e, k, grid_interp)[0]
    n = int(psobs['n_valid'][k])
    base = ((qc & 7) == 0).cpu().numpy()[:n]
    inn_f, inn_e, err = to_np(inn_f)[:n], to_np(inn_e)[:n], to_np(err)[:n]
    types = psobs['obtype'][k].cpu().numpy()[:n]
    ok = base & gross_ok(table, 'ps', types, np.zeros_like(types), inn_e, err)
    acc.add('ps', k, psobs['lat'][k].cpu().numpy()[:n], np.where(ok, inn_f, np.nan), rejected=(base & ~ok).sum())


def score_raobs(acc, k, d_f, d_e, raobs, table):
    res_f = utils._compute_raobs_observation_diagnostics_at_time(model, d_f, raobs, grid_interp, k)
    res_e = utils._compute_raobs_observation_diagnostics_at_time(model, d_e, raobs, grid_interp, k)
    lat = raobs['lat'][k].cpu().numpy()
    plev = {b: raobs['levels_hpa'][b].cpu().numpy()[:, None] for b in raobs['variables']}
    used = {}
    for b, (inn_f, _, err, use) in res_f.items():
        inn_f, inn_e, err, use = to_np(inn_f), to_np(res_e[b][0]), to_np(err), use.cpu().numpy()
        if b == 'q':
            t_e = to_np(res_e['t'][1]) if 't' in res_e and len(plev['t']) else None
            t_q = t_e[[lev['t'].index(float(p)) for p in plev['q'][:, 0]]] if t_e is not None else 280.
            qs = qsat(t_q, plev['q'])
            row = gsi_convinfo.lookup(table, 'q', 120)
            ok = use & gsi_convinfo.gross_check(inn_e / qs, np.full_like(qs, SFC_ERR['q']),
                                                dict(row, ermin=row['ermin'] / 100., ermax=row['ermax'] / 100.))
            inn_f = inn_f * 1e3
        else:
            row = gsi_convinfo.lookup(table, 't' if b == 't' else 'uv', 120 if b == 't' else 220)
            if b in ('u', 'v'):  # GSI checks the vector departure
                dv = np.hypot(to_np(res_e['u'][0]), to_np(res_e['v'][0]))
                ok = use & gsi_convinfo.gross_check(dv, err, row)
            else:
                ok = use & gsi_convinfo.gross_check(inn_e, err, row)
        used[b] = ok
        acc.add('raob_' + b, k, lat, np.where(ok, inn_f, np.nan), rejected=(use & ~ok).sum())
    if 'u' in res_f and 'v' in res_f:
        ok = used['u'] & used['v']
        uo, vo = to_np(raobs['values_u'][k]), to_np(raobs['values_v'][k])
        uf, vf = to_np(res_f['u'][1]), to_np(res_f['v'][1])
        acc.add('raob_ws', k, lat, np.where(ok, np.hypot(uo, vo) - np.hypot(uf, vf), np.nan))
        acc.add_vector('raob_uv', k, lat, np.where(ok, uo - uf, np.nan), np.where(ok, vo - vf, np.nan))


def sfc_operator(d, sfc, idx, wts, orog):
    g = lambda name: to_np(grid_interp.interp(idx, wts, d[name]))  # noqa: E731
    sp = g('sp') / 100.
    t = g('2t') - LAPSE * (sfc['elev'] - orog)
    return {'t': t, 'q': qsat(g('2d'), sp), 'u': g('10u'), 'v': g('10v'), 'sp': sp}


def score_sfc(acc, k, d_f, d_e, sfc, idx, wts, orog, table):
    hf, he = sfc_operator(d_f, sfc, idx, wts, orog), sfc_operator(d_e, sfc, idx, wts, orog)
    net, typ, sub = sfc['network'], sfc['obtype'], sfc['subtype']
    station = net != sfcobs_parquet.NETWORK_CODES['scat']
    oro_ok = ~station | (np.abs(sfc['elev'] - orog) <= args.zthresh)
    p = np.where(np.isfinite(sfc['p']), sfc['p'], he['sp'])
    checks = {
        't': (sfc['t'] - he['t'], np.full(len(net), SFC_ERR['t']), 't', 1.),
        'q': ((sfc['q'] - he['q']) / qsat(he['t'], p), np.full(len(net), SFC_ERR['q']), 'q', 100.),
        'uv': (np.hypot(sfc['u'] - he['u'], sfc['v'] - he['v']),
               np.where(station, SFC_ERR['uv'], SCAT_ERR_UV), 'uv', 1.),
    }
    ok = {}
    for v, (dep, err, otype, scale) in checks.items():
        ok[v] = oro_ok & gross_ok(table, otype, typ, sub, dep, err, scale)
    for name, code in sfcobs_parquet.NETWORK_CODES.items():
        m = net == code
        if not m.any():
            continue
        lat = sfc['lat'][m]
        for v, scale in (('t', 1.), ('q', 1e3)):
            if name != 'scat':
                good = ok[v][m]
                acc.add(f'sfc_{name}_{v}', k, lat, np.where(good, (sfc[v][m] - hf[v][m]) * scale, np.nan),
                        rejected=(np.isfinite(sfc[v][m]) & ~good).sum())
        good = ok['uv'][m]
        uo, vo, uf, vf = sfc['u'][m], sfc['v'][m], hf['u'][m], hf['v'][m]
        acc.add(f'sfc_{name}_ws', k, lat, np.where(good, np.hypot(uo, vo) - np.hypot(uf, vf), np.nan),
                rejected=(np.isfinite(uo) & ~good).sum())
        acc.add_vector(f'sfc_{name}_uv', k, lat, np.where(good, uo - uf, np.nan), np.where(good, vo - vf, np.nan))


def vinterp(d, b, idx, wts, p_obs):
    """model field `b` at the obs: horizontal interpolation, then linear in ln p between the
    model's levels; NaN outside them"""
    h = to_np(grid_interp.interp(idx, wts, d[b]))  # (nlev, nobs)
    lnp = np.log(np.asarray(lev[b]))
    order = np.argsort(lnp)
    lnp, h = lnp[order], h[order]
    x = np.log(p_obs)
    j = np.clip(np.searchsorted(lnp, x) - 1, 0, len(lnp) - 2)
    w = (x - lnp[j]) / (lnp[j + 1] - lnp[j])
    n = np.arange(len(x))
    return np.where((x >= lnp[0]) & (x <= lnp[-1]), (1 - w) * h[j, n] + w * h[j + 1, n], np.nan)


class ProfileAcc:
    """n, sum and sum of squares of O - F per (lead, region, platform group, pressure layer)"""

    def __init__(self):
        shape = (3, LEADS + 1, len(REGIONS), len(CONV_GROUPS), len(PBIN_MID))
        self.data = {v: np.zeros(shape) for v in ('t', 'q', 'uv')}
        self.nrej = {v: np.zeros(LEADS + 1) for v in self.data}

    def add(self, v, k, group, pbin, lat, d, rejected):
        ok = np.isfinite(d)
        for i, m in enumerate(region_masks(lat)):
            s = ok & m
            for j, x in enumerate((np.ones_like(d), d, d ** 2)):
                np.add.at(self.data[v][j, k, i], (group[s], pbin[s]), x[s])
        self.nrej[v][k] += rejected


conv_cache = {}
GROUP_INDEX = np.full(len(convobs_parquet.GROUP_CODES), -1)
for _i, _name in enumerate(CONV_GROUPS):
    GROUP_INDEX[convobs_parquet.GROUP_CODES[_name]] = _i


def conv_at(vt, d_e, table):
    """upper-air obs at valid time vt with their QC (vs ERA5 at vt, so the same for every run)"""
    if vt not in conv_cache:
        o = convobs_parquet.load_convobs_parquet(exp['obspath'], [vt], args.conv_tolerance_hours, groups=CONV_GROUPS,
                                                 logger=logger)
        c = {f: v[0] for f, v in o.items()}
        idx, wts = grid_interp.weights(c['lon'], c['lat'], k=4, device=device)
        c = {f: v.astype(np.float64) if v.dtype.kind == 'f' else v for f, v in c.items()}
        p = c['p']
        he = {b: vinterp(d_e, b, idx, wts, p) for b in ('t', 'q', 'u', 'v')}
        above = p <= to_np(grid_interp.interp(idx, wts, d_e['sp'])) / 100.
        pbin = np.digitize(p, PBIN_EDGES) - 1
        above &= (pbin >= 0) & (pbin < len(PBIN_MID))
        typ, sub = c['obtype'], c['subtype']
        err = lambda name: np.where(np.isfinite(c['err_' + name]) & (c['err_' + name] > 0), c['err_' + name],  # noqa: E731
                                    CONV_ERR[name])
        dep = {'t': c['t'] - he['t'], 'q': (c['q'] - he['q']) / qsat(he['t'], p),
               'uv': np.hypot(c['u'] - he['u'], c['v'] - he['v'])}
        have = {v: above & np.isfinite(dep[v]) for v in dep}
        have['q'] &= p >= args.q_min_pressure
        ok = {v: have[v] & gross_ok(table, v, typ, sub, dep[v], err(v), 100. if v == 'q' else 1.) for v in dep}
        conv_cache[vt] = (c, idx, wts, ok, {v: int((have[v] & ~ok[v]).sum()) for v in ok}, pbin,
                          GROUP_INDEX[c['group']])
        for old in [d for d in conv_cache if d < utils.add_hours(vt, -6 * LEADS)]:
            del conv_cache[old]
    return conv_cache[vt]


def score_conv(pacc, k, d_f, d_e, table, vt):
    c, idx, wts, ok, nrej, pbin, group = conv_at(vt, d_e, table)
    hf = {b: vinterp(d_f, b, idx, wts, c['p']) for b in ('t', 'q', 'u', 'v')}
    pacc.add('t', k, group, pbin, c['lat'], np.where(ok['t'], c['t'] - hf['t'], np.nan), nrej['t'])
    pacc.add('q', k, group, pbin, c['lat'], np.where(ok['q'], (c['q'] - hf['q']) * 1e3, np.nan), nrej['q'])
    pacc.add('uv', k, group, pbin, c['lat'],
             np.where(ok['uv'], np.hypot(c['u'] - hf['u'], c['v'] - hf['v']), np.nan), nrej['uv'])


def done(ofile):
    if not os.path.exists(ofile):
        return False
    try:
        with xr.open_dataset(ofile) as ds:
            return ds.attrs.get('skill_version', 1) >= SKILL_VERSION
    except (OSError, ValueError):  # e.g. a job killed mid-write: recompute
        return False


def initial_state(T):
    if args.source == 'era5':
        state, _ = utils.get_input(dict(exp, date=T, nhrs_back=0, restart_window=False), model, logger)
        return state
    t0 = utils.add_hours(T, -args.window_hours)
    # a weak-constraint run saves its window-end analysis (with the model-error
    # jumps); advancing its optimal_inputs (valid at window start + 6 h) would drop them
    files = glob.glob(f'{args.source}/{t0}_wcend_inputs_{args.window_hours}h_*it.pkl')
    if files:
        logger.info(f'weak-constraint window-end analysis {files[0]}')
    else:
        files = glob.glob(f'{args.source}/{t0}_optimal_inputs_{args.window_hours}h_*it.pkl')
    if len(files) != 1:
        raise FileNotFoundError(f'expected one {args.source}/{t0}_optimal_inputs_{args.window_hours}h_*it.pkl, found {files}')
    with open(files[0], 'rb') as fh:
        state = model.prime_from_state(pickle.load(fh))
    hours = (datetime.strptime(T, '%Y-%m-%dT%H') - state.date.replace(tzinfo=None)).total_seconds() / 3600
    state = model.advance(state, steps=int(round(hours / 6)), use_checkpoint=False)
    assert state.date.replace(tzinfo=None) == datetime.strptime(T, '%Y-%m-%dT%H'), (state.date, T)
    return utils.reset_skt_over_ocean(model, state, utils.get_verif(dict(exp, date=T), model, logger),
                                      lsm_threshold=exp.get('skt_ocean_lsm_threshold', 0.5))


sfc_cache = {}


def sfcobs_at(vt):
    if vt not in sfc_cache:
        o = sfcobs_parquet.load_sfcobs_parquet(exp['obspath'], [vt], args.sfc_tolerance_hours, logger=logger)
        sfc = {f: v[0] for f, v in o.items()}
        idx, wts = grid_interp.weights(sfc['lon'], sfc['lat'], k=4, device=device)
        sfc_cache[vt] = (sfc, idx, wts)
        for old in [d for d in sfc_cache if d < utils.add_hours(vt, -6 * LEADS)]:
            del sfc_cache[old]
    return sfc_cache[vt]


T = args.start
while T <= args.end:
    ofile = f'{OUT}/{T}.nc'
    if done(ofile):
        T = utils.add_hours(T, args.every)
        continue
    e = dict(exp, date=T)
    table = gsi_convinfo.read_convinfo(gsi_convinfo.find_convinfo(args.gsinfo, T))
    verif0 = utils.get_verif(e, model, logger)
    orog_grid = verif0['geopotential_at_surface'] / utils.GRAV
    psobs = utils.get_psobs(e, model, logger, grid_interp)
    raobs = utils.get_raobs(e, model, logger, grid_interp, psobs)
    acc, pacc = Acc(), ProfileAcc()
    era = {}
    with torch.no_grad():
        state = initial_state(T)
        for k in range(LEADS + 1):
            if k > 0:
                state = model.advance(state, steps=1, use_checkpoint=False)
            vt = utils.add_hours(T, 6 * k)
            d_f = model.decode_state(state)
            d_e = utils.get_verif(e, model, logger, date_override=vt)
            for key, val in era5_scores(fields(d_f), fields(d_e)).items():
                era[key + (k,)] = val
            score_ps(acc, k, d_f, d_e, verif0, psobs, e, table)
            score_raobs(acc, k, d_f, d_e, raobs, table)
            sfc, idx, wts = sfcobs_at(vt)
            orog = to_np(grid_interp.interp(idx, wts, orog_grid))
            score_sfc(acc, k, d_f, d_e, sfc, idx, wts, orog, table)
            score_conv(pacc, k, d_f, d_e, table, vt)
    ds = xr.Dataset(coords={'lead_hours': 6 * np.arange(LEADS + 1), 'region': list(REGIONS)},
                    attrs={'init_time': T, 'source': args.source, 'label': args.label, 'config': args.config,
                           'convinfo': gsi_convinfo.find_convinfo(args.gsinfo, T), 'zthresh_m': args.zthresh,
                           'sfc_tolerance_hours': args.sfc_tolerance_hours, 'skill_version': SKILL_VERSION,
                           'conv_tolerance_hours': args.conv_tolerance_hours, 'q_min_pressure_hpa': args.q_min_pressure,
                           'conv_level_edges_hpa': PBIN_EDGES,
                           'era5_vars': 'z500 m, t850 K, q850 g/kg, msl hPa, t2m K, q2m g/kg, ws/vw m/s; error = forecast - ERA5',
                           'obs_vars': 'O - F; ps hPa, t K, q g/kg, wind m/s; *_uv: s1 = sum |dV|, s2 = sum |dV|^2'})
    for name in ERA_FIELDS:
        ds[f'rmse_{name}'] = (('lead_hours', 'region'), np.array([[era[name, i, k][0] for i in range(len(REGIONS))] for k in range(LEADS + 1)]))
        ds[f'bias_{name}'] = (('lead_hours', 'region'), np.array([[era[name, i, k][1] for i in range(len(REGIONS))] for k in range(LEADS + 1)]))
    for b in raobs['variables']:
        ds = ds.assign_coords({f'level_{b}': raobs['levels_hpa'][b].cpu().numpy()})
    for name, a in acc.data.items():
        if name.startswith('raob_'):
            b = name[5:] if name[5:] in raobs['variables'] else 'u'
            dims = ('lead_hours', 'region', f'level_{b}')
        else:
            dims, a = ('lead_hours', 'region'), a[..., 0]
        for s, x in zip(('n', 's1', 's2'), a):
            ds[f'obs_{name}_{s}'] = (dims, x)
        ds[f'obs_{name}_nrej'] = (('lead_hours',), acc.nrej[name])
    ds = ds.assign_coords(conv_group=list(CONV_GROUPS), conv_level=PBIN_MID)
    for v, a in pacc.data.items():
        for s, x in zip(('n', 's1', 's2'), a):
            ds[f'conv_{v}_{s}'] = (('lead_hours', 'region', 'conv_group', 'conv_level'), x)
        ds[f'conv_{v}_nrej'] = (('lead_hours',), pacc.nrej[v])
    ds.to_netcdf(ofile)
    logger.info(f'wrote {ofile}: z500 rmse 0/{6 * LEADS} h {era["z500", 0, 0][0]:.2f} / {era["z500", 0, LEADS][0]:.2f} m')
    T = utils.add_hours(T, args.every)
