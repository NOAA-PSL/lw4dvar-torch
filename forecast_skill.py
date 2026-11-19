"""Daily 5-day forecast skill from the end of the 4D-Var window vs from ERA5
(notes/integrate_atms_obs.md, "forecast test protocol").

For each 00Z date T in [START, END] (daily):
  era5  IC = ERA5 at T (the checkpoint's lagged pair T-6h, T; no spin-up)
  run   IC = the analysis of the window ending at T: <RUN_DIR>/<T-120h>_optimal_inputs_120h_100it.pkl
        (valid at window start + 6 h) advanced to T, then skt over ocean reset from ERA5 at T, as the
        cycling does at each analysis time
then a 120-h forecast, scored every 6 h (lead 0..120 h):
  vs ERA5: area-weighted RMSE and mean error (forecast - ERA5) of z500 (m), t850 (K), q850 (g/kg),
           ws250 (m/s, speed), vw250 (m/s, vector wind), msl (hPa), t2m (K); regions Global/NH/Tropics/SH
  vs obs:  PREPBUFR surface pressure (the solver's operator and orography screen, no gross check) and
           radiosonde t/u/v/q on the model levels (read-time QC only, no background check), from the
           solver's readers; per lead, region (station latitude) and level: n, sum and sum of squares of
           O - F, units as stored (ps hPa, t K, u/v m/s, q kg/kg)
Lead 0 of a run's forecast is the window's last analysis time: its obs were assimilated.

Writes output/fcst_skill/<LABEL>/<T>.nc (dates already written are skipped).

Usage: python forecast_skill.py CONFIG LABEL era5|RUN_DIR START END
  CONFIG: an lwatms config supplying the model, ic_cache and observations (raobs enabled)
"""

import os
import pickle
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr

import long_window_4dvar_utils as utils

cfg, label, src, start, end = sys.argv[1:6]
OUT = f'output/fcst_skill/{label}'
LEADS = 20  # 6-h steps
REGIONS = ('Global', 'NH', 'Tropics', 'SH')
os.makedirs(OUT, exist_ok=True)

exp, windows = utils.load_config(cfg)
exp['date'] = exp['sdate']
exp = utils.get_window(exp, list(windows)[0], windows)
exp['n_verif'], exp['dt_verif'], exp['dt_obs'] = LEADS, 6, 6
exp.pop('raobs_q_bg_check', None)  # score every sonde q, not only those near the forecast
logger = utils.get_logger()
model = utils.get_model(exp)
grid_interp = utils.get_grid_interpolator(model, exp)
area = np.asarray(model.area_weights, dtype=np.float64)
lats = np.asarray(model.lats)
rmask = {'Global': np.ones_like(lats, bool), 'NH': lats > 20, 'Tropics': np.abs(lats) <= 20, 'SH': lats < -20}
lev = {b: [float(p) for p in model.pressure_levels(b)] for b in ('z', 't', 'q', 'u', 'v')}


def fields(d):
    g = lambda b, p: d[b][lev[b].index(p)].detach().cpu().numpy().astype(np.float64)  # noqa: E731
    u, v = g('u', 250.), g('v', 250.)
    return {'z500': g('z', 500.) / utils.GRAV, 't850': g('t', 850.), 'q850': g('q', 850.) * 1e3,
            'u250': u, 'v250': v, 'msl': d['msl'].detach().cpu().numpy() / 100., 't2m': d['2t'].detach().cpu().numpy()}


def era5_scores(f, e):
    out = {}
    for name in ('z500', 't850', 'q850', 'msl', 't2m', 'ws250', 'vw250'):
        if name == 'ws250':
            diff = np.hypot(f['u250'], f['v250']) - np.hypot(e['u250'], e['v250'])
        elif name == 'vw250':
            diff = None
            sq = (f['u250'] - e['u250']) ** 2 + (f['v250'] - e['v250']) ** 2
        else:
            diff = f[name] - e[name]
        for r in REGIONS:
            w = area * rmask[r]
            if diff is not None:
                out[name, r] = (np.sqrt((w * diff ** 2).sum() / w.sum()), (w * diff).sum() / w.sum())
            else:
                out[name, r] = (np.sqrt((w * sq).sum() / w.sum()), np.nan)
    return out


def region_of(lat):
    return {'Global': np.ones_like(lat, bool), 'NH': lat > 20, 'Tropics': np.abs(lat) <= 20, 'SH': lat < -20}


def initial_state(T):
    if src == 'era5':
        e = dict(exp, date=T, nhrs_back=0, restart_window=False)
        state, _ = utils.get_input(e, model, logger)
        return state
    t0 = utils.add_hours(T, -LEADS * 6)
    with open(f'{src}/{t0}_optimal_inputs_120h_100it.pkl', 'rb') as fh:
        state = model.prime_from_state(pickle.load(fh))
    hours = (datetime.strptime(T, '%Y-%m-%dT%H') - state.date.replace(tzinfo=None)).total_seconds() / 3600
    state = model.advance(state, steps=int(round(hours / 6)), use_checkpoint=False)
    assert state.date.replace(tzinfo=None) == datetime.strptime(T, '%Y-%m-%dT%H'), (state.date, T)
    return utils.reset_skt_over_ocean(model, state, utils.get_verif(dict(exp, date=T), model, logger),
                                      lsm_threshold=exp.get('skt_ocean_lsm_threshold', 0.5))


T = start
while T <= end:
    ofile = f'{OUT}/{T}.nc'
    if os.path.exists(ofile):
        T = utils.add_hours(T, 24)
        continue
    e = dict(exp, date=T)
    verif0 = utils.get_verif(e, model, logger)
    psobs = utils.get_psobs(e, model, logger, grid_interp)
    raobs = utils.get_raobs(e, model, logger, grid_interp, psobs)
    obs_vars = ['ps'] + list(raobs['variables'])
    nlev = {b: len(raobs['levels_hpa'][b]) for b in raobs['variables']}
    nlev['ps'] = 1
    acc = {(v, s): np.zeros((LEADS + 1, len(REGIONS), max(nlev.values()))) for v in obs_vars for s in ('n', 's1', 's2')}
    era = {}
    with torch.no_grad():
        state = initial_state(T)
        for k in range(LEADS + 1):
            if k > 0:
                state = model.advance(state, steps=1, use_checkpoint=False)
            vt = utils.add_hours(T, 6 * k)
            d = model.decode_state(state)
            for key, val in era5_scores(fields(d), fields(utils.get_verif(e, model, logger, date_override=vt))).items():
                era[key + (k,)] = val
            innov, _, _, _, _, qc = utils._compute_ps_observation_diagnostics_at_time(model, d, verif0, psobs, e, k, grid_interp)
            ok = ((qc & 7) == 0).cpu().numpy()
            o, la = innov.cpu().numpy()[ok], psobs['lat'][k].cpu().numpy()[ok]
            for i, (r, m) in enumerate(region_of(la).items()):
                acc['ps', 'n'][k, i, 0] += m.sum()
                acc['ps', 's1'][k, i, 0] += o[m].sum()
                acc['ps', 's2'][k, i, 0] += (o[m] ** 2).sum()
            res = utils._compute_raobs_observation_diagnostics_at_time(model, d, raobs, grid_interp, k)
            la = raobs['lat'][k].cpu().numpy()
            for b, (inn, _, _, used) in res.items():
                inn, used = inn.cpu().numpy(), used.cpu().numpy()
                for i, (r, m) in enumerate(region_of(la).items()):
                    um = used & m[None, :]
                    acc[b, 'n'][k, i, :nlev[b]] += um.sum(1)
                    acc[b, 's1'][k, i, :nlev[b]] += np.where(um, inn, 0).sum(1)
                    acc[b, 's2'][k, i, :nlev[b]] += np.where(um, inn ** 2, 0).sum(1)
    ds = xr.Dataset(coords={'lead_hours': 6 * np.arange(LEADS + 1), 'region': list(REGIONS)},
                    attrs={'init_time': T, 'source': src, 'label': label, 'config': cfg,
                           'era5_vars': 'z500 m, t850 K, q850 g/kg, msl hPa, t2m K, ws250 / vw250 m/s; error = forecast - ERA5',
                           'obs_vars': 'O - F; ps hPa, t K, u/v m/s, q kg/kg'})
    for name in ('z500', 't850', 'q850', 'msl', 't2m', 'ws250', 'vw250'):
        ds[f'rmse_{name}'] = (('lead_hours', 'region'), np.array([[era[name, r, k][0] for r in REGIONS] for k in range(LEADS + 1)]))
        ds[f'bias_{name}'] = (('lead_hours', 'region'), np.array([[era[name, r, k][1] for r in REGIONS] for k in range(LEADS + 1)]))
    for v in obs_vars:
        n = nlev[v]
        if v != 'ps':
            ds = ds.assign_coords({f'level_{v}': raobs['levels_hpa'][v].cpu().numpy()})
        dims = ('lead_hours', 'region') + (() if v == 'ps' else (f'level_{v}',))
        for s in ('n', 's1', 's2'):
            a = acc[v, s][..., :n]
            ds[f'obs_{v}_{s}'] = (dims, a[..., 0] if v == 'ps' else a)
    ds.to_netcdf(ofile)
    logger.info(f'wrote {ofile}: z500 rmse 0/120 h {era["z500", "Global", 0][0]:.2f} / {era["z500", "Global", LEADS][0]:.2f} m')
    T = utils.add_hours(T, 24)
