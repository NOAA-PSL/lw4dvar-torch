"""Fit and evaluate the static per-scene bias correction (lwatms/scene_bc.py) for one decoder bundle.

d = y - H(x_ERA5) on the 2015-16 ERA5 pair files (y = ObsBC = y_o - GSI replay BC, or raw y for `raw` decoders, masked
like ObsBC); over land / snow / sea ice (MW) the emissivity control z is fitted per ob to the window channels as in
cycling (as era5_ref.py), over ocean z = 0. Least squares per scene and channel (ridge 1e-3, |d| < DMAX, then one refit
without |d - fit| > 4 sd), on every STRIDE-th file of 2015-16 except September 2016 (the decoders' held-out month).
Evaluation (before / after) of the mean and sd of d by latitude band x scene x obs-cloud bin on (a) September 2016 and
(b) Jan-May 2015 (the cycling period).

Usage (lwaifs2-rocm venv, from lwatms):
  python tools/decoder_tables/fit_scene_bc.py SENSOR BUNDLE PRE TARGET OUT_PREFIX [STRIDE=4] [NPROC=16]
  TARGET: obsbc | raw.  Writes OUT_PREFIX.json (scene_bc table) and OUT_PREFIX_eval.npz; prints the tables."""
import glob
import json
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))  # the lwatms repo
import scene_bc as sbc  # noqa: E402
from obs_atms_operator import AtmsOperator  # noqa: E402

SENSOR, BUNDLE, PRE, TARGET, OUT = sys.argv[1:6]
STRIDE = int(sys.argv[6]) if len(sys.argv) > 6 else 4
OPTS = {k: v for k, v in (('clw_cap', float(os.environ['SCBC_CLW_CAP']) if os.environ.get('SCBC_CLW_CAP') else None),
                          ('clw_lat', os.environ.get('SCBC_CLW_LAT') == '1')) if v}   # 2026-11-23 overshoot fix options
NPROC = int(sys.argv[7]) if len(sys.argv) > 7 else 16
WIN = {'atms': [1, 2, 3, 4, 16, 17], 'amsua': [1, 2, 3, 15], 'iasi': []}[SENSOR]
DMAX = {'atms': 15.0, 'amsua': 15.0, 'iasi': 8.0}[SENSOR]
BANDS = [('SP <65S', -90, -65), ('SO 65-45S', -65, -45), ('SH 45-25S', -45, -25), ('TR', -25, 25), ('NH 25-45N', 25, 45),
         ('NA 45-70N', 45, 70), ('NP >70N', 70, 90.01)]
CB = [0.0, 0.05, 0.2, 0.5, 3.01]
torch.set_num_threads(2)
_OP = {}


def op():
    if 'op' not in _OP:
        _OP['op'] = AtmsOperator.load(BUNDLE)
    return _OP['op']


def fit_z(o, x, y, sig_w, iw, iters=4):
    with torch.no_grad():
        _, _, log_sig = o(x)
    pinv = torch.exp(-2 * log_sig)
    z = torch.zeros(x.shape[0], o.K)
    for _ in range(iters):
        zz = z.clone().requires_grad_(True)
        tb = o(x, zz)[0][:, iw]
        J = torch.stack([torch.autograd.grad(tb[:, w].sum(), zz, retain_graph=True)[0] for w in range(len(iw))], 1)
        r = (y[:, iw] - tb.detach()) / sig_w
        Jn = J / sig_w[None, :, None]
        A = Jn.transpose(1, 2) @ Jn + torch.diag_embed(pinv)
        g = (Jn.transpose(1, 2) @ r[..., None])[..., 0] - pinv * z
        z = z + torch.linalg.solve(A, g[..., None])[..., 0]
    return z.detach()


def residual(f):
    """d (nobs, nch), scene codes, predictors, lat, clw_obs for one pair file."""
    o = op(); nch = len(o.sigma_o)
    pre = pd.read_parquet(f)
    x = torch.tensor(pre[o.input_features].to_numpy(np.float32)); x[:, o.input_features.index('Obs_Time')] = 0.0
    yb = pre[[f'ObsBC-{i}' for i in range(nch)]].to_numpy(np.float64)
    y = yb if TARGET == 'obsbc' else np.where(np.isfinite(yb), pre[[f'Observation-{i}' for i in range(nch)]].to_numpy(np.float64), np.nan)
    wf, ice, snow = pre.water_frac.to_numpy(), pre.ice_frac.to_numpy(), pre.snow_frac.to_numpy()
    clw = np.nan_to_num(pre.clw_obs.to_numpy(np.float64), nan=0.0) if 'clw_obs' in pre else np.zeros(len(pre))
    with torch.no_grad():
        hx = o(x)[0]
    zmask = ~((wf >= 0.99) & (ice <= 0))
    if WIN and zmask.any():
        iw = [c - 1 for c in WIN]
        zi = torch.as_tensor(np.nonzero(zmask)[0])
        yt = torch.tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32)
        z = fit_z(o, x[zi], yt[zi], o.sigma_o[iw].clone(), iw)
        with torch.no_grad():
            hx[zi] = o(x[zi], z)[0]
    d = y - hx.numpy().astype(np.float64)
    d[np.abs(d) > DMAX] = np.nan
    sc = sbc.scene_codes(wf, ice, snow, clw)
    doy = datetime.strptime(os.path.basename(f)[:8], '%Y%m%d').timetuple().tm_yday
    P = sbc.predictors(SENSOR, pre.lat.to_numpy(), pre.Scan_Angle.to_numpy(), pre.sat_id.to_numpy(), np.full(len(pre), doy),
                       clw=clw, tb89=yb[:, 15] if SENSOR == 'atms' else None, tb165=yb[:, 16] if SENSOR == 'atms' else None, opts=OPTS)
    return d, sc, P, pre.lat.to_numpy(), clw


def normal_eq(args):
    f, beta = args
    d, sc, P, _, _ = residual(f)
    ns, nch, npr = len(sbc.SCENES), d.shape[1], P.shape[1]
    A = np.zeros((ns, nch, npr, npr)); b = np.zeros((ns, nch, npr)); ss = np.zeros((ns, nch, 3))
    for k in range(ns):
        m = sc == k
        if not m.any():
            continue
        Pk, dk = P[m], d[m]
        for c in range(nch):
            ok = np.isfinite(dk[:, c])
            if beta is not None:   # refit pass: drop |d - fit| > 4 sd of the first-pass residual
                r = dk[:, c] - Pk @ beta['beta'][k, c]
                ok &= np.abs(r) < 4 * beta['sd'][k, c]
            if ok.sum() == 0:
                continue
            A[k, c] += Pk[ok].T @ Pk[ok]; b[k, c] += Pk[ok].T @ dk[ok, c]
            r = dk[ok, c] - (Pk[ok] @ beta['beta'][k, c] if beta is not None else 0)
            ss[k, c] += [ok.sum(), r.sum(), (r ** 2).sum()]
    return A, b, ss


def evaluate(args):
    f, beta = args
    d, sc, P, lat, clw = residual(f)
    bias = sbc.apply({'beta': beta}, sc, P)
    e = d - bias
    nch = d.shape[1]
    S = np.zeros((5, len(BANDS), len(sbc.SCENES), len(CB) - 1, nch))   # n, sum d, sumsq d, sum e, sumsq e
    bi = np.full(len(lat), -1)
    for i, (_, lo, hi) in enumerate(BANDS):
        bi[(lat >= lo) & (lat < hi)] = i
    ci = np.clip(np.digitize(clw, CB) - 1, 0, len(CB) - 2)
    for i in range(len(BANDS)):
        for k in range(len(sbc.SCENES)):
            for j in range(len(CB) - 1):
                m = (bi == i) & (sc == k) & (ci == j)
                if not m.any():
                    continue
                dm, em = d[m], e[m]; ok = np.isfinite(dm)
                S[0, i, k, j] += ok.sum(0)
                S[1, i, k, j] += np.where(ok, dm, 0).sum(0); S[2, i, k, j] += np.where(ok, dm ** 2, 0).sum(0)
                S[3, i, k, j] += np.where(ok, em, 0).sum(0); S[4, i, k, j] += np.where(ok, em ** 2, 0).sum(0)
    return S


def solve(A, b, ridge=1e-3):
    ns, nch, npr = b.shape
    beta = np.zeros((ns, nch, npr))
    for k in range(ns):
        for c in range(nch):
            if A[k, c, 0, 0] < 200:   # too few obs: no correction
                continue
            beta[k, c] = np.linalg.solve(A[k, c] + ridge * A[k, c, 0, 0] * np.eye(npr) / npr, b[k, c])
    return beta


if __name__ == '__main__':
    allf = sorted(glob.glob(f'{PRE}/201[56]*.parquet'))
    train = [f for f in allf if not os.path.basename(f).startswith('201609')][::STRIDE]
    test = [f for f in allf if os.path.basename(f).startswith('201609')][::2]
    cyc = [f for f in allf if '20150101' <= os.path.basename(f)[:8] <= '20150531'][::STRIDE * 2]
    print(f'{SENSOR} {TARGET} {BUNDLE}: fit on {len(train)} files, eval Sep 2016 {len(test)} / Jan-May 2015 {len(cyc)}', flush=True)
    with Pool(NPROC) as p:
        A, b, ss = (sum(z) for z in zip(*p.map(normal_eq, [(f, None) for f in train])))
        beta1 = solve(A, b)
        sd1 = np.sqrt(np.maximum(ss[..., 2] / np.maximum(ss[..., 0], 1) - (ss[..., 1] / np.maximum(ss[..., 0], 1)) ** 2, 1e-6))
        # first-pass sd of the residual after the fit (recomputed on the refit pass's kept obs below)
        A, b, ss2 = (sum(z) for z in zip(*p.map(normal_eq, [(f, {'beta': beta1, 'sd': sd1 + 1e-3}) for f in train])))
        beta = solve(A, b)
        Ste = sum(p.map(evaluate, [(f, beta) for f in test]))
        Scy = sum(p.map(evaluate, [(f, beta) for f in cyc]))
    o = AtmsOperator.load(BUNDLE)
    json.dump({'sensor': SENSOR, 'target': TARGET, 'bundle': BUNDLE, 'pre': PRE, 'predictors': sbc.predictor_names(SENSOR, OPTS), 'options': OPTS,
               'scenes': list(sbc.SCENES), 'channels': list(range(1, len(o.sigma_o) + 1)), 'beta': beta.round(6).tolist(),
               'n_fit': A[:, :, 0, 0].astype(int).tolist(), 'fitted': datetime.now().isoformat(timespec='minutes'),
               'source': 'tools/decoder_tables/fit_scene_bc.py: 2015-16 ERA5 pairs except Sep 2016, d = y - H(x_ERA5)'},
              open(OUT + '.json', 'w'))
    np.savez(OUT + '_eval.npz', Ste=Ste, Scy=Scy, bands=np.array([x[0] for x in BANDS]), scenes=np.array(sbc.SCENES), cb=CB)
    chs = {'atms': {'5-7': [4, 5, 6], '10-13': [9, 10, 11, 12], '18-22': [17, 18, 19, 20, 21]},
           'amsua': {'4-7': [3, 4, 5, 6], '8-13': [7, 8, 9, 10, 11, 12]},
           'iasi': {'1-50': list(range(50)), '51-175': list(range(50, 175))}}[SENSOR]
    for name, S in (('Sep 2016 (held out)', Ste), ('Jan-May 2015', Scy)):
        print(f'\n=== {name}: mean d before -> after per channel group (|mean| averaged over channels), sd after; n')
        for g, cols in chs.items():
            print(f'-- channels {g}')
            for i, (bn, _, _) in enumerate(BANDS):
                for k, sn in enumerate(sbc.SCENES):
                    for j in range(len(CB) - 1):
                        n = S[0, i, k, j, cols]
                        if n.sum() < 500:
                            continue
                        mb = S[1, i, k, j, cols] / np.maximum(n, 1); ma = S[3, i, k, j, cols] / np.maximum(n, 1)
                        sa = np.sqrt(np.maximum(S[4, i, k, j, cols] / np.maximum(n, 1) - ma ** 2, 0))
                        print(f'{bn:10s} {sn:12s} clw {CB[j]:.2f}-{CB[j + 1]:.2f}  mean {np.mean(mb):+.3f} -> {np.mean(ma):+.3f}'
                              f'  |mean| {np.mean(np.abs(mb)):.3f} -> {np.mean(np.abs(ma)):.3f}  sd {np.mean(sa):.3f}  n {int(n.sum())}')
