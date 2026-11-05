"""Error samples in the latent control space by inversion: for a model state x at t - 6 h, the
latent increment delta (added to the encoder output on the one 6-h step to t, as
compute_loss_4dvar injects it) that best fits ERA5 at t -- min over delta of the area-weighted
mean square of the normalized state difference step(x, delta) - ERA5(t), all variables. The
spread of the fitted delta per latent channel and by latitude sizes B and Q in the units the
solver uses (estimate_latent_errors.py's encoder differences are not that: the decoder adds its
output onto x's own last level, so the increment acts on the 6-h tendency).

Samples, for each 00Z t (needs the lagged ERA5 GRIB at t):
  B      x = the background of the run's window starting t - 12 h, advanced 6 h
  Q24    x = ERA5(t - 24 h) advanced 18 h (the 24-h model error a 24-h jump has to absorb)
  A+18h  x = the run's analysis trajectory of the window starting t - 24 h (+18 h at x)
  A+90h  x = the same, window starting t - 96 h (+90 h at x)
(the run's windows start at 00Z and 12Z)
Writes OUT.npz (per-channel and per-hidden-node rms of the fitted delta, fit statistics) and
prints a summary; the fit fraction is the share of the mean-square misfit the delta removes.

--optimizer lbfgs fits from zero with L-BFGS, which stays in the range of the step's Jacobian
(a minimum-norm fit); Adam moves every element ~lr per step, null-space ones included, so its
delta carries noise the step does not see. --save DIR writes each fitted delta (float16,
DIR/<kind>_<T>.pt) for building a structured B / Q (latent_eof.py).

Usage (GPU): python fit_latent_to_era5.py CONFIG RUN_DIR OUT FIRST_T LAST_T [--iters 200] [--lr 0.01]
             [--optimizer adam|lbfgs] [--kinds B,Q24,A+18h,A+90h] [--every 24] [--save DIR]
"""

import argparse
import glob
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import torch

import long_window_4dvar_utils as utils

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('config')
ap.add_argument('run')
ap.add_argument('out')
ap.add_argument('first')
ap.add_argument('last')
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--lr', type=float, default=0.01)
ap.add_argument('--optimizer', choices=['adam', 'lbfgs'], default='adam')
ap.add_argument('--kinds', default='B,Q24,A+18h,A+90h')
ap.add_argument('--every', type=int, default=24)
ap.add_argument('--save', default='')
args = ap.parse_args()

logger = utils.get_logger()
exp, windows = utils.load_config(args.config)
exp = utils.get_window(exp, list(windows)[0], windows)
model = utils.get_model(exp)
m = model.interface.model
areaw = torch.as_tensor(model.area_weights, dtype=torch.float32, device=model.device)
areaw = areaw / areaw.sum()


def fmt(T):
    return T.strftime('%Y-%m-%dT%H')


def era5(T):
    state, _ = utils.get_input(dict(exp, date=fmt(T), nhrs_back=0, restart_window=False), model, logger)
    return state


def load(pattern):
    with open(glob.glob(pattern)[0], 'rb') as fh:
        return model.prime_from_state(pickle.load(fh))


def adv(state, hours):
    with torch.no_grad():
        return model.advance(state, steps=hours // 6, use_checkpoint=False)


def norm_last(state):
    """the state's last time level, normalized as the model's input, (n_points, n_vars)"""
    x = state.state[:, -1:, None, ...]
    return model.interface.pre_processors(x, in_place=False)[0, 0, 0]


def fit(x, target):
    """fitted delta, initial and final misfit"""
    tgt = norm_last(target).detach()
    ok = torch.isfinite(tgt).all(0)  # variables with a finite normalized value everywhere

    def misfit(d):
        y = norm_last(model.advance(x, steps=1, latent_increment=d, use_checkpoint=False))
        return (areaw[:, None] * (y - tgt)[:, ok] ** 2).sum() / ok.sum()

    delta = torch.zeros(model.latent_shape, device=model.device, requires_grad=True)
    with torch.no_grad():
        j0 = misfit(None).item()
    if args.optimizer == 'lbfgs':
        # optimize v, delta = scale * v: L-BFGS's first step is lr / ||grad||_1 along -grad, about
        # 1e-8 per element over 41M elements (below the model's bf16 resolution, so the line search
        # stalls); scale makes it ~args.lr rms
        scale = args.lr * 0.8 * delta.numel()
        v = torch.zeros_like(delta, requires_grad=True)
        opt = torch.optim.LBFGS([v], lr=1.0, max_iter=args.iters, history_size=20,
                                line_search_fn='strong_wolfe', tolerance_grad=0., tolerance_change=0.)
        hist = []

        def closure():
            opt.zero_grad()
            j = misfit(scale * v)
            j.backward()
            hist.append(j.item())
            return j
        opt.step(closure)
        print(f'   lbfgs {len(hist)} evals, misfit ' + ' '.join(f'{v:.4g}' for v in hist[::max(1, len(hist) // 8)]),
              flush=True)
        with torch.no_grad():
            d = (scale * v).detach().clone()
            return d, j0, misfit(d).item()
    opt = torch.optim.Adam([delta], lr=args.lr)
    best, jbest = None, j0
    for _ in range(args.iters):
        opt.zero_grad()
        j = misfit(delta)
        if j.item() < jbest:
            best, jbest = delta.detach().clone(), j.item()
        j.backward()
        opt.step()
    return (best if best is not None else torch.zeros_like(delta)), j0, jbest


sumsq, stats = {}, {}
T = datetime.strptime(args.first, '%Y-%m-%dT%H')
while T <= datetime.strptime(args.last, '%Y-%m-%dT%H'):
    target = era5(T)
    makers = {
        'B': lambda: adv(load(f'{args.run}/{fmt(T - timedelta(hours=12))}_control_inputs_*it.pkl'), 6),
        'Q24': lambda: adv(era5(T - timedelta(hours=24)), 18),
        'A+18h': lambda: adv(load(f'{args.run}/{fmt(T - timedelta(hours=24))}_optimal_inputs_*it.pkl'), 12),
        'A+90h': lambda: adv(load(f'{args.run}/{fmt(T - timedelta(hours=96))}_optimal_inputs_*it.pkl'), 84),
    }
    for kind in args.kinds.split(','):
        x = makers[kind]()
        assert x.date.replace(tzinfo=None) == T - timedelta(hours=6), (kind, x.date, T)
        d, j0, j1 = fit(x, target)
        s, n = sumsq.get(kind, (torch.zeros(model.latent_shape, device=model.device), 0))
        sumsq[kind] = (s + d ** 2, n + 1)
        stats.setdefault(kind, []).append((j0, j1))
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            torch.save(d.half().cpu(), f'{args.save}/{kind}_{fmt(T)}.pt')
        print(f'{fmt(T)} {kind:6s} misfit {j0:.4g} -> {j1:.4g} ({100 * (1 - j1 / j0):.0f}% removed), '
              f'delta rms {d.pow(2).mean().sqrt().item():.4g}', flush=True)
    T += timedelta(hours=args.every)

ms = {k: (v[0] / v[1]).cpu().numpy() for k, v in sumsq.items()}
try:
    lat, lon = np.degrees(m._graph_data[m._graph_name_hidden]['x'].cpu().numpy()).T
except Exception as err:  # noqa: BLE001 -- graph layout differs between anemoi versions
    print('no hidden-node coordinates:', err)
    lat = lon = np.full(model.latent_shape[0], np.nan)
np.savez(args.out, lat=lat, lon=lon, **{f'{k}_chan': np.sqrt(v.mean(0)) for k, v in ms.items()},
         **{f'{k}_node': np.sqrt(v.mean(1)) for k, v in ms.items()},
         **{f'{k}_misfit': np.array(v) for k, v in stats.items()})
bands = [(-90, -20), (-20, 20), (20, 90)]
bch = np.sqrt(ms['B'].mean(0))
print(f"\n{'kind':6s} {'n':>3s} {'removed':>8s} {'rms':>8s} {'chan p10':>9s} {'chan p90':>9s}  "
      + ' '.join(f'{a:>4d}..{b:<4d}' for a, b in bands) + '  corr(chan, B)')
for k, v in ms.items():
    ch, node = np.sqrt(v.mean(0)), np.sqrt(v.mean(1))
    st = np.array(stats[k])
    lb = ' '.join(f'{np.sqrt(np.mean(node[(lat >= a) & (lat < b)] ** 2)):10.4g}' for a, b in bands)
    print(f'{k:6s} {len(st):3d} {100 * (1 - st[:, 1].sum() / st[:, 0].sum()):7.0f}% {np.sqrt(v.mean()):8.4g} '
          f'{np.percentile(ch, 10):9.4g} {np.percentile(ch, 90):9.4g}  {lb}  {np.corrcoef(ch, bch)[0, 1]:.2f}')
print('wrote', args.out)
