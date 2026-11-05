"""Error statistics in the latent control space (AIFS encoder hidden-mesh output, where
compute_loss_4dvar adds its increments), from differences with ERA5, to size B (window-start
increment) and Q (weak-constraint model-error jumps):

  B:     enc(ERA5(T0)) - enc(background(T0))           (a run's control_inputs, T0 = window start)
  Q24:   enc(ERA5(t))  - enc(M_24h(ERA5(t - 24 h)))    (24-h model error, from ERA5)
  A+h:   enc(ERA5(t))  - enc(analysis trajectory(t)),  t = T0 + 24/48/72/96 h
  incr:  the run's saved window-start latent increments (latent_increment_*.pt / latent_scale)

enc(x) is the encoder output for the lagged pair x (the processor input before the increment).
Writes OUT.npz (per-channel and per-hidden-node rms of each kind) and prints a summary:
global rms, per-channel spread, rms by latitude band of the hidden nodes.

Usage (GPU): python estimate_latent_errors.py CONFIG RUN_DIR OUT FIRST_T0 LAST_T0 [--every 24]
Needs the lagged ERA5 GRIB (era5_<T>_lagged.grib in the config's ic_cache) at T0 and T0 + 24..96 h.
"""

import argparse
import glob
import os
import pickle
import sys
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
ap.add_argument('--every', type=int, default=24)
args = ap.parse_args()

logger = utils.get_logger()
exp, windows = utils.load_config(args.config)
exp = utils.get_window(exp, list(windows)[0], windows)
model = utils.get_model(exp)
utils._ensure_backend_on_path('aifs')
import aifs_model  # noqa: E402

m = model.interface.model


def enc(state):
    """encoder hidden-mesh output for a model state, (n_hidden, n_channels), float32"""
    with torch.no_grad():
        xin = state.state[:, 0:model.multi_step, None, ...]
        xin = model.interface.pre_processors(xin, in_place=False)
        with torch.autocast(device_type=model.device.type, dtype=model._autocast_dtype):
            x_data_latent, _, shard_shapes_data = aifs_model._assemble_input_compat(m, xin, xin.shape[0])
            x_hidden = m.node_attributes(m._graph_name_hidden, batch_size=xin.shape[0])
            shard_hidden = aifs_model.get_shard_shapes(x_hidden, 0, None)
            _, x_latent = aifs_model._run_mapper_compat(
                m, m.encoder, (x_data_latent, x_hidden), batch_size=xin.shape[0],
                shard_shapes=(shard_shapes_data, shard_hidden), x_src_is_sharded=False,
                x_dst_is_sharded=False, keep_x_dst_sharded=True)
    return x_latent.float().reshape(model.latent_shape)


def era5(T):
    state, _ = utils.get_input(dict(exp, date=T, nhrs_back=0, restart_window=False), model, logger)
    return state


def load(f):
    with open(f, 'rb') as fh:
        return model.prime_from_state(pickle.load(fh))


def adv(state, hours):
    with torch.no_grad():
        return model.advance(state, steps=hours // 6, use_checkpoint=False)


sumsq = {}  # kind -> (sum over samples of d^2 per node and channel, n)


def add(kind, d):
    s, n = sumsq.get(kind, (torch.zeros(model.latent_shape, device=d.device), 0))
    sumsq[kind] = (s + d ** 2, n + 1)


T0 = datetime.strptime(args.first, '%Y-%m-%dT%H')
while T0 <= datetime.strptime(args.last, '%Y-%m-%dT%H'):
    t0 = T0.strftime('%Y-%m-%dT%H')
    bg = load(glob.glob(f'{args.run}/{t0}_control_inputs_*it.pkl')[0])
    e0 = enc(era5(t0))
    add('B', e0 - enc(bg))
    inc = glob.glob(f'{args.run}/{t0}_latent_increment_*it.pt')
    if inc:
        st = torch.load(inc[0], map_location=model.device)
        add('incr', st['increment'] / float(st.get('latent_scale', 1.0)))
    traj = load(glob.glob(f'{args.run}/{t0}_optimal_inputs_*it.pkl')[0])  # valid T0 + 6 h
    traj = adv(traj, 18)
    for h in (24, 48, 72, 96):
        t = (T0 + timedelta(hours=h)).strftime('%Y-%m-%dT%H')
        et = enc(era5(t))
        add(f'A+{h}h', et - enc(traj))
        if h == 24:
            add('Q24', et - enc(adv(era5(t0), 24)))
        traj = adv(traj, 24)
    print(t0, {k: round(float((v[0] / v[1]).mean().sqrt()), 4) for k, v in sumsq.items()}, flush=True)
    T0 += timedelta(hours=args.every)

ms = {k: (v[0] / v[1]).cpu().numpy() for k, v in sumsq.items()}  # mean square per node, channel
try:
    xy = m._graph_data[m._graph_name_hidden]['x'].cpu().numpy()  # (lat, lon) radians
    lat = np.degrees(xy[:, 0])
except Exception as err:  # noqa: BLE001 -- graph layout differs between anemoi versions
    print('no hidden-node coordinates:', err)
    lat = np.full(model.latent_shape[0], np.nan)
np.savez(args.out, lat=lat, n={k: v[1] for k, v in sumsq.items()},
         **{f'{k}_chan': np.sqrt(v.mean(0)) for k, v in ms.items()},
         **{f'{k}_node': np.sqrt(v.mean(1)) for k, v in ms.items()})
bands = [(-90, -20), (-20, 20), (20, 90)]
print(f"\n{'kind':8s} {'rms':>8s} {'chan p10':>9s} {'chan p50':>9s} {'chan p90':>9s}  "
      + ' '.join(f'{a:>4d}..{b:<4d}' for a, b in bands) + '   corr(chan rms, B chan rms)')
bch = np.sqrt(ms['B'].mean(0))
for k, v in ms.items():
    ch = np.sqrt(v.mean(0))
    node = np.sqrt(v.mean(1))
    lb = ' '.join(f'{np.sqrt(np.mean(node[(lat >= a) & (lat < b)] ** 2)):10.4f}' for a, b in bands)
    print(f'{k:8s} {np.sqrt(v.mean()):8.4f} {np.percentile(ch, 10):9.4f} {np.percentile(ch, 50):9.4f} '
          f'{np.percentile(ch, 90):9.4f}  {lb}   {np.corrcoef(ch, bch)[0, 1]:.2f}')
print('wrote', args.out)
