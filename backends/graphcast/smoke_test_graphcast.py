"""Smoke test of the graphcast backend (GPU, graphcast-torch env):

  sbatch -t 01:00:00 run_graphcast.sh backends/graphcast/smoke_test_graphcast.py [DATE]

1. Bridged rollout vs lw4dvar-jax's own adapter (its graphcast_model.py,
   eager) from the same two input files: 2 steps, max |diff| per variable.
2. z500 rms vs ERA5 at +6 h .. +24 h (area-weighted).
3. Gradient of a 24 h z500 cost w.r.t. the latent vs central finite
   differences along smooth perturbations of single families.
4. Timing: forward and forward+backward of a 16-step (4-day) rollout.
"""

import datetime
import importlib.util
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'backends', 'graphcast'))
import graphcast_model  # noqa: E402

BASE = '/scratch4/BMC/gsienkf/Jeffrey.Whitaker/long-window-4dvar-jswhit2'
CKPT = f'{BASE}/models/graphcast_small.npz'
STATS = f'{BASE}/stats'
INPUT = f'{BASE}/input/2015-gc'
REF = '/scratch4/BMC/gsienkf/Jeffrey.Whitaker/lw4dvar-jax/backends/graphcast/graphcast_model.py'

date = datetime.datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else '2015-01-02T00')
model = graphcast_model.GraphCastModel(CKPT, STATS, INPUT, device='cuda')
w = torch.as_tensor(model.area_weights, dtype=torch.float32, device='cuda')
w = w / w.sum()
s0 = model.prepare_initial_state(model.read_inputs(date), date)
print(f'state {tuple(s0.state.shape)}, latent {model.latent_shape}, frozen {model._frozen}', flush=True)

# 1. vs lw4dvar-jax's adapter
import xarray as xr  # noqa: E402
spec = importlib.util.spec_from_file_location('gcm_ref', REF)
gcm_ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gcm_ref)
ref = gcm_ref.GraphCastModel(CKPT, STATS)
fn = lambda d: os.path.join(INPUT, d.strftime('%Y-%m-%dT%H') + '.nc')
f0 = gcm_ref.encode_from_files(xr.open_dataset(fn(date - datetime.timedelta(hours=6))), xr.open_dataset(fn(date)))
forc = gcm_ref.precompute_forcings(np.datetime64(date), 2, ref.timestep)
t0 = time.time()
_, traj = gcm_ref.unroll(ref, f0, forc, 2)
print(f'reference (eager) 2 steps: {time.time() - t0:.1f} s', flush=True)
with torch.no_grad():
    s = s0
    for k in range(2):
        s = model.advance(s, 1)
        dec = model.decode_state(s)
        for v, short in (('geopotential', 'z'), ('temperature', 't'), ('u_component_of_wind', 'u'),
                         ('specific_humidity', 'q'), ('mean_sea_level_pressure', 'msl'), ('2m_temperature', 't2m')):
            r = np.asarray(traj[v][k])                                   # (level, lon, lat) or (lon, lat)
            r = np.swapaxes(r, -1, -2).reshape(*r.shape[:-2], -1)       # -> (level, lat*lon) lat-major
            if r.ndim == 2:
                r = r[::-1]                                              # surface first
            o = dec[short].cpu().numpy()
            print(f'  step {k + 1} {v:26s} max|diff| {np.abs(o - r).max():.3e}  (field rms {np.sqrt((r ** 2).mean()):.3e})')

# 2. z500 vs ERA5
with torch.no_grad():
    s = s0
    for k in range(4):
        s = model.advance(s, 1)
        truth = model.read_verif(s.date)['z500']
        e = model.decode_state(s)['z500'] - truth
        print(f'z500 rms +{6 * (k + 1)}h: {torch.sqrt((w * e ** 2).sum()).item() / 9.80665:.2f} m', flush=True)
truth24 = model.read_verif(date + datetime.timedelta(hours=24))['z500']


# 3. gradient vs finite differences
def cost(latent):
    s = model.advance(s0, 4, latent_increment=latent)
    return ((w * (model.decode_state(s)['z500'] - truth24) ** 2).sum() / 9.80665 ** 2)


lat = torch.as_tensor(model.lats, device='cuda').reshape(model.latent_shape[1:])
lon = torch.as_tensor(model.lons, device='cuda').reshape(model.latent_shape[1:])
bump = (torch.exp(-((lat - 45) / 15) ** 2) * torch.cos(torch.deg2rad(lon - 180)).clamp(min=0) ** 2).float()
latent = torch.zeros(model.latent_shape, device='cuda', requires_grad=True)
t0 = time.time()
J = cost(latent)
J.backward()
print(f'cost {J.item():.4f}, first fwd+bwd {time.time() - t0:.1f} s', flush=True)
g = latent.grad
with torch.no_grad():
    z0 = torch.zeros(model.latent_shape, device='cuda')
    rep = [cost(z0).item() for _ in range(4)]
print(f'  repeat cost spread (GPU nondeterminism): {max(rep) - min(rep):.2e}', flush=True)
for fam in ('t', 'z', 'u', 'msl'):
    for amp in (0.003, 0.01, 0.03, 0.1, 0.3):
        d = torch.zeros(model.latent_shape, device='cuda')
        d[model.resolve_columns(fam)] = bump
        d = d * amp
        with torch.no_grad():
            fd = (cost(d) - cost(-d)).item() / 2
        an = (g * d).sum().item()
        print(f'  {fam:4s} amp {amp:<5g} grad.d {an:+.4e}  FD {fd:+.4e}  rel diff {abs(an - fd) / max(abs(fd), 1e-12):.2%}',
              flush=True)


# 4. timing (16 steps = 4-day window)
for label in ('forward', 'forward+backward'):
    torch.cuda.synchronize()
    t0 = time.time()
    latent = torch.zeros(model.latent_shape, device='cuda', requires_grad=label != 'forward')
    with torch.set_grad_enabled(label != 'forward'):
        s = model.advance(s0, 16, latent_increment=latent)
        J = (w * model.decode_state(s)['z500']).sum()
        if label != 'forward':
            J.backward()
    torch.cuda.synchronize()
    print(f'{label} 16 steps: {time.time() - t0:.1f} s; peak torch mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB',
          flush=True)
