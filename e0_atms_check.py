"""E0 (notes/integrate_atms_obs.md): check lw4dvar's ATMS operator path.

(a) H(x) on the ERA5 state at the window start, through get_atmsobs +
    _compute_atms_observation_diagnostics_at_time, against the decoder run
    offline on radianced's ERA5 pairs for the same obs (same Obs_Time = 0 and
    q_50 = training mean as the 4D-Var). Differences come only from the two
    horizontal-interpolation codes.
(b) Jo_atms (assimilated channels) finite-difference vs autograd derivative
    along a random direction in t, q (levels not detached), skt, 2t, tcw.

Usage: python e0_atms_check.py config_lwatms_atms_smoke.yml [era5_pre_dir]
"""

import sys

import numpy as np
import pandas as pd
import torch

import long_window_4dvar_utils as utils

cfg = sys.argv[1]
pre_dir = sys.argv[2] if len(sys.argv) > 2 else '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/era5_pre_2015'
exp, windows = utils.load_config(cfg)
exp['date'] = exp['sdate']
exp = utils.get_window(exp, list(windows)[0], windows)
logger = utils.get_logger()
model = utils.get_model(exp)
grid_interp = utils.get_grid_interpolator(model, exp)
verif = utils.get_verif(exp, model, logger)
missing = [k for k in utils.ATMS_MODEL_FIELDS if k not in verif]
print('ERA5 verif fields missing for ATMS:', missing, flush=True)
if missing:
    state_fields = model.decode_state(utils.get_input(exp, model, logger)[0])
    for k in missing:
        verif[k] = state_fields[k]
    print('  filled from the model state (constant fields only expected):', missing, flush=True)

slots = {'step_lo': torch.tensor([0]), 'alpha': torch.tensor([0.0]), 'n_steps': 1}
atms = utils.get_atmsobs(exp, model, logger, grid_interp, slots)
innov, tb, screened, used = utils._compute_atms_observation_diagnostics_at_time(model, verif, atms, grid_interp, 0)
n = int(atms['n_valid'][0])
tb = tb[:n].detach().cpu().numpy()
scr = screened[:n].cpu().numpy()

# (a) offline: same obs rows (cycle file, dt < tol, same scene screen), radianced's ERA5 inputs
op = atms['op']
cyc = utils.get_YYYYMMDDHH(exp['date'])
obsf = pd.read_parquet(f"{exp['atms_path']}/{cyc}.parquet")
pre = pd.read_parquet(f"{pre_dir}/{cyc}.parquet")
assert len(obsf) == len(pre)
tol = exp['dt_obs'] / 2
keep = ((obsf.dt_hours >= -tol) & (obsf.dt_hours < tol) & (obsf.water_frac >= exp.get('atms_water_frac_min', 0.99))
        & (obsf.ice_frac <= 0) & (obsf.clw_obs <= exp.get('atms_clw_max', 0.05))).to_numpy()
pre = pre[keep]
assert len(pre) == n, (len(pre), n)
x = torch.tensor(pre[op.input_features].to_numpy(np.float32), device=op.x_mean.device)
x[:, op.input_features.index('spfh-1')] = op.x_mean[op.input_features.index('spfh-1')]
x[:, op.input_features.index('Obs_Time')] = 0.0
with torch.no_grad():
    tb_off = op(x)[0].cpu().numpy()
d = np.abs(tb - tb_off)[scr]
print(f'\n(a) {n} obs in slot 0, {int(scr.sum())} screened; |H lw4dvar - H offline| (K) on screened obs')
print('  ch    ' + ' '.join(f'{c:6d}' for c in range(1, 23)))
print('  mean  ' + ' '.join(f'{v:6.3f}' for v in d.mean(0)))
print('  max   ' + ' '.join(f'{v:6.3f}' for v in d.max(0)))
y = atms['y'][0, :n].cpu().numpy()
for name, h in (('lw4dvar', tb), ('offline', tb_off)):
    r = np.sqrt(((y - h)[scr] ** 2).mean(0))
    print(f'  O-H rmse {name:8s} ch5-11 ' + ' '.join(f'{v:6.3f}' for v in r[4:11]))

# (b) finite difference vs autograd, Jo_atms on the assimilated channels
a = atms['assimilate_idx']
mask = used[:, a].detach()
sig = atms['sigma_o'][a]
names = ['t', 'q', 'skt', '2t', 'tcw']
base = {k: verif[k].detach().clone() for k in names}
torch.manual_seed(0)
scale = {'t': 0.5, 'q': 1e-4, 'skt': 0.5, '2t': 0.5, 'tcw': 1.0}
direction = {k: scale[k] * torch.randn_like(base[k]) for k in names}
detached_q = [model.pressure_levels('q').tolist().index(p) for p in model.pressure_levels('q') if p < exp.get('atms_detach_q_above_hpa', 250.)]
direction['q'][detached_q] = 0.0


def jo(fields):
    dec = dict(verif, **fields)
    inn, _, _, _ = utils._compute_atms_observation_diagnostics_at_time(model, dec, atms, grid_interp, 0)
    i = torch.where(mask, inn[:, a], torch.zeros_like(inn[:, a]))
    return ((i / sig) ** 2).sum()


leaf = {k: base[k].clone().requires_grad_() for k in names}
J0 = jo(leaf)
J0.backward()
ad = sum(float((leaf[k].grad * direction[k]).sum()) for k in names)
print(f'\n(b) Jo_atms = {J0.item():.6g} over {int(mask.sum())} ob-channels; autograd dJ.d = {ad:.6g}')
with torch.no_grad():
    for eps in (1e-1, 3e-2, 1e-2, 3e-3):
        jp = jo({k: base[k] + eps * direction[k] for k in names}).item()
        jm = jo({k: base[k] - eps * direction[k] for k in names}).item()
        fd = (jp - jm) / (2 * eps)
        print(f'  eps {eps:7.0e}: FD {fd:.6g}  rel diff {abs(fd - ad) / abs(ad):.2e}')
