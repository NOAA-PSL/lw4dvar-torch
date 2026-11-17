"""Golden runs on the toy model: the solver's minimization (`compute_optimal`)
on the one-window toy problem of tests/math/problem.py, with QC active, for
a few configs. `run(name)` returns a JSON-able summary that
tests/golden/test_golden_toy.py compares with tests/data/golden/toy_<name>.json
and `python tests/make_fixtures.py golden` rewrites.

Float64 and deterministic on CPU, so the comparison is tight (RTOL): any
change to the numbers -- a default, a QC threshold, the optimizer or its
schedule, a weighting -- shows up, whether or not the gradients stay right.
"""

import copy
import json
import logging
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'math'))

import long_window_4dvar_utils as utils  # noqa: E402
from problem import N_OBS, Problem  # noqa: E402
from golden_compare import compare, report  # noqa: E402,F401

GOLDEN_DIR = os.path.join(os.path.dirname(HERE), 'data', 'golden')
RTOL = 1e-8

# The solver settings every config shares: a 24 h window, 3-hourly obs, the
# real QC (background check and station-height check) and the optimizer with
# its warmup + cosine schedule.
BASE = {
    'bg_check': 4.0, 'zthresh': 500.0, 'zconst': 3e-3,
    'learn_rate': 0.05, 'weight_decay': 1e-3, 'max_epoch': 20,
    'warmup_steps': 3, 'start_factor': 0.1, 'end_factor': 0.1,
    'print_every': -1, 'latent_scale': 1.0,
    'date': '2015010100', 'window_name': '1', 'suffix': '',
}
CONFIGS = {
    'default': {},
    'obden_jcps': {'obs_density_box_deg': 10.0, 'jc_ps_weight': 0.01},
    'control_stride2': {'control_variables': ['z', 'sp'], 'checkpoint_stride': 2},
    # the MAPS reduction from a mid-level t, q (the toy model has no 700 hPa;
    # the driver sets nlev700 from the model's levels)
    'ps_operator': {'ps_operator': 'ps', 'nlev700': 1},
    # Jz ~20% of the first-epoch cost
    'jc_z': {'jc_z_weight': 5.0},
}
# obs given a gross error (hPa), so the background check rejects them
GROSS = [(0, 3, 40.0), (2, 17, -35.0), (5, 8, 60.0), (8, 20, -50.0)]


def _problem(name):
    p = Problem(**{**BASE, **CONFIGS[name]})
    for slot, k, err in GROSS:
        p.psobs_traj['ob'][slot, k] += err
    return p


def _diagnostics(p, state, increment):
    """Per-slot obs diagnostics of the trajectory from `state` with
    `increment` injected at the first step: obs used, mean and rms
    innovation over the used obs."""
    m, traj = p.model, p.psobs_traj
    out = {'used': [], 'innov_mean': [], 'innov_rms': []}
    with torch.no_grad():
        states = [state, m.advance(state, steps=1, use_checkpoint=False, latent_increment=increment)]
        for _ in range(traj['n_steps'] - 1):
            states.append(m.advance(states[-1], steps=1, use_checkpoint=False))
        decoded = [m.decode_state(s) for s in states]
        for j in range(len(traj['step_lo'])):
            s, a = int(traj['step_lo'][j]), float(traj['alpha'][j])
            d = utils._interp_decoded(decoded[s], decoded[s + 1], a, p.interp_specs)
            innov, _, _, _, used, _ = utils._compute_ps_observation_diagnostics_at_time(
                m, d, p.verif_ic, traj, p.exp, j, p.grid_interp)
            iu = innov[used]
            out['used'].append(int(used.sum()))
            out['innov_mean'].append(float(iu.mean()) if len(iu) else 0.0)
            out['innov_rms'].append(float(iu.pow(2).mean().sqrt()) if len(iu) else 0.0)
    return out


def run(name):
    p = _problem(name)
    exp = copy.deepcopy(p.exp)
    with tempfile.TemporaryDirectory() as tmp:
        exp['path_output'] = tmp + '/'
        analysis, lsave = utils.compute_optimal(exp, p.model, p.state, p.verif_ic, p.psobs_traj,
                                                p.grid_interp, logging.getLogger('golden'))
        stem = tmp + '/' + exp['date'] + '_%s_' + exp['window_name'] + 'h_' + str(exp['max_epoch']) + 'it'
        with open(stem % 'loss_latent' + '.json') as f:
            history = json.load(f)
        inc = torch.load(stem % 'latent_increment' + '.pt')['increment']

    flat = inc.flatten()
    a = analysis.state[0].reshape(analysis.state.shape[1], -1)
    resolved = {k: v for k, v in exp.items() if k != 'path_output'}
    return {
        'config': resolved,
        'n_obs_available': int(p.psobs_traj['n_valid'].sum()),
        'loss': [h['loss'] for h in history],
        'jo_per_slot_first': history[0]['all_loss'],
        'jo_per_slot_last': history[-1]['all_loss'],
        'best_loss': lsave,
        'increment': {
            'norm': float(flat.norm()), 'sum': float(flat.sum()),
            'min': float(flat.min()), 'max': float(flat.max()),
            'sample': flat[::37].tolist(),
        },
        'analysis_channel_mean': a.mean(dim=1).tolist(),
        'analysis_channel_std': a.std(dim=1).tolist(),
        'background_obs': _diagnostics(p, p.state, None),
        'analysis_obs': _diagnostics(p, p.state, inc),
        'n_obs_per_slot': N_OBS,
    }


def golden_path(name):
    return os.path.join(GOLDEN_DIR, f'toy_{name}.json')


def write(name):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(golden_path(name), 'w') as f:
        json.dump(run(name), f, indent=1)
        f.write('\n')
    return golden_path(name)
