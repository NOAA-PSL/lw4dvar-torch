"""Golden runs on the toy model: the solver's minimization (`compute_optimal`)
on the one-window toy problem of tests/math/problem.py, with QC active, for
a few configs. `run(name)` returns a JSON-able summary that
tests/golden/test_golden_toy.py compares with tests/data/golden/toy_<name>.json
and `python tests/make_fixtures.py golden` rewrites.

Float64 (control and optimizer state included) and deterministic on CPU,
so the comparison is tight (RTOL): any change to the numbers -- a default, a
QC threshold, the optimizer or its schedule, a weighting, an operator --
shows up, whether or not the gradients stay right.

The obs values are frozen in tests/data/golden/toy_obs.json, not rebuilt
from the solver's forward operator on each run: obs made from the current
operator would move with it (a bias added to the operator would shift obs
and H(x) alike and leave the innovations unchanged). The obs locations,
errors and station heights come from a fixed numpy seed in problem.py,
independent of the solver. `python tests/make_fixtures.py golden
--toy-obs` rewrites the obs file -- only when the toy problem itself
changes, never to absorb a solver change.
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
OBS_PATH = os.path.join(GOLDEN_DIR, 'toy_obs.json')
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
    # no ps_operator: the solver's own fallback (problem.py's BASE_EXP sets one)
    'default': {'ps_operator': None},
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


# One frozen obs set per forward operator, each made with that operator so
# the obs are consistent with it (the 'ps' operator's H(x) differs from
# logpinterp's by more than the background check allows at most stations).
OBS_SETS = {'logpinterp': {'ps_operator': 'logpinterp'}, 'ps': CONFIGS['ps_operator']}


def _obs_set(name):
    return 'ps' if CONFIGS[name].get('ps_operator') == 'ps' else 'logpinterp'


def make_obs(obs_set):
    """The toy obs as problem.py builds them (the background's H(x), with
    `obs_set`'s operator, plus noise), with the gross errors added:
    (n_slots, N_OBS), as the solver reads them."""
    ob = Problem(**{**BASE, **OBS_SETS[obs_set]}).psobs_traj['ob'][:, :N_OBS].clone()
    for slot, k, err in GROSS:
        ob[slot, k] += err
    return ob


def write_obs():
    obs = {k: make_obs(k).tolist() for k in OBS_SETS}   # before truncating the file
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(OBS_PATH, 'w') as f:
        json.dump(obs, f, indent=1)
        f.write('\n')
    return OBS_PATH


def _problem(name):
    p = Problem(**{**BASE, **CONFIGS[name]})
    with open(OBS_PATH) as f:
        ob = torch.tensor(json.load(f)[_obs_set(name)], dtype=torch.float64)
    assert ob.shape == p.psobs_traj['ob'][:, :N_OBS].shape, ob.shape
    p.psobs_traj['ob'][:, :N_OBS] = ob
    return p


def _diagnostics(p, state, increment):
    """Per-slot obs diagnostics of the trajectory from `state` with
    `increment` injected at the first step: obs used, mean and rms
    innovation over the used obs."""
    m, traj = p.model, p.psobs_traj
    out = {'used': [], 'innov_mean': [], 'innov_rms': []}
    with torch.no_grad():
        s1 = m.advance(state, steps=1, use_checkpoint=False, latent_increment=increment)
        if increment is not None and p.keep_mask is not None:
            # as the solver: uncontrolled columns revert to the background
            s1 = m.wrap_state(utils._apply_control_mask(s1.state, p.keep_mask, p.ref_state1.state,
                                                        m.state_layout), s1.date)
        states = [state, s1]
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
    # compute_optimal makes its control (and so AdamW's state) in the
    # default dtype: float64 here, like the toy model
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            exp['path_output'] = tmp + '/'
            analysis, lsave = utils.compute_optimal(exp, p.model, p.state, p.verif_ic, p.psobs_traj,
                                                    p.grid_interp, logging.getLogger('golden'))
            stem = tmp + '/' + exp['date'] + '_%s_' + exp['window_name'] + 'h_' + str(exp['max_epoch']) + 'it'
            with open(stem % 'loss_latent' + '.json') as f:
                history = json.load(f)
            inc = torch.load(stem % 'latent_increment' + '.pt')['increment']
    finally:
        torch.set_default_dtype(dtype)
    assert inc.dtype == torch.float64, inc.dtype

    flat = inc.flatten()
    a = analysis.state[0].reshape(analysis.state.shape[1], -1)
    # area-weighted, as the solver's grid statistics (toy rows: equal point
    # counts, unequal cell areas)
    w = torch.as_tensor(p.model.area_weights, dtype=a.dtype)
    w = w / w.sum()
    a_mean = (a * w).sum(dim=1)
    a_std = (((a - a_mean[:, None]) ** 2) * w).sum(dim=1).sqrt()
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
        'analysis_channel_mean': a_mean.tolist(),
        'analysis_channel_std': a_std.tolist(),
        # a spatial sample of the returned analysis itself (every 7th point of
        # every channel: the stride is coprime with the 16-point rows, so a
        # shift or permutation in longitude or latitude moves it; the stats
        # above and the O-A replay of the increment would not see that)
        'analysis_sample': a[:, ::7].tolist(),
        'background_obs': _diagnostics(p, p.state, None),
        'analysis_obs': _diagnostics(p, p.state, inc),
        'n_obs_per_slot': N_OBS,
    }


def golden_path(name):
    return os.path.join(GOLDEN_DIR, f'toy_{name}.json')


def write(name):
    summary = run(name)          # before truncating the file: a failed run keeps the old one
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(golden_path(name), 'w') as f:
        json.dump(summary, f, indent=1)
        f.write('\n')
    return golden_path(name)
