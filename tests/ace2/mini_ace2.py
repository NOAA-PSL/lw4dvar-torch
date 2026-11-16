"""The mini-ACE2 solver run shared by tests/ace2/test_ace2_mini.py and
`python tests/make_fixtures.py golden --ace2`: its config, the checkpoint
lookup, the run itself (long_window_4dvar.py in a subprocess, as a user
runs it) and the golden summary of its outputs.

The mini ACE2 runs in float32, so its summary compares with per-quantity
tolerances (`tolerance`) rather than the toy model's 1e-8. Measured by
running at 1 and at 4 threads (different reduction order, as on a different
machine): everything computed from the background is bit-identical; the
cost agrees to ~5e-6 at the first epochs, then fp32 rounding is amplified by
the minimization to ~2e-3 by epoch 50 and ~3e-4 in the analysis O-A. The
run is pinned to one thread anyway, so the core count does not matter. The
tight checks -- background and QC exactly or to 1e-5, the first epochs to
1e-4 -- catch a changed setting, input or operator; the loose ones catch a
minimization that goes somewhere else. The toy model's golden runs
(tests/golden/) are the precise check on the minimization itself.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

import numpy as np
import xarray as xr
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO_ROOT, 'tests', 'data', 'mini_ace2')
GOLDEN = os.path.join(REPO_ROOT, 'tests', 'data', 'golden', 'mini_ace2.json')
EPOCHS = 50
# epochs (1-based) whose cost is checked tightly, before fp32 drift grows
TIGHT_EPOCHS = 5


def tolerance(path):
    """(rtol, atol) for a summary value, by its path; see the docstring."""
    if path.startswith('/config'):
        return 0.0, 0.0
    if path.startswith('/obs/background/innov_mean'):
        return 1e-5, 1e-4      # hPa; the mean O-B is near zero (observed 3e-6)
    if path.startswith('/obs/background') or path.startswith('/obs/available') or \
            path.startswith('/obs/qc_flag_counts') or path.startswith('/jo_per_slot_first'):
        return 1e-5, 0.0
    if path.startswith('/loss['):
        epoch = int(path[len('/loss['):-1]) + 1
        return (1e-4, 0.0) if epoch <= TIGHT_EPOCHS else (1e-2, 0.0)
    if path.startswith('/obs/analysis/used'):
        return 0.0, 5          # a borderline ob may flip
    if path.startswith('/obs/analysis/innov_mean'):
        return 0.0, 0.01       # hPa; the mean O-A is near zero
    if path.startswith('/analysis_forecast'):
        return 1e-3, 0.0
    return 1e-2, 0.0           # jo_per_slot_last, analysis O-A rms
# fields of the analysis forecast summarized at each output time
FORECAST_FIELDS = ['h500', 'TMP850', 'PRESsfc']


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def manifest():
    with open(os.path.join(DATA, 'manifest.json')) as f:
        return json.load(f)


def checkpoint_path():
    """The checkpoint: $LW4DVAR_MINI_ACE2_CKPT, else the cache directory
    ($LW4DVAR_FIXTURE_CACHE, default ~/.cache/lw4dvar), downloading the
    pinned release asset there if needed. Checked against its sha256."""
    ck = manifest()['checkpoint']
    path = os.environ.get('LW4DVAR_MINI_ACE2_CKPT')
    if not path:
        cache = os.environ.get('LW4DVAR_FIXTURE_CACHE', os.path.expanduser('~/.cache/lw4dvar'))
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, ck['name'])
        if not os.path.exists(path) or _sha256(path) != ck['sha256']:
            tmp = path + '.part'
            urllib.request.urlretrieve(ck['url'], tmp)
            os.replace(tmp, path)
    got = _sha256(path)
    assert got == ck['sha256'], f"{path}: sha256 {got} != manifest {ck['sha256']}"
    return path


def config(tmp_path, ckpt):
    m = manifest()
    exp = {
        'model_backend': 'ace2',
        'device': 'cpu',
        'path_model': os.path.dirname(ckpt) + '/',
        'model_name': os.path.basename(ckpt),
        'forcing_dir': os.path.join(DATA, 'forcing_data') + '/',
        'ic_cache': os.path.join(DATA, 'ic_cache') + '/',
        'obspath': os.path.join(DATA, 'psobs'),
        'path_output': os.path.join(str(tmp_path), 'output') + '/',
        'exp_name': 'test_ace2_mini',
        'rpath': os.path.join(str(tmp_path), 'unused') + '/',
        'restart': False,
        'cycle': False,
        'sdate': m['sdate'],
        'nhrs_back': m['spinup_h'],
        'n_init': 1,
        'dt_init': 6,
        'nobs_max': 15000,
        'ic_suffix': f"{m['window_h']}h_{EPOCHS}it",
        'save_levs': [1000, 850, 500, 250, 50],
        'print_every': 1,
        'oberrstart': -1,
        'oberrdeltaperday': 0,
        'checkpoint_stride': 0,
        'jc_ps_weight': 10,
    }
    windows = {str(m['window_h'] // 24): {
        'n_verif': m['window_h'] // 6, 'dt_verif': 6, 'dt_obs': 6,
        'learn_rate': 1.0e-2, 'weight_decay': 1.0e-3, 'max_epoch': EPOCHS,
        'warmup_steps': 5, 'start_factor': 0.1, 'end_factor': 1,
        'bg_check': 4.0, 'zthresh': 500.0, 'zconst': 0.003, 'suffix': '',
    }}
    path = os.path.join(str(tmp_path), 'config_test_ace2_mini.yml')
    with open(path, 'w') as f:
        yaml.safe_dump({'exp': exp, 'windows': windows}, f, sort_keys=False)
    return path, exp, windows


def run(tmp_path, ckpt):
    """Run the solver on the mini ACE2. Returns (returncode, log, exp,
    windows, output directory)."""
    cfg, exp, windows = config(tmp_path, ckpt)
    # one thread: the result must not depend on the machine's core count
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    proc = subprocess.run([sys.executable, '-u', 'long_window_4dvar.py', cfg], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    out = os.path.join(exp['path_output'], exp['exp_name'])
    return proc.returncode, proc.stdout + proc.stderr, exp, windows, out


def epoch_losses(log):
    return {int(e): float(v) for e, v in re.findall(r'epoch=(\d+), lr=[^,]+, loss=(\S+)', log)}


def _area_weights(lat):
    w = np.cos(np.radians(lat))
    return w / w.sum()


def summarize(exp, windows, out):
    """The golden summary of a run's outputs: the cost history (total and
    per obs slot), the obs diagnostics per slot (counts, O-B and O-A), and
    area-weighted mean and std of a few analysis-forecast fields per time."""
    m = manifest()
    stem = os.path.join(out, f"{m['sdate']}_%s_{m['window_h']}h_{EPOCHS}it")
    with open(stem % 'loss_latent' + '.json') as f:
        history = json.load(f)

    obs = {}
    with xr.open_dataset(stem % 'observation_diagnostics' + '.nc') as ds:
        ob = ds['surface_pressure_observation'].values.astype(np.float64)
        for kind, used_var in (('background', 'surface_pressure_used'),
                               ('analysis', 'surface_pressure_analysis_used')):
            hx = ds[f'surface_pressure_{kind}'].values.astype(np.float64)
            used = ds[used_var].values.astype(bool)
            # no used obs at a slot (the analysis at window start: it is
            # valid from the first step on) -> null
            d = [(ob[j] - hx[j])[used[j]] for j in range(len(ob))]
            obs[kind] = {
                'used': used.sum(axis=1).tolist(),
                'innov_mean': [float(x.mean()) if len(x) else None for x in d],
                'innov_rms': [float(np.sqrt((x ** 2).mean())) if len(x) else None for x in d],
            }
        obs['available'] = ds['surface_pressure_available'].values.astype(bool).sum(axis=1).tolist()
        qc = ds['surface_pressure_qc_flag'].values
        obs['qc_flag_counts'] = {str(f): int((qc == f).sum()) for f in np.unique(qc)}

    forecast = {}
    with xr.open_dataset(stem % 'optimal_forecast' + '.nc') as ds:
        lat = ds['lat'].values if 'lat' in ds else ds['latitude'].values
        w = _area_weights(lat)
        for name in FORECAST_FIELDS:
            v = ds[name].values.astype(np.float64)          # (time, values)
            mean = (v * w).sum(axis=-1)
            std = np.sqrt((((v - mean[:, None]) ** 2) * w).sum(axis=-1))
            forecast[name] = {'mean': mean.tolist(), 'std': std.tolist()}

    return {
        'config': {'exp': {k: v for k, v in exp.items() if not k.startswith('path') and k not in
                           ('forcing_dir', 'ic_cache', 'obspath', 'rpath')},
                   'windows': windows},
        'loss': [h['loss'] for h in history],
        'jo_per_slot_first': history[0]['all_loss'],
        'jo_per_slot_last': history[-1]['all_loss'],
        'obs': obs,
        'analysis_forecast': forecast,
    }
