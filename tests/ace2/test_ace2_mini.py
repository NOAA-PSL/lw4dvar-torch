"""Smoke run of the full solver with a real (tiny) model: the mini ACE2
(2.5 deg, 16-layer CRTM+land variables, SFNO embed_dim 64) on CPU, one 24 h
window of ps obs, 50 epochs, through long_window_4dvar.py exactly as a user
runs it.

Inputs are tests/data/mini_ace2 (tests/make_fixtures.py mini_ace2) plus the
checkpoint, a GitHub release asset pinned by sha256 in that directory's
manifest.json: found at $LW4DVAR_MINI_ACE2_CKPT, else in the cache directory
($LW4DVAR_FIXTURE_CACHE, default ~/.cache/lw4dvar), else downloaded there.

Needs fme (envs/requirements-ci-ace2.txt); run with `make check-ace2`. It checks
that the code runs end to end and that the minimization works (a large,
steady cost drop), not analysis quality: this model is far too small for
that (its z500 analysis gets slightly worse).
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.request

import pytest
import yaml

pytest.importorskip('fme')

pytestmark = [pytest.mark.ace2, pytest.mark.slow]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO_ROOT, 'tests', 'data', 'mini_ace2')
EPOCHS = 50
# observed: 1979 -> 891 (-55%, synthetic parquet psobs fixture); the margin
# absorbs CPU/BLAS differences
MAX_FINAL_TO_FIRST = 0.6


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope='module')
def checkpoint():
    with open(os.path.join(DATA, 'manifest.json')) as f:
        ck = json.load(f)['checkpoint']
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


def _config(tmp_path, ckpt):
    with open(os.path.join(DATA, 'manifest.json')) as f:
        m = json.load(f)
    exp = {
        'model_backend': 'ace2',
        'device': 'cpu',
        'path_model': os.path.dirname(ckpt) + '/',
        'model_name': os.path.basename(ckpt),
        'forcing_dir': os.path.join(DATA, 'forcing_data') + '/',
        'ic_cache': os.path.join(DATA, 'ic_cache') + '/',
        'observations': {'psobs': {
            'path': os.path.join(DATA, 'psobs'), 'nobs_max': 15000, 'use_source_error': True,
        }},
        'path_output': str(tmp_path / 'output') + '/',
        'exp_name': 'test_ace2_mini',
        'rpath': str(tmp_path / 'unused') + '/',
        'restart': False,
        'cycle': False,
        'sdate': m['sdate'],
        'nhrs_back': m['spinup_h'],
        'n_init': 1,
        'dt_init': 6,
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
    path = tmp_path / 'config_test_ace2_mini.yml'
    path.write_text(yaml.safe_dump({'exp': exp, 'windows': windows}, sort_keys=False))
    return path, exp


def test_ace2_mini_window(tmp_path, checkpoint):
    cfg, exp = _config(tmp_path, checkpoint)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    run = subprocess.run([sys.executable, '-u', 'src/long_window_4dvar.py', str(cfg)], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=1200)
    log = run.stdout + run.stderr
    assert run.returncode == 0, log[-4000:]

    losses = {int(e): float(v) for e, v in re.findall(r'epoch=(\d+), lr=[^,]+, loss=(\S+)', log)}
    assert sorted(losses) == list(range(1, EPOCHS + 1)), f'epochs logged: {sorted(losses)}'
    assert all(math.isfinite(v) for v in losses.values()), losses
    assert losses[EPOCHS] <= MAX_FINAL_TO_FIRST * losses[1], (
        f'cost fell only {losses[1]:.0f} -> {losses[EPOCHS]:.0f} in {EPOCHS} epochs')

    out = os.path.join(exp['path_output'], exp['exp_name'])
    stem = f"{exp['sdate']}_"
    for kind in ('latent_increment', 'observation_diagnostics', 'control_forecast', 'optimal_forecast'):
        hits = [f for f in os.listdir(out) if f.startswith(stem + kind)]
        assert hits, f'no {kind} output in {out}: {sorted(os.listdir(out))}'
