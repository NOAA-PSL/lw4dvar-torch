"""load_config / get_window: every tracked config loads, and the validation
rules that guard against silent misconfiguration still fire."""

import glob
import os

import pytest
import yaml

import long_window_4dvar_utils as utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACKED_CONFIGS = sorted(glob.glob(os.path.join(REPO_ROOT, 'config_test_*.yml'))
                         + [os.path.join(REPO_ROOT, 'config.yml.template')])

WINDOW = {'n_verif': 4, 'dt_verif': 6, 'learn_rate': 1e-3, 'weight_decay': 0.0, 'max_epoch': 10,
          'warmup_steps': 0, 'start_factor': 1.0, 'end_factor': 1.0, 'suffix': ''}


def write_config(tmp_path, windows=None, **exp):
    exp = {'model_backend': 'aifs', 'exp_name': 'unit', 'path_output': './output/', **exp}
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump({'exp': exp, 'windows': windows or {'1': dict(WINDOW)}}))
    return str(path)


@pytest.mark.parametrize('path', TRACKED_CONFIGS, ids=os.path.basename)
def test_tracked_configs_load(path):
    exp, windows = utils.load_config(path)
    assert exp['path_output'].endswith(exp['exp_name'] + '/')
    for name in windows:
        utils.get_window(dict(exp), name, windows)


def test_path_output_includes_exp_name(tmp_path):
    exp, _ = utils.load_config(write_config(tmp_path))
    assert exp['path_output'] == './output/unit/'


def test_missing_file_returns_none(tmp_path):
    assert utils.load_config(str(tmp_path / 'nope.yml')) == (None, None)


@pytest.mark.parametrize('exp, match', [
    ({'model_backend': 'gpt'}, 'model_backend'),
    ({'n_init': 4}, 'cycle: True'),
    ({'jc_ps_sigma': 0.1}, 'jc_ps_sigma has been removed'),
    ({'jc_ps_weight': -1}, 'jc_ps_weight must be > 0'),
    ({'jb_samples': 'x.pt'}, r'jb_samples .*only implemented for the .*got .aifs.'),
    ({'jb_weight': 2.0}, 'jb_samples is not'),
    ({'obs_density_box_deg': -5}, 'obs_density_box_deg'),
    ({'ps_operator': 'ps_native'}, r"ps_operator='ps_native' is only supported for .*got .aifs."),
    ({'model_backend': 'fcn3', 'reset_skt_ocean': True}, 'reset_skt_ocean'),
    ({'model_backend': 'fcn3', 'ps_operator': 'ps'}, "ps_operator='ps'"),
    ({'model_backend': 'ace2', 'ps_operator': 'ps'}, "ps_operator='ps'"),
    ({'model_backend': 'fcn3', 'autocast_dtype': 'bf16'}, 'not implemented'),
    ({'autocast_dtype': 'int8'}, 'must be one of'),
    ({'model_backend': 'ace2', 'dry_mass_fixer': True}, 'dry_mass_fixer'),
])
def test_invalid_configs_rejected(tmp_path, exp, match):
    with pytest.raises(ValueError, match=match):
        utils.load_config(write_config(tmp_path, **exp))


def test_ace2_defaults_to_ps_native(tmp_path):
    exp, _ = utils.load_config(write_config(tmp_path, model_backend='ace2'))
    assert exp['ps_operator'] == 'ps_native'


def test_autocast_alias_normalized(tmp_path):
    exp, _ = utils.load_config(write_config(tmp_path, aifs_autocast_dtype='bf16'))
    assert exp['autocast_dtype'] == 'bfloat16' and 'aifs_autocast_dtype' not in exp


def test_get_window_defaults_dt_obs(tmp_path):
    exp, windows = utils.load_config(write_config(tmp_path))
    exp = utils.get_window(exp, '1', windows)
    assert (exp['dt_obs'], exp['window_name'], exp['n_verif']) == (6, '24', 4)


@pytest.mark.parametrize('dt_obs', [4, 5])
def test_get_window_rejects_bad_dt_obs(tmp_path, dt_obs):
    exp, windows = utils.load_config(write_config(tmp_path, windows={'1': {**WINDOW, 'dt_obs': dt_obs}}))
    with pytest.raises(ValueError, match='dt_obs'):
        utils.get_window(exp, '1', windows)
