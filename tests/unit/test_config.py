"""load_config / get_window: every tracked config loads, and the validation
rules that guard against silent misconfiguration still fire."""

import glob
import os

import pytest
import yaml

import long_window_4dvar_utils as utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(REPO_ROOT, 'configs')
TRACKED_CONFIGS = sorted(glob.glob(os.path.join(CONFIGS_DIR, 'config_test_*.yml'))
                         + [os.path.join(CONFIGS_DIR, 'config.yml.template')])

WINDOW = {'n_verif': 4, 'dt_verif': 6, 'learn_rate': 1e-3, 'weight_decay': 0.0, 'max_epoch': 10,
          'warmup_steps': 0, 'start_factor': 1.0, 'end_factor': 1.0, 'suffix': ''}


def write_config(tmp_path, windows=None, **exp):
    exp = {
        'model_backend': 'aifs', 'exp_name': 'unit', 'path_output': './output/',
        'observations': {'psobs': {'path': './psobs'}}, **exp,
    }
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
    ({'jc_z_weight': -1}, 'jc_z_weight must be > 0'),
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
    ({'observations': {'bogus': {}}}, "unknown keys \\['bogus'\\]"),
    ({'observations': {'psobs': {'bogus': 1}}}, 'observations.psobs has unknown keys'),
    ({'observations': {}}, 'observations.psobs.path is required'),
    ({'observations': {'psobs': {'path': './psobs'}, 'raobs': {'enabled': True, 'bogus': 1}}},
     'observations.raobs has unknown keys'),
    ({'model_backend': 'fcn3', 'observations': {'psobs': {'path': './psobs'}, 'raobs': {'enabled': True}}},
     r"observations.raobs.enabled is only supported for model_backend in .*got 'fcn3'"),
    ({'observations': {'psobs': {'path': './psobs'}, 'raobs': {'enabled': True, 'variables': ['bogus']}}},
     r"observations.raobs.variables entries must be a subset of .*got unsupported entries \['bogus'\]"),
    ({'observations': {'psobs': {'path': './psobs'}, 'raobs': {'enabled': True, 'variables': []}}},
     'observations.raobs.variables must be non-empty'),
    ({'observations': {'psobs': {'path': './psobs'}, 'raobs': {'enabled': True, 'tuv_bg_check': -1.0}}},
     'observations.raobs.tuv_bg_check must be positive and finite'),
    ({'observations': {'psobs': {'path': './psobs'}, 'raobs': {'enabled': True, 'q_bg_check': 0.0}}},
     'observations.raobs.q_bg_check must be positive and finite'),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'error_std': {'t': {1000.: -1.0}}}}},
     r"observations.raobs.error_std\['t'\]\[1000.0\] must be positive and finite"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'error_std': {'t': {-1.0: 1.2}}}}},
     r"observations.raobs.error_std\['t'\] has a non-positive/non-finite pressure key -1.0"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'q_error_rh_percent': {1000.: -1.0}}}},
     r"observations.raobs.q_error_rh_percent\[1000.0\] must be positive and finite"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'bg_check': {'z': {'threshold': 7.0}}}}},
     r"observations.raobs.bg_check keys must be a subset of .*got \['z'\]"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'bg_check': {'t': {'absolute': 9.0}}}}},
     r"observations.raobs.bg_check\['t'\] must be a dict with a 'threshold' key"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'bg_check': {'t': {'threshold': 7.0, 'bogus': 1}}}}},
     r"observations.raobs.bg_check\['t'\] has unknown keys \['bogus'\]"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'bg_check': {'t': {'threshold': -7.0}}}}},
     r"observations.raobs.bg_check\['t'\].threshold must be positive and finite"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'bg_check': {'t': {'threshold': 7.0, 'absolute': 0.0}}}}},
     r"observations.raobs.bg_check\['t'\].absolute must be positive and finite"),
    ({'observations': {'psobs': {'path': './psobs'},
                       'raobs': {'enabled': True, 'vertical_interpolation': 'bogus'}}},
     r"observations.raobs.vertical_interpolation must be 'exact', 'nearest', or 'log-linear', got 'bogus'"),
])
def test_invalid_configs_rejected(tmp_path, exp, match):
    with pytest.raises(ValueError, match=match):
        utils.load_config(write_config(tmp_path, **exp))


def test_observations_psobs_resolved_to_flat_keys(tmp_path):
    exp, _ = utils.load_config(write_config(
        tmp_path, observations={'psobs': {
            'path': '/archive/root', 'nobs_max': 1234, 'use_source_error': True,
            'quality_mark_max': 2,
        }},
    ))
    assert exp['obspath'] == '/archive/root'
    assert exp['nobs_max'] == 1234
    assert exp['psobs_use_source_error'] is True
    assert exp['psobs_quality_mark_max'] == 2
    assert 'psobs_time_tolerance_hours' not in exp  # left unset -> get_psobs's own default applies


def test_observations_raobs_resolved_to_flat_keys(tmp_path):
    exp, _ = utils.load_config(write_config(
        tmp_path, observations={
            'psobs': {'path': './psobs'},
            'raobs': {
                'enabled': True, 'variables': ['temperature', 'u_component_of_wind'],
                'quality_mark_max': 2, 'q_bg_check': None, 'tuv_bg_check': 3.5,
                'use_source_error': True, 'use_drift_time': True, 'use_drift_position': True,
            },
        },
    ))
    assert exp['raobs_enabled'] is True
    assert exp['raobs_variables'] == ['temperature', 'u_component_of_wind']
    assert exp['raobs_quality_mark_max'] == 2
    assert exp['raobs_q_bg_check'] is None
    assert exp['raobs_tuv_bg_check'] == 3.5
    assert exp['raobs_use_source_error'] is True
    assert exp['raobs_use_drift_time'] is True
    assert exp['raobs_use_drift_position'] is True
    assert exp['raobs_path'] == exp['obspath']  # defaults to psobs's resolved path
    assert exp['raobs_vertical_interpolation'] == 'exact'  # left unset -> defaults to today's behavior


def test_observations_raobs_use_drift_position_left_unset_by_default(tmp_path):
    exp, _ = utils.load_config(write_config(
        tmp_path, observations={
            'psobs': {'path': './psobs'},
            'raobs': {'enabled': True},
        },
    ))
    assert 'raobs_use_drift_position' not in exp  # left unset -> get_raobs's own False default applies


@pytest.mark.parametrize('vertical_interpolation', ['exact', 'nearest', 'log-linear'])
def test_observations_raobs_vertical_interpolation_accepted(tmp_path, vertical_interpolation):
    exp, _ = utils.load_config(write_config(
        tmp_path, observations={
            'psobs': {'path': './psobs'},
            'raobs': {'enabled': True, 'vertical_interpolation': vertical_interpolation},
        },
    ))
    assert exp['raobs_vertical_interpolation'] == vertical_interpolation


def test_observations_raobs_error_std_table_accepted(tmp_path):
    # error_std may be a per-variable {hPa: value} table (same convention
    # as q_error_rh_percent) instead of a flat scalar -- e.g. a UFO/GSI sonde
    # error table (PR #29's review comment).
    table = {1000.: 1.2, 500.: 0.8, 10.: 1.5}
    exp, _ = utils.load_config(write_config(
        tmp_path, observations={
            'psobs': {'path': './psobs'},
            'raobs': {'enabled': True, 'error_std': {'t': table, 'u': 2.0}},
        },
    ))
    assert exp['raobs_error_std'] == {'t': table, 'u': 2.0}


def test_observations_raobs_q_error_rh_percent_table_accepted(tmp_path):
    # q's own {hPa: percent} RH-error table override -- GSI's real
    # mechanism (RH_fraction * qsat(background)), replacing the previous
    # Desroziers-based q_error_g_kg (PR #29 review discussion).
    table = {1000.: 10., 500.: 15., 10.: 20.}
    exp, _ = utils.load_config(write_config(
        tmp_path, observations={
            'psobs': {'path': './psobs'},
            'raobs': {'enabled': True, 'q_error_rh_percent': table},
        },
    ))
    assert exp['raobs_q_error_rh_percent'] == table


def test_observations_raobs_bg_check_accepted(tmp_path):
    # UFO/GSI's per-variable gross-check form (PR #29's review comment):
    # threshold + an optional absolute limit, per variable.
    cfg = {'t': {'threshold': 7.0, 'absolute': 9.0}, 'q': {'threshold': 8.0}}
    exp, _ = utils.load_config(write_config(
        tmp_path, observations={'psobs': {'path': './psobs'}, 'raobs': {'enabled': True, 'bg_check': cfg}},
    ))
    assert exp['raobs_bg_check'] == cfg


def test_observations_raobs_disabled_by_default(tmp_path):
    exp, _ = utils.load_config(write_config(tmp_path))
    assert exp['raobs_enabled'] is False


@pytest.mark.parametrize('backend', ['aurora', 'ace2', 'ngcm', 'neuralgcm', 'graphcast'])
def test_raobs_enabled_backends_accepted(tmp_path, backend):
    extra = {}
    if backend == 'ngcm':
        extra = {'ngcm_exp': 'x', 'ngcm_step': 1, 'ngcm_data': 'y'}
    elif backend == 'neuralgcm':
        extra = {'neuralgcm_checkpoint': 'x', 'neuralgcm_input_dir': 'y'}
    elif backend == 'graphcast':
        extra = {'graphcast_checkpoint': 'x', 'graphcast_stats_dir': 'y', 'graphcast_input_dir': 'z'}
    exp, _ = utils.load_config(write_config(
        tmp_path, model_backend=backend,
        observations={'psobs': {'path': './psobs'}, 'raobs': {'enabled': True}},
        **extra,
    ))
    assert exp['raobs_enabled'] is True


def test_get_raobs_rejects_specific_humidity_for_total_water_checkpoint():
    """get_raobs must reject raobs.variables including specific_humidity
    against a checkpoint whose native humidity is specific_total_water
    (vapor + cloud condensate) -- not comparable to a real radiosonde's
    vapor-only specific humidity (the released ACE2-ERA5 checkpoint; see
    docs/integrate_multiple_parquetobs.md's ACE2-specific caveat). Must
    fire before any archive I/O, so a fake model with no other backend
    behavior is enough."""
    class FakeModel:
        humidity_family = 'specific_total_water'

    exp = {'raobs_enabled': True, 'raobs_variables': ['temperature', 'specific_humidity'],
           'dt_obs': 6, 'date': '2015-01-01T00'}
    with pytest.raises(ValueError, match='specific_total_water'):
        utils.get_raobs(exp, FakeModel(), utils.get_logger(), None, {'step_lo': [0]})


def test_get_raobs_allows_tuv_for_total_water_checkpoint():
    """The same checkpoint must NOT be rejected for t/u/v-only raobs --
    only specific_humidity is the problem. The guard passing through
    cleanly is confirmed by failing later, on model.pressure_levels (which
    a real backend would have, but FakeModel deliberately doesn't) --
    not on the humidity_family check."""
    class FakeModel:
        humidity_family = 'specific_total_water'

    exp = {'raobs_enabled': True, 'raobs_variables': ['temperature'],
           'dt_obs': 6, 'date': '2015-01-01T00'}
    with pytest.raises(AttributeError, match='pressure_levels'):
        utils.get_raobs(exp, FakeModel(), utils.get_logger(), None, {'step_lo': [0]})


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
