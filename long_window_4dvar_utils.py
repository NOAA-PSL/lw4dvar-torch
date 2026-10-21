"""
Originator: Greg Hakim
            University of Washington
            January 2025
            rev. December 2025
            4dvar test code: April 2026

Jeff Whitaker modified to assimilate real surface pressure obs June-August 2026

Merged (September 2026) from two independent single-backend ports of this
same long-window 4D-Var solver: `long-window-4dvar-aifsv2` (NeuralGCM/JAX ->
AIFS-single-2.0/PyTorch, August 2026) and `long-window-4dvar-fcstnetv3`
(AIFS-single-2.0/PyTorch -> FourCastNet3/PyTorch, September 2026), then
extended with a third backend, Microsoft Aurora (AuroraV1p5, also September
2026). A single top-level `exp['model_backend']` config key ('aifs', 'fcn3',
or 'aurora') now selects the forecast model at runtime;
`backends/aifs/`/`backends/fcn3/`/`backends/aurora/` hold each backend's
model/grid/IC-fetching modules, imported LAZILY (inside the function that
needs them, never at this module's top level) so that a process running
with only one backend's conda env active (`aifs2` vs `fcstnet3` vs `aurora`
-- see integrate_multiple_backends.md; the three envs are not co-installed) never needs the
other backends' dependencies importable at all.

All three backends implement `forecast_model.LatentForecastModel` with no
interface violations (see forecast_model.py) and share almost all of this
file unchanged -- `get_psobs`, `get_surface_pressure`, `preduce`,
`_interp_decoded`, `save_trajectory_diagnostics`, `save_xr_trajectory`,
`save_inputs_pkl`/`_nc`, `make_forecasts` are all backend-agnostic, built
entirely on `model.advance`/`model.decode_state`/`model.n_points`/
`model.lats`/`model.lons`/`model._levels_by_base`/`model._single_by_base`/
`model.wrap_state`, which all three backends implement with the same
names/shapes. The control variable in all three is a one-time LATENT-space
increment injected as a forcing at the first rollout step (see
`compute_loss_4dvar`'s docstring) -- never a physical-space perturbation --
so `compute_optimal` always returns a state dated `window_start +
dt_verif`, never `window_start` itself, for any backend.

Where the three backends genuinely differ, and how this file handles it:

  - **Packed-state tensor layout.** AIFS and Aurora both pack
    `(1, multi_step, n_points, n_vars)` (`n_vars` trailing --
    `model.state_layout == 'channels_last'`; Aurora's `multi_step` is 2
    lagged time levels, same convention as AIFS, just a different
    `n_points`/`n_vars` count). FCN3 packs `(1, n_vars, H, W)` (`n_vars` at
    dim 1 -- `model.state_layout == 'channels_first'`, since FCN3 has no
    history to pack, `n_history == 0` is hard-enforced by the checkpoint).
    Anywhere this file needs to broadcast a per-variable-column mask
    against the packed state directly (`control_variables`, via
    `_resolve_control_mask` / `_apply_control_mask`), it branches on
    `model.state_layout` rather than assuming any one packing.
  - **Grid regularity.** AIFS's grid is an irregular N320 octahedral mesh,
    interpolated via a k-d-tree/inverse-distance scheme
    (`aifs_grid.GridInterpolator`) with no alternative. FCN3's and Aurora's
    grids are both regular 0.25 deg equiangular lat/lon grids (FCN3:
    721x1440, including both poles; Aurora: 720x1440, `Batch.crop()` drops
    the South Pole row), so both additionally support an exact,
    ~50x-faster closed-form bilinear interpolator
    (`fcn3_grid.BilinearGridInterpolator`, the default for either, selected
    via the `grid_interp` config key -- Aurora deliberately reuses FCN3's
    grid module rather than duplicating it, see `get_grid_interpolator`).
  - **`ps_operator`.** `'logpinterp'` (pressure-level geopotential search,
    needing only 'z') works identically for all three and is the default.
    `'ps'` (native surface-pressure + t700/q700 station-pressure reduction)
    is currently AIFS-only in practice: FCN3 has no native surface-pressure
    ('sp') decode_state field at all, so `load_config` raises up front if
    `ps_operator: 'ps'` is set under `model_backend: 'fcn3'`. Aurora DOES
    have a native 'sp' field (unlike FCN3) and this path may work for it in
    principle, but it has only ever been exercised for AIFS -- `load_config`
    raises for `model_backend: 'aurora'` too, until it's actually been
    tested, rather than silently claim support that hasn't been verified.
    `'ps_native'` (ACE2 only) is `'ps'` with the lowest NATIVE model
    level's t/q/pressure in place of t/q interpolated to 700 hPa --
    `load_config` raises for any other backend.
  - **`reset_skt_ocean`.** Supported for AIFS and Aurora (both have
    `skt`/`lsm` decode_state fields -- Aurora's `lsm` was added to its
    packed state specifically to support this). FCN3 has no equivalent at
    all (no sea-surface-temperature-like prognostic channel, and its
    land-sea mask is a static preprocessor input, not something
    `resolve_columns` can find); `load_config` raises if this is set at all
    under `model_backend: 'fcn3'`.
  - **FCN3's stochastic noise.** FCN3 is a stochastic/ensemble model run
    deterministically by design: a fixed noise realization must be
    RE-PRIMED (reset) before every independent rollout a later comparison
    depends on being evaluated consistently -- most importantly, once per
    epoch inside `compute_optimal`'s AdamW loop (confirmed necessary, not
    just tidy, by the FCN3 model-backend smoke test: without it, every
    epoch sees a different, uncorrelated noise draw and AdamW chases a
    moving target instead of optimizing a well-defined loss surface).
    Neither AIFS nor Aurora has any stochastic input; both models'
    `_prime_noise()` is a no-op, so every `model._prime_noise()` call site
    in this file (three: before `ref_state1`, the top of each AdamW epoch,
    before the final analysis materialization) needs no backend
    conditional at all. **Not every forward-pass call site re-primes** --
    only ones that directly affect optimization well-definedness or the
    reported "best" state. Diagnostic/forecast call sites
    (`compute_ps_observation_hx`, `make_forecasts`, the driver's z500
    before/after check) do not, and (for FCN3) their exact run-to-run
    noise reproducibility has not been separately audited.
  - **Aurora's fp16-vs-bf16 autocast.** Aurora's `AuroraV1p5` runs its
    encoder/backbone/decoder under `torch.autocast`; its own default dtype
    (`float16`) was confirmed (2026-09-18) to produce a NaN gradient w.r.t.
    `latent_increment` when backpropagating through a 16+ step chained
    rollout, deterministically, from the all-zero initial increment --
    `backends/aurora/aurora_model.py` now defaults to `bfloat16` instead
    (see integrate_multiple_backends.md "Aurora 16-step divergence: root cause and fix"). Purely
    an internal Aurora backend concern, invisible to this file, EXCEPT that
    this file's own `torch.isfinite(loss)` check (before `.backward()`) is
    not sufficient to catch this failure mode by itself -- a finite loss
    does not guarantee a finite gradient, since a corrupted increment's
    downstream obs terms get correctly QC-rejected as non-finite rather
    than producing a NaN loss. `compute_optimal` also checks
    `torch.isfinite(increment.grad)` right after `.backward()`, for any
    backend, as a defense-in-depth backstop.

Observations may be sampled finer than the model's 6h rollout step via a
per-window `dt_obs` config key (hours, default `dt_verif`; must divide both
`dt_verif` and the fixed 6h timestep) -- entirely backend-agnostic (works
purely through `model.advance`/`model.decode_state`).
"""

import copy
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from functools import partial

import numpy as np
import torch
import xarray
import yaml

GRAV = 9.80665  # m/s^2
RD = 287.05  # J/(kg K), dry air gas constant
RV = 461.5  # J/(kg K), water vapor gas constant
FV = RV / RD - 1.0

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _ensure_backend_on_path(backend):
    """Add `backends/<backend>/` to sys.path so that module's flat
    `import forecast_model` / `import {aifs,fcn3,aurora}_grid` / etc.
    resolve -- idempotent, and never touches any OTHER backend's
    directory, so a process that only ever calls one backend's branch
    never needs the other backends' dependencies importable."""
    d = os.path.join(_REPO_ROOT, 'backends', backend)
    if d not in sys.path:
        sys.path.insert(0, d)


# decode_state produces both a base-name key and a NeuralGCM/AIFS-style alias
# for a few families (see {AIFS,FCN3,Aurora}Model.decode_state).
# `loss_variables` is given in base names; the forward operator reads the
# aliases -- so a time-interpolated field must be stored under both. 'sp' is
# AIFS/Aurora-only (FCN3 has no native surface-pressure channel) but
# harmless to list unconditionally here -- it's simply never populated in
# FCN3's decode_state output.
_DECODE_ALIASES = {
    "z": ["geopotential"],
    "t": ["temperature"],
    "q": ["specific_humidity"],
    "sp": ["surface_pressure"],
}

# decode_state keys each `ps_operator`'s H(x) reads. 'z' (geopotential) is
# ALWAYS added on top of these (see _resolve_loss_interp_specs) because the
# shared QC/setup in _compute_ps_observation_diagnostics_at_time reads it
# unconditionally (`.device`, and the `interpolation_failed` finiteness
# check on the obs-space geopotential) -- so 'ps', whose operator only needs
# t/q/sp, is not listed with 'z' here. 'ps' is only exercised for AIFS in
# practice -- FCN3 has no native surface-pressure channel at all, so
# load_config() rejects `model_backend: 'fcn3'` + `ps_operator: 'ps'` up
# front; Aurora DOES have a native 'sp' field but this path hasn't been
# tested for it yet, so load_config() rejects
# `model_backend: 'aurora'` + `ps_operator: 'ps'` too, for now.
# 'ps_native' (ACE2 only) reads the lowest NATIVE model level's t/q/pressure
# ('t_lowest'/'q_lowest'/'p_lowest', which only ACE2Model.decode_state
# supplies) instead of t/q on the 700 hPa pressure level.
# autocast_dtype config values (canonical names) and which each backend
# accepts; backends not listed run in fp32 only.
_AUTOCAST_NAMES = {'bfloat16': 'bfloat16', 'bf16': 'bfloat16', 'float16': 'float16', 'fp16': 'float16',
                   'float32': 'float32', 'fp32': 'float32'}
_AUTOCAST_ALLOWED = {'aifs': {'bfloat16', 'float16'}, 'aurora': {'bfloat16', 'float16'},
                     'sfno': {'bfloat16', 'float32'}}
_AUTOCAST_DEFAULT = {'aifs': 'bfloat16', 'aurora': 'bfloat16', 'sfno': 'float32'}


def _autocast_torch_dtype(exp, backend):
    """The configured (or backend-default) autocast dtype as a torch.dtype;
    None means no autocast (fp32)."""
    name = exp.get('autocast_dtype') or _AUTOCAST_DEFAULT[backend]
    return None if name == 'float32' else getattr(torch, name)


_PS_OPERATOR_REQUIRES = {
    "logpinterp": {"z"},
    "ps": {"t", "q", "sp"},
    "ps_native": {"t_lowest", "q_lowest", "p_lowest", "sp"},
}

PSOBS_QC_FLAG_MASKS = np.array([1, 2, 4, 8], dtype=np.int16)
PSOBS_QC_FLAG_MEANINGS = (
    'invalid_or_padded model_interpolation_failed '
    'orography_difference background_gross_check'
)

# raobs (radiosonde profile) obs-variable name -> model packed-state base
# name. AIFS and Aurora share this base-name convention ('t'/'u'/'v'/'q' --
# see this module's docstring); FCN3/ACE2 have not been checked and
# load_config() rejects raobs_enabled for them until they have been (see
# RAOBS_SUPPORTED_BACKENDS below). 'specific_humidity' (q) has no
# reference-repo precedent -- see raobs_parquet.py's module docstring.
RAOBS_VARIABLE_MODEL_BASE = {
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'specific_humidity': 'q',
}
RAOBS_SUPPORTED_BACKENDS = ('aifs', 'aurora')

# ATMS radiances through the radianced two-head decoder (obs_atms_operator.py,
# notes/integrate_atms_obs.md). The decoder was trained on the AIFS 1-degree
# checkpoint's variables, so AIFS only. ATMS_MODEL_FIELDS: decode_state keys
# the operator reads, interpolated to each ob; the keys of
# _ATMS_OPERATOR_FIELDS are the operator's own field names.
ATMS_SUPPORTED_BACKENDS = ('aifs',)
_ATMS_OPERATOR_FIELDS = {
    'sp': 'sp', 'msl': 'msl', 'skt': 'skt', '2t': '2t', '2d': '2d', '10u': '10u', '10v': '10v',
    'tcw': 'tcw', 'lsm': 'lsm', 'z': 'geopotential_at_surface', 'sdor': 'sdor', 'slor': 'slor',
}
ATMS_MODEL_FIELDS = ('t', 'q') + tuple(_ATMS_OPERATOR_FIELDS.values())
# obs-file columns the operator takes as inputs (Obs_Time is left at its
# default 0: the state is interpolated to the slot time)
_ATMS_OBS_INPUTS = ('land_frac', 'water_frac', 'ice_frac', 'snow_frac', 'Sat_Zenith_Angle',
                    'Scan_Angle', 'Sat_Azimuth_Angle', 'sat_id')
# Microwave sensors with a two-head decoder, each its own `observations.<sensor>`
# block (same keys as atms, flattened to `<sensor>_<key>`) and its own
# trajectory; default channel sets (1-based) from the radianced hand-offs.
# AMSU-A 4-8 are the frequencies of ATMS 5-9; AMSU-A 1, 2 are ATMS 1, 2
# (23.8 / 31.4 GHz), which the cloud retrieval `_atms_clw` uses.
MW_SENSORS = ('atms', 'amsua')
_MW_DEFAULT_CHANNELS = {
    'atms': {'assimilate': [5, 6, 7], 'monitor': [8, 9, 10, 11], 'emis': [1, 2, 3, 4, 16, 17]},
    'amsua': {'assimilate': [4, 5, 6, 7, 8], 'monitor': [9, 10], 'emis': [1, 2, 3, 15]},
}


# ---------------------------------------------------------------------------
# logging / dates / config
# ---------------------------------------------------------------------------

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.addFilter(lambda record: True)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def add_hours(date_str, nhours):
    from datetime import timedelta
    date_format = '%Y-%m-%dT%H'
    date_obj = datetime.strptime(date_str, date_format)
    new_date_obj = date_obj + timedelta(hours=nhours)
    return new_date_obj.strftime(date_format)


def get_YYYYMMDDHH(date_str, date_format_in='%Y-%m-%dT%H', date_format_out='%Y%m%d%H'):
    date_obj = datetime.strptime(date_str, date_format_in)
    return date_obj.strftime(date_format_out)


def _to_utc_datetime(date_str, backend, date_format='%Y-%m-%dT%H'):
    """Parse a config date string to a `datetime`. FCN3 REQUIRES a tz-aware
    UTC datetime (its zenith-angle forcing subtracts a tz-aware epoch
    internally -- see fcn3_model.py) -- a naive datetime only fails much
    later, inside the model's forward pass. AIFS/anemoi has never been
    shown to need (or to break on) tz-awareness either way; rather than
    assume a tz-aware datetime is a safe superset for AIFS's code path too
    (unverified -- anemoi's dynamic-forcings computation is real,
    non-trivial date arithmetic this wasn't audited against), this stays
    backend-conditional: naive for 'aifs' (matches its own, validated,
    original behavior exactly), tz-aware UTC for 'fcn3' (required)."""
    dt = datetime.strptime(date_str, date_format)
    if backend in ('fcn3', 'sfno'):  # makani's zenith-angle input needs tz-aware UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# Nested `exp.observations.psobs.<key>` / `exp.observations.raobs.<key>` ->
# flat internal `exp[<value>]` key it's resolved into (see
# `_resolve_observations_config` and config.yml.template). Grouping each
# observation type's path/QC/error settings together on disk -- `path`
# included, exactly like `report_types`/`error_std_hpa`/etc. -- mirrors the
# sibling JAX/NeuralGCM solver's config.yml.template (greg-long-window-4dvar/
# long-window-4dvar's `observations:` block, where every stream carries its
# own `path:` with no shared top-level fallback -- see integrate_prepbufr_obs.md
# "How the reference repo does this") -- but this repo has no fixed-shape/
# capacity-manifest machinery to canonicalize streams through (PyTorch, not
# JAX), so there's nothing to gain from threading a second, nested
# representation through the rest of the solver too: this mapping is applied
# once, at load time, and everything downstream still reads the flat keys it
# always has.
_PSOBS_CONFIG_KEYS = {
    'path': 'obspath',
    'nobs_max': 'nobs_max',
    'prepbufr_classes': 'psobs_prepbufr_classes',
    'report_types': 'psobs_report_types',
    'quality_mark_max': 'psobs_quality_mark_max',
    'time_tolerance_hours': 'psobs_time_tolerance_hours',
    'use_source_error': 'psobs_use_source_error',
    'error_std_hpa': 'surface_pressure_error_std_hpa',
    'error_scale': 'psobs_error_scale',
}
_RAOBS_CONFIG_KEYS = {
    'path': 'raobs_path',
    'variables': 'raobs_variables',
    'prepbufr_classes': 'raobs_prepbufr_classes',
    'time_tolerance_hours': 'raobs_time_tolerance_hours',
    'quality_mark_max': 'raobs_quality_mark_max',
    'pressure_match_tolerance_hpa': 'raobs_pressure_match_tolerance_hpa',
    'height_min_m': 'raobs_height_min_m',
    'height_max_m': 'raobs_height_max_m',
    'below_station_tolerance_m': 'raobs_below_station_tolerance_m',
    'error_std': 'raobs_error_std',
    'q_error_fraction': 'raobs_q_error_fraction',
    'q_error_floor_kg_kg': 'raobs_q_error_floor_kg_kg',
    'q_error_g_kg': 'raobs_q_error_g_kg',
    'q_bg_check': 'raobs_q_bg_check',
    'humidity_min_pressure_hpa': 'raobs_humidity_min_pressure_hpa',
    'station_capacity': 'raobs_station_capacity',
}
# observations.<sensor> keys for every MW_SENSORS sensor, flattened to <sensor>_<key>
_MW_CONFIG_KEYS = (
    'path', 'bundle', 'assimilate_channels', 'monitor_channels', 'sigma_o_scale', 'sigma_o', 'bg_check',
    'huber_c', 'water_frac_min', 'clw_max', 'lsm_max', 'detach_q_above_hpa', 'time_tolerance_hours',
    'capacity', 'scenes', 'emis_control', 'emis_channels', 'z_learn_rate', 'oberr_model', 'oberr_table',
    'cloud_max', 'window_state_scenes', 'corr_r', 'corr_cond_max',
)


def _resolve_observations_config(exp):
    """Parse `exp['observations']` (`psobs:`/`raobs:` blocks) into the flat
    `obspath`/`nobs_max`/`psobs_*`/`raobs_*`/`surface_pressure_error_std_hpa`
    keys the rest of this module reads -- see the module-level key-mapping
    tables above and config.yml.template.

    `observations.psobs.path` is required (there is no top-level `obspath`
    fallback -- every stream names its own archive path, same as the
    reference repo's `observations:` streams). `observations.raobs.path` is
    optional and defaults to `psobs`'s resolved path, since both normally
    read the same parquet archive (just different PREPBUFR classes) -- set
    it explicitly only if raobs ever needs to diverge from psobs.

    Both blocks are optional in the sense that an absent/empty `raobs` block
    (or `enabled: False`) leaves `raobs_enabled` False, same as before this
    config was nested.
    """
    observations = exp.get('observations', {}) or {}
    unknown = set(observations) - {'psobs', 'raobs'} - set(MW_SENSORS)
    if unknown:
        raise ValueError(f"exp.observations has unknown keys {sorted(unknown)} "
                         f"(expected 'psobs', 'raobs', {', '.join(repr(m) for m in MW_SENSORS)})")

    psobs = observations.get('psobs', {}) or {}
    unknown_psobs = set(psobs) - set(_PSOBS_CONFIG_KEYS)
    if unknown_psobs:
        raise ValueError(f"exp.observations.psobs has unknown keys {sorted(unknown_psobs)}")
    if 'path' not in psobs:
        raise ValueError("exp.observations.psobs.path is required (the parquet archive root)")
    for src_key, dst_key in _PSOBS_CONFIG_KEYS.items():
        if src_key in psobs:
            exp[dst_key] = psobs[src_key]

    raobs = observations.get('raobs', {}) or {}
    unknown_raobs = set(raobs) - set(_RAOBS_CONFIG_KEYS) - {'enabled'}
    if unknown_raobs:
        raise ValueError(f"exp.observations.raobs has unknown keys {sorted(unknown_raobs)}")
    exp['raobs_enabled'] = bool(raobs.get('enabled', False))
    if exp['raobs_enabled']:
        for src_key, dst_key in _RAOBS_CONFIG_KEYS.items():
            if src_key in raobs:
                exp[dst_key] = raobs[src_key]
        exp.setdefault('raobs_variables', ['temperature', 'u_component_of_wind', 'v_component_of_wind', 'specific_humidity'])
        exp.setdefault('raobs_path', exp['obspath'])

    for sensor in MW_SENSORS:
        block = observations.get(sensor, {}) or {}
        unknown_mw = set(block) - set(_MW_CONFIG_KEYS) - {'enabled'}
        if unknown_mw:
            raise ValueError(f"exp.observations.{sensor} has unknown keys {sorted(unknown_mw)}")
        exp[f'{sensor}_enabled'] = bool(block.get('enabled', False))
        if exp[f'{sensor}_enabled']:
            for key in ('path', 'bundle'):
                if key not in block:
                    raise ValueError(f"exp.observations.{sensor}.{key} is required when {sensor} is enabled")
            for key in _MW_CONFIG_KEYS:
                if key in block:
                    exp[f'{sensor}_{key}'] = block[key]


def load_config(config_path='config.yml'):
    """Loads configuration from a YAML file and returns exp and windows dictionaries."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None, None
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None, None

    exp = config.get('exp', {})
    windows = config.get('windows', {})
    exp['path_output'] = exp['path_output'] + exp['exp_name'] + '/'
    _resolve_observations_config(exp)

    backend = exp.get('model_backend')
    if backend not in ('aifs', 'fcn3', 'aurora', 'ace2', 'sfno'):
        raise ValueError(
            f"exp.model_backend must be 'aifs', 'fcn3', 'aurora', 'ace2', or 'sfno', got {backend!r} -- "
            "this key is required (see config.yml.template)"
        )
    # reset_skt_ocean: AIFS and Aurora both have 'skt'/'lsm' packed-state
    # columns (Aurora's 'lsm' added specifically to support this -- see
    # aurora_model.py), so reset_skt_over_ocean works unmodified for either.
    # FCN3 has no equivalent at all -- fail loudly here rather than
    # silently ignore a config key that looks like it should do something.
    if backend in ('fcn3', 'sfno') and exp.get('reset_skt_ocean', False):
        raise ValueError(
            "reset_skt_ocean is not supported for the FCN3 backend -- FCN3 has no "
            "'skt'/'lsm' decode_state field to reset (see integrate_multiple_backends.md)."
        )
    if backend == 'fcn3' and exp.get('ps_operator', 'logpinterp') == 'ps':
        raise ValueError(
            "ps_operator='ps' is not supported for the FCN3 backend -- FCN3 has no "
            "native surface-pressure ('sp') decode_state field; use the default "
            "'logpinterp' instead (see integrate_multiple_backends.md)."
        )
    # Aurora DOES have a native 'sp' field (unlike FCN3), so 'ps' might work
    # here in principle -- but _compute_ps_observation_diagnostics_at_time's
    # 'ps' branch has only ever been exercised for AIFS. Block it for Aurora
    # too until it's actually been tested, rather than silently claim
    # support that hasn't been verified.
    if backend == 'aurora' and exp.get('ps_operator', 'logpinterp') == 'ps':
        raise ValueError(
            "ps_operator='ps' is not yet supported for the Aurora backend -- Aurora "
            "has a native 'sp' field so this may work in principle, but the 'ps' "
            "forward-operator path has only been tested for AIFS so far. Use the "
            "default 'logpinterp' instead (see integrate_multiple_backends.md)."
        )
    # ACE2: 'ps_native' is the default operator. Its layers are hybrid
    # sigma-pressure, so 'logpinterp' would have to search a hydrostatically
    # DERIVED geopotential that is ~19 m low vs ERA5 (~2 hPa of ps) -- still
    # allowed if set explicitly, for comparison.
    if backend == 'ace2':
        exp.setdefault('ps_operator', 'ps_native')
        if exp.get('reset_skt_ocean', False):
            raise ValueError(
                "reset_skt_ocean is not supported for the ACE2 backend -- ACE2's own ocean "
                "scheme already prescribes surface_temperature over ocean from the forcing "
                "file's SST at every step."
            )
        if exp['ps_operator'] == 'ps':
            raise ValueError("ps_operator='ps' is not supported for ACE2 -- use 'ps_native' (the default)")
    # jc_ps_sigma (the original, un-normalized penalty) was removed -- fail
    # loudly rather than silently run without the penalty.
    if 'jc_ps_sigma' in exp:
        raise ValueError(
            "jc_ps_sigma has been removed -- use jc_ps_weight (kappa, hPa^-2): "
            "Jc = kappa * N_obs * area-weighted mean of Delta^2. The equivalent of "
            "jc_ps_sigma = s is kappa = n_points / (s^2 * N_obs) -- e.g. s = 0.1 on "
            "ACE2 (64800 points) with a 5-day window (~2.1e5 obs) is kappa ~ 30."
        )
    if exp.get('jc_ps_weight') and float(exp['jc_ps_weight']) <= 0:
        raise ValueError(f"jc_ps_weight must be > 0 (hPa^-2), got {exp['jc_ps_weight']!r}")
    # More than one DA cycle requires cycle: True. With cycle: False the driver
    # never advances the previous analysis (valid at window_start + dt_verif)
    # to the next cycle time -- it falls through to the next-window-in-the-
    # same-cycle branch and reuses it as is. That is only accidentally right
    # when dt_init == dt_verif; for dt_init = 12 every cycle fell a further 6 h
    # behind its label (found 2026-10-01: a 240-cycle SFNO run verified Jan 25
    # cycles against a Jan 13 state). Fail loudly instead.
    if int(exp.get('n_init', 1)) > 1 and not exp.get('cycle', False):
        raise ValueError(
            f"n_init={exp.get('n_init')} needs cycle: True -- with cycle: False the previous "
            "analysis is not advanced to the next cycle time (only correct by accident when "
            "dt_init == dt_verif). Set cycle: True for cycling DA, or n_init: 1."
        )
    # autocast_dtype: one key for every backend that supports mixed precision.
    # aifs_autocast_dtype is a deprecated alias (existing aifs1 configs use it).
    if 'aifs_autocast_dtype' in exp:
        old = exp.pop('aifs_autocast_dtype')
        if exp.get('autocast_dtype', old) != old:
            raise ValueError(f"aifs_autocast_dtype={old!r} conflicts with autocast_dtype={exp['autocast_dtype']!r}")
        exp['autocast_dtype'] = old
        print("NOTE: 'aifs_autocast_dtype' is deprecated -- use 'autocast_dtype' (same values)")
    if exp.get('autocast_dtype') is not None:
        name = _AUTOCAST_NAMES.get(str(exp['autocast_dtype']).lower())
        allowed = _AUTOCAST_ALLOWED.get(backend)
        if allowed is None:
            raise ValueError(f"autocast_dtype is not implemented for model_backend {backend!r} (it runs in fp32)")
        if name not in allowed:
            raise ValueError(f"autocast_dtype for {backend!r} must be one of {sorted(allowed)}, got {exp['autocast_dtype']!r}")
        exp['autocast_dtype'] = name
    # ps_native needs decode_state's lowest-native-level keys
    # (t_lowest/q_lowest/p_lowest), which only the ACE2 backend provides.
    if exp.get('ps_operator', 'logpinterp') == 'ps_native' and backend != 'ace2':
        raise ValueError(
            f"ps_operator='ps_native' is only supported for model_backend 'ace2', got {backend!r} "
            "(it needs decode_state's lowest-native-level t_lowest/q_lowest/p_lowest fields)."
        )
    # raobs (radiosonde profiles): off by default -- surface-pressure-only
    # configs are untouched by any of this (see integrate_prepbufr_obs.md).
    # `_resolve_observations_config` (above) already applied the
    # `raobs_variables` default and the `raobs_path` fallback to `obspath`;
    # this just enforces the backend/variable support guard.
    for sensor in MW_SENSORS:
        if exp.get(f'{sensor}_enabled', False) and backend not in ATMS_SUPPORTED_BACKENDS:
            raise ValueError(f"observations.{sensor} is supported for model_backend in {ATMS_SUPPORTED_BACKENDS}, "
                             f"got {backend!r} -- the decoder was trained on the AIFS 1-degree variables")
    if exp.get('raobs_enabled', False):
        if backend not in RAOBS_SUPPORTED_BACKENDS:
            raise ValueError(
                f"observations.raobs.enabled is only supported for model_backend in "
                f"{RAOBS_SUPPORTED_BACKENDS}, got {backend!r} -- the 't'/'u'/'v'/'q' base-name "
                "forward operator has only been exercised for AIFS/Aurora so far "
                "(see RAOBS_VARIABLE_MODEL_BASE)."
            )
        variables = exp['raobs_variables']
        unsupported = set(variables) - set(RAOBS_VARIABLE_MODEL_BASE)
        if unsupported:
            raise ValueError(
                f"observations.raobs.variables entries must be a subset of "
                f"{sorted(RAOBS_VARIABLE_MODEL_BASE)}, got unsupported entries {sorted(unsupported)}"
            )
        if not variables:
            raise ValueError("observations.raobs.variables must be non-empty when observations.raobs.enabled is True")
        exp['raobs_variables'] = list(variables)
    return exp, windows


def get_window(exp, window, windows):
    dt_verif = windows[window]['dt_verif']
    exp['dt_verif'] = dt_verif
    exp['window'] = window
    if 'bg_check' in windows[window]: exp['bg_check'] = windows[window]['bg_check']
    if 'zthresh' in windows[window]: exp['zthresh'] = windows[window]['zthresh']
    if 'zconst' in windows[window]: exp['zconst'] = windows[window]['zconst']
    exp['n_verif'] = windows[window]['n_verif']
    # dt_obs: observation sampling interval in hours. Defaults to dt_verif
    # (one obs time per 6h model step). Must divide both dt_verif and the
    # model's 6h timestep (all three backends' checkpoints use the same 6h
    # step) so an obs is always bracketed by two adjacent 6h model states.
    dt_obs = windows[window].get('dt_obs', dt_verif)
    if dt_verif % dt_obs != 0 or 6 % dt_obs != 0:
        raise ValueError(
            f"dt_obs ({dt_obs}) must divide both dt_verif ({dt_verif}) and 6"
        )
    exp['dt_obs'] = dt_obs
    exp['learn_rate'] = windows[window]['learn_rate']
    exp['weight_decay'] = windows[window]['weight_decay']
    exp['max_epoch'] = windows[window]['max_epoch']
    exp['warmup_steps'] = windows[window]['warmup_steps']
    exp['start_factor'] = windows[window]['start_factor']
    exp['end_factor'] = windows[window]['end_factor']
    if 'beta1' in windows[window]: exp['beta1'] = windows[window]['beta1']
    if 'beta2' in windows[window]: exp['beta2'] = windows[window]['beta2']
    exp['window_name'] = str(exp['n_verif'] * dt_verif)
    exp['suffix'] = windows[window]['suffix']
    return exp


def log_window(exp, logger):
    logger.info('   model_backend: ' + str(exp['model_backend']))
    logger.info('   window: ' + str(exp['window']))
    logger.info('   n_verif: ' + str(exp['n_verif']))
    logger.info('   dt_verif: ' + str(exp['dt_verif']))
    logger.info('   dt_obs: ' + str(exp['dt_obs']))
    logger.info('   loss_variables: ' + str(exp.get('loss_variables', '(default from ps_operator)')))
    logger.info('   control_variables: ' + str(exp.get('control_variables', '(all -- increment unrestricted)')))
    logger.info('   latent_scale: ' + str(exp.get('latent_scale', 1.0)))
    logger.info('   checkpoint_stride: ' + str(exp.get('checkpoint_stride', 1)) + ' (1 = checkpoint every step)')
    logger.info('   jc_ps_weight: ' + (str(exp['jc_ps_weight']) + ' hPa^-2 (x N_obs)'
                                        if exp.get('jc_ps_weight') else '(off)'))
    logger.info('   raobs_enabled: ' + (str(exp['raobs_variables']) if exp.get('raobs_enabled', False) else '(off, psobs only)'))
    for sensor in MW_SENSORS:
        if exp.get(f'{sensor}_enabled', False):
            ch = _MW_DEFAULT_CHANNELS[sensor]
            logger.info(f"   {sensor}: assimilate ch {exp.get(f'{sensor}_assimilate_channels', ch['assimilate'])}, "
                        f"monitor ch {exp.get(f'{sensor}_monitor_channels', ch['monitor'])}, "
                        f"sigma_o {exp.get(f'{sensor}_sigma_o') or 'bundle'} x {exp.get(f'{sensor}_sigma_o_scale', 1.0)}, "
                        f"huber_c {exp.get(f'{sensor}_huber_c') or '(off)'}, bundle {exp[f'{sensor}_bundle']}")
        else:
            logger.info(f'   {sensor}: (off)')
    if exp['model_backend'] == 'fcn3':
        logger.info('   atmo_chunk_size: ' + str(exp.get('atmo_chunk_size', 2)))
        logger.info('   grid_interp: ' + str(exp.get('grid_interp', 'bilinear')))
    if 'bg_check' in exp: logger.info('   bg_check: ' + str(exp['bg_check']))
    if 'zthresh' in exp: logger.info('   zthresh: ' + str(exp['zthresh']))
    if 'zconst' in exp: logger.info('   zconst: ' + str(exp['zconst']))
    logger.info('   learn_rate: ' + str(exp['learn_rate']))
    logger.info('   weight_decay: ' + str(exp['weight_decay']))
    if 'beta1' in exp: logger.info('   beta1: ' + str(exp['beta1']))
    if 'beta2' in exp: logger.info('   beta2: ' + str(exp['beta2']))
    logger.info('   max_epoch: ' + str(exp['max_epoch']))
    logger.info('   start_factor: ' + str(exp['start_factor']))
    logger.info('   end_factor: ' + str(exp['end_factor']))
    logger.info('   warmup_steps: ' + str(exp['warmup_steps']))
    logger.info('   window_name: ' + str(exp['window_name']))
    logger.info('   suffix: ' + str(exp['suffix']))
    return


# ---------------------------------------------------------------------------
# model / grid
# ---------------------------------------------------------------------------

def get_model(exp):
    backend = exp['model_backend']
    if backend == 'aifs':
        _ensure_backend_on_path('aifs')
        import aifs_model
        return aifs_model.AIFSModel(
            checkpoint_path=exp['path_model'] + exp['model_name'],
            config_path=exp.get('aifs_config', 'aifs_inference.yaml'),
            device=exp.get('device', 'cuda'),
            autocast_dtype=_autocast_torch_dtype(exp, 'aifs'),
            compile_wrapper=exp.get('compile_wrapper', False),
            attention=exp.get('aifs_attention'),
        )
    elif backend == 'fcn3':
        _ensure_backend_on_path('fcn3')
        import fcn3_model
        return fcn3_model.FCN3Model(
            package_root=exp['path_model'],
            device=exp.get('device', 'cuda'),
            atmo_chunk_size=exp.get('atmo_chunk_size', 2),
        )
    elif backend == 'sfno':
        # NVIDIA SFNO-73ch-small (NGC), run with makani in the fcstnet3 env;
        # path_model = the converted package dir (backends/sfno/ngc_cache/
        # sfno_73ch_small by default, see sfno_prefetch_checkpoint.py).
        _ensure_backend_on_path('sfno')
        import sfno_model
        return sfno_model.SFNOModel(
            package_root=exp.get('path_model', sfno_model.DEFAULT_PACKAGE),
            device=exp.get('device', 'cuda'),
            latent_block=exp.get('sfno_latent_block', -2),
            autocast_dtype=_autocast_torch_dtype(exp, 'sfno'),
        )
    elif backend == 'aurora':
        _ensure_backend_on_path('aurora')
        import aurora_model
        return aurora_model.AuroraModel(
            package_root=exp['path_model'],
            device=exp.get('device', 'cuda'),
            autocast_dtype=_autocast_torch_dtype(exp, 'aurora'),
            compile_wrapper=exp.get('compile_wrapper', False),
        )
    elif backend == 'ace2':
        _ensure_backend_on_path('ace2')
        import ace2_model
        return ace2_model.ACE2Model(
            checkpoint_path=exp['path_model'] + exp.get('model_name', 'ace2_era5_ckpt.tar'),
            forcing_dir=exp.get('forcing_dir', exp['path_model'] + 'forcing_data/'),
            device=exp.get('device', 'cuda'),
        )
    raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', 'ace2', or 'sfno')")


def get_grid_interpolator(model, exp):
    """AIFS's irregular N320 octahedral grid always uses the k-d-tree/k-NN
    approach (no alternative exists). FCN3's and Aurora's regular grids both
    default to the exact, ~50x-faster BilinearGridInterpolator -- see
    fcn3_grid.py's module docstring and verify_bilinear_grid_interp.py (in
    the original long-window-4dvar-fcstnetv3 repo) for the
    ported-from-NeuralGCM math and the correctness/speed checks.
    `grid_interp: 'kdtree'` in the config falls back to the k-d-tree
    implementation for either regular-grid backend too (kept as an A/B
    check; there's no reason to use it for a regular grid)."""
    backend = exp['model_backend']
    if backend == 'aifs':
        _ensure_backend_on_path('aifs')
        import aifs_grid
        return aifs_grid.GridInterpolator(model.lons, model.lats)
    elif backend in ('fcn3', 'aurora', 'sfno'):
        # Aurora shares FCN3's regular-grid machinery directly (same 0.25deg
        # ERA5 grid, just 720 rather than 721 rows -- fcn3_grid.py's classes
        # derive nlat/nlon from the data itself). Deliberately imports from
        # backends/fcn3/ even when backend=='aurora' -- the one intentional
        # exception to "each backend's imports stay in its own directory":
        # fcn3_grid.py's own dependencies (numpy/torch/scipy) are already
        # present in both envs, so this carries none of the cross-backend
        # dependency risk _ensure_backend_on_path exists to avoid.
        _ensure_backend_on_path('fcn3')
        import fcn3_grid
        kind = exp.get('grid_interp', 'bilinear')
        if kind == 'bilinear':
            return fcn3_grid.BilinearGridInterpolator(model.lons, model.lats)
        if kind == 'kdtree':
            return fcn3_grid.GridInterpolator(model.lons, model.lats)
        raise ValueError(f"unknown grid_interp kind: {kind!r} (expected 'bilinear' or 'kdtree')")
    elif backend == 'ace2':
        # F90 Gaussian grid: unevenly spaced latitudes, so fcn3_grid's
        # uniform-latitude BilinearGridInterpolator doesn't apply --
        # ace2_grid's rectilinear version is the same scheme on the real
        # latitudes. 'kdtree' falls back to fcn3_grid.GridInterpolator.
        kind = exp.get('grid_interp', 'bilinear')
        if kind == 'bilinear':
            _ensure_backend_on_path('ace2')
            import ace2_grid
            return ace2_grid.RectilinearBilinearInterpolator(model.lons, model.lats)
        if kind == 'kdtree':
            _ensure_backend_on_path('fcn3')
            import fcn3_grid
            return fcn3_grid.GridInterpolator(model.lons, model.lats)
        raise ValueError(f"unknown grid_interp kind: {kind!r} (expected 'bilinear' or 'kdtree')")
    raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', 'ace2', or 'sfno')")


# ---------------------------------------------------------------------------
# initial conditions / verification
# ---------------------------------------------------------------------------

def get_input(exp, model, logger):
    """Get the initial model state.

    AIFS: two lagged time levels packed into one tensor, with forcings
    living inside the packed state itself (refreshed automatically by
    `model.advance()`). FCN3: a single physical-space time level
    (`n_history == 0` is hard-enforced by the checkpoint).
    """
    backend = exp['model_backend']
    date = exp['date']
    cache_dir = exp.get('ic_cache', exp['path_output'] + 'ic_cache/')

    if exp['restart_window']:
        ic_file = exp['rpath'] + date + '_optimal_inputs_' + exp['ic_suffix'] + '.pkl'
        logger.info('restarting from this initial condition file: ' + ic_file)
        with open(ic_file, 'rb') as handle:
            input_encoded = pickle.load(handle)
        # A pickled state carries only the packed tensor + date; per-process
        # bookkeeping model.advance() depends on for determinism (AIFS's
        # anemoi Runner input-tensor cache; FCN3's stochastic-noise state)
        # is not part of the pickle -- rebuild it.
        input_encoded = model.prime_from_state(input_encoded)
        exp['restart_window'] = False
    else:
        back_date = add_hours(date, -exp['nhrs_back'])
        logger.info('perfect model experiment: making initial condition from ERA5: ' + back_date)
        back_dt = _to_utc_datetime(back_date, backend)
        if backend == 'aifs':
            _ensure_backend_on_path('aifs')
            import aifs_ic
            input_state = aifs_ic.build_input_state(model.runner, back_dt, cache_dir, lagged=True)
        elif backend == 'fcn3':
            _ensure_backend_on_path('fcn3')
            import fcn3_ic
            input_state = fcn3_ic.build_input_state(back_dt, cache_dir)
        elif backend == 'sfno':
            # FCN3's cached ERA5 fields + a small per-date 'sp' file (sfno_ic.py)
            _ensure_backend_on_path('sfno')
            import sfno_ic
            input_state = sfno_ic.build_input_state(back_dt, cache_dir)
        elif backend == 'aurora':
            _ensure_backend_on_path('aurora')
            import aurora_ic
            input_state = aurora_ic.build_input_state(back_dt, cache_dir)
        elif backend == 'ace2':
            # ICs are prefetched from a LOGIN node (ace2ic env):
            # `python backends/ace2/ace2_ic.py YYYY-MM-DDTHH` -- this only
            # reads the cached netCDF.
            _ensure_backend_on_path('ace2')
            import ace2_ic
            input_state = ace2_ic.build_input_state(back_dt, cache_dir)
        else:
            raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', 'ace2', or 'sfno')")
        input_encoded = model.prepare_initial_state(input_state, back_dt)
        dt_hours = exp['nhrs_back']
        logger.info('generating input state from a forecast...')
        input_encoded = model.advance(input_encoded, steps=int(round(dt_hours / (model.timestep.total_seconds() / 3600))))

    return input_encoded, exp


def z500_field(decoded, nlev500):
    """500 hPa geopotential (m^2/s^2, (n_points,)) from a decode_state dict or
    a get_verif dict: a backend-supplied 'z500' if present (ACE2 -- its own
    h500 output / ERA5 z500, since its pressure-level 'geopotential' is only
    a hydrostatic derivation), else the pressure-level 'geopotential' family
    at index nlev500 (AIFS/FCN3/Aurora)."""
    if 'z500' in decoded:
        return decoded['z500']
    return decoded['geopotential'][nlev500, :]


def get_verif(exp, model, logger, date_override=None):
    """Get the ERA5 verification state at the window's start date (used for
    the ps-obs forward operator's orography and the driver's z500 diagnostic).

    `date_override` (a '%Y-%m-%dT%H' string) fetches truth at a date other than
    `exp['date']`. Used for the z500 before/after
    diagnostic, whose analysis is valid at `window_start + dt_verif`, not
    `window_start` -- the driver scores it against ERA5 truth fetched there.
    """
    backend = exp['model_backend']
    date = date_override if date_override is not None else exp['date']
    cache_dir = exp.get('ic_cache', exp['path_output'] + 'ic_cache/')
    logger.info('reading verification (ERA5) data for ' + date)
    date_dt = _to_utc_datetime(date, backend)

    verif_ic = {}
    if backend == 'aifs':
        _ensure_backend_on_path('aifs')
        import aifs_ic
        raw_fields = aifs_ic.read_single_date_fields(model.runner, date_dt, cache_dir)
        # Single-date snapshot: pack directly rather than through
        # model.prepare_initial_state() (which expects a full
        # multi_step-length state), mirroring AIFSModel.decode_state()'s
        # per-level stacking and base-name aliasing.
        raw = {
            name: torch.as_tensor(np.asarray(field), dtype=torch.float32, device=model.device)
            for name, field in raw_fields.items()
        }
        for base, levels in model._levels_by_base.items():
            names = [f"{base}_{lev}" for lev, _ in levels]
            if all(n in raw for n in names):
                verif_ic[base] = torch.stack([raw[n] for n in names], dim=0)
        for name in model._single_by_base:
            # AIFS's checkpoint has a real orography INPUT column read under
            # the raw 'z' (surface geopotential) name -- resolve it the same
            # way any other model input is resolved.
            src = "z" if name == "geopotential_at_surface" else name
            if src in raw:
                verif_ic[name] = raw[src]
    elif backend in ('fcn3', 'sfno'):
        # same 721x1440 grid and channel naming; SFNO's reader adds native 'sp'
        if backend == 'sfno':
            _ensure_backend_on_path('sfno')
            import sfno_ic as ic_module
        else:
            _ensure_backend_on_path('fcn3')
            import fcn3_ic as ic_module
        raw_fields = ic_module.read_single_date_fields(date_dt, cache_dir)
        # decode_state's flat n_points convention (see FCN3Model.decode_state)
        # is a row-major flatten of the native (721, 1440) grid -- flatten
        # here too so verif_ic's tensors line up with decoded model fields.
        raw = {
            name: torch.as_tensor(np.asarray(field), dtype=torch.float32, device=model.device).reshape(-1)
            for name, field in raw_fields.items()
        }
        for base, levels in model._levels_by_base.items():
            names = [f"{base}{lev}" for lev, _ in levels]
            if all(n in raw for n in names):
                verif_ic[base] = torch.stack([raw[n] for n in names], dim=0)
        for name in model._single_by_base:
            if name in raw:
                verif_ic[name] = raw[name]
        # FCN3 has no orography INPUT channel (unlike AIFS) -- fcn3_ic.py
        # fetches ERA5's own surface geopotential directly as an extra
        # field specifically to give the ps-obs QC code a
        # 'geopotential_at_surface' to compare station elevations against.
        verif_ic['geopotential_at_surface'] = raw['geopotential_at_surface']
    elif backend == 'aurora':
        _ensure_backend_on_path('aurora')
        import aurora_ic
        raw_fields = aurora_ic.read_single_date_fields(date_dt, cache_dir)
        # aurora_ic.py already returns atmos fields pre-stacked per family
        # (13, 720, 1440), in the same descending-level order as
        # model._levels_by_base -- no per-level key assembly needed here,
        # unlike AIFS/FCN3's flat per-level keys.
        for base in model._levels_by_base:
            if base in raw_fields:
                arr = torch.as_tensor(raw_fields[base], dtype=torch.float32, device=model.device)
                verif_ic[base] = arr.reshape(arr.shape[0], -1)  # (13, n_points)
        for name in model._single_by_base:
            if name in raw_fields:
                verif_ic[name] = torch.as_tensor(
                    raw_fields[name], dtype=torch.float32, device=model.device
                ).reshape(-1)
        # Like FCN3 (and unlike AIFS), Aurora's own checkpoint-bundled
        # orography is static, not a per-date fetch -- aurora_ic.py fetches
        # ERA5's own surface geopotential fresh, as an extra field, rather
        # than relying on the unverified assumption that Aurora's static
        # 'z' equals real ERA5 orography (see aurora_ic.py's docstring).
        verif_ic['geopotential_at_surface'] = torch.as_tensor(
            raw_fields['geopotential_at_surface'], dtype=torch.float32, device=model.device
        ).reshape(-1)
    elif backend == 'ace2':
        # Orography for the ps-obs QC/station reduction: ACE2's OWN static
        # HGTsfc (what its PRESsfc is consistent with), not a separate ERA5
        # fetch. Truth for the z500 diagnostic: ERA5 z500/t850 conservatively
        # regridded to F90 (prefetched from a login node:
        # `python backends/ace2/ace2_ic.py --verif YYYY-MM-DDTHH`), stored
        # under 'z500' so the driver compares it with ACE2's own h500.
        _ensure_backend_on_path('ace2')
        import ace2_ic
        verif_ic['geopotential_at_surface'] = model.surface_geopotential
        truth = ace2_ic.load_verif(date_dt, cache_dir)
        verif_ic['z500'] = GRAV * torch.as_tensor(truth['h500'], dtype=torch.float32, device=model.device).reshape(-1)
        verif_ic['TMP850'] = torch.as_tensor(truth['TMP850'], dtype=torch.float32, device=model.device).reshape(-1)
    else:
        raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', 'ace2', or 'sfno')")

    if "z" in verif_ic:
        verif_ic["geopotential"] = verif_ic["z"]
    if "t" in verif_ic:
        verif_ic["temperature"] = verif_ic["t"]
    if "q" in verif_ic:
        verif_ic["specific_humidity"] = verif_ic["q"]
    if "sp" in verif_ic:
        verif_ic["surface_pressure"] = verif_ic["sp"]
    return verif_ic


def reset_skt_over_ocean(model, model_state, verif_ic, lsm_threshold=0.5):
    """AIFS and Aurora only -- never called for FCN3 (load_config rejects
    `reset_skt_ocean: True` under `model_backend: 'fcn3'` up front, since
    FCN3 has no `skt`/`lsm` decode_state field at all).

    Overwrite `model_state`'s `skt` (skin temperature) at ocean grid points
    (`lsm < lsm_threshold`) with fresh ERA5 truth from `verif_ic`, at both
    lagged time levels, instead of letting the model's own carried-forward/
    predicted skt persist there across DA cycles.

    This checkpoint has no separate SST or sea-ice variable -- `skt` is its
    only surface-temperature-like prognostic field, covering land, ocean, and
    ice alike (checked directly against the checkpoint metadata: `sst` does
    not appear anywhere in it, including training provenance). ECMWF's own
    operational/ERA5 `skt` field is already defined as the SST analysis over
    open ocean (land-sea-mask-blended upstream of this whole pipeline, not by
    any code here), so resetting `skt` at ocean points from ERA5 is the
    closest available proxy for "reset SST from ERA5 every analysis time" --
    see the long-window-4dvar-aifsv2 repo's CLAUDE.md. No separate sea-ice handling: there's no sea-ice variable
    in this checkpoint at all (`lsm` is a static land-sea mask, constant in
    time, with no time-varying ice-extent field to reset).

    `lsm` is itself part of the packed state (constant across both time
    levels and across the whole rollout), so no extra fetch is needed for
    the mask -- only `skt`'s replacement values come from `verif_ic`.

    Indexes the `(1, multi_step, n_points, n_vars)` ('channels_last')
    packing directly -- this function is only ever reachable for AIFS or
    Aurora (both use this same layout, just with different `n_points`/
    `n_vars` counts; see this module's docstring), never FCN3, so it
    doesn't need to branch on `model.state_layout` the way backend-agnostic
    code in this file does.
    """
    skt_idx = model.resolve_columns("skt")[0]
    lsm_idx = model.resolve_columns("lsm")[0]
    state = model_state.state.clone()
    ocean_mask = state[0, 0, :, lsm_idx] < lsm_threshold  # constant across time levels
    skt_era5 = verif_ic["skt"].to(device=state.device, dtype=state.dtype)
    skt_col = state[0, :, :, skt_idx]  # (multi_step, n_points)
    state[0, :, :, skt_idx] = torch.where(ocean_mask, skt_era5, skt_col)
    return model.wrap_state(state, model_state.date)


def get_psobs(exp, model, logger, grid_interp):
    """Read the ps observations spanning the window from the normalized
    PREPBUFR parquet archive (`exp['obspath']` points at the archive root,
    e.g. `.../prepbufr_to_parquet/data/prepbufr_parquet/`; see
    `psobs_parquet.py` and `integrate_prepbufr_obs.md`).

    Entirely backend-agnostic (only uses `model.device`/`model.timestep`/
    `grid_interp.weights`). Obs are sampled every `dt_obs` hours (default
    `dt_verif`), which may be finer than the model's 6h timestep. Each obs
    slot `j` (time `t_j = j*dt_obs`) is tagged with the lower bracketing 6h
    model-step index `step_lo[j]` and the in-step fraction `alpha[j] in [0,
    1]`, so the loss / diagnostics code can linearly interpolate the model
    state in time between steps `step_lo` and `step_lo+1` before applying
    the forward operator. `dt_obs == dt_verif` reproduces the mainline (one
    slot per 6h step, `alpha == 0` everywhere).
    """
    n_verif = exp['n_verif']
    dt_verif = exp['dt_verif']
    dt_obs = exp['dt_obs']
    timestep_h = int(round(model.timestep.total_seconds() / 3600))
    n_steps = n_verif * dt_verif // timestep_h          # total 6h model steps in the window
    n_obs = n_verif * dt_verif // dt_obs + 1            # number of obs slots (inclusive of both ends)
    obspath = exp['obspath']
    oberrstart = exp.get('oberrstart', 0.)
    if oberrstart > 0:
        logger.info('ob error at start of window:' + str(oberrstart))
    oberrdeltaperday = exp.get('oberrdeltaperday', 0.)
    if oberrdeltaperday > 0:
        logger.info('ob error increase per day:' + str(oberrdeltaperday))
    if exp.get('psobs_error_scale', 1.0) != 1.0:
        logger.info('ps ob error scale: ' + str(exp['psobs_error_scale']))
    date = exp['date']
    device = model.device
    psobs_traj = {}
    psobs_datestrings = []
    vdate = copy.copy(date)
    verification_times = []
    for j in range(n_obs):
        yyyymmddhh = get_YYYYMMDDHH(vdate)
        psobs_datestrings.append(yyyymmddhh)
        verification_times.append(vdate)
        vdate = add_hours(vdate, dt_obs)

    from psobs_parquet import load_psobs_parquet
    time_tolerance_hours = exp.get('psobs_time_tolerance_hours', dt_obs / 2.)
    parquet_data = load_psobs_parquet(
        obspath,
        verification_times,
        time_tolerance_hours,
        prepbufr_classes=exp.get('psobs_prepbufr_classes', ('ADPSFC', 'SFCSHP')),
        report_types=exp.get('psobs_report_types', (180, 181, 187)),
        quality_mark_max=exp.get('psobs_quality_mark_max', 3),
        observation_error_std_hpa=exp.get('surface_pressure_error_std_hpa', 1.),
        use_source_error=exp.get('psobs_use_source_error', False),
        logger=logger,
    )
    nobs_list = [arr.size for arr in parquet_data['ob']]
    if 'nobs_max' in exp:
        nobs_max = exp['nobs_max']
    else:
        nobs_max = max(nobs_list)
    logger.info('max number of obs at each slot in window:' + str(nobs_max))

    for k in ['obtype', 'lon', 'lat', 'elev', 'ob', 'oberr', 'oberr_qc']:
        if k in ('ob', 'elev'):
            psobs_traj[k] = torch.zeros((n_obs, nobs_max), dtype=torch.float32, device=device)
        elif k == 'obtype':
            psobs_traj[k] = torch.full((n_obs, nobs_max), 999, device=device)
        else:
            psobs_traj[k] = torch.full((n_obs, nobs_max), 1.e10, dtype=torch.float32, device=device)
    psobs_traj['n_valid'] = torch.as_tensor(nobs_list, dtype=torch.int32, device=device)

    for j in range(n_obs):
        nobs = nobs_list[j]
        if nobs > nobs_max:
            raise ValueError(
                'psobs slot '+str(j)+' has '+str(nobs)
                +' observations, exceeding configured nobs_max '+str(nobs_max)+'.'
            )
        psobs_traj['obtype'][j, :nobs] = torch.as_tensor(parquet_data['obtype'][j], device=device)
        psobs_traj['lon'][j, :nobs] = torch.as_tensor(parquet_data['lon'][j], dtype=torch.float32, device=device)
        psobs_traj['lat'][j, :nobs] = torch.as_tensor(parquet_data['lat'][j], dtype=torch.float32, device=device)
        psobs_traj['elev'][j, :nobs] = torch.as_tensor(parquet_data['elev'][j], dtype=torch.float32, device=device)
        psobs_traj['ob'][j, :nobs] = torch.as_tensor(parquet_data['ob'][j], dtype=torch.float32, device=device)
        tday = j * dt_obs / 24.
        if oberrstart > 0:
            base_err = torch.full((nobs,), float(oberrstart), dtype=torch.float32, device=device)
        else:
            base_err = torch.as_tensor(parquet_data['oberr'][j], dtype=torch.float32, device=device)
        base_err = base_err * float(exp.get('psobs_error_scale', 1.0))  # before the orography term
        psobs_traj['oberr'][j, :nobs] = base_err + oberrdeltaperday * tday
        # start-of-window error, without the oberrdeltaperday growth -- used by
        # the gross check instead of 'oberr' when qc_fixed_oberr is set
        psobs_traj['oberr_qc'][j, :nobs] = base_err
    logger.info('ps ob times:' + str(psobs_datestrings))

    # Per-slot time-bracketing metadata: step_lo = floor(t_j / 6h), alpha =
    # the leftover fraction into that 6h step. The final slot lands exactly on
    # step n_steps (alpha would be 0 with no upper bracket) -- pin it to
    # (n_steps-1, alpha=1) instead so every slot has a valid (step_lo,
    # step_lo+1) pair.
    step_lo, alpha = [], []
    for j in range(n_obs):
        t_hours = j * dt_obs
        s = t_hours // timestep_h
        a = (t_hours - s * timestep_h) / timestep_h
        if s >= n_steps:
            s, a = n_steps - 1, 1.0
        step_lo.append(int(s))
        alpha.append(float(a))
    psobs_traj['step_lo'] = torch.as_tensor(step_lo, dtype=torch.int64, device=device)
    psobs_traj['alpha'] = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    psobs_traj['n_steps'] = n_steps
    logger.info('ps ob slot step_lo:' + str(step_lo))
    logger.info('ps ob slot alpha:' + str([round(a, 4) for a in alpha]))

    # Precompute k-NN obs->grid interpolation indices/weights once per window
    # (not per epoch): the loss function's inner loop then does a fixed
    # gather + fixed-weight dot product, never a repeated tree query.
    idx_list, wts_list = [], []
    lon_np = psobs_traj['lon'].detach().cpu().numpy()
    lat_np = psobs_traj['lat'].detach().cpu().numpy()
    for j in range(n_obs):
        idx, wts = grid_interp.weights(lon_np[j], lat_np[j], k=4, device=device)
        idx_list.append(idx)
        wts_list.append(wts)
    psobs_traj['interp_idx'] = torch.stack(idx_list, dim=0)  # (n_obs, nobs_max, k)
    psobs_traj['interp_wts'] = torch.stack(wts_list, dim=0)

    return psobs_traj


def get_raobs(exp, model, logger, grid_interp, psobs_traj):
    """Read radiosonde (ADPUPA) profile observations for the SAME obs-slot
    grid `get_psobs` already built (`psobs_traj['step_lo']`/`['alpha']`,
    same `dt_obs`-spaced verification times) -- real ADPUPA launches are
    synoptic (00/06/12/18Z), matching the slot times `dt_obs` is normally
    set to anyway, so this piggybacks on psobs's rollout/time-bracketing
    instead of needing a second one (see `_resolve_loss_interp_specs`,
    `compute_loss_4dvar`). Returns `None` if `exp['raobs_enabled']` is not
    True -- callers (`compute_optimal`, `compute_loss_4dvar`,
    `compute_ps_observation_hx`, `save_trajectory_diagnostics`) all accept
    `raobs_traj=None` and skip the raobs term entirely, so a config that
    never sets `raobs_enabled` assimilates surface pressure only, exactly as
    before.

    Mandatory-level matching (see raobs_parquet.py and integrate_prepbufr_obs.md):
    each requested variable is matched against the MODEL's own pressure
    levels for that variable's family (`model.pressure_levels(base)`), so
    the forward operator is a plain horizontal-interp + level-index gather,
    no vertical interpolation.
    """
    if not exp.get('raobs_enabled', False):
        return None
    from raobs_parquet import load_raobs_parquet

    variables = exp['raobs_variables']
    bases = [RAOBS_VARIABLE_MODEL_BASE[v] for v in variables]
    levels_hpa = {base: model.pressure_levels(base) for base in bases}

    dt_obs = exp['dt_obs']
    date = exp['date']
    vdate = copy.copy(date)
    verification_times = []
    for _ in range(len(psobs_traj['step_lo'])):
        verification_times.append(vdate)
        vdate = add_hours(vdate, dt_obs)

    time_tolerance_hours = exp.get('raobs_time_tolerance_hours', dt_obs / 2.)
    data = load_raobs_parquet(
        exp.get('raobs_path', exp['obspath']),
        verification_times,
        time_tolerance_hours,
        levels_hpa,
        bases,
        prepbufr_classes=exp.get('raobs_prepbufr_classes', ('ADPUPA',)),
        quality_mark_max=exp.get('raobs_quality_mark_max', 3),
        geopotential_height_min_m=exp.get('raobs_height_min_m', -1000.),
        geopotential_height_max_m=exp.get('raobs_height_max_m', 60000.),
        below_station_tolerance_m=exp.get('raobs_below_station_tolerance_m', 100.),
        pressure_match_tolerance_hpa=exp.get('raobs_pressure_match_tolerance_hpa', 0.05),
        humidity_min_pressure_hpa=exp.get('raobs_humidity_min_pressure_hpa', 300.),
        logger=logger,
    )

    device = model.device
    n_obs = len(verification_times)
    station_capacity = [arr.size for arr in data['lat']]
    capacity = exp.get('raobs_station_capacity', max(station_capacity))
    if max(station_capacity) > capacity:
        raise ValueError(
            f"raobs station count {max(station_capacity)} exceeds configured "
            f"raobs_station_capacity {capacity}"
        )

    raobs_traj = {
        'n_valid': torch.as_tensor(station_capacity, dtype=torch.int32, device=device),
        'levels_hpa': {base: torch.as_tensor(levels_hpa[base], dtype=torch.float32, device=device) for base in bases},
    }
    for name in ('lat', 'lon', 'elev'):
        padded = torch.zeros((n_obs, capacity), dtype=torch.float32, device=device)
        for j in range(n_obs):
            padded[j, :data[name][j].size] = torch.as_tensor(data[name][j], device=device)
        raobs_traj[name] = padded

    # t/u/v: a fixed absolute error (K, m/s). q: error scales with the
    # observed value instead -- a flat absolute error makes no physical
    # sense across q's full range (near-zero in the upper troposphere,
    # several g/kg at the surface) -- so 'q' is a FRACTION of |observation|
    # (floored, to avoid a near-zero effective error whenever q itself is
    # near zero). No reference-repo precedent for any of q's handling here.
    error_std = exp.get('raobs_error_std', {'t': 1.0, 'u': 2.0, 'v': 2.0})
    q_error_fraction = exp.get('raobs_q_error_fraction', 0.2)
    q_error_floor = exp.get('raobs_q_error_floor_kg_kg', 1.e-5)

    for base in bases:
        n_levels = levels_hpa[base].size
        values = torch.zeros((n_obs, n_levels, capacity), dtype=torch.float32, device=device)
        mask = torch.zeros((n_obs, n_levels, capacity), dtype=torch.bool, device=device)
        for j in range(n_obs):
            n = data['values_'+base][j].shape[-1]
            values[j, :, :n] = torch.as_tensor(data['values_'+base][j], device=device)
            mask[j, :, :n] = torch.as_tensor(data['mask_'+base][j], device=device)
        raobs_traj['values_'+base] = values
        raobs_traj['mask_'+base] = mask
        if base == 'q':
            raobs_traj['error_q_fraction'] = float(q_error_fraction)
            raobs_traj['error_q_floor'] = float(q_error_floor)
            # q_error_g_kg {hPa: g/kg}: a fixed per-level error instead of the
            # fraction of the ob (which weights dry obs most and pulls the
            # analysis dry); levels between table entries interpolate in ln(p),
            # beyond them take the nearest entry
            table = exp.get('raobs_q_error_g_kg')
            if table:
                tp = np.array(sorted(float(k) for k in table))
                tv = np.array([float(table[k] if k in table else table[int(k)]) for k in sorted(table, key=float)])
                lev = np.asarray(levels_hpa['q'], dtype=float)
                err = np.interp(np.log(lev), np.log(tp), tv) * 1.e-3
                raobs_traj['error_q_level'] = torch.as_tensor(err, dtype=torch.float32, device=device)
            bgc = exp.get('raobs_q_bg_check')
            raobs_traj['q_bg_check'] = float(bgc) if bgc else None
        else:
            raobs_traj['error_'+base] = float(error_std[base])

    # Shared k-NN horizontal interpolation (same mechanism as psobs, same
    # station locations across every variable in a slot).
    idx_list, wts_list = [], []
    lon_np = raobs_traj['lon'].detach().cpu().numpy()
    lat_np = raobs_traj['lat'].detach().cpu().numpy()
    for j in range(n_obs):
        idx, wts = grid_interp.weights(lon_np[j], lat_np[j], k=4, device=device)
        idx_list.append(idx)
        wts_list.append(wts)
    raobs_traj['interp_idx'] = torch.stack(idx_list, dim=0)
    raobs_traj['interp_wts'] = torch.stack(wts_list, dim=0)
    raobs_traj['variables'] = bases
    raobs_traj['n_steps'] = psobs_traj['n_steps']
    raobs_traj['step_lo'] = psobs_traj['step_lo']
    raobs_traj['alpha'] = psobs_traj['alpha']
    return raobs_traj


_ATMS_OPERATORS = {}


def _load_atms_operator(bundle, device):
    """One AtmsOperator per (bundle, device), kept across windows."""
    key = (os.path.abspath(bundle), str(device))
    if key not in _ATMS_OPERATORS:
        from obs_atms_operator import AtmsOperator
        _ATMS_OPERATORS[key] = AtmsOperator.load(bundle, device)
    return _ATMS_OPERATORS[key]


ATMS_SCENES = ('ocean_clear', 'land', 'snow', 'seaice', 'ocean_cloudy')


def _atms_scene_codes(d, water_frac_min, clw_max):
    """Scene class per ob (index into ATMS_SCENES, -1 = none), as radianced
    `scripts/twohead.py` `scene_masks` defines them on the ERA5 pairs: clear
    ocean (all water, no ice, clw <= clw_max), land (water_frac <= 0.01, no
    snow, no ice), snow (water_frac <= 0.01, snow), sea ice (ice_frac >= 0.5).
    Cloudy ocean (all water, no ice, clw > clw_max) is code 4 ("ocean_cloudy");
    mixed coastal footprints are -1."""
    wf, ice, snow, clw = (d[c].to_numpy() for c in ('water_frac', 'ice_frac', 'snow_frac', 'clw_obs'))
    land = wf <= 0.01
    code = np.full(len(d), -1, dtype=np.int8)
    code[(wf >= water_frac_min) & (ice <= 0) & (clw <= clw_max)] = 0
    code[(wf >= water_frac_min) & (ice <= 0) & (clw > clw_max)] = 4
    code[land & (snow <= 0) & (ice <= 0)] = 1
    code[land & (snow > 0)] = 2
    code[ice >= 0.5] = 3
    return code


def get_mwobs(exp, model, logger, grid_interp, psobs_traj):
    """`get_atmsobs` for every enabled MW_SENSORS sensor: a list of
    trajectories (one per sensor, `traj['sensor']`), or None if none is on."""
    trajs = [get_atmsobs(exp, model, logger, grid_interp, psobs_traj, sensor) for sensor in MW_SENSORS]
    trajs = [t for t in trajs if t is not None]
    return trajs or None


def get_atmsobs(exp, model, logger, grid_interp, psobs_traj, sensor='atms'):
    """Read microwave radiances of `sensor` (ATMS, AMSU-A) for psobs's
    obs-slot grid (`step_lo`/`alpha`, as `get_raobs` does). Returns `None`
    unless `exp['<sensor>_enabled']`. Config keys are `observations.<sensor>`
    (flattened to `<sensor>_<key>`); the docstrings below say `atms_*`.

    Obs files (radianced `scripts/build_lw4dvar_obs.py` for ATMS,
    `build_mw_obs.py` for the others): one parquet per 6-h cycle,
    `<atms_path>/<YYYYMMDDHH>.parquet`, with `dt_hours` (obs time minus cycle
    time), `yobc_01..NN` (y_o - BC, K; NaN where the channel is off for that
    satellite), `sat_id`, scan geometry, FOV surface fractions, `clw_obs`
    and `withheld`. Slot `j` (time `t_j`) takes
    every ob with obs time in `[t_j - tol, t_j + tol)`, `tol =
    atms_time_tolerance_hours` (default `dt_obs / 2`), from the cycles
    `t_j - 6h`, `t_j`, `t_j + 6h`.

    Read-time scene screen (E1, clear-sky ocean): `water_frac >=
    atms_water_frac_min` (0.99), `ice_frac == 0`, `clw_obs <= atms_clw_max`
    (0.05 kg/m2). The model-side screens (all 4 interpolation neighbours
    with lsm <= `atms_lsm_max`, background check) are applied in
    `_compute_atms_observation_diagnostics_at_time`. Withheld obs are kept,
    scored in the diagnostics and never enter the loss.
    """
    if not exp.get(f'{sensor}_enabled', False):
        return None
    import pandas as pd

    def cfg(key, default=None):
        return exp.get(f'{sensor}_{key}', default)

    device = model.device
    op = _load_atms_operator(cfg('bundle'), device)
    if op.sensor != sensor:
        raise ValueError(f"observations.{sensor}.bundle {cfg('bundle')} is a {op.sensor!r} decoder")
    dt_obs = exp['dt_obs']
    tol = cfg('time_tolerance_hours', dt_obs / 2.)
    water_frac_min = cfg('water_frac_min', 0.99)
    clw_max = cfg('clw_max', 0.05)
    scenes = list(cfg('scenes', ['ocean_clear']))
    unknown = set(scenes) - set(ATMS_SCENES)
    if unknown:
        raise ValueError(f"observations.{sensor}.scenes: unknown {sorted(unknown)}; known {list(ATMS_SCENES)}")
    scene_codes = [ATMS_SCENES.index(sc) for sc in scenes]
    n_obs = len(psobs_traj['step_lo'])

    cache = {}

    def read_cycle(date_str):
        if date_str not in cache:
            f = os.path.join(cfg('path'), get_YYYYMMDDHH(date_str) + '.parquet')
            cache[date_str] = pd.read_parquet(f) if os.path.exists(f) else None
        return cache[date_str]

    slots = []
    vdate = copy.copy(exp['date'])
    for j in range(n_obs):
        parts = []
        for h in (-6, 0, 6):
            d = read_cycle(add_hours(vdate, h))
            if d is None:
                continue
            offset = h + d['dt_hours'].to_numpy()
            parts.append(d[(offset >= -tol) & (offset < tol)])
        if not parts:
            raise FileNotFoundError(f"no {sensor} obs file within 6h of {vdate} under {cfg('path')}")
        d = pd.concat(parts, ignore_index=True)
        d['scene'] = _atms_scene_codes(d, water_frac_min, clw_max)
        slots.append(d[d['scene'].isin(scene_codes)].reset_index(drop=True))
        vdate = add_hours(vdate, dt_obs)

    counts = [len(d) for d in slots]
    capacity = cfg('capacity', max(max(counts), 1))
    if max(counts) > capacity:
        raise ValueError(f"{sensor} slot count {max(counts)} exceeds configured {sensor} capacity {capacity}")
    logger.info(f'{sensor} obs per slot after scene screen: ' + str(counts)
                + ', withheld: ' + str([int(d['withheld'].sum()) for d in slots]))
    allobs = pd.concat(slots)
    logger.info(f'{sensor} obs by scene: ' + str({sc: int((allobs['scene'] == ATMS_SCENES.index(sc)).sum()) for sc in scenes}))

    nch = op.nch
    f32 = dict(dtype=torch.float32, device=device)
    atms_traj = {
        'sensor': sensor,
        'op': op,
        'n_valid': torch.as_tensor(counts, dtype=torch.int32, device=device),
        'lat': torch.zeros((n_obs, capacity), **f32),
        'lon': torch.zeros((n_obs, capacity), **f32),
        'y': torch.zeros((n_obs, capacity, nch), **f32),
        'valid': torch.zeros((n_obs, capacity), dtype=torch.bool, device=device),
        'withheld': torch.zeros((n_obs, capacity), dtype=torch.bool, device=device),
        'scene': torch.full((n_obs, capacity), -1, dtype=torch.int8, device=device),
    }
    for name in _ATMS_OBS_INPUTS:
        atms_traj['obs_'+name] = torch.zeros((n_obs, capacity), **f32)
    ycols = [f'yobc_{c:02d}' for c in range(1, nch + 1)]
    for j, d in enumerate(slots):
        n = len(d)
        atms_traj['lat'][j, :n] = torch.as_tensor(d['lat'].to_numpy(copy=True), **f32)
        atms_traj['lon'][j, :n] = torch.as_tensor(d['lon'].to_numpy(copy=True), **f32)
        atms_traj['y'][j, :n] = torch.as_tensor(d[ycols].to_numpy(copy=True), **f32)
        atms_traj['valid'][j, :n] = True
        atms_traj['withheld'][j, :n] = torch.as_tensor(d['withheld'].to_numpy(copy=True), device=device)
        atms_traj['scene'][j, :n] = torch.as_tensor(d['scene'].to_numpy(copy=True), dtype=torch.int8, device=device)
        for name in _ATMS_OBS_INPUTS:
            atms_traj['obs_'+name][j, :n] = torch.as_tensor(d[name].to_numpy(copy=True), **f32)

    # model levels -> the operator's level order (top -> bottom); q is
    # detached (no Jo gradient) above atms_detach_q_above_hpa, where the
    # decoder's q sensitivity is a learned proxy, not physics
    t_levels = [float(p) for p in model.pressure_levels('t')]
    q_levels = [float(p) for p in model.pressure_levels('q')]
    op_q_levels = op.q_levels_hpa if op.q_levels_hpa[0] in q_levels else op.q_levels_hpa[1:]
    atms_traj['t_order'] = [t_levels.index(p) for p in op.t_levels_hpa]
    atms_traj['q_order'] = [q_levels.index(p) for p in op_q_levels]
    detach_above = cfg('detach_q_above_hpa', 250.)
    atms_traj['q_detach'] = torch.as_tensor([p < detach_above for p in op_q_levels], device=device)

    defaults = _MW_DEFAULT_CHANNELS[sensor]
    assimilate = list(cfg('assimilate_channels', defaults['assimilate']))
    monitor = [c for c in cfg('monitor_channels', defaults['monitor']) if c not in assimilate]
    atms_traj['assimilate_channels'] = assimilate
    atms_traj['monitor_channels'] = monitor
    atms_traj['assimilate_idx'] = [c - 1 for c in assimilate]
    # emissivity control (plan E2): one z per land/snow/sea-ice ob, moved only
    # by the window channels, which see the state detached
    emis_control = cfg('emis_control', 'none')
    if emis_control not in ('none', 'per_obs'):
        raise ValueError(f"observations.{sensor}.emis_control must be 'none' or 'per_obs', got {emis_control!r}")
    atms_traj['emis_control'] = emis_control
    emis = list(cfg('emis_channels', defaults['emis'])) if emis_control == 'per_obs' else []
    if set(emis) & set(assimilate):
        raise ValueError(f"{sensor} emis_channels {emis} overlap assimilate_channels {assimilate}")
    atms_traj['emis_channels'] = emis
    atms_traj['emis_idx'] = [c - 1 for c in emis]
    atms_traj['monitor_channels'] = [c for c in monitor if c not in emis]
    atms_traj['z_active'] = atms_traj['valid'] & (atms_traj['scene'] > 0) & (atms_traj['scene'] < 4)  # z = 0 over open ocean
    # scenes where the window channels also act on the state (plan E4), not only on z
    window_state = list(cfg('window_state_scenes', []))
    if [sc for sc in window_state if sc not in ATMS_SCENES]:
        raise ValueError(f"observations.{sensor}.window_state_scenes must be among {ATMS_SCENES}, got {window_state}")
    if window_state and emis_control != 'per_obs':
        raise ValueError(f"observations.{sensor}.window_state_scenes needs emis_control: 'per_obs' (it uses emis_channels)")
    ws_idx = torch.as_tensor([ATMS_SCENES.index(sc) for sc in window_state], dtype=torch.long, device=device)
    atms_traj['window_state'] = atms_traj['valid'] & torch.isin(atms_traj['scene'].long(), ws_idx)
    atms_traj['window_state_scenes'] = window_state
    # per-channel overrides {channel: K} (e.g. Desroziers estimates), then the global scale
    sigma_o = op.sigma_o.clone()
    for c, v in (cfg('sigma_o') or {}).items():
        sigma_o[int(c) - 1] = float(v)
    atms_traj['sigma_o'] = sigma_o * float(cfg('sigma_o_scale', 1.0))
    atms_traj['bg_check'] = float(cfg('bg_check', 4.0))
    # Huber norm on Jo (assimilated and window channels): quadratic for
    # |d| / sigma <= huber_c, linear beyond; None = plain quadratic
    huber_c = cfg('huber_c')
    atms_traj['huber_c'] = float(huber_c) if huber_c else None
    # channels the decoder predicts (NaN y_mean = a channel without training
    # targets, e.g. AMSU-A 14): only these enter the finite-H(x) screen
    atms_traj['tb_ok_idx'] = torch.isfinite(op.y_mean).nonzero().flatten().tolist()
    _setup_atms_oberr(exp, atms_traj, nch, device)
    _setup_atms_corr_r(cfg('corr_r'), float(cfg('corr_cond_max', 50.)), atms_traj, logger, device)
    atms_traj['lsm_max'] = float(cfg('lsm_max', 0.01))

    idx_list, wts_list = [], []
    lon_np = atms_traj['lon'].detach().cpu().numpy()
    lat_np = atms_traj['lat'].detach().cpu().numpy()
    for j in range(n_obs):
        idx, wts = grid_interp.weights(lon_np[j], lat_np[j], k=4, device=device)
        idx_list.append(idx)
        wts_list.append(wts)
    atms_traj['interp_idx'] = torch.stack(idx_list, dim=0)
    atms_traj['interp_wts'] = torch.stack(wts_list, dim=0)
    atms_traj['n_steps'] = psobs_traj['n_steps']
    atms_traj['step_lo'] = psobs_traj['step_lo']
    atms_traj['alpha'] = psobs_traj['alpha']
    return atms_traj


# ---------------------------------------------------------------------------
# forward operator (station-elevation surface pressure) + QC
# ---------------------------------------------------------------------------

def get_surface_pressure(pressure_levels, geopotential, orography):
    """Surface pressure from geopotential on pressure levels, by
    interpolating/extrapolating log(pressure) linearly in
    surface_geopotential-geopotential (unlimited linear extrapolation at
    either end, via clamping the searchsorted bracket to the first/last
    interval). Fully vectorized across observations via a batched
    torch.searchsorted, rather than a per-observation Python loop with
    repeated in-place index-assignment into a shared buffer.

    pressure_levels : (n_levels,)
    geopotential : (n_levels, n_obs)
    orography : (n_obs,)  (already * grav)
    returns : (n_obs,)
    """
    n_levels, n_obs = geopotential.shape
    # torch.searchsorted requires its search array to be ascending.
    # {AIFS,FCN3}Model.pressure_levels()/decode_state() order levels
    # surface-first (1000hPa, ..., 10 or 50hPa) -- geopotential *increases*
    # with altitude along that ordering, so relative_height = orography -
    # geopotential *decreases* along it. Flip to top-of-atmosphere-first so
    # relative_height is ascending, as this function's search requires.
    # Without this, searchsorted's result on a descending array is
    # undefined (not an error -- it silently returns nonsense bracket
    # indices), which produced a systematic ~200-280 hPa high bias in the
    # AIFS port before this fix (confirmed via saved diagnostics) -- ported
    # to FCN3 already fixed, not rediscovered there.
    pressure_levels = torch.flip(pressure_levels, dims=(0,))
    geopotential = torch.flip(geopotential, dims=(0,))
    relative_height = orography[None, :] - geopotential  # (n_levels, n_obs)
    log_p = -torch.log(pressure_levels)  # (n_levels,)

    xp = relative_height.transpose(0, 1).contiguous()  # (n_obs, n_levels)
    x = torch.zeros((n_obs, 1), dtype=xp.dtype, device=xp.device)
    u = torch.searchsorted(xp, x, right=True).squeeze(-1)  # (n_obs,)
    u = torch.clamp(u, 1, n_levels - 1)

    idx_lo = (u - 1).unsqueeze(-1)
    idx_hi = u.unsqueeze(-1)
    xp_lo = torch.gather(xp, 1, idx_lo).squeeze(-1)
    xp_hi = torch.gather(xp, 1, idx_hi).squeeze(-1)
    fp_lo = log_p[idx_lo.squeeze(-1)]
    fp_hi = log_p[idx_hi.squeeze(-1)]

    w = (0.0 - xp_lo) / (xp_hi - xp_lo)
    out = (1 - w) * fp_lo + w * fp_hi
    return torch.exp(-out)


def preduce(ps, tpress, t, q, zmodel, zob, rlapse=0.0065, grav=GRAV, rd=RD, rv=RV):
    """MAPS pressure reduction from model to station elevation.
    See Benjamin and Miller (1990, MWR, p. 2100). Used by
    `ps_operator='ps'` (AIFS; `tpress` = the scalar 700 hPa level) and
    `ps_operator='ps_native'` (ACE2; `tpress` = the lowest native level's
    per-point pressure) -- elementwise, so either works.
    """
    alpha = rd * rlapse / grav
    fv = rv / rd - 1.
    tv = t * (1. + fv * q)
    t0 = tv * (ps / tpress) ** alpha
    return ps * ((t0 + rlapse * (zmodel - zob)) / t0) ** (1.0 / alpha)


def _compute_ps_observation_diagnostics_at_time(model, decoded, verif_ic, psobs_traj, exp, oind, grid_interp):
    """Evaluate surface-pressure H(x) and the loss QC masks at time index oind."""
    zthresh = exp['zthresh']
    zconst = exp['zconst']
    bg_check = exp['bg_check']
    nobs_max = exp['nobs_max']

    device = decoded['geopotential'].device
    lon = psobs_traj['lon'][oind, :]
    lat = psobs_traj['lat'][oind, :]
    elevation = psobs_traj['elev'][oind, :]
    observation = psobs_traj['ob'][oind, :]
    assigned_error = psobs_traj['oberr'][oind, :]
    nobs_valid = psobs_traj['n_valid'][oind]
    available = torch.arange(nobs_max, device=device) < nobs_valid

    idx = psobs_traj['interp_idx'][oind]
    wts = psobs_traj['interp_wts'][oind]

    era5_orography = verif_ic['geopotential_at_surface'] / GRAV  # (n_points,) or scalar broadcastable
    geopotential_obspace = grid_interp.interp(idx, wts, decoded['geopotential'])  # (n_levels, n_obs)
    model_orography = grid_interp.interp(idx, wts, era5_orography)  # (n_obs,)

    interpolation_failed = ~torch.isfinite(geopotential_obspace[0]) | ~torch.isfinite(model_orography)
    preliminary_rejected = ~available | interpolation_failed

    safe_elevation = torch.where(preliminary_rejected, torch.zeros_like(elevation), elevation)
    safe_model_orography = torch.where(preliminary_rejected, torch.zeros_like(model_orography), model_orography)

    ps_operator = exp.get('ps_operator', 'logpinterp')
    if ps_operator == 'ps':
        # AIFS ONLY in practice -- unreachable for FCN3 (no native 'sp' at
        # all) and currently also blocked for Aurora (has a native 'sp',
        # but this path hasn't been tested for it yet); load_config()
        # rejects both combinations up front, see module docstring.
        nlev700 = exp['nlev700']
        plevs = torch.as_tensor(model.pressure_levels('z'), device=device)
        tpress = plevs[nlev700]
        t700 = decoded['temperature'][nlev700, :]
        q700 = decoded['specific_humidity'][nlev700, :]
        # AIFS's `sp` is in Pa (~98526); observations, tpress and the
        # `safe_ps` fallback below are all hPa -- convert.
        ps = decoded['surface_pressure'][:] / 100.0
        ps_obspace = grid_interp.interp(idx, wts, ps)
        t700_obspace = grid_interp.interp(idx, wts, t700)
        q700_obspace = grid_interp.interp(idx, wts, q700)
        safe_ps = torch.where(preliminary_rejected, torch.full_like(ps_obspace, 985.), ps_obspace)
        safe_t700 = torch.where(preliminary_rejected, torch.full_like(t700_obspace, 300.), t700_obspace)
        safe_q700 = torch.where(preliminary_rejected, torch.full_like(q700_obspace, 1.e-5), q700_obspace)
        model_equivalent = preduce(safe_ps, tpress, safe_t700, safe_q700, safe_model_orography, safe_elevation)
    elif ps_operator == 'ps_native':
        # ACE2 only (load_config enforces): same MAPS reduction as 'ps', but
        # with the lowest NATIVE model level's t/q and its own per-point
        # pressure instead of t/q interpolated to the fixed 700 hPa level --
        # ACE2's layers are hybrid sigma-pressure, so no pressure-level
        # interpolation at all. preduce is elementwise, so a per-point
        # `tpress` works unchanged. Pressures in Pa -> hPa.
        ps_obspace = grid_interp.interp(idx, wts, decoded['surface_pressure'] / 100.0)
        pl_obspace = grid_interp.interp(idx, wts, decoded['p_lowest'] / 100.0)
        tl_obspace = grid_interp.interp(idx, wts, decoded['t_lowest'])
        ql_obspace = grid_interp.interp(idx, wts, decoded['q_lowest'])
        safe_ps = torch.where(preliminary_rejected, torch.full_like(ps_obspace, 985.), ps_obspace)
        safe_pl = torch.where(preliminary_rejected, torch.full_like(pl_obspace, 925.), pl_obspace)
        safe_tl = torch.where(preliminary_rejected, torch.full_like(tl_obspace, 285.), tl_obspace)
        safe_ql = torch.where(preliminary_rejected, torch.full_like(ql_obspace, 1.e-3), ql_obspace)
        model_equivalent = preduce(safe_ps, safe_pl, safe_tl, safe_ql, safe_model_orography, safe_elevation)
    elif ps_operator == 'logpinterp':
        areaw = torch.as_tensor(model.area_weights, dtype=decoded['geopotential'].dtype, device=device)
        geopotential_mean = (areaw[None, :] * decoded['geopotential']).sum(dim=-1) / areaw.sum()
        safe_geopotential = torch.where(
            preliminary_rejected[None, :], geopotential_mean[:, None], geopotential_obspace
        )
        plevs = torch.as_tensor(model.pressure_levels('z'), device=device)
        model_equivalent = get_surface_pressure(plevs, safe_geopotential, GRAV * safe_elevation)
    else:
        raise ValueError(f"unknown ps_operator {ps_operator!r} (expected 'ps', 'ps_native' or 'logpinterp')")

    orography_difference = torch.where(preliminary_rejected, torch.zeros_like(safe_elevation), torch.abs(safe_model_orography - safe_elevation))
    orography_failed = orography_difference > zthresh
    effective_error = assigned_error + zconst * orography_difference
    # qc_fixed_oberr: gross-check against the START-of-window error (no
    # oberrdeltaperday growth), so down-weighting late-window obs in the loss
    # doesn't also loosen their QC (bg_check * sigma would otherwise grow from
    # e.g. 4 to 14 hPa by 120h at oberrdeltaperday=0.5). Opt-in, so earlier
    # runs with oberrdeltaperday > 0 stay reproducible.
    if exp.get('qc_fixed_oberr', False):
        qc_error = psobs_traj['oberr_qc'][oind, :] + zconst * orography_difference
    else:
        qc_error = effective_error
    base_rejected = preliminary_rejected | interpolation_failed | orography_failed
    innovation = observation - model_equivalent
    gross_check_failed = (~base_rejected) & (torch.abs(innovation / qc_error) > bg_check)
    total_rejected = preliminary_rejected | interpolation_failed | orography_failed | gross_check_failed
    used = ~total_rejected

    qc_flag = torch.zeros(nobs_max, dtype=torch.int16, device=device)
    qc_flag = torch.where(~available, qc_flag | 1, qc_flag)
    qc_flag = torch.where(interpolation_failed, qc_flag | 2, qc_flag)
    qc_flag = torch.where(orography_failed, qc_flag | 4, qc_flag)
    qc_flag = torch.where(gross_check_failed, qc_flag | 8, qc_flag)

    return innovation, model_equivalent, effective_error, available, used, qc_flag


def _compute_raobs_observation_diagnostics_at_time(model, decoded, raobs_traj, grid_interp, oind):
    """Evaluate radiosonde H(x) for every configured variable at slot `oind`.

    No vertical interpolation (see get_raobs/raobs_parquet.py's module
    docstrings): `raobs_traj['values_<base>'][oind]`'s level axis was
    matched against `model.pressure_levels(base)` AT READ TIME, in the same
    order `decoded[base]`'s rows are in -- so the forward operator is just
    horizontal interpolation, no level gather needed.

    Returns `{base: (innovation, model_equivalent, effective_error, used)}`,
    one entry per `raobs_traj['variables']`. Unlike the ps forward
    operator, there is no orography/background-gross-error QC here (just
    read-time QC, already folded into `mask_<base>`) and no elevation-based
    reduction -- each variable is compared against the model's own native
    level directly.
    """
    idx = raobs_traj['interp_idx'][oind]
    wts = raobs_traj['interp_wts'][oind]
    results = {}
    for base in raobs_traj['variables']:
        model_obspace = grid_interp.interp(idx, wts, decoded[base])  # (n_levels, capacity)
        observation = raobs_traj['values_'+base][oind]
        mask = raobs_traj['mask_'+base][oind]
        interpolation_failed = ~torch.isfinite(model_obspace)
        used = mask & ~interpolation_failed
        safe_model_obspace = torch.where(interpolation_failed, torch.zeros_like(model_obspace), model_obspace)
        innovation = observation - safe_model_obspace
        if base == 'q':
            if raobs_traj.get('error_q_level') is not None:
                effective_error = raobs_traj['error_q_level'][:, None].expand_as(observation)
            else:
                effective_error = torch.clamp(
                    raobs_traj['error_q_fraction'] * observation.abs(),
                    min=raobs_traj['error_q_floor'],
                )
            if raobs_traj.get('q_bg_check'):
                used = used & (innovation.abs() <= raobs_traj['q_bg_check'] * effective_error).detach()
        else:
            effective_error = torch.full_like(observation, raobs_traj['error_'+base])
        results[base] = (innovation, safe_model_obspace, effective_error, used)
    return results


def _compute_atms_observation_diagnostics_at_time(model, decoded, atms_traj, grid_interp, oind, z=None,
                                                  return_inputs=False):
    """ATMS H(x) and QC at obs slot `oind`.

    Returns `(innovation, model_equivalent, screened, used)`:
    innovation/model_equivalent `(capacity, 22)` K (y_o - BC minus H(x));
    `screened` `(capacity,)`: a real ob whose 4 interpolation neighbours are
    all ocean (lsm <= atms_lsm_max) and whose H(x) is finite; `used`
    `(capacity, 22)`: screened, passing the background check
    `|innovation| <= atms_bg_check * sigma_o` for that channel, and not
    withheld. The loss uses `used` on the assimilated channels; the
    diagnostics also score withheld obs. The all-ocean-neighbour screen
    applies to clear-ocean obs only (land/snow/ice obs come with `z`).

    `z` `(capacity, K)`: emissivity control in PC units (`e = e_hat + U z`);
    None = 0. If None and `atms_traj['zeta_eval']` is set (the analysis
    diagnostics), z = zeta_eval * sigma_k from the surface head.
    `return_inputs`: also return `(x, log_sig)` (decoder inputs, surface-head
    log sigma_k) for the loss's emissivity term.
    """
    op = atms_traj['op']
    idx = atms_traj['interp_idx'][oind]
    wts = atms_traj['interp_wts'][oind]
    t = grid_interp.interp(idx, wts, decoded['t'][atms_traj['t_order']])
    q = grid_interp.interp(idx, wts, decoded['q'][atms_traj['q_order']])
    q = torch.where(atms_traj['q_detach'][:, None], q.detach(), q)
    fields = {'t': t.float(), 'q': q.float()}
    for name, key in _ATMS_OPERATOR_FIELDS.items():
        fields[name] = grid_interp.interp(idx, wts, decoded[key]).float()
    obs = {name: atms_traj['obs_'+name][oind] for name in _ATMS_OBS_INPUTS}
    x = op.assemble(fields, obs)
    model_equivalent, _, log_sig = op(x, z)
    if z is None and atms_traj.get('zeta_eval') is not None:
        z = _atms_z(atms_traj, atms_traj['zeta_eval'], oind, log_sig)
        model_equivalent, _, _ = op(x, z)
    innovation = atms_traj['y'][oind] - model_equivalent

    ocean = (atms_traj['scene'][oind] == 0) | (atms_traj['scene'][oind] == 4)
    all_ocean = decoded['lsm'][idx].max(dim=-1).values <= atms_traj['lsm_max']
    finite = torch.isfinite(model_equivalent[:, atms_traj['tb_ok_idx']]).all(dim=-1)
    screened = atms_traj['valid'][oind] & (all_ocean | ~ocean) & finite
    sigma = _atms_sigma(atms_traj, oind, model_equivalent)
    if atms_traj['oberr_model'] != 'none':
        screened = screened & atms_traj['cloud_ok'][oind]
    passed = (innovation.abs() <= atms_traj['bg_check'] * sigma).detach()
    used = screened[:, None] & passed & ~atms_traj['withheld'][oind][:, None]
    if return_inputs:
        return innovation, model_equivalent, screened, used, (x, log_sig)
    return innovation, model_equivalent, screened, used


def _atms_clw(tb1, tb2, zenith_deg):
    """GSI `ret_amsua` cloud liquid water (kg/m2) over ocean from the 23.8 and
    31.4 GHz brightness temperatures (K), clipped to [0, 3]."""
    mu = torch.cos(torch.deg2rad(zenith_deg))
    c = mu * (8.240 - (2.622 - 1.846 * mu) * mu + 0.754 * torch.log((285.0 - tb1).clamp(min=1e-3))
              - 2.265 * torch.log((285.0 - tb2).clamp(min=1e-3)))
    return torch.nan_to_num(c, nan=3.0).clamp(0.0, 3.0)


def _setup_atms_oberr(exp, atms_traj, nch, device):
    """Observation-error model (`observations.atms.oberr_model`).

    'none': sigma_o per channel. 'scene_cloud': sigma = sigma_o * r per ob,
    with r from `oberr_table` (atms_cloud_error.py; r = 1 for channels it
    does not list): over ocean the Geer & Bauer (2011) ramp in the symmetric
    cloud amount C = (clw(y_o) + clw(H(x_b))) / 2, r = 1 for C <= c0,
    ramp_ratio for C >= c1, linear between; over land, snow and sea ice the
    per-scene ratio; r >= 1 (never tighter than sigma_o). r is set from the
    background, the first time each slot is evaluated in a cycle, then held
    fixed. Ocean obs with C > `cloud_max` (kg/m2, default 0.5) are not used.
    """
    sensor = atms_traj['sensor']
    model_name = exp.get(f'{sensor}_oberr_model', 'none')
    if model_name not in ('none', 'scene_cloud'):
        raise ValueError(f"observations.{sensor}.oberr_model must be 'none' or 'scene_cloud', got {model_name!r}")
    atms_traj['oberr_model'] = model_name
    if model_name == 'none':
        return
    import json
    with open(exp[f'{sensor}_oberr_table']) as fh:
        table = json.load(fh)['channels']
    f32 = dict(dtype=torch.float32, device=device)
    scene_ratio = torch.ones((len(ATMS_SCENES), nch), **f32)
    c0, c1, ramp = torch.zeros(nch, **f32), torch.ones(nch, **f32), torch.ones(nch, **f32)
    for c, v in table.items():
        k = int(c) - 1
        for sc in ('land', 'snow', 'seaice'):
            scene_ratio[ATMS_SCENES.index(sc), k] = v[sc]
        c0[k], c1[k], ramp[k] = v['c0'], v['c1'], v['ramp_ratio']
    atms_traj.update(oberr_scene_ratio=scene_ratio.clamp(min=1.0), oberr_c0=c0, oberr_c1=c1, oberr_ramp=ramp.clamp(min=1.0),
                     cloud_max=float(exp.get(f'{sensor}_cloud_max', 0.5)))
    y = atms_traj['y']
    atms_traj['clw_obs_tb'] = _atms_clw(y[..., 0], y[..., 1], atms_traj['obs_Sat_Zenith_Angle'])
    shape = atms_traj['valid'].shape
    atms_traj['sigma_obs'] = torch.zeros(shape + (nch,), **f32)
    atms_traj['cloud_sym'] = torch.zeros(shape, **f32)
    atms_traj['cloud_ok'] = torch.ones(shape, dtype=torch.bool, device=device)
    atms_traj['sigma_set'] = [False] * shape[0]


def _setup_atms_corr_r(path, cond_max, atms_traj, logger, device):
    """Inter-channel correlated R for the assimilated channels
    (`observations.<sensor>.corr_r`): per scene, the correlation matrix C
    from `path` (atms_decoder_error.py's json: `channels`, `scenes.<scene>.corr`),
    reconditioned by C' = (C + d I) / (1 + d) with d set so cond(C') <=
    `corr_cond_max` (unit diagonal kept, so sigma is unchanged). Stores
    `corr_whiten` (n_scenes, na, na) = L^-1, C' = L L^T; identity for scenes
    the file does not list. None = diagonal R."""
    atms_traj['corr_whiten'] = None
    if not path:
        return
    import json
    with open(path) as fh:
        d = json.load(fh)
    chans = list(d['channels'])
    assim = atms_traj['assimilate_channels']
    if [c for c in assim if c not in chans]:
        raise ValueError(f"corr_r {path} lacks assimilated channels {[c for c in assim if c not in chans]}")
    ii = [chans.index(c) for c in assim]
    na = len(assim)
    W = torch.eye(na, dtype=torch.float64).repeat(len(ATMS_SCENES), 1, 1)
    for sc, v in d['scenes'].items():
        C = torch.tensor(v['corr'], dtype=torch.float64)[ii][:, ii]
        lam = torch.linalg.eigvalsh(C)
        dl = max(0.0, float((lam[-1] - cond_max * lam[0]) / (cond_max - 1)))
        Cr = (C + dl * torch.eye(na, dtype=torch.float64)) / (1 + dl)
        W[ATMS_SCENES.index(sc)] = torch.linalg.inv(torch.linalg.cholesky(Cr))
        logger.info(f"{atms_traj['sensor']} corr_r {sc}: cond {float(lam[-1] / lam[0]):.0f} -> "
                    f"{float(torch.linalg.cond(Cr)):.0f} (d {dl:.3f}), channels {assim}")
    atms_traj['corr_whiten'] = W.to(device=device, dtype=torch.float32)


def _atms_sigma(atms_traj, oind, model_equivalent=None):
    """Observation error at slot `oind`: `(22,)` sigma_o, or `(capacity, 22)`
    per ob for 'scene_cloud' (set from `model_equivalent` on the slot's first
    evaluation, see `_setup_atms_oberr`)."""
    if atms_traj['oberr_model'] == 'none':
        return atms_traj['sigma_o']
    if not atms_traj['sigma_set'][oind]:
        if model_equivalent is None:
            raise RuntimeError(f'ATMS sigma for slot {oind} used before the slot was evaluated')
        hb = model_equivalent.detach()
        clw_b = _atms_clw(hb[:, 0], hb[:, 1], atms_traj['obs_Sat_Zenith_Angle'][oind])
        scene = atms_traj['scene'][oind].long()
        ocean = (scene == 0) | (scene == 4)
        C = torch.where(ocean, 0.5 * (atms_traj['clw_obs_tb'][oind] + clw_b), torch.zeros_like(clw_b))
        f = ((C[:, None] - atms_traj['oberr_c0']) / (atms_traj['oberr_c1'] - atms_traj['oberr_c0'])).clamp(0.0, 1.0)
        r_cloud = 1.0 + (atms_traj['oberr_ramp'] - 1.0) * f
        r = torch.where(ocean[:, None], r_cloud, atms_traj['oberr_scene_ratio'][scene.clamp(min=0)])
        atms_traj['sigma_obs'][oind] = atms_traj['sigma_o'] * r
        atms_traj['cloud_sym'][oind] = C
        atms_traj['cloud_ok'][oind] = ~ocean | (C <= atms_traj['cloud_max'])
        atms_traj['sigma_set'][oind] = True
    return atms_traj['sigma_obs'][oind]


def _rho(r, huber_c=None):
    """Jo per normalized departure r: r^2, or the Huber norm (r^2 for
    |r| <= c, 2c|r| - c^2 beyond: the same value and slope at c, linear in
    the tails, so outliers pull with a bounded gradient)."""
    if huber_c is None:
        return r ** 2
    a = r.abs()
    return torch.where(a <= huber_c, r ** 2, 2. * huber_c * a - huber_c ** 2)


def _atms_z(atms_traj, zeta, oind, log_sig):
    """Emissivity control at slot `oind` in PC units: z = zeta * sigma_k
    (sigma_k from the surface head, detached), 0 where `z_active` is off."""
    return zeta[oind] * torch.exp(log_sig.detach()) * atms_traj['z_active'][oind][:, None]


def _resolve_loss_interp_specs(model, exp):
    """Which decoded fields must be linearly time-interpolated between the two
    bracketing 6h model states for a sub-6h obs slot.

    Returns a list of `(base_key, [alias_keys])` -- e.g. `('z', ['geopotential'])`
    -- covering exactly the `loss_variables` config list (base names), plus
    'z' (always: `_compute_ps_observation_diagnostics_at_time` reads
    `decoded['geopotential']` unconditionally -- for its `.device` and the
    obs-space finiteness QC check -- even for `ps_operator='ps'`, whose H(x)
    itself needs only t/q/sp). Defaults `loss_variables` to what the active
    `ps_operator` reads, and errors if an explicit list omits something the
    operator needs or names a non-variable.
    """
    ps_operator = exp.get('ps_operator', 'logpinterp')
    required = _PS_OPERATOR_REQUIRES.get(ps_operator)
    if required is None:
        raise ValueError(f"unknown ps_operator {ps_operator!r}")
    names = exp.get('loss_variables')
    if names is None:
        names = set(required)
    else:
        names = set(names)
        missing = required - names
        if missing:
            raise ValueError(
                f"loss_variables {sorted(names)} is missing {sorted(missing)} "
                f"required by ps_operator={ps_operator!r}"
            )
    names |= {'z'}
    # raobs variable bases ('t'/'u'/'v'/'q') need the same time-interpolated
    # decode -- folded in here rather than a second decode_state call, since
    # raobs observations share psobs's obs-slot grid (see get_raobs).
    if exp.get('raobs_enabled', False):
        names |= {RAOBS_VARIABLE_MODEL_BASE[v] for v in exp['raobs_variables']}
    # microwave sensors read profiles, surface and static fields at the same slots
    if any(exp.get(f'{sensor}_enabled', False) for sensor in MW_SENSORS):
        names |= set(ATMS_MODEL_FIELDS)
    specs = []
    for base in sorted(names):
        if not model.is_known_variable(base):
            raise KeyError(f"loss_variables entry {base!r} is not a model variable or family")
        specs.append((base, list(_DECODE_ALIASES.get(base, []))))
    return specs


def _interp_decoded(decoded_lo, decoded_hi, alpha, interp_specs):
    """Linear time interpolation of just the fields in `interp_specs` between
    two decoded model states. `alpha` in [0, 1]: 0 -> `decoded_lo`, 1 ->
    `decoded_hi`. The result carries the base key and each alias key pointing
    at the same interpolated tensor (the forward operator reads the aliases).
    """
    out = {}
    for base, aliases in interp_specs:
        v = (1.0 - alpha) * decoded_lo[base] + alpha * decoded_hi[base]
        out[base] = v
        for al in aliases:
            out[al] = v
    return out


def _resolve_control_mask(model, exp, n_vars, device):
    """Bool `(n_vars,)` packed-state column mask for `control_variables`
    (config, family/single base names).

    True  -> the latent increment MAY modify this variable at the injection
             step.
    False -> the variable is overwritten there with the uncorrected
             first-step forecast, so the increment's *direct* effect on it is
             exactly zeroed (it can still evolve downstream in response to the
             controlled variables).

    A backend's `diagnostic_columns` (output-only fields, if it declares
    any) are always True, whatever `control_variables` says -- see below.

    Returns `None` when `control_variables` is absent (no masking -- the
    increment is free on every variable, the default behaviour).

    Column lookup (`model.resolve_columns`) checks pressure-level families
    before any single-level/raw-checkpoint fallback -- see the long-window-4dvar-aifsv2 repo's CLAUDE.md for the
    bug that shipped from getting that order backwards, which is exactly
    what `resolve_columns` closes off at the source for every backend behind
    the `LatentForecastModel` interface, not just AIFS.

    `n_vars` (and hence the mask's meaning) is layout-dependent -- callers
    pass `input_encoded.state.shape[1]` for `model.state_layout ==
    'channels_first'` (FCN3) or `input_encoded.state.shape[-1]` for
    `'channels_last'` (AIFS); see `compute_optimal` and `_apply_control_mask`.
    """
    names = exp.get('control_variables')
    if names is None:
        return None
    mask = torch.zeros(n_vars, dtype=torch.bool, device=device)
    for base in names:
        try:
            cols = model.resolve_columns(base)
        except KeyError:
            raise KeyError(f"control_variables entry {base!r} is not a model variable or family")
        for i in cols:
            mask[i] = True
    # Output-only diagnostic columns (backend-declared, e.g. ACE2's h500/
    # TMP850/fluxes) are never reset: they don't feed the next step, so
    # resetting them protects nothing and only replaces the analysis's own
    # diagnosis with the background's -- which made ACE2's h500-based z500
    # "after" line identical to "before" whenever control_variables omitted
    # h500.
    for i in getattr(model, 'diagnostic_columns', ()):
        mask[i] = True
    return mask


@torch.no_grad()
def compute_ps_observation_hx(model, model_state, verif_ic, psobs_traj, exp, grid_interp,
                              interp_specs, step_start=0, raobs_traj=None, mw_trajs=None, wc_state=None):
    """Evaluate a trajectory at every surface-pressure observation slot.
    Post-optimization diagnostics only -- no gradients anywhere here.

    Returns two or three output grids.
      - `model_trajectory`: the decoded model state at every 6h step, from
        `step_start` to `n_steps` inclusive (native rollout cadence).
      - `obs_trajectory`: H(x)/QC at every obs slot `j` with
        `step_lo[j] >= step_start`, using the same time-interpolation as the
        loss (`_interp_decoded` over `interp_specs`). Slots earlier than
        `step_start` are omitted -- the caller pads them.
      - `raobs_trajectory` (only when `raobs_traj` is not None): the same
        shape of thing, per raobs variable -- see `_record_raobs`.
      - `mw_trajectories` (only when `mw_trajs` is not None): per microwave
        sensor, in `mw_trajs` order, `model_equivalent` `(slot, capacity,
        nch)`, `screened`, `used`.

    `step_start`: the 6h-step index `model_state` sits at. 0 for the
    background; `dt_verif // 6` for a latent analysis (valid at
    `window_start + dt_verif` -- see `compute_optimal`'s docstring).

    `wc_state` (an analysis under weak constraint, `exp['wc_state']`): the
    model-error jumps added along the way (see `_wc_step`).

    Entirely backend-agnostic (only uses `model.advance`/`model.decode_state`).
    Does NOT re-prime FCN3's stochastic noise state before running (see
    module docstring's note on which call sites do and don't) -- this
    trajectory's exact noise realization is whatever the model happens to
    carry at call time. A no-op for AIFS.
    """
    n_steps = psobs_traj['n_steps']
    step_lo = psobs_traj['step_lo'].tolist()
    alpha = psobs_traj['alpha'].tolist()
    n_obs = len(step_lo)
    nobs_max = exp['nobs_max']

    slots_by_step = {}
    for j in range(n_obs):
        if step_lo[j] >= step_start:
            slots_by_step.setdefault(step_lo[j], []).append(j)

    diagnostics = {}
    model_outputs = {}
    raobs_diagnostics = {base: {} for base in raobs_traj['variables']} if raobs_traj is not None else None
    mw_diagnostics = [{} for _ in mw_trajs] if mw_trajs is not None else None

    def _record_obs(j, decoded_j):
        innovation, model_equivalent, effective_error, available, used, qc_flag = \
            _compute_ps_observation_diagnostics_at_time(model, decoded_j, verif_ic, psobs_traj, exp, j, grid_interp)
        for name, value in {
            'model_equivalent': model_equivalent,
            'effective_error': effective_error,
            'available': available,
            'used': used,
            'qc_flag': qc_flag,
        }.items():
            diagnostics.setdefault(name, []).append(value.detach().cpu().numpy())
        if raobs_traj is not None:
            raobs_results = _compute_raobs_observation_diagnostics_at_time(
                model, decoded_j, raobs_traj, grid_interp, j
            )
            for base, (innovation_r, model_equivalent_r, effective_error_r, used_r) in raobs_results.items():
                for name, value in {
                    'model_equivalent': model_equivalent_r,
                    'effective_error': effective_error_r,
                    'used': used_r,
                }.items():
                    raobs_diagnostics[base].setdefault(name, []).append(value.detach().cpu().numpy())
        for mw_traj, mw_diag in zip(mw_trajs or [], mw_diagnostics or []):
            _, model_equivalent_a, screened_a, used_a = _compute_atms_observation_diagnostics_at_time(
                model, decoded_j, mw_traj, grid_interp, j
            )
            for name, value in {'model_equivalent': model_equivalent_a, 'screened': screened_a, 'used': used_a}.items():
                mw_diag.setdefault(name, []).append(value.detach().cpu().numpy())

    f_state = model_state
    decoded_prev = model.decode_state(f_state)
    for name, value in decoded_prev.items():
        model_outputs.setdefault(name, []).append(value.detach().cpu().numpy())

    for s in range(step_start + 1, n_steps + 1):
        f_state = _wc_step(model, f_state, s, wc_state)
        decoded_cur = model.decode_state(f_state)
        for name, value in decoded_cur.items():
            model_outputs.setdefault(name, []).append(value.detach().cpu().numpy())
        for j in slots_by_step.get(s - 1, []):
            a = alpha[j]
            if a == 0.0:
                decoded_j = decoded_prev
            elif a == 1.0:
                decoded_j = decoded_cur
            else:
                decoded_j = _interp_decoded(decoded_prev, decoded_cur, a, interp_specs)
            _record_obs(j, decoded_j)
        decoded_prev = decoded_cur

    model_trajectory = {name: np.stack(v, axis=0) for name, v in model_outputs.items()}
    if not diagnostics:
        # No obs slot at or after step_start (only possible if step_start
        # covers the whole window -- e.g. dt_verif >= window length). Return
        # empty arrays so a caller that pads still has the expected keys.
        diagnostics = {k: [] for k in
                       ('model_equivalent', 'effective_error', 'available', 'used', 'qc_flag')}
    obs_trajectory = {
        name: (np.stack(v, axis=0) if v else np.empty((0, nobs_max)))
        for name, v in diagnostics.items()
    }
    raobs_trajectory = None
    if raobs_traj is not None:
        raobs_trajectory = {
            base: {name: np.stack(v, axis=0) for name, v in fields.items()}
            for base, fields in raobs_diagnostics.items()
        }
    mw_trajectories = None
    if mw_trajs is not None:
        mw_trajectories = [{name: np.stack(v, axis=0) for name, v in d.items()} for d in mw_diagnostics]
    return obs_trajectory, model_trajectory, raobs_trajectory, mw_trajectories


# ---------------------------------------------------------------------------
# rollout / loss / optimizer
# ---------------------------------------------------------------------------

def forecast(model, model_state, steps):
    return model.advance(model_state, steps=steps)


def _resolve_weak_constraint(exp, n_steps, timestep_h):
    """Window hours of the weak-constraint model-error jumps
    (`exp.weak_constraint.every_hours`: one every N h, first at N h, last
    before the window end), or [] for strong constraint. A jump at hour h is
    a latent increment injected on the h -> h + 6 h step, exactly like the
    window-start increment on the 0 -> 6 h step."""
    every = int((exp.get('weak_constraint') or {}).get('every_hours', 0) or 0)
    if every <= 0:
        return []
    if every % timestep_h:
        raise ValueError(f'weak_constraint.every_hours ({every}) must be a multiple of the {timestep_h} h model step')
    # the next cycle's background is this analysis advanced to +dt_init; a
    # restart from the saved +dt_verif state advances it without jumps, so
    # no jump may fall before dt_init
    if every < int(exp.get('dt_init', 0)):
        raise ValueError(f"weak_constraint.every_hours ({every}) must be >= dt_init ({exp.get('dt_init')})")
    return list(range(every, n_steps * timestep_h, every))


def _wc_step(model, state, s, wc_state, use_checkpoint=False):
    """One 6h step to window step `s`, adding the weak-constraint jump
    injected on that step if there is one. `wc_state`: None (plain step) or
    {'step_index': {s: k}, 'forcing': (n_jumps, *latent_shape) already
    divided by latent_scale, 'keep_mask': control_variables mask or None}.
    With a mask, the uncontrolled columns take the no-jump step (not
    detached: they still depend on the earlier controls)."""
    k = wc_state['step_index'].get(s) if wc_state else None
    if k is None:
        return model.advance(state, steps=1, use_checkpoint=use_checkpoint)
    new = model.advance(state, steps=1, latent_increment=wc_state['forcing'][k], use_checkpoint=use_checkpoint)
    if wc_state['keep_mask'] is not None:
        ref = model.advance(state, steps=1, use_checkpoint=use_checkpoint)
        new = model.wrap_state(_apply_control_mask(new.state, wc_state['keep_mask'], ref.state, model.state_layout),
                               new.date)
    return new


def advance_with_jumps(model, state, step_start, steps, wc_state):
    """`model.advance` (no grad, no checkpointing) of `state`, sitting at
    window step `step_start`, by `steps` steps, adding the weak-constraint
    jumps of `wc_state` (see `_wc_step`) that fall in that range."""
    with torch.no_grad():
        if not wc_state:
            return model.advance(state, steps=steps, use_checkpoint=False)
        for s in range(step_start + 1, step_start + steps + 1):
            state = _wc_step(model, state, s, wc_state)
    return state


def _wc_file(exp, date, max_epoch):
    return (exp['path_output'] + date + '_wc_forcing_' + exp['window_name'] + 'h' + exp['suffix']
            + '_' + str(max_epoch) + 'it.pt')


def _previous_wc_controls(exp, prev_state, max_epoch, logger):
    """The previous cycle's unit jump controls v and their window hours, for
    `weak_constraint.warm_start`: from the in-process `exp['wc_state']` of the
    cycle dt_init earlier, else from its saved `*_wc_forcing_*.pt` (a restart
    or resubmit). None if neither exists (the first cycle)."""
    prev_date = add_hours(exp['date'], -int(exp['dt_init']))
    if prev_state and prev_state.get('date') == prev_date and prev_state.get('controls') is not None:
        return prev_state['controls'], prev_state['hours']
    f = _wc_file(exp, prev_date, max_epoch)
    if os.path.exists(f):
        st = torch.load(f, map_location='cpu', weights_only=False)
        if st.get('controls') is not None:
            return st['controls'], st['hours']
        logger.warning(f'weak_constraint.warm_start: {f} has no unit controls (written before warm_start)')
    logger.info(f'weak_constraint.warm_start: no jumps from {prev_date} -- prior mean 0 this cycle')
    return None


def _shift_wc_controls(prev, prev_hours, hours, dt):
    """Move the previous window's jump controls `prev` (n_prev, ...) at window
    hours `prev_hours` to this window's `hours`, a window starting `dt` h later:
    jump k sits at the previous window's hour hours[k] + dt (the same valid
    time), linearly interpolated between the previous jumps, with 0 at hour 0
    (the window-start increment is not a jump) and one jump spacing past the
    last jump (no information beyond the previous window)."""
    every = prev_hours[1] - prev_hours[0] if len(prev_hours) > 1 else prev_hours[0]
    knots = [0] + list(prev_hours) + [prev_hours[-1] + every]
    vals = torch.cat([torch.zeros_like(prev[:1]), prev, torch.zeros_like(prev[:1])])
    out = []
    for h in hours:
        t = min(max(h + dt, knots[0]), knots[-1])
        i = max(j for j in range(len(knots) - 1) if knots[j] <= t)
        w = (t - knots[i]) / (knots[i + 1] - knots[i])
        out.append((1 - w) * vals[i] + w * vals[i + 1])
    return torch.stack(out)


def _apply_control_mask(state, keep_mask, ref_state, state_layout):
    """Broadcast `keep_mask` (n_vars,) against the packed state and select
    between `state`/`ref_state`. `state_layout` (`model.state_layout` --
    'channels_first' for FCN3's `(1, n_vars, H, W)` packing, 'channels_last'
    for AIFS's `(1, multi_step, n_points, n_vars)` packing, `n_vars`
    trailing) determines whether the mask needs an explicit reshape or
    already broadcasts as-is."""
    if state_layout == 'channels_first':
        return torch.where(keep_mask.view(1, -1, 1, 1), state, ref_state)
    elif state_layout == 'channels_last':
        return torch.where(keep_mask, state, ref_state)
    raise ValueError(f"unknown state_layout {state_layout!r} (expected 'channels_first' or 'channels_last')")


def compute_loss_4dvar(model, latent_increment, latent_scale, input_state, verif_ic,
                       psobs_traj, epoch, exp, grid_interp, print_every, interp_specs,
                       keep_mask=None, ref_state1=None, jc_ref=None, raobs_traj=None, mw_trajs=None, wc=None):
    """Latent-space 4D-Var cost function, with sub-6h observations.

    `latent_increment` lives in the model's latent space -- AIFS's hidden-
    mesh encoder-output space, or FCN3's processed-latent space (see
    `model.latent_shape`). It is injected once, as a forcing on the first
    6h rollout step -- there is no decoder path that reconstructs a
    corrected t=0 physical state, so:

      - the window-start (t=0) obs term is scored against the UNMODIFIED
        background (no gradient path to `latent_increment`);
      - obs in the first 6h bracket `background(0) <-> analysis(6h)`; only the
        `analysis(6h)` end depends on the increment, weighted by `alpha`.

    `keep_mask` / `ref_state1` (both from `compute_optimal`, present iff
    `control_variables` is configured): after the injection step, packed-state
    columns with `keep_mask == False` are overwritten with `ref_state1` (the
    detached uncorrected first-step forecast), so the increment's direct
    effect -- and hence its gradient -- is confined to the controlled
    variables. Non-controlled variables still evolve freely on steps 2..N.
    Broadcast against the packed state via `_apply_control_mask`, which
    branches on `model.state_layout` (FCN3 packs `n_vars` at a different
    tensor dim than AIFS/Aurora, which share the same 'channels_last'
    layout -- see module docstring).

    Sub-6h obs: the rollout is one 6h model step at a time (`n_steps` total).
    Obs slot `j` is bracketed by steps `step_lo[j]` and `step_lo[j]+1` with
    in-step fraction `alpha[j]` (from `get_psobs`); the decoded fields named
    in `interp_specs` are linearly interpolated in time between those two
    steps before the ordinary spatial interp + forward operator + QC
    (`_compute_ps_observation_diagnostics_at_time`). `dt_obs == dt_verif`
    reduces to one slot per step with every `alpha == 0`.

    `exp['checkpoint_stride']` (default 1, i.e. every step -- unchanged
    behavior): only every `checkpoint_stride`-th step is wrapped in
    `torch.utils.checkpoint` (trading memory for speed -- a non-checkpointed
    step keeps its activations for backward instead of recomputing them). Set
    to 0 to disable checkpointing entirely for a memory-rich GPU/short
    window; validate with the usual "does AdamW still monotonically reduce
    the loss" check before trusting a non-default value on a new window
    length.

    `raobs_traj` (from `get_raobs`, present iff `exp['raobs_enabled']` is
    True): radiosonde temperature/wind/humidity terms, added to `J` the same
    way as the ps term, using the SAME obs-slot grid/time-interpolated
    `decoded_j` (see get_raobs's docstring for why no second rollout is
    needed) -- `None` (the default) assimilates surface pressure only,
    unchanged from before raobs existed.

    `mw_trajs` (from `get_mwobs`: one per enabled microwave sensor): the
    radiance terms, sum over used obs and the assimilated channels of
    rho((y_o - BC - H(x)) / sigma_o), on the same slots and `decoded_j`;
    rho(r) = r^2, or the Huber norm with the sensor's `huber_c`.

    `wc` (weak constraint, from `compute_optimal`, present iff
    `exp.weak_constraint` is on): `{'step_index', 'forcing', 'keep_mask'}`
    as in `_wc_step` -- model-error jumps, latent increments injected on
    later steps the same way as the window-start one. Their penalty is
    AdamW weight decay on their own optimizer group, like the window-start
    increment's, so they add no term to `J` here.

    `jc_ref` (from `compute_optimal`, present iff `exp['jc_ps_weight']` is
    set): `(field, bg_p_6h, bg_p_12h, weights, factor)` -- the pressure
    field used ('sp' where the backend has it, else 'msl' -- FCN3), the
    uncorrected background's values of it (Pa) one and two steps in, area
    weights summing to 1, and factor = kappa * N_obs (N_obs = obs available
    in the window). Adds a balance penalty on how fast the increment changes
    over the first step after injection, which damps the transient
    mass-field part of the increment the model otherwise sheds within ~a
    day:
        Jc = kappa * N_obs * sum_i weights_i * Delta_i^2,
        Delta = (an - bg)(t+12h) - (an - bg)(t+6h)   [pressure, hPa]
    i.e. the analysis pressure tendency minus the background's over the same
    step, so tides/real tendencies common to both cancel. The grid mean
    makes kappa independent of resolution; the N_obs factor keeps Jc / Jo
    roughly fixed as obs counts change.

    Callers (`compute_optimal`) must re-prime the model's stochastic noise
    state (`model._prime_noise()`) immediately before calling this, once per
    epoch -- not once before the epoch loop -- so every epoch evaluates the
    same underlying noise realization (see this module's docstring; a no-op
    for AIFS); this function itself does not re-prime, since it has no way
    to know whether it is being called as part of a fresh epoch or a
    diagnostic re-evaluation.
    """
    scaled_increment = latent_increment / latent_scale
    dt_verif = exp['dt_verif']
    n_verif = exp['n_verif']
    dt_obs = exp['dt_obs']
    timestep_h = int(round(model.timestep.total_seconds() / 3600))
    n_steps = n_verif * dt_verif // timestep_h
    checkpoint_stride = int(exp.get('checkpoint_stride', 1))

    loss_bases = [b for b, _ in interp_specs]

    step_lo = psobs_traj['step_lo'].tolist()
    alpha = psobs_traj['alpha'].tolist()
    n_obs = len(step_lo)
    slots_by_step = {}
    for j in range(n_obs):
        slots_by_step.setdefault(step_lo[j], []).append(j)

    do_print = print_every > 0 and (epoch % print_every == 0 or epoch == 1)
    Jol = [None] * n_obs
    Jraobs = [None] * n_obs if raobs_traj is not None else None
    Jmw = [[None] * n_obs for _ in mw_trajs] if mw_trajs is not None else None

    def _obs_term(j, decoded_j):
        innovation, model_equivalent, effective_error, available, used, qc_flag = \
            _compute_ps_observation_diagnostics_at_time(model, decoded_j, verif_ic, psobs_traj, exp, j, grid_interp)
        innov = torch.where(used, innovation, torch.zeros_like(innovation))
        oberr_stdev = torch.where(used, effective_error, torch.full_like(effective_error, 1.e10))
        Jt = ((innov / oberr_stdev) ** 2).sum()
        if do_print:
            tag = ' [background, uncorrected]' if (step_lo[j] == 0 and alpha[j] == 0.0) else ''
            print(f"epoch {epoch}, oind {j} (t+{j * dt_obs}h): J {Jt.item()}, "
                  f"{int(used.sum().item())} obs used (out of {int(psobs_traj['n_valid'][j])}){tag}")
        return Jt

    def _raobs_term(j, decoded_j):
        results = _compute_raobs_observation_diagnostics_at_time(model, decoded_j, raobs_traj, grid_interp, j)
        Jt = 0.
        counts = {}
        for base, (innovation, model_equivalent, effective_error, used) in results.items():
            innov = torch.where(used, innovation, torch.zeros_like(innovation))
            err = torch.where(used, effective_error, torch.full_like(effective_error, 1.e10))
            Jt = Jt + ((innov / err) ** 2).sum()
            counts[base] = int(used.sum().item())
        if do_print:
            print(f"epoch {epoch}, oind {j} (t+{j * dt_obs}h): raobs J {Jt.item() if torch.is_tensor(Jt) else Jt}, "
                  f"obs used {counts}")
        return Jt

    def _atms_term(atms_traj, j, decoded_j):
        zeta = atms_traj.get('zeta')
        if zeta is None:
            innovation, _, screened, used = _compute_atms_observation_diagnostics_at_time(
                model, decoded_j, atms_traj, grid_interp, j)
        else:
            # z from the previous evaluation's sigma_k is not needed: sigma_k
            # depends on x only, so one surface-head pass gives it, then the
            # sounding channels see z detached and the window channels see x
            # detached (plan E2)
            _, _, _, _, (x, log_sig) = _compute_atms_observation_diagnostics_at_time(
                model, decoded_j, atms_traj, grid_interp, j, return_inputs=True)
            z = _atms_z(atms_traj, zeta, j, log_sig)
            innovation, _, screened, used = _compute_atms_observation_diagnostics_at_time(
                model, decoded_j, atms_traj, grid_interp, j, z=z.detach())
        a = atms_traj['assimilate_idx']
        u = used[:, a]
        innov = torch.where(u, innovation[:, a], torch.zeros_like(innovation[:, a]))
        sigma = _atms_sigma(atms_traj, j)
        huber_c = atms_traj['huber_c']
        rn = innov / sigma[..., a]
        W = atms_traj.get('corr_whiten')
        if W is not None:
            # correlated R: whiten the normalized departures of obs used in
            # all assimilated channels (r^T C^-1 r = |L^-1 r|^2); obs with
            # some channels rejected keep the diagonal R; Huber acts on the
            # whitened components
            full = u.all(-1)
            w = torch.einsum('nij,nj->ni', W[atms_traj['scene'][j].long().clamp(min=0)], rn)
            rn = torch.where(full[:, None], w, rn)
        Jt = _rho(rn, huber_c).sum()
        if zeta is not None:
            e = atms_traj['emis_idx']
            tb_e, _, _ = atms_traj['op'](x.detach(), z)
            act = atms_traj['z_active'][j]
            ws = atms_traj['window_state'][j]
            if ws.any():
                # window_state_scenes: these obs see x and z both attached
                tb_s, _, _ = atms_traj['op'](x, z)
                tb_e = torch.where(ws[:, None], tb_s, tb_e)
                if 'cloud_ok' in atms_traj:
                    ws = ws & atms_traj['cloud_ok'][j]
                act = act | ws
            innov_e = atms_traj['y'][j][:, e] - tb_e[:, e]
            sig_e = sigma[..., e]
            ok = screened[:, None] & act[:, None] & (innov_e.abs() <= atms_traj['bg_check'] * sig_e).detach()
            Je = _rho(torch.where(ok, innov_e, torch.zeros_like(innov_e)) / sig_e, huber_c).sum()
            Jt = Jt + Je
        if do_print:
            msg = (f"epoch {epoch}, oind {j} (t+{j * dt_obs}h): {atms_traj['sensor']} J {Jt.item()}, "
                   f"obs used {dict(zip(atms_traj['assimilate_channels'], u.sum(0).tolist()))} "
                   f"of {int(screened.sum().item())} screened")
            if W is not None:
                msg += f"; corr_r on {int(full.sum())} obs"
            if zeta is not None:
                msg += f"; window-channel Jo {Je.item():.1f} over {int(ok.any(-1).sum())} obs"
            print(msg)
        return Jt

    jc_sp = {}
    f_state = input_state
    decoded_prev = model.decode_state(f_state, only=loss_bases)   # step 0 = uncorrected background
    for s in range(1, n_steps + 1):
        use_ckpt = (checkpoint_stride > 0) and (s % checkpoint_stride == 0)
        if s > 1:
            f_state = _wc_step(model, f_state, s, wc, use_checkpoint=use_ckpt)
        else:
            f_state = model.advance(f_state, steps=1, latent_increment=scaled_increment, use_checkpoint=use_ckpt)
        if s == 1 and keep_mask is not None:
            # Confine the increment's direct effect to the controlled
            # columns; the rest revert to the uncorrected forecast
            # (ref_state1 is detached, so those columns carry no gradient to
            # `latent_increment`).
            f_state = model.wrap_state(
                _apply_control_mask(f_state.state, keep_mask, ref_state1.state, model.state_layout),
                f_state.date,
            )
        decoded_cur = model.decode_state(f_state, only=loss_bases)
        if jc_ref is not None and s in (1, 2):
            jc_sp[s] = model.decode_state(f_state, only=[jc_ref[0]])[jc_ref[0]]
        for j in slots_by_step.get(s - 1, []):
            a = alpha[j]
            if a == 0.0:
                decoded_j = decoded_prev
            elif a == 1.0:
                decoded_j = decoded_cur
            else:
                decoded_j = _interp_decoded(decoded_prev, decoded_cur, a, interp_specs)
            Jol[j] = _obs_term(j, decoded_j)
            if raobs_traj is not None:
                Jraobs[j] = _raobs_term(j, decoded_j)
            for k, mw_traj in enumerate(mw_trajs or []):
                Jmw[k][j] = _atms_term(mw_traj, j, decoded_j)
        decoded_prev = decoded_cur

    J = sum(Jol)
    if raobs_traj is not None:
        J = J + sum(Jraobs)
    for k, mw_traj in enumerate(mw_trajs or []):
        J = J + sum(Jmw[k])
        if mw_traj.get('zeta') is not None:
            Jz = ((mw_traj['zeta'] ** 2) * mw_traj['z_active'][..., None]).sum()
            J = J + Jz
            if do_print:
                print(f"epoch {epoch}, {mw_traj['sensor']} Jz = {Jz.item()}")
    if jc_ref is not None:
        jc_field, bg1, bg2, jc_w, jc_factor = jc_ref
        delta = ((jc_sp[2] - bg2) - (jc_sp[1] - bg1)) / 100.0   # hPa
        Jc = jc_factor * (jc_w * delta ** 2).sum()
        J = J + Jc
        if do_print:
            rms = torch.sqrt((jc_w * delta ** 2).sum() / jc_w.sum()).item()
            print(f"epoch {epoch}, Jc = {Jc.item()} (first-step {jc_field} increment change {rms:.4f} hPa rms)")
    if do_print:
        print(f"epoch {epoch}, Jtot = {J.item()}")

    Jol_np = np.array([j.item() for j in Jol])
    return J, Jol_np


def load_optstate_background(exp, model, input_encoded, logger):
    """The background state saved in an `opt_resume` file, in place of the
    freshly built `input_encoded`: the 48-h spin-up forecast that builds the
    first background is not bit-reproducible on GPU (Jo of the uncorrected
    background varies ~0.4% between processes), so a resumed optimization
    must reuse the saved one to continue the same problem."""
    st = torch.load(exp['opt_resume'], map_location='cpu', weights_only=False)
    if 'background' not in st:
        logger.warning(f"{exp['opt_resume']} has no saved background; resuming on a rebuilt one")
        return input_encoded
    if st['background_date'] != input_encoded.date:
        raise ValueError(f"opt_resume background is dated {st['background_date']}, this run's is {input_encoded.date}")
    state = st['background'].to(input_encoded.state.device, input_encoded.state.dtype)
    d = float((state - input_encoded.state).abs().max())
    logger.info(f"opt_resume: using the saved background (max |diff| vs the rebuilt one {d:.4g})")
    return model.prime_from_state(model.wrap_state(state, input_encoded.date))


def compute_optimal(exp, model, input_encoded, verif_ic, psobs_traj, grid_interp, logger, raobs_traj=None,
                    mw_trajs=None):
    """Optimize a latent-space increment against the ps obs.

    No backend's encoder/decoder is used to produce a corrected t=0
    physical state from a latent-space increment -- it is only ever
    injected as a one-time forcing at the model's first forward step, so
    the earliest a corrected physical state exists is
    `input_encoded.date + dt_verif hours`. (AIFS and FCN3 were both
    checked directly for whether an earlier injection point could exist at
    all -- AIFS's decoder structurally cannot, since it only ever produces
    a residual onto the most recent input state, never a t=0
    reconstruction; FCN3's t=0 reconstruction path was tested empirically
    and found badly damped/biased, ~7-9x worse than a real 6h step, since
    `decode()` was never trained downstream of anything but the processor
    step -- see the long-window-4dvar-fcstnetv3 repo's CLAUDE.md. Aurora has not been separately investigated;
    the same `t+timestep` injection point is used uniformly for all three
    backends by design, not because it has been confirmed to be the only
    option for each one.) This therefore returns a state dated at
    `input_encoded.date + dt_verif`, NOT at `input_encoded.date` --
    callers
    (`save_trajectory_diagnostics`, the driver's cycling loop,
    `printz500err`) all use its `.date` rather than assuming it matches the
    background's. Also saves `*_latent_increment_*.pt` for offline
    inspection of the raw increment.
    """
    logger.info('starting optimization (latent control space)...')
    lr = exp['learn_rate']
    wd = exp['weight_decay']
    max_epoch = exp['max_epoch']
    print_every = exp.get('print_every', -1)
    beta1 = exp.get('beta1',0.9)
    beta2 = exp.get('beta2',0.99)
    latent_scale = float(exp.get('latent_scale', 1.0))

    # Which decoded fields must be time-interpolated for a sub-6h obs slot
    # (dt_obs < 6h). Resolved once here, not per epoch.
    interp_specs = _resolve_loss_interp_specs(model, exp)
    logger.info('loss time-interp fields: ' + str([b for b, _ in interp_specs]))

    device = input_encoded.state.device
    increment = torch.zeros(model.latent_shape, device=device, requires_grad=True)

    # weak constraint (exp.weak_constraint): model-error jumps, one latent
    # increment per jump hour, each injected like the window-start one
    timestep_h = int(round(model.timestep.total_seconds() / 3600))
    wc_hours = _resolve_weak_constraint(exp, exp['n_verif'] * exp['dt_verif'] // timestep_h, timestep_h)
    wc_steps = [h // timestep_h + 1 for h in wc_hours]
    prev_wc_state = exp.get('wc_state')
    exp['wc_state'] = None
    exp['wc_prior_state'] = None
    eta = None
    if wc_hours:
        eta = torch.zeros((len(wc_hours),) + tuple(model.latent_shape), device=device, requires_grad=True)

    # control_prior: explicit Gaussian prior -- `increment` and `eta` are unit-variance controls
    # u, v with injected delta = sigma_b * u, eta_k = sigma_q * v_k (injected latent units, as
    # fit_latent_to_era5.py sizes them), J += sum u^2 + sum v^2 (the factor-2 convention of Jo),
    # and no weight decay. With b_cov / q_cov (latent_eof.py output: channel eigvecs, eigvals),
    # each node's increment is correlated across latent channels: delta_n = sigma_b * L u_n,
    # L = E diag(sqrt(lam / mean(lam))) -- unit mean variance, so sigma_b stays the rms per
    # element. to_b / to_q map the optimized tensors to the latent_scale'd increments the rest
    # of this function works with.
    prior = exp.get('control_prior') or {}
    to_b = to_q = lambda t: t  # noqa: E731
    if prior:
        def _prior_map(sigma, cov_file):
            if not cov_file:
                return lambda t: t * (sigma * latent_scale)
            ev = np.load(cov_file)
            lam = torch.as_tensor(ev['eigvals'], dtype=torch.float64).clamp(min=0)
            L = torch.as_tensor(ev['eigvecs'], dtype=torch.float64) * (lam / lam.mean()).sqrt()
            Lt = (L.T * (sigma * latent_scale)).to(device=device, dtype=torch.float32).contiguous()
            return lambda t: t @ Lt
        to_b = _prior_map(float(prior['sigma_b']), prior.get('b_cov'))
        if eta is not None:
            to_q = _prior_map(float(prior['sigma_q']), prior.get('q_cov'))
        logger.info(f"control_prior: sigma_b {prior['sigma_b']} ({prior.get('b_cov') or 'diagonal'})"
                    + (f", sigma_q {prior['sigma_q']} ({prior.get('q_cov') or 'diagonal'})" if eta is not None else '')
                    + ' (no weight decay)')

    # weak_constraint.warm_start (needs control_prior): the jumps' prior mean, and first guess,
    # is the previous cycle's jumps moved to the same valid times, Jq = sum (v - v_prior)^2
    v_prior = None
    if eta is not None and (exp['weak_constraint'] or {}).get('warm_start'):
        if not prior:
            raise ValueError('weak_constraint.warm_start needs control_prior (a prior mean for the jumps)')
        prev = _previous_wc_controls(exp, prev_wc_state, exp['max_epoch'], logger)
        if prev is not None:
            v_prior = _shift_wc_controls(prev[0].to(device), prev[1], wc_hours, int(exp['dt_init']))
            with torch.no_grad():
                eta.copy_(v_prior)
            logger.info('weak_constraint.warm_start: prior-mean jump rms '
                        + ', '.join(f'{h} h {to_q(v_prior[k]).pow(2).mean().sqrt().item():.4g}'
                                    for k, h in enumerate(wc_hours)))
            exp['wc_prior_state'] = {'step_index': {s: k for k, s in enumerate(wc_steps)},
                                     'forcing': (to_q(v_prior) / latent_scale).detach(), 'keep_mask': None}

    # control_variables: restrict which packed-state columns the increment may
    # modify at the injection step (see _resolve_control_mask /
    # compute_loss_4dvar). `ref_state1` -- the detached uncorrected first-step
    # forecast the masked-out columns revert to -- is built once here.
    # n_vars's position in the packed state depends on model.state_layout
    # (dim 1 for FCN3's 'channels_first', the trailing dim for AIFS's and
    # Aurora's shared 'channels_last').
    n_vars = (input_encoded.state.shape[1] if model.state_layout == 'channels_first'
              else input_encoded.state.shape[-1])
    keep_mask = _resolve_control_mask(model, exp, n_vars, device)
    if exp['wc_prior_state'] is not None:
        exp['wc_prior_state']['keep_mask'] = keep_mask
    ref_state1 = None
    if keep_mask is not None:
        n_ctl = int(keep_mask.sum())
        logger.info(f'control_variables: {list(exp["control_variables"])} '
                    f'-> {n_ctl}/{keep_mask.numel()} packed-state columns controlled')
        if not any(b in exp['control_variables'] for b, _ in interp_specs):
            logger.warning('control_variables contains none of the forward-operator fields '
                           f'{[b for b, _ in interp_specs]} -- the first-6h obs terms will have '
                           'no gradient to the increment (only downstream-coupled terms will)')
        model._prime_noise()  # no-op for AIFS
        with torch.no_grad():
            ref_state1 = model.advance(input_encoded, steps=1, use_checkpoint=False)

    # jc_ps_weight (kappa, hPa^-2): balance penalty on the first-step change
    # of the pressure increment -- see compute_loss_4dvar's docstring. Needs
    # the uncorrected background's pressure one and two steps in (no
    # gradient), computed once here.
    jc_ref = None
    if exp.get('jc_ps_weight'):
        # Pressure field for the penalty: surface pressure where the backend
        # decodes it (ACE2, AIFS, Aurora), else mean sea-level pressure
        # (FCN3, which has no surface-pressure channel). Both in Pa.
        if model.is_known_variable('sp'):
            jc_field = 'sp'
        elif model.is_known_variable('msl'):
            jc_field = 'msl'
        else:
            raise ValueError("jc_ps_weight needs a surface ('sp') or mean-sea-level ('msl') pressure field")
        model._prime_noise()
        with torch.no_grad():
            bg1 = model.advance(input_encoded, steps=1, use_checkpoint=False)
            bg2 = model.advance(bg1, steps=1, use_checkpoint=False)
            p1 = model.decode_state(bg1, only=[jc_field])[jc_field].detach()
            p2 = model.decode_state(bg2, only=[jc_field])[jc_field].detach()
        # Jc = kappa * N_obs * mean_w(Delta^2): area-weighted grid mean
        # (resolution-independent) scaled by the number of obs AVAILABLE in
        # the window (fixed for the whole optimization, unlike the used
        # count), so Jc / Jo stays ~constant as obs counts change.
        areaw = torch.as_tensor(model.area_weights, dtype=p1.dtype, device=p1.device)
        jc_w = areaw / areaw.sum()
        n_obs = int(psobs_traj['n_valid'].sum())
        jc_factor = float(exp['jc_ps_weight']) * n_obs
        logger.info(f"jc_ps_weight: {exp['jc_ps_weight']} hPa^-2 x N_obs={n_obs} on '{jc_field}' -- "
                    f"first-step pressure balance penalty on (a per-point sigma of "
                    f"{(model.n_points / jc_factor) ** 0.5:.4g} hPa on this grid)")
        jc_ref = (jc_field, p1, p2, jc_w, jc_factor)

    groups = [{'params': [increment]} if not prior else {'params': [increment], 'weight_decay': 0.0}]
    if eta is not None:
        # the jumps' penalty (Q) is their own AdamW weight decay, as the
        # window-start increment's (B) is the window's weight_decay -- or, under
        # control_prior, the explicit sum v^2
        wc_cfg = exp['weak_constraint']
        groups.append({'params': [eta], 'lr': lr * float(wc_cfg.get('learn_rate_factor', 1.0)),
                       'weight_decay': 0.0 if prior else float(wc_cfg.get('weight_decay', wd))})
        logger.info(f"weak constraint: {len(wc_hours)} model-error jumps at window hours {wc_hours}, "
                    f"lr {groups[-1]['lr']}, weight_decay {groups[-1]['weight_decay']} "
                    f"(window-start increment: lr {lr}, weight_decay {wd})")
    zeta_groups = []  # optimizer group indices of the zeta controls
    zeta = {}  # sensor -> per-ob emissivity control, one optimizer group each
    for mw_traj in mw_trajs or []:
        if mw_traj.get('emis_control') != 'per_obs':
            continue
        # emissivity control in prior-std units (z = zeta * sigma_k, Jz = sum zeta^2);
        # window-local: a fresh zero zeta every cycle
        sensor = mw_traj['sensor']
        K = mw_traj['op'].K
        zeta[sensor] = torch.zeros(mw_traj['y'].shape[:2] + (K,), device=device, requires_grad=True)
        mw_traj['zeta'] = zeta[sensor]
        mw_traj['zeta_eval'] = None
        groups.append({'params': [zeta[sensor]], 'lr': float(exp.get(f'{sensor}_z_learn_rate', 0.1)),
                       'weight_decay': 0.0})
        zeta_groups.append(len(groups) - 1)
        logger.info(f"{sensor} emissivity control: per-ob zeta ({K} PCs) on {int(mw_traj['z_active'].sum())} "
                    f"land/snow/ice obs, lr {groups[-1]['lr']}, window channels {mw_traj['emis_channels']}")
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(beta1,beta2))

    # linear warmup/cosine decay
    # (parameters: warmup_steps, start_factor, end_factor)
    total_steps = exp['max_epoch']
    warmup_steps = exp['warmup_steps']
    start_factor = exp['start_factor']
    eta_min      = lr*exp['end_factor']
    decay_steps = total_steps - warmup_steps
    # Create the individual schedulers
    # Linear Warmup: starts from lr * start_factor and scales up to lr
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
     optimizer,
     start_factor=start_factor,
     end_factor=1.0,
     total_iters=warmup_steps
    )
    # Cosine Decay: decays from lr down to eta_min over decay_steps
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
     optimizer,
     T_max=decay_steps,
     eta_min=eta_min
    )
    # Combine them sequentially
    # milestones indicates the step index at which to switch schedulers
    scheduler = torch.optim.lr_scheduler.SequentialLR(
     optimizer,
     schedulers=[warmup_scheduler, decay_scheduler],
     milestones=[warmup_steps]
    )

    lsave = []
    loss_min = 1e19
    epoch_min = 1
    increment_best = None
    eta_best = None
    zeta_best = {sensor: None for sensor in zeta}

    # '_loss_latent_' kept (not '_loss_') for continuity with existing
    # loss-history file naming.
    json_file = exp['path_output'] + exp['date'] + '_loss_latent_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(max_epoch) + 'it.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []

    epoch = 0
    # Optimizer checkpoints (exp.opt_checkpoint_every > 0): the full state --
    # current and best increment, AdamW moments, scheduler, epoch, history --
    # saved every N epochs and after the last one, as
    # <date>_optstate_<window>h<suffix>_e<epoch>.pt. exp.opt_resume: path to
    # such a file continues that optimization (same date) up to this run's
    # max_epoch. With the same max_epoch it picks up the saved schedule
    # exactly; with a larger one the extra epochs run at a constant learning
    # rate: exp.opt_resume_learn_rate if set, else the rate the saved run
    # ended at (the cosine schedule is not re-stretched, which would jump
    # the rate back up).
    ckpt_every = int(exp.get('opt_checkpoint_every', 0) or 0)
    ckpt_base = exp['path_output'] + exp['date'] + '_optstate_' + exp['window_name'] + 'h' + exp['suffix']
    if exp.get('opt_resume'):
        st = torch.load(exp['opt_resume'], map_location=device, weights_only=False)
        if st['date'] != exp['date'] or st['window_name'] != exp['window_name']:
            raise ValueError(f"opt_resume {exp['opt_resume']} is for {st['date']} / {st['window_name']}h, "
                             f"this run is {exp['date']} / {exp['window_name']}h")
        with torch.no_grad():
            increment.copy_(st['increment'])
        optimizer.load_state_dict(st['optimizer'])
        epoch = st['epoch']
        loss_min, epoch_min, lsave, history = st['loss_min'], st['epoch_min'], st['lsave'], st['history']
        increment_best = st['increment_best'].to(device) if st['increment_best'] is not None else None
        if eta is not None:
            if st.get('eta') is None or st['eta'].shape != eta.shape:
                raise ValueError(f"opt_resume {exp['opt_resume']} has no weak-constraint jumps matching "
                                 f"this run's {len(wc_hours)}")
            with torch.no_grad():
                eta.copy_(st['eta'])
            eta_best = st['eta_best'].to(device) if st['eta_best'] is not None else None
        saved_zeta, saved_best = st.get('zeta'), st.get('zeta_best')
        if torch.is_tensor(saved_zeta):  # written before AMSU-A: ATMS only
            saved_zeta, saved_best = {'atms': saved_zeta}, {'atms': saved_best}
        for sensor in zeta:
            if (saved_zeta or {}).get(sensor) is not None:
                with torch.no_grad():
                    zeta[sensor].copy_(saved_zeta[sensor])
                best = (saved_best or {}).get(sensor)
                zeta_best[sensor] = best.to(device) if best is not None else None
        if max_epoch == st['max_epoch']:
            scheduler.load_state_dict(st['scheduler'])
        else:
            lr_cont = float(exp.get('opt_resume_learn_rate') or st['last_lr'])
            for g in optimizer.param_groups:
                g['lr'] = lr_cont
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        logger.info(f"resumed optimization from {exp['opt_resume']} at epoch {epoch} "
                    f"(saved max_epoch {st['max_epoch']}, best loss {loss_min} at epoch {epoch_min}), "
                    f"continuing to {max_epoch} at lr {optimizer.param_groups[0]['lr']}")

    def save_optstate():
        f = f'{ckpt_base}_e{epoch}.pt'
        torch.save({'date': exp['date'], 'window_name': exp['window_name'], 'epoch': epoch,
                    'max_epoch': max_epoch, 'increment': increment.detach().clone(),
                    'increment_best': increment_best, 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'last_lr': optimizer.param_groups[0]['lr'],
                    'loss_min': loss_min, 'epoch_min': epoch_min, 'lsave': lsave, 'history': history,
                    'latent_scale': latent_scale, 'background': input_encoded.state.detach().cpu(),
                    'zeta': {k: v.detach().clone() for k, v in zeta.items()}, 'zeta_best': dict(zeta_best),
                    'eta': eta.detach().clone() if eta is not None else None, 'eta_best': eta_best,
                    'background_date': input_encoded.date}, f)
        logger.info(f'saved optimizer state at epoch {epoch} to {f}')

    while epoch < max_epoch:
        epoch += 1
        # Re-prime FCN3's stochastic noise state at the START of every
        # epoch, not just once before the loop -- confirmed necessary (not
        # just tidy) by the FCN3 model-backend smoke test: without this,
        # every epoch evaluates a different, uncorrelated noise draw and
        # AdamW chases a moving target instead of optimizing a well-defined
        # loss surface for the fixed `increment`. No-op for AIFS.
        model._prime_noise()
        optimizer.zero_grad()
        wc = None
        if eta is not None:
            wc = {'step_index': {s: k for k, s in enumerate(wc_steps)}, 'forcing': to_q(eta) / latent_scale,
                  'keep_mask': keep_mask}
        loss, all_loss = compute_loss_4dvar(model, to_b(increment), latent_scale, input_encoded, verif_ic, psobs_traj, epoch, exp, grid_interp, print_every, interp_specs, keep_mask, ref_state1, jc_ref, raobs_traj,
                                            mw_trajs=mw_trajs, wc=wc)
        Jb = Jq = None
        if prior:
            Jb = (increment ** 2).sum()
            loss = loss + Jb
            if eta is not None:
                Jq = ((eta if v_prior is None else eta - v_prior) ** 2).sum()
                loss = loss + Jq

        if not torch.isfinite(loss):
            logger.warning(f'epoch={epoch}: non-finite loss ({loss.item()}), stopping optimization early')
            break

        loss.backward()

        # A finite loss does not guarantee a finite gradient -- confirmed
        # 2026-09-18 (Aurora, 16-step window): a NaN gradient w.r.t.
        # `increment` survives `clip_grad_norm_` unchanged (NaN propagates
        # through the norm computation, so the rescale factor is NaN too),
        # `optimizer.step()` then corrupts `increment` to NaN, and the
        # corrupted increment's downstream obs terms are correctly QC-
        # rejected as non-finite (`interpolation_failed`) rather than
        # producing a NaN loss -- so the *loss* itself looks deceptively
        # finite and lower than before, and the corrupted increment gets
        # saved as "new best" rather than caught here. See integrate_multiple_backends.md
        # "Aurora 16-step divergence" for the full root-cause writeup.
        #
        # Originally a skip-and-continue (warn, zero the grad, proceed to
        # the next epoch) rather than fatal -- changed 2026-09-21 after a
        # real aifs-single-1.1 5-day/100-epoch run hit this on 4+
        # consecutive epochs: skip-and-continue silently burned through
        # the rest of the epoch budget (and would have done so again every
        # subsequent cycle) with zero further progress, then exited 0 as
        # if the run had succeeded. Every prior occurrence of this guard
        # firing in this codebase's history (see the Aurora writeup above)
        # was likewise a persistent, deterministic condition, not a
        # transient one-off recoverable by skipping a single step -- so
        # there is no known case where skip-and-continue actually helped.
        # Raising here makes that failure visible (nonzero exit, full
        # traceback) instead of silently masking a stalled optimization.
        if not torch.isfinite(increment.grad).all() or (eta is not None and not torch.isfinite(eta.grad).all()):
            raise RuntimeError(
                f'epoch={epoch}: non-finite gradient in increment -- stopping. '
                f'This indicates a real numerical instability (e.g. an overly '
                f'aggressive learn_rate/window-length for this backend/checkpoint), '
                f'not a transient blip -- see the "non-finite gradient" comment above.'
            )

        history.append({
            'time': time.ctime(),
            'epoch': epoch,
            'loss': float(loss.item()),
            'all_loss': all_loss.tolist(),
            **({'Jb': float(Jb.item())} if Jb is not None else {}),
            **({'Jq': float(Jq.item())} if Jq is not None else {}),
        })
        with open(json_file, 'w') as f:
            json.dump(history, f, indent=4)

        if loss.item() < loss_min:
            logger.info('new best optimal...')
            lsave.append(loss.item())
            loss_min = loss.item()
            epoch_min = epoch
            increment_best = increment.detach().clone()
            if eta is not None:
                eta_best = eta.detach().clone()
            for sensor, z in zeta.items():
                zeta_best[sensor] = z.detach().clone()
        else:
            lsave.append(lsave[-1])

        torch.nn.utils.clip_grad_norm_([increment], max_norm=1.0)
        if eta is not None:
            torch.nn.utils.clip_grad_norm_([eta], max_norm=1.0)

        optimizer.step()
        scheduler.step()
        # zeta keeps a constant rate: its per-ob problem is small and is re-fit from zero each cycle
        for k in zeta_groups:
            optimizer.param_groups[k]['lr'] = groups[k]['initial_lr']
        logger.info(f'epoch={epoch}, lr={scheduler.get_last_lr()[0]}, loss={loss.item()}'
                    + (f', Jb={Jb.item():.6g}' if Jb is not None else '')
                    + (f', Jq={Jq.item():.6g}' if Jq is not None else ''))
        if ckpt_every > 0 and (epoch % ckpt_every == 0 or epoch == max_epoch):
            save_optstate()

    if eta is not None:  # the unit controls, for the next cycle's warm_start
        eta_controls = (eta_best if eta_best is not None
                        else v_prior if v_prior is not None else torch.zeros_like(eta)).detach().cpu()
    if prior:  # unit controls -> the latent_scale'd increments everything downstream expects
        increment_best = to_b(increment_best) if increment_best is not None else None
        eta_best = to_q(eta_best) if eta_best is not None else None
    if increment_best is not None:
        pt_file = exp['path_output'] + exp['date'] + '_latent_increment_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(max_epoch) + 'it.pt'
        torch.save({'increment': increment_best, 'latent_scale': latent_scale}, pt_file)
        logger.info(f'saved best latent increment (for epoch={epoch_min}) to {pt_file}')
    else:
        logger.warning('no epoch improved on the background -- returning an UNCORRECTED forecast advance')
        lsave = lsave or [float('nan')]
    if eta is not None:
        # the analysis trajectory past +dt_verif is the analysis state advanced
        # WITH these jumps (save_trajectory_diagnostics, the driver's cycling
        # and window-end state all read exp['wc_state'])
        if eta_best is not None:
            forcing = eta_best
        else:
            forcing = to_q(v_prior).detach() if v_prior is not None else torch.zeros_like(eta)
        exp['wc_state'] = {'hours': wc_hours, 'steps': wc_steps,
                           'step_index': {s: k for k, s in enumerate(wc_steps)},
                           'forcing': forcing / latent_scale, 'keep_mask': keep_mask,
                           'date': exp['date'], 'controls': eta_controls}
        wc_file = _wc_file(exp, exp['date'], max_epoch)
        torch.save({'hours': wc_hours, 'steps': wc_steps, 'forcing': forcing, 'latent_scale': latent_scale,
                    'controls': eta_controls}, wc_file)
        rms0 = increment_best.pow(2).mean().sqrt().item() if increment_best is not None else 0.
        logger.info(f'saved weak-constraint jumps to {wc_file}; latent rms: window-start increment {rms0:.4g}, '
                    + ', '.join(f'{h} h {forcing[k].pow(2).mean().sqrt().item():.4g}' for k, h in enumerate(wc_hours)))
    for mw_traj in mw_trajs or []:
        sensor = mw_traj['sensor']
        if sensor not in zeta:
            continue
        # the analysis diagnostics evaluate H with the best zeta; the loss's live zeta is retired
        mw_traj['zeta_best'] = zeta_best[sensor]
        mw_traj['zeta'] = None
        if zeta_best[sensor] is not None:
            za = zeta_best[sensor][mw_traj['z_active']]
            logger.info(f'{sensor} zeta at the analysis: rms {za.pow(2).mean().sqrt().item():.3f}, '
                        f'|zeta| > 2 for {(za.abs() > 2).float().mean().item() * 100:.1f}% of components')

    # Materialize the actual corrected physical state once, under no_grad
    # (training is done -- increment_best is already detached). This is the
    # earliest point a real physical analysis exists (see docstring). Falls
    # back to a plain (uncorrected) forecast advance if no epoch improved on
    # the background, keeping the contract "always returns a state dated
    # `+dt_verif`" regardless of optimization success. Re-primes FCN3's
    # noise once more first (no-op for AIFS), so this materialization is a
    # deterministic function of increment_best rather than whatever noise
    # state the epoch loop happened to leave behind.
    dt_verif = exp['dt_verif']
    steps_per_verif = dt_verif // int(round(model.timestep.total_seconds() / 3600))
    scaled_increment_best = increment_best / latent_scale if increment_best is not None else None
    model._prime_noise()
    with torch.no_grad():
        if keep_mask is None:
            analysis_state = model.advance(
                input_encoded, steps=steps_per_verif, use_checkpoint=False,
                latent_increment=scaled_increment_best,
            )
        else:
            # Split the advance so the same control_variables mask the loss
            # used is applied between the injection step and the rest.
            analysis_state = model.advance(
                input_encoded, steps=1, use_checkpoint=False,
                latent_increment=scaled_increment_best,
            )
            analysis_state = model.wrap_state(
                _apply_control_mask(analysis_state.state, keep_mask, ref_state1.state, model.state_layout),
                analysis_state.date,
            )
            if steps_per_verif > 1:
                analysis_state = model.advance(
                    analysis_state, steps=steps_per_verif - 1, use_checkpoint=False,
                )

    return analysis_state, lsave


# ---------------------------------------------------------------------------
# diagnostics / output
# ---------------------------------------------------------------------------

def _pad_leading(arr, n_lead, fill):
    """Pad an (obs-slot, ...) array with `n_lead` copies of `fill` at the
    front -- used because a latent analysis trajectory only starts at
    `window_start + dt_verif` (see compute_ps_observation_hx's `step_start`),
    so its obs-slot arrays are shorter than the full-window background's and
    must be padded before the two are stacked into one Dataset.
    """
    if n_lead == 0:
        return arr
    pad = np.full((n_lead,) + arr.shape[1:], fill, dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


def save_trajectory_diagnostics(exp, model, input_encoded, analysis_state, verif_ic, psobs_traj, save_levs, grid_interp, logger, raobs_traj=None,
                                mw_trajs=None):
    interp_specs = _resolve_loss_interp_specs(model, exp)
    timestep_h = int(round(model.timestep.total_seconds() / 3600))
    dt_obs = exp['dt_obs']
    n_obs = int(psobs_traj['n_valid'].shape[0])
    step_lo = psobs_traj['step_lo'].tolist()

    for mw_traj in mw_trajs or []:
        mw_traj['zeta_eval'] = None  # background: z = 0
    # background: the free run, or under weak_constraint.warm_start the first guess with the
    # prior-mean jumps (the previous cycle's)
    background, initial_traj, raobs_background, mw_background = compute_ps_observation_hx(
        model, input_encoded, verif_ic, psobs_traj, exp, grid_interp, interp_specs, step_start=0,
        raobs_traj=raobs_traj, mw_trajs=mw_trajs, wc_state=exp.get('wc_prior_state'),
    )
    for mw_traj in mw_trajs or []:
        mw_traj['zeta_eval'] = mw_traj.get('zeta_best')  # analysis: the fitted z (None without control)

    # The latent analysis_state is valid at input_encoded.date + dt_verif
    # (see compute_optimal). compute_ps_observation_hx starts its trajectory
    # at that 6h-step index; obs slots earlier than it have no analysis value
    # and are NaN/0-padded here so the analysis arrays line up with the
    # full-window background arrays.
    shift_hours = (analysis_state.date - input_encoded.date).total_seconds() / 3600.0
    step_start = int(round(shift_hours / timestep_h))
    j_start = sum(1 for s in step_lo if s < step_start)
    analysis, final_traj, raobs_analysis, mw_analysis = compute_ps_observation_hx(
        model, analysis_state, verif_ic, psobs_traj, exp, grid_interp, interp_specs, step_start=step_start,
        raobs_traj=raobs_traj, mw_trajs=mw_trajs, wc_state=exp.get('wc_state'),
    )
    analysis['model_equivalent'] = _pad_leading(analysis['model_equivalent'], j_start, np.nan)
    analysis['used'] = _pad_leading(analysis['used'], j_start, False)
    analysis['qc_flag'] = _pad_leading(analysis['qc_flag'], j_start, 0)
    if raobs_traj is not None:
        for base in raobs_traj['variables']:
            raobs_analysis[base]['model_equivalent'] = _pad_leading(raobs_analysis[base]['model_equivalent'], j_start, np.nan)
            raobs_analysis[base]['used'] = _pad_leading(raobs_analysis[base]['used'], j_start, False)
    for atms_analysis in mw_analysis or []:
        atms_analysis['model_equivalent'] = _pad_leading(atms_analysis['model_equivalent'], j_start, np.nan)
        atms_analysis['screened'] = _pad_leading(atms_analysis['screened'], j_start, False)
        atms_analysis['used'] = _pad_leading(atms_analysis['used'], j_start, False)

    available = background['available'].astype(bool)
    used = background['used'].astype(bool)
    analysis_used = analysis['used'].astype(bool)
    observation = psobs_traj['ob'].detach().cpu().numpy().astype(np.float32)
    background_hx = background['model_equivalent'].astype(np.float32)
    analysis_hx = analysis['model_equivalent'].astype(np.float32)
    valid_times = np.asarray([
        np.datetime64(add_hours(exp['date'], j * dt_obs), 'h')
        for j in range(n_obs)
    ])
    lead_hours = np.arange(n_obs, dtype=np.int32) * dt_obs
    observation_slots = np.arange(observation.shape[1], dtype=np.int32)
    dims = ('time', 'observation_slot')

    ds = xarray.Dataset(
        coords={
            'time': ('time', valid_times),
            'observation_slot': ('observation_slot', observation_slots),
            'lead_time_hours': ('time', lead_hours),
        },
        attrs={
            'title': f'4D-Var surface-pressure observation diagnostics ({type(model).__name__})',
            'window_start': str(exp['date']),
            'window_length_hours': int(exp['n_verif'] * exp['dt_verif']),
            'dt_obs_hours': int(dt_obs),
            'analysis_valid_from': str(analysis_state.date),
            'innovation_convention': 'observation_minus_model',
            'quality_control_reference': 'background',
            'history': 'Created by long_window_4dvar.py',
        },
    )
    ds['lead_time_hours'].attrs.update({'long_name': 'elapsed time from assimilation-window start', 'units': 'h'})

    location_fields = {
        'observation_latitude': (psobs_traj['lat'], 'observation latitude', 'degrees_north'),
        'observation_longitude': (psobs_traj['lon'], 'observation longitude', 'degrees_east'),
        'station_elevation': (psobs_traj['elev'], 'station elevation above mean sea level', 'm'),
    }
    for name, (values, long_name, units) in location_fields.items():
        values = values.detach().cpu().numpy().astype(np.float32)
        ds[name] = (dims, np.where(available, values, np.nan))
        ds[name].attrs.update({'long_name': long_name, 'units': units})

    fields = {
        'surface_pressure_observation': (observation, 'observed surface pressure'),
        'surface_pressure_background': (background_hx, 'background model equivalent H(x_b)'),
        'surface_pressure_analysis': (analysis_hx, 'analysis model equivalent H(x_a)'),
        'surface_pressure_omb': (observation - background_hx, 'observation minus background'),
        'surface_pressure_oma': (observation - analysis_hx, 'observation minus analysis'),
        'surface_pressure_error_std': (background['effective_error'], 'observation-error standard deviation after orography adjustment'),
    }
    for name, (values, long_name) in fields.items():
        values = np.squeeze(np.asarray(values, dtype=np.float32))
        ds[name] = (dims, np.where(available, values, np.nan))
        units = 'm' if name in ('model_orography', 'orography_difference') else 'hPa'
        ds[name].attrs.update({'long_name': long_name, 'units': units})

    for name, values, long_name in (
        ('surface_pressure_available', available, 'valid source observation'),
        ('surface_pressure_used', used, 'observation passing background quality control'),
        ('surface_pressure_analysis_used', analysis_used, 'observation passing quality control when reevaluated at the analysis'),
    ):
        ds[name] = (dims, np.squeeze(values.astype(np.int8)))
        ds[name].attrs.update({'long_name': long_name, 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})

    for name, values, long_name in (
        ('surface_pressure_qc_flag', background['qc_flag'], 'background quality-control bit mask'),
        ('surface_pressure_analysis_qc_flag', analysis['qc_flag'], 'quality-control bit mask reevaluated at the analysis'),
    ):
        ds[name] = (dims, np.squeeze(np.asarray(values, dtype=np.int16)))
        ds[name].attrs.update({'long_name': long_name, 'flag_masks': PSOBS_QC_FLAG_MASKS, 'flag_meanings': PSOBS_QC_FLAG_MEANINGS})

    ofile1 = exp['path_output'] + exp['date'] + '_observation_diagnostics_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
    encoding = {name: {'zlib': True, 'complevel': 2} for name in ds.data_vars}
    ds.to_netcdf(ofile1, encoding=encoding)
    logger.info('saved observation diagnostics: ' + ofile1)

    if raobs_traj is not None:
        ofile_raobs = (
            exp['path_output'] + exp['date'] + '_raobs_diagnostics_' + exp['window_name'] + 'h'
            + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
        )
        save_raobs_diagnostics(
            exp, raobs_traj, raobs_background, raobs_analysis, valid_times, lead_hours, ofile_raobs, logger,
        )

    for mw_traj, mw_b, mw_a in zip(mw_trajs or [], mw_background or [], mw_analysis or []):
        ofile_mw = (
            exp['path_output'] + exp['date'] + f"_{mw_traj['sensor']}_diagnostics_" + exp['window_name'] + 'h'
            + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
        )
        save_atms_diagnostics(exp, mw_traj, mw_b, mw_a, valid_times, lead_hours, ofile_mw, logger)

    ofile2 = exp['path_output'] + exp['date'] + '_control_forecast_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
    ofile3 = exp['path_output'] + exp['date'] + '_optimal_forecast_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
    _ = save_xr_trajectory(model, initial_traj, save_levs, ofile2, start_date=str(input_encoded.date))
    logger.info('saved initial trajectory: ' + ofile2)
    # initial_traj / final_traj are on the native 6h model grid (NOT the
    # dt_obs obs-slot grid). final_traj's bare 'time' dim starts at
    # analysis_state.date (window_start + dt_verif), not window_start --
    # recorded as an attribute so a later reader isn't misled.
    _ = save_xr_trajectory(model, final_traj, save_levs, ofile3, start_date=str(analysis_state.date))
    logger.info('saved final trajectory: ' + ofile3)

    return ofile1, ofile2, ofile3


_RAOBS_BASE_LONG_NAME = {v: k for k, v in RAOBS_VARIABLE_MODEL_BASE.items()}
_RAOBS_BASE_UNITS = {'t': 'K', 'u': 'm s-1', 'v': 'm s-1', 'q': 'kg kg-1'}


def save_raobs_diagnostics(exp, raobs_traj, background, analysis, valid_times, lead_hours, ofile, logger):
    """Save per-variable radiosonde O-B/O-A diagnostics, mirroring the ps
    obs diagnostics netCDF but with an extra per-variable pressure-level
    dimension (no vertical interpolation -- see get_raobs) and no
    cross-time station identity (`station_slot` is a plain index, same
    convention as psobs's `observation_slot`).
    """
    dt_obs = exp['dt_obs']
    n_obs, _, capacity = raobs_traj['values_'+raobs_traj['variables'][0]].shape
    ds = xarray.Dataset(
        coords={
            'time': ('time', valid_times),
            'station_slot': ('station_slot', np.arange(capacity, dtype=np.int32)),
            'lead_time_hours': ('time', lead_hours),
        },
        attrs={
            'title': f'4D-Var radiosonde observation diagnostics ({", ".join(raobs_traj["variables"])})',
            'window_start': str(exp['date']),
            'window_length_hours': int(exp['n_verif'] * exp['dt_verif']),
            'dt_obs_hours': int(dt_obs),
            'innovation_convention': 'observation_minus_model',
            'forward_operator': 'horizontal interpolation only -- obs matched to model pressure levels at read time, no vertical interpolation',
            'history': 'Created by long_window_4dvar.py',
        },
    )
    for name in ('lat', 'lon', 'elev'):
        ds['station_'+name] = (('time', 'station_slot'), raobs_traj[name].detach().cpu().numpy().astype(np.float32))

    for base in raobs_traj['variables']:
        level_dim = 'level_'+base
        ds = ds.assign_coords({level_dim: raobs_traj['levels_hpa'][base].detach().cpu().numpy().astype(np.float32)})
        dims = ('time', level_dim, 'station_slot')
        long_name = _RAOBS_BASE_LONG_NAME[base]
        units = _RAOBS_BASE_UNITS[base]

        observation = raobs_traj['values_'+base].detach().cpu().numpy().astype(np.float32)
        mask = raobs_traj['mask_'+base].detach().cpu().numpy().astype(bool)
        background_hx = background[base]['model_equivalent'].astype(np.float32)
        analysis_hx = analysis[base]['model_equivalent'].astype(np.float32)
        background_used = background[base]['used'].astype(bool)
        analysis_used = analysis[base]['used'].astype(bool)

        ds[base+'_observation'] = (dims, np.where(mask, observation, np.nan))
        ds[base+'_observation'].attrs.update({'long_name': long_name+' observation', 'units': units})
        ds[base+'_background'] = (dims, np.where(mask, background_hx, np.nan))
        ds[base+'_background'].attrs.update({'long_name': long_name+' background model equivalent H(x_b)', 'units': units})
        ds[base+'_analysis'] = (dims, np.where(mask, analysis_hx, np.nan))
        ds[base+'_analysis'].attrs.update({'long_name': long_name+' analysis model equivalent H(x_a)', 'units': units})
        ds[base+'_omb'] = (dims, np.where(mask, observation - background_hx, np.nan))
        ds[base+'_omb'].attrs.update({'long_name': long_name+' observation minus background', 'units': units})
        ds[base+'_oma'] = (dims, np.where(mask, observation - analysis_hx, np.nan))
        ds[base+'_oma'].attrs.update({'long_name': long_name+' observation minus analysis', 'units': units})
        ds[base+'_available'] = (dims, mask.astype(np.int8))
        ds[base+'_available'].attrs.update({'long_name': long_name+' valid source observation after read-time QC', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})
        ds[base+'_used'] = (dims, background_used.astype(np.int8))
        ds[base+'_used'].attrs.update({'long_name': long_name+' observation used in the background loss', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})
        ds[base+'_analysis_used'] = (dims, analysis_used.astype(np.int8))
        ds[base+'_analysis_used'].attrs.update({'long_name': long_name+' observation used when reevaluated at the analysis', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})

    encoding = {name: {'zlib': True, 'complevel': 2} for name in ds.data_vars}
    ds.to_netcdf(ofile, encoding=encoding)
    logger.info('saved raobs diagnostics: ' + ofile)
    return ofile


def save_atms_diagnostics(exp, atms_traj, background, analysis, valid_times, lead_hours, ofile, logger):
    """Save a microwave sensor's O-B/O-A for the assimilated, monitored and
    window channels: one `(time, obs_slot, channel)` array per quantity,
    `obs_slot` a plain index within each time (same convention as psobs's
    `observation_slot`). Channels are 1-based channel numbers of the sensor.
    """
    sensor = atms_traj['sensor']
    channels = atms_traj['assimilate_channels'] + atms_traj['monitor_channels'] + atms_traj.get('emis_channels', [])
    ci = [c - 1 for c in channels]
    capacity = atms_traj['y'].shape[1]
    ds = xarray.Dataset(
        coords={
            'time': ('time', valid_times),
            'obs_slot': ('obs_slot', np.arange(capacity, dtype=np.int32)),
            'channel': ('channel', np.asarray(channels, dtype=np.int32)),
            'lead_time_hours': ('time', lead_hours),
        },
        attrs={
            'title': f'4D-Var {sensor.upper()} observation diagnostics (radianced two-head decoder)',
            'sensor': sensor,
            'window_start': str(exp['date']),
            'window_length_hours': int(exp['n_verif'] * exp['dt_verif']),
            'dt_obs_hours': int(exp['dt_obs']),
            'assimilated_channels': np.asarray(atms_traj['assimilate_channels'], dtype=np.int32),
            'monitored_channels': np.asarray(atms_traj['monitor_channels'], dtype=np.int32),
            'emissivity_channels': np.asarray(atms_traj.get('emis_channels', []), dtype=np.int32),
            'emis_control': str(atms_traj.get('emis_control', 'none')),
            'window_state_scenes': ' '.join(atms_traj.get('window_state_scenes', [])) or 'none',
            'scenes': ' '.join(f'{i}={n}' for i, n in enumerate(ATMS_SCENES)),
            'bundle': str(exp[f'{sensor}_bundle']),
            'huber_c': float(atms_traj['huber_c'] or 0.),
            'observation': 'y_o - BC (GSI replay bias correction, fixed)',
            'innovation_convention': 'observation_minus_model',
            'history': 'Created by long_window_4dvar.py',
        },
    )
    valid = atms_traj['valid'].detach().cpu().numpy()
    ds['latitude'] = (('time', 'obs_slot'), np.where(valid, atms_traj['lat'].detach().cpu().numpy(), np.nan))
    ds['longitude'] = (('time', 'obs_slot'), np.where(valid, atms_traj['lon'].detach().cpu().numpy(), np.nan))
    ds['withheld'] = (('time', 'obs_slot'), atms_traj['withheld'].detach().cpu().numpy().astype(np.int8))
    ds['scene'] = (('time', 'obs_slot'), atms_traj['scene'].detach().cpu().numpy().astype(np.int8))
    ds['scene'].attrs['long_name'] = 'scene class, index into the scenes attribute (-1 = padding)'
    if atms_traj.get('zeta_best') is not None:
        zb = atms_traj['zeta_best'].detach().cpu().numpy().astype(np.float32)
        ds['zeta'] = (('time', 'obs_slot', 'emis_pc'), np.where(valid[..., None], zb, np.nan))
        ds['zeta'].attrs['long_name'] = 'analysis emissivity control in prior-std units (z = zeta * sigma_k)'
    ds['screened'] = (('time', 'obs_slot'), background['screened'].astype(np.int8))
    ds['screened'].attrs['long_name'] = 'real ob, finite H(x_b); clear-ocean obs also need all-ocean interpolation neighbours'
    ds['sigma_o'] = (('channel',), atms_traj['sigma_o'].detach().cpu().numpy()[ci])
    ds['sigma_o'].attrs.update({'long_name': 'observation error standard deviation', 'units': 'K'})
    ds.attrs['oberr_model'] = str(atms_traj['oberr_model'])
    if atms_traj['oberr_model'] != 'none':
        so = atms_traj['sigma_obs'].detach().cpu().numpy()[..., ci]
        ds['sigma_obs'] = (('time', 'obs_slot', 'channel'), np.where(valid[..., None], so, np.nan).astype(np.float32))
        ds['sigma_obs'].attrs.update({'long_name': 'per-ob observation error (oberr_model scene_cloud), set at x_b',
                                      'units': 'K'})
        cs = atms_traj['cloud_sym'].detach().cpu().numpy()
        ds['cloud_sym'] = (('time', 'obs_slot'), np.where(valid, cs, np.nan).astype(np.float32))
        ds['cloud_sym'].attrs.update({'long_name': 'symmetric cloud liquid water (clw(y_o) + clw(H(x_b))) / 2, ocean',
                                      'units': 'kg m-2'})
    dims = ('time', 'obs_slot', 'channel')
    mask = valid[..., None]
    y = atms_traj['y'].detach().cpu().numpy()[..., ci]
    hb = background['model_equivalent'][..., ci].astype(np.float32)
    ha = analysis['model_equivalent'][..., ci].astype(np.float32)
    for name, values, long_name in (
        ('observation', y, 'observed brightness temperature minus bias correction'),
        ('background', hb, 'background model equivalent H(x_b)'),
        ('analysis', ha, 'analysis model equivalent H(x_a)'),
        ('omb', y - hb, 'observation minus background'),
        ('oma', y - ha, 'observation minus analysis'),
    ):
        ds[name] = (dims, np.where(mask, values, np.nan).astype(np.float32))
        ds[name].attrs.update({'long_name': long_name, 'units': 'K'})
    for name, values, long_name in (
        ('used', background['used'][..., ci], 'screened, passing the background check at x_b, not withheld'),
        ('analysis_used', analysis['used'][..., ci], 'the same, reevaluated at the analysis'),
    ):
        ds[name] = (dims, values.astype(np.int8))
        ds[name].attrs.update({'long_name': long_name, 'flag_values': np.array([0, 1], dtype=np.int8),
                               'flag_meanings': 'not_used used'})
    encoding = {name: {'zlib': True, 'complevel': 2} for name in ds.data_vars}
    ds.to_netcdf(ofile, encoding=encoding)
    if 'sat_id' in _ATMS_OBS_INPUTS:
        ds['sat_id'] = (('time', 'obs_slot'), np.where(valid, atms_traj['obs_sat_id'].detach().cpu().numpy(), -1).astype(np.int8))
    logger.info(f'saved {sensor} diagnostics: ' + ofile)
    return ofile


def save_xr_trajectory(model, model_trajectory, olevels, ofile, save=True, start_date=None):
    """Save a (time, ...) dict-of-arrays trajectory (from compute_ps_observation_hx)
    to netCDF on the model's flat unstructured "values" grid -- lat/lon are
    1D coordinate arrays over that dimension (a row-major flatten of FCN3's
    native 721x1440 grid, or AIFS's native irregular N320 point ordering),
    not separate lon/lat dimensions.

    `start_date`: the 'time' dim has no coordinate values of its own (just an
    index) -- if given, recorded as a dataset attribute so a caller can tell
    what real date `time=0` corresponds to (needed when the trajectory
    doesn't start at the nominal window-start date
    -- see save_trajectory_diagnostics).
    """
    ds_new = xarray.Dataset(
        coords={
            'values': np.arange(model.n_points),
            'latitude': ('values', model.lats),
            'longitude': ('values', model.lons),
        },
        attrs={'trajectory_start_date': start_date} if start_date is not None else {},
    )
    for base, levels in model._levels_by_base.items():
        if base not in model_trajectory:
            continue
        all_levels = np.array([lev for lev, _ in levels])
        sel = [i for i, lev in enumerate(all_levels) if olevels == 'all' or lev in olevels]
        if not sel:
            continue
        ds_new[base] = (['time', 'level_' + base, 'values'], model_trajectory[base][:, sel, :])
        ds_new = ds_new.assign_coords({'level_' + base: all_levels[sel]})
    for name in model._single_by_base:
        if name in model_trajectory:
            ds_new[name] = (['time', 'values'], model_trajectory[name])
    if save:
        print('saving to: ', ofile)
        ds_new.to_netcdf(ofile)
    return ds_new


def save_inputs_pkl(exp, input_encoded, analysis_state, lsave):
    ofile_base = exp['path_output'] + exp['date'] + '_{}_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it'
    with open(ofile_base.format('control_inputs') + '.pkl', 'wb') as f:
        pickle.dump(input_encoded, f)
    with open(ofile_base.format('optimal_inputs') + '.pkl', 'wb') as f:
        pickle.dump(analysis_state, f)
    ofile = exp['path_output'] + exp['date'] + '_loss_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.txt'
    with open(ofile, 'w') as f:
        for L in lsave:
            f.write(f"{L}\n")
    return


def save_window_end_state(exp, model, input_encoded, analysis_state, logger):
    """Weak constraint: the analysis at the window end -- the analysis state
    advanced through the window WITH the model-error jumps -- as
    `<date>_wcend_inputs_<window>h..it.pkl`. Forecasts from the window end
    must start from it (diagnostics/forecast_skill.py does): advancing the
    `optimal_inputs` state (valid at +dt_verif) would drop the jumps."""
    timestep = model.timestep.total_seconds()
    step_start = int(round((analysis_state.date - input_encoded.date).total_seconds() / timestep))
    n_steps = exp['n_verif'] * exp['dt_verif'] * 3600 // int(timestep)
    end_state = advance_with_jumps(model, analysis_state, step_start, n_steps - step_start, exp['wc_state'])
    ofile = (exp['path_output'] + exp['date'] + '_wcend_inputs_' + exp['window_name'] + 'h' + exp['suffix']
             + '_' + str(exp['max_epoch']) + 'it.pkl')
    with open(ofile, 'wb') as f:
        pickle.dump(end_state, f)
    logger.info(f'saved window-end analysis (valid {end_state.date}): {ofile}')


def save_inputs_nc(exp, model, input_encoded, analysis_state, logger):
    max_epoch = exp['max_epoch']

    def _state_to_dict(model_state):
        decoded = model.decode_state(model_state)
        return {k: v.detach().cpu().numpy() for k, v in decoded.items()}

    ds_control = _save_xr_state(model, _state_to_dict(input_encoded), date=str(input_encoded.date))
    ofile = exp['path_output'] + exp['date'] + '_control_inputs_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(max_epoch) + 'it.nc'
    ds_control.to_netcdf(ofile)

    # analysis_state.date may differ from input_encoded.date
    # recorded as an attribute
    # since this is a single-time snapshot with no 'time' dim of its own.
    ds_optimal = _save_xr_state(model, _state_to_dict(analysis_state), date=str(analysis_state.date))
    ofile = exp['path_output'] + exp['date'] + '_optimal_inputs_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(max_epoch) + 'it.nc'
    ds_optimal.to_netcdf(ofile)
    return


def _save_xr_state(model, decoded, date=None):
    ds = xarray.Dataset(
        coords={
            'values': np.arange(model.n_points),
            'latitude': ('values', model.lats),
            'longitude': ('values', model.lons),
        },
        attrs={'valid_date': date} if date is not None else {},
    )
    for base, levels in model._levels_by_base.items():
        if base not in decoded:
            continue
        all_levels = np.array([lev for lev, _ in levels])
        ds[base] = (['level_' + base, 'values'], decoded[base])
        ds = ds.assign_coords({'level_' + base: all_levels})
    for name in model._single_by_base:
        if name in decoded:
            ds[name] = (['values'], decoded[name])
    return ds


def make_forecasts(exp, model, input_encoded, analysis_state, save_levs, logger):
    max_epoch = exp['max_epoch']
    dt_forecast = exp['dt_forecast']
    steps_per_output = int(round(dt_forecast / (model.timestep.total_seconds() / 3600)))
    logger.info('optimal forecast...')
    outputs = {}
    state = analysis_state
    with torch.no_grad():
        for n in range(exp['n_forecast']):
            state = model.advance(state, steps=steps_per_output, use_checkpoint=False)
            decoded = model.decode_state(state)
            for k, v in decoded.items():
                outputs.setdefault(k, []).append(v.detach().cpu().numpy())
    outputs = {k: np.stack(v, axis=0) for k, v in outputs.items()}
    logger.info('...forecasts complete')
    ofile = exp['path_output'] + exp['date'] + '_optimal_longforecast_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(max_epoch) + 'it.nc'
    # First output is `steps_per_output` (dt_forecast hours) past
    # analysis_state.date, which may itself already be shifted from
    # input_encoded.date  -- record it rather than
    # assume the window-start date.
    first_output_date = analysis_state.date + steps_per_output * model.timestep
    _ = save_xr_trajectory(model, outputs, save_levs, ofile, start_date=str(first_output_date))
    return
