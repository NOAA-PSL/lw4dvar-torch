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
-- see CLAUDE.md; the three envs are not co-installed) never needs the
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
    (see CLAUDE.md "Aurora 16-step divergence: root cause and fix"). Purely
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
    if backend == 'fcn3':
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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

    backend = exp.get('model_backend')
    if backend not in ('aifs', 'fcn3', 'aurora', 'ace2'):
        raise ValueError(
            f"exp.model_backend must be 'aifs', 'fcn3', 'aurora', or 'ace2', got {backend!r} -- "
            "this key is required (see config.yml.template)"
        )
    # reset_skt_ocean: AIFS and Aurora both have 'skt'/'lsm' packed-state
    # columns (Aurora's 'lsm' added specifically to support this -- see
    # aurora_model.py), so reset_skt_over_ocean works unmodified for either.
    # FCN3 has no equivalent at all -- fail loudly here rather than
    # silently ignore a config key that looks like it should do something.
    if backend == 'fcn3' and exp.get('reset_skt_ocean', False):
        raise ValueError(
            "reset_skt_ocean is not supported for the FCN3 backend -- FCN3 has no "
            "'skt'/'lsm' decode_state field to reset (see CLAUDE.md)."
        )
    if backend == 'fcn3' and exp.get('ps_operator', 'logpinterp') == 'ps':
        raise ValueError(
            "ps_operator='ps' is not supported for the FCN3 backend -- FCN3 has no "
            "native surface-pressure ('sp') decode_state field; use the default "
            "'logpinterp' instead (see CLAUDE.md)."
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
            "default 'logpinterp' instead (see CLAUDE.md)."
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
    # ps_native needs decode_state's lowest-native-level keys
    # (t_lowest/q_lowest/p_lowest), which only the ACE2 backend provides.
    if exp.get('ps_operator', 'logpinterp') == 'ps_native' and backend != 'ace2':
        raise ValueError(
            f"ps_operator='ps_native' is only supported for model_backend 'ace2', got {backend!r} "
            "(it needs decode_state's lowest-native-level t_lowest/q_lowest/p_lowest fields)."
        )
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
            autocast_dtype=exp.get('aifs_autocast_dtype', None),
            compile_wrapper=exp.get('compile_wrapper', False),
        )
    elif backend == 'fcn3':
        _ensure_backend_on_path('fcn3')
        import fcn3_model
        return fcn3_model.FCN3Model(
            package_root=exp['path_model'],
            device=exp.get('device', 'cuda'),
            atmo_chunk_size=exp.get('atmo_chunk_size', 2),
        )
    elif backend == 'aurora':
        _ensure_backend_on_path('aurora')
        import aurora_model
        return aurora_model.AuroraModel(
            package_root=exp['path_model'],
            device=exp.get('device', 'cuda'),
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
    raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', or 'ace2')")


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
    elif backend in ('fcn3', 'aurora'):
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
    raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', or 'ace2')")


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
            raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', or 'ace2')")
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
    elif backend == 'fcn3':
        _ensure_backend_on_path('fcn3')
        import fcn3_ic
        raw_fields = fcn3_ic.read_single_date_fields(date_dt, cache_dir)
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
        raise ValueError(f"unknown model_backend {backend!r} (expected 'aifs', 'fcn3', 'aurora', or 'ace2')")

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
    see CLAUDE.md. No separate sea-ice handling: there's no sea-ice variable
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
    """Read the ps observation files spanning the window.

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
    date = exp['date']
    device = model.device
    psobs_traj = {}
    psobs_datestrings = []
    vdate = copy.copy(date)

    if 'nobs_max' in exp:
        nobs_max = exp['nobs_max']
    else:
        nobs_max = -1
    nobs_list = []
    for j in range(n_obs):
        yyyymmddhh = get_YYYYMMDDHH(vdate)
        psobs_datestrings.append(yyyymmddhh)
        psobs_filename = os.path.join(obspath, 'psobs1_%s.txt' % yyyymmddhh)
        logger.info('reading ' + str(psobs_filename))
        with open(psobs_filename, 'r') as f:
            nobs = sum(1 for line in f)
        if nobs > nobs_max: nobs_max = nobs
        nobs_list.append(nobs)
        vdate = add_hours(vdate, dt_obs)
    logger.info('max number of obs at each slot in window:' + str(nobs_max))

    for k in ['obtype', 'lon', 'lat', 'elev', 'ob', 'oberr']:
        if k in ('ob', 'elev'):
            psobs_traj[k] = torch.zeros((n_obs, nobs_max), dtype=torch.float32, device=device)
        elif k == 'obtype':
            psobs_traj[k] = torch.full((n_obs, nobs_max), 999, device=device)
        else:
            psobs_traj[k] = torch.full((n_obs, nobs_max), 1.e10, dtype=torch.float32, device=device)
    psobs_traj['n_valid'] = torch.as_tensor(nobs_list, dtype=torch.int32, device=device)

    for j in range(n_obs):
        yyyymmddhh = psobs_datestrings[j]
        psobs_filename = os.path.join(obspath, 'psobs1_%s.txt' % yyyymmddhh)
        with open(psobs_filename) as f:
            psobs_data = np.loadtxt((line[8:] for line in f))
        nobs = psobs_data.shape[0]
        psobs_traj['obtype'][j, :nobs] = torch.as_tensor(psobs_data[:, 0], device=device)
        psobs_traj['lon'][j, :nobs] = torch.as_tensor(psobs_data[:, 1], dtype=torch.float32, device=device)
        psobs_traj['lat'][j, :nobs] = torch.as_tensor(psobs_data[:, 2], dtype=torch.float32, device=device)
        psobs_traj['elev'][j, :nobs] = torch.as_tensor(psobs_data[:, 3], dtype=torch.float32, device=device)
        psobs_traj['ob'][j, :nobs] = torch.as_tensor(psobs_data[:, 5], dtype=torch.float32, device=device)
        tday = j * dt_obs / 24.
        if oberrstart > 0:
            psobs_traj['oberr'][j, :nobs] = oberrstart + oberrdeltaperday * tday
        else:
            psobs_traj['oberr'][j, :nobs] = torch.as_tensor(psobs_data[:, 7], dtype=torch.float32, device=device) + oberrdeltaperday * tday
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
        coslat = torch.as_tensor(model.coslat if hasattr(model, 'coslat') else grid_interp.coslat, device=device)
        geopotential_mean = (coslat[None, :] * decoded['geopotential']).sum(dim=-1) / coslat.sum()
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
    base_rejected = preliminary_rejected | interpolation_failed | orography_failed
    innovation = observation - model_equivalent
    gross_check_failed = (~base_rejected) & (torch.abs(innovation / effective_error) > bg_check)
    total_rejected = preliminary_rejected | interpolation_failed | orography_failed | gross_check_failed
    used = ~total_rejected

    qc_flag = torch.zeros(nobs_max, dtype=torch.int16, device=device)
    qc_flag = torch.where(~available, qc_flag | 1, qc_flag)
    qc_flag = torch.where(interpolation_failed, qc_flag | 2, qc_flag)
    qc_flag = torch.where(orography_failed, qc_flag | 4, qc_flag)
    qc_flag = torch.where(gross_check_failed, qc_flag | 8, qc_flag)

    return innovation, model_equivalent, effective_error, available, used, qc_flag


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

    Returns `None` when `control_variables` is absent (no masking -- the
    increment is free on every variable, the default behaviour).

    Column lookup (`model.resolve_columns`) checks pressure-level families
    before any single-level/raw-checkpoint fallback -- see CLAUDE.md for the
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
    return mask


@torch.no_grad()
def compute_ps_observation_hx(model, model_state, verif_ic, psobs_traj, exp, grid_interp,
                              interp_specs, step_start=0):
    """Evaluate a trajectory at every surface-pressure observation slot.
    Post-optimization diagnostics only -- no gradients anywhere here.

    Returns two output grids.
      - `model_trajectory`: the decoded model state at every 6h step, from
        `step_start` to `n_steps` inclusive (native rollout cadence).
      - `obs_trajectory`: H(x)/QC at every obs slot `j` with
        `step_lo[j] >= step_start`, using the same time-interpolation as the
        loss (`_interp_decoded` over `interp_specs`). Slots earlier than
        `step_start` are omitted -- the caller pads them.

    `step_start`: the 6h-step index `model_state` sits at. 0 for the
    background; `dt_verif // 6` for a latent analysis (valid at
    `window_start + dt_verif` -- see `compute_optimal`'s docstring).

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

    f_state = model_state
    decoded_prev = model.decode_state(f_state)
    for name, value in decoded_prev.items():
        model_outputs.setdefault(name, []).append(value.detach().cpu().numpy())

    for s in range(step_start + 1, n_steps + 1):
        f_state = model.advance(f_state, steps=1, use_checkpoint=False)
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
    return obs_trajectory, model_trajectory


# ---------------------------------------------------------------------------
# rollout / loss / optimizer
# ---------------------------------------------------------------------------

def forecast(model, model_state, steps):
    return model.advance(model_state, steps=steps)


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
                       keep_mask=None, ref_state1=None):
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

    f_state = input_state
    decoded_prev = model.decode_state(f_state, only=loss_bases)   # step 0 = uncorrected background
    for s in range(1, n_steps + 1):
        inj = scaled_increment if s == 1 else None
        use_ckpt = (checkpoint_stride > 0) and (s % checkpoint_stride == 0)
        f_state = model.advance(f_state, steps=1, latent_increment=inj, use_checkpoint=use_ckpt)
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
        for j in slots_by_step.get(s - 1, []):
            a = alpha[j]
            if a == 0.0:
                decoded_j = decoded_prev
            elif a == 1.0:
                decoded_j = decoded_cur
            else:
                decoded_j = _interp_decoded(decoded_prev, decoded_cur, a, interp_specs)
            Jol[j] = _obs_term(j, decoded_j)
        decoded_prev = decoded_cur

    J = sum(Jol)
    if do_print:
        print(f"epoch {epoch}, Jtot = {J.item()}")

    Jol_np = np.array([j.item() for j in Jol])
    return J, Jol_np


def compute_optimal(exp, model, input_encoded, verif_ic, psobs_traj, grid_interp, logger):
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
    step -- see CLAUDE.md. Aurora has not been separately investigated;
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

    optimizer = torch.optim.AdamW([increment], lr=lr, weight_decay=wd, betas=(beta1,beta2))

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
        loss, all_loss = compute_loss_4dvar(model, increment, latent_scale, input_encoded, verif_ic, psobs_traj, epoch, exp, grid_interp, print_every, interp_specs, keep_mask, ref_state1)

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
        # saved as "new best" rather than caught here. See CLAUDE.md
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
        if not torch.isfinite(increment.grad).all():
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
        })
        with open(json_file, 'w') as f:
            json.dump(history, f, indent=4)

        if loss.item() < loss_min:
            logger.info('new best optimal...')
            lsave.append(loss.item())
            loss_min = loss.item()
            epoch_min = epoch
            increment_best = increment.detach().clone()
        else:
            lsave.append(lsave[-1])

        torch.nn.utils.clip_grad_norm_([increment], max_norm=1.0)

        optimizer.step()
        scheduler.step()
        logger.info(f'epoch={epoch}, lr={scheduler.get_last_lr()[0]}, loss={loss.item()}')

    if increment_best is not None:
        pt_file = exp['path_output'] + exp['date'] + '_latent_increment_' + exp['window_name'] + 'h' + exp['suffix'] + '_' + str(max_epoch) + 'it.pt'
        torch.save({'increment': increment_best, 'latent_scale': latent_scale}, pt_file)
        logger.info(f'saved best latent increment (for epoch={epoch_min}) to {pt_file}')
    else:
        logger.warning('no epoch improved on the background -- returning an UNCORRECTED forecast advance')
        lsave = lsave or [float('nan')]

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


def save_trajectory_diagnostics(exp, model, input_encoded, analysis_state, verif_ic, psobs_traj, save_levs, grid_interp, logger):
    interp_specs = _resolve_loss_interp_specs(model, exp)
    timestep_h = int(round(model.timestep.total_seconds() / 3600))
    dt_obs = exp['dt_obs']
    n_obs = int(psobs_traj['n_valid'].shape[0])
    step_lo = psobs_traj['step_lo'].tolist()

    background, initial_traj = compute_ps_observation_hx(
        model, input_encoded, verif_ic, psobs_traj, exp, grid_interp, interp_specs, step_start=0
    )

    # The latent analysis_state is valid at input_encoded.date + dt_verif
    # (see compute_optimal). compute_ps_observation_hx starts its trajectory
    # at that 6h-step index; obs slots earlier than it have no analysis value
    # and are NaN/0-padded here so the analysis arrays line up with the
    # full-window background arrays.
    shift_hours = (analysis_state.date - input_encoded.date).total_seconds() / 3600.0
    step_start = int(round(shift_hours / timestep_h))
    j_start = sum(1 for s in step_lo if s < step_start)
    analysis, final_traj = compute_ps_observation_hx(
        model, analysis_state, verif_ic, psobs_traj, exp, grid_interp, interp_specs, step_start=step_start
    )
    analysis['model_equivalent'] = _pad_leading(analysis['model_equivalent'], j_start, np.nan)
    analysis['used'] = _pad_leading(analysis['used'], j_start, False)
    analysis['qc_flag'] = _pad_leading(analysis['qc_flag'], j_start, 0)

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
