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

import gpsro_operator

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
# name. AIFS, Aurora, and ACE2 share this base-name convention ('t'/'u'/'v'/
# 'q' -- see this module's docstring); FCN3/SFNO have not been checked and
# load_config() rejects raobs_enabled for them until they have been (see
# RAOBS_SUPPORTED_BACKENDS below). 'specific_humidity' (q) has no
# reference-repo precedent -- see raobs_parquet.py's module docstring.
# ACE2, 2026-10-21 (see integrate_prepbufr_obs.md): get_raobs/get_amv have no
# backend-conditional code, and ACE2Model derives 't'/'u'/'v'/'q'/'z' on the
# same 13 ERA5 pressure levels, bottom-to-top, that AIFS/Aurora natively
# carry (ace2_model.py's _to_pressure_levels/decode_state), with a
# .weights(k=4) grid interpolator (ace2_grid.RectilinearBilinearInterpolator)
# matching the AMV_HORIZONTAL_K/GPSRO_HORIZONTAL_K=4 convention -- so both
# worked by construction once this guard was lifted. OFFLINE-validated only
# (no real checkpoint in this worktree's pointer-only ACE2-ERA5 submodule, so
# no GPU run yet): get_psobs/get_raobs/get_amv run end-to-end against the
# real parquet archive + a fake model exposing ACE2's actual F90
# Gauss-Legendre 180x360 grid and DEFAULT_PLEVS, with finite values, sane
# physical ranges, and interpolation weights summing to 1 -- see
# integrate_prepbufr_obs.md for the probe. ACE2's 'q' is specific_total_water
# (vapor + condensate), not a pure water-vapor specific humidity -- usually a
# small difference outside cloudy columns, but NOT verified against ERA5 q
# for this repo; treat raobs-q results on ACE2 as unreviewed until checked.
RAOBS_VARIABLE_MODEL_BASE = {
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'specific_humidity': 'q',
}
RAOBS_SUPPORTED_BACKENDS = ('aifs', 'aurora', 'ace2')

# amv (satellite-derived wind, read from the PREPBUFR archive's 'SATWND'
# class) obs-variable name -> model packed-state base name. 'u'/'v' decoding
# is offline-validated for AIFS/Aurora/ACE2 (get_amv has no backend-
# conditional code and ACE2's derived 'u'/'v' share AIFS's base-name/level-
# order convention, same as raobs above -- see the ACE2 note there);
# FCN3/SFNO have not been checked. See integrate_amv_satwnd.md.
AMV_VARIABLE_MODEL_BASE = {
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
}
AMV_SUPPORTED_BACKENDS = ('aifs', 'aurora', 'ace2')
# Horizontal k-NN neighbors per observation (same convention as psobs/
# raobs's grid_interp.weights(..., k=4)). Not a config knob -- changing it
# changes the composite-stencil node count (2*k) throughout get_amv/
# _compute_amv_observation_diagnostics_at_time together, so it stays a
# single source of truth here rather than a free-floating YAML value.
AMV_HORIZONTAL_K = 4

# aircraft (AIRCAR/AIRCFT) obs-variable name -> model packed-state base name.
# Same backend list as raobs/amv (get_aircraft has no backend-conditional
# code; offline-validated only, no real GPU run yet -- see
# integrate_aircraft_obs.md). AIRCAR_HORIZONTAL_K reuses AMV_HORIZONTAL_K's
# k=4 convention -- same stencil mechanism (_amv_horizontal_vertical_stencil
# is shared verbatim).
AIRCRAFT_VARIABLE_MODEL_BASE = {
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
}
AIRCRAFT_SUPPORTED_BACKENDS = ('aifs', 'aurora', 'ace2')

# gpsro (GPS radio-occultation bending angle) needs temperature, specific
# humidity, and geopotential on every model pressure level. Extended to
# 'ace2' 2026-10-22 (still not checked for FCN3/SFNO) -- confirmed directly
# (not assumed from the raobs/amv precedent) that ACE2Model.decode_state's
# 't'/'q'/'z' share one level set (self._plevs, no AIFS-style q-subset
# padding needed), are in the required bottom-to-top order, and that
# backpropagating through the 'z' (geopotential) derivation -- exercised
# here for the FIRST time, raobs/amv never read 'z' -- produces finite
# gradients (a CPU-only probe, since this derivation's code path differs
# structurally from the t/u/v/q block the 2026-10-21 NaN-gradient fix
# patched). See integrate_gpsro.md "Plan: extend GPSRO to the ACE2
# backend" for the full checklist, including the still-open ACE2 'q' vs
# ERA5 specific-humidity bias this inherits from raobs (now load-bearing,
# not just an optional caveat) and ACE2's much lower model top (50 hPa,
# ~20 km) requiring its own impact_height_range_km/top_extrapolation_km
# tuning rather than reusing AIFS's defaults.
GPSRO_SUPPORTED_BACKENDS = ('aifs', 'aurora', 'ace2')
GPSRO_HORIZONTAL_K = 4


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
}
# Default per-level raob q error (hPa: g/kg): Desroziers estimates from a
# ps + raobs run, cycles 2-10 (integrate_prepbufr_obs.md)
RAOBS_Q_ERROR_G_KG = {1000: 1.4, 925: 1.0, 850: 1.05, 700: 0.95, 600: 0.8, 500: 0.6, 400: 0.33, 300: 0.13}
RAOBS_Q_BG_CHECK = 4.0

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

# Nested `exp.observations.amv.<key>` -> flat `exp[<value>]` key, same
# convention as psobs/raobs above. Ported from the `long-window-4dvar`
# reference repo's `observation_sources.py` ('upper_air', source=
# 'prepbufr', prepbufr_classes=['SATWND']) -- see integrate_amv_satwnd.md
# for the full design and why the defaults below are what they are.
#
# Thinning (rmesh_km/pmesh_hpa) and the background gross check
# (background_check_stddev/error_min_ms/error_max_ms) are ported from
# GSI's actual SATWND treatment (`read_satwnd.f90`'s `rmesh`/`pmesh`
# box-thinning via `convthin`'s `map3grids_m`, and `setupw.f90`'s
# clamped-ratio gross check) -- see integrate_amv_satwnd.md "GSI
# comparison" for the exact source lines and why composite-stencil
# superobbing was replaced with this.
_AMV_CONFIG_KEYS = {
    'path': 'amv_path',
    'prepbufr_classes': 'amv_prepbufr_classes',
    'report_types': 'amv_report_types',
    'quality_mark_max': 'amv_quality_mark_max',
    'time_tolerance_hours': 'amv_time_tolerance_hours',
    'pressure_range_hpa': 'amv_pressure_range_hpa',
    'use_source_error': 'amv_use_source_error',
    'fallback_error_std_ms': 'amv_fallback_error_std_ms',
    'exclude_satellite_ids': 'amv_exclude_satellite_ids',
    'separate_report_types': 'amv_separate_report_types',
    'separate_satellite_ids': 'amv_separate_satellite_ids',
    'rmesh_km': 'amv_rmesh_km',
    'pmesh_hpa': 'amv_pmesh_hpa',
    'vertical_bin_mode': 'amv_vertical_bin_mode',
    'error_floor_ms': 'amv_error_floor_ms',
    'background_check_stddev': 'amv_background_check_stddev',
    'error_min_ms': 'amv_error_min_ms',
    'error_max_ms': 'amv_error_max_ms',
    'capacity': 'amv_capacity',
}

# Nested `exp.observations.aircraft.<key>` -> flat `exp[<value>]` key, same
# convention as amv above. Aircraft (AIRCAR/AIRCFT) is read, thinned, and
# forward-operated exactly like amv (vectorized read, native-pressure point
# stencil, GSI box thinning) EXCEPT it carries two independent sub-streams
# -- temperature ('t', report_type 133-style mass reports) and wind ('uv',
# report_type 233-style wind reports) -- because the archive reports them
# as mutually exclusive rows (see aircraft_parquet.py/
# integrate_aircraft_obs.md). `rmesh_km`/`pmesh_hpa`/`vertical_bin_mode`/
# `background_check_stddev` apply to BOTH sub-streams identically (one
# thinning policy, not two); error floors/clamps are per-sub-stream since
# temperature (K) and wind (m/s) are different units.
_AIRCRAFT_CONFIG_KEYS = {
    'path': 'aircraft_path',
    'prepbufr_classes': 'aircraft_prepbufr_classes',
    'report_types': 'aircraft_report_types',
    'quality_mark_max': 'aircraft_quality_mark_max',
    'time_tolerance_hours': 'aircraft_time_tolerance_hours',
    'pressure_range_hpa': 'aircraft_pressure_range_hpa',
    'use_source_error': 'aircraft_use_source_error',
    'fallback_error_std_t_k': 'aircraft_fallback_error_std_t_k',
    'fallback_error_std_ms': 'aircraft_fallback_error_std_ms',
    'separate_report_types': 'aircraft_separate_report_types',
    'rmesh_km': 'aircraft_rmesh_km',
    'pmesh_hpa': 'aircraft_pmesh_hpa',
    'vertical_bin_mode': 'aircraft_vertical_bin_mode',
    'error_floor_t_k': 'aircraft_error_floor_t_k',
    'error_floor_ms': 'aircraft_error_floor_ms',
    'background_check_stddev': 'aircraft_background_check_stddev',
    'error_min_t_k': 'aircraft_error_min_t_k',
    'error_max_t_k': 'aircraft_error_max_t_k',
    'error_min_ms': 'aircraft_error_min_ms',
    'error_max_ms': 'aircraft_error_max_ms',
    'capacity': 'aircraft_capacity',
    # Static pressure x region temperature bias-correction table -- see
    # build_aircraft_temperature_bias_table.py / integrate_aircraft_obs.md.
    # NOT per-tail (per-tail needs either this repo's own, admittedly-
    # imperfect ACE2 cycling to estimate online, or an external GSI run's
    # tail numbers, which share no overlap with this archive's own tail
    # IDs -- checked directly, zero overlap even for the same cycle date).
    'bias_correction_enabled': 'aircraft_bias_correction_enabled',
    'bias_correction_table': 'aircraft_bias_correction_table',
}

# Nested `exp.observations.gpsro.<key>` -> flat `exp[<value>]` key, same
# convention as the others above. Ported from the `long-window-4dvar`
# reference repo's `observation_sources.py` ('gpsro', source='parquet') --
# see integrate_gpsro.md for the full design and simplifications made
# relative to the reference (no frozen/quantile-scaled background check,
# no profile-shape QC, only the 'ecmwf_seed_v1' error model).
#
# error_min_rad/error_max_rad, top_extrapolation_km, and the SR
# (super-refraction) keys below are ported from GSI's actual GPSRO
# treatment (`read_gps.f90`/`setupbend.f90`), not from the JAX reference
# -- see integrate_gpsro.md "GSI comparison". `impact_height_spacing_m`
# (vertical thinning) is KEPT despite GSI not thinning GPSRO at all
# (confirmed directly) -- briefly removed, then restored the same day,
# 2026-10-20: GSI's own forecast models carry 60-150+ vertical levels, so
# an unthinned profile still has real degrees of freedom to fit against;
# this repo's 14-level model backends don't, so removing thinning
# inflated gpsro's effective weight in the loss by ~(levels-per-profile /
# 14), confirmed in a real run (gpsro's Jo grew ~6-7x and dominated the
# total loss) -- see integrate_gpsro.md "GSI comparison".
_GPSRO_CONFIG_KEYS = {
    'path': 'gpsro_path',
    'time_tolerance_hours': 'gpsro_time_tolerance_hours',
    'impact_height_range_km': 'gpsro_impact_height_range_km',
    'impact_height_spacing_m': 'gpsro_impact_height_spacing_m',
    'maximum_bending_angle_rad': 'gpsro_maximum_bending_angle_rad',
    'quality_flags_disqualifying_mask': 'gpsro_quality_flags_disqualifying_mask',
    'minimum_percent_confidence': 'gpsro_minimum_percent_confidence',
    'error_model': 'gpsro_error_model',
    'maximum_simulated_bending_angle_rad': 'gpsro_maximum_simulated_bending_angle_rad',
    'background_check_stddev': 'gpsro_background_check_stddev',
    'error_min_rad': 'gpsro_error_min_rad',
    'error_max_rad': 'gpsro_error_max_rad',
    'top_extrapolation_km': 'gpsro_top_extrapolation_km',
    'critical_gradient_n_per_km': 'gpsro_critical_gradient_n_per_km',
    'sr_detection_fraction': 'gpsro_sr_detection_fraction',
    'sr_screen_height_km': 'gpsro_sr_screen_height_km',
    'profile_capacity': 'gpsro_profile_capacity',
    'level_capacity': 'gpsro_level_capacity',
}


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
    unknown = set(observations) - {'psobs', 'raobs', 'amv', 'aircraft', 'gpsro'}
    if unknown:
        raise ValueError(f"exp.observations has unknown keys {sorted(unknown)} (expected 'psobs', 'raobs', 'amv', 'aircraft', and/or 'gpsro')")

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

    amv = observations.get('amv', {}) or {}
    unknown_amv = set(amv) - set(_AMV_CONFIG_KEYS) - {'enabled'}
    if unknown_amv:
        raise ValueError(f"exp.observations.amv has unknown keys {sorted(unknown_amv)}")
    exp['amv_enabled'] = bool(amv.get('enabled', False))
    if exp['amv_enabled']:
        for src_key, dst_key in _AMV_CONFIG_KEYS.items():
            if src_key in amv:
                exp[dst_key] = amv[src_key]
        exp.setdefault('amv_path', exp['obspath'])

    aircraft = observations.get('aircraft', {}) or {}
    unknown_aircraft = set(aircraft) - set(_AIRCRAFT_CONFIG_KEYS) - {'enabled'}
    if unknown_aircraft:
        raise ValueError(f"exp.observations.aircraft has unknown keys {sorted(unknown_aircraft)}")
    exp['aircraft_enabled'] = bool(aircraft.get('enabled', False))
    if exp['aircraft_enabled']:
        for src_key, dst_key in _AIRCRAFT_CONFIG_KEYS.items():
            if src_key in aircraft:
                exp[dst_key] = aircraft[src_key]
        exp.setdefault('aircraft_path', exp['obspath'])
        exp.setdefault('aircraft_bias_correction_enabled', False)
        exp.setdefault(
            'aircraft_bias_correction_table',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aircraft_temperature_bias_table.json'),
        )

    gpsro = observations.get('gpsro', {}) or {}
    unknown_gpsro = set(gpsro) - set(_GPSRO_CONFIG_KEYS) - {'enabled'}
    if unknown_gpsro:
        raise ValueError(f"exp.observations.gpsro has unknown keys {sorted(unknown_gpsro)}")
    exp['gpsro_enabled'] = bool(gpsro.get('enabled', False))
    if exp['gpsro_enabled']:
        for src_key, dst_key in _GPSRO_CONFIG_KEYS.items():
            if src_key in gpsro:
                exp[dst_key] = gpsro[src_key]
        if 'gpsro_path' not in exp:
            raise ValueError("exp.observations.gpsro.path is required (the GPSRO parquet archive root -- a different archive than observations.psobs.path)")


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
    if exp.get('raobs_enabled', False):
        if backend not in RAOBS_SUPPORTED_BACKENDS:
            raise ValueError(
                f"observations.raobs.enabled is only supported for model_backend in "
                f"{RAOBS_SUPPORTED_BACKENDS}, got {backend!r} -- the 't'/'u'/'v'/'q' base-name "
                "forward operator has only been exercised for AIFS/Aurora (real runs) and ACE2 "
                "(offline-validated, no GPU run yet) so far (see RAOBS_VARIABLE_MODEL_BASE)."
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

    # amv (satellite-derived winds, SATWND class): off by default, same
    # backend guard pattern as raobs -- see integrate_amv_satwnd.md.
    if exp.get('amv_enabled', False) and backend not in AMV_SUPPORTED_BACKENDS:
        raise ValueError(
            f"observations.amv.enabled is only supported for model_backend in "
            f"{AMV_SUPPORTED_BACKENDS}, got {backend!r} -- the 'u'/'v' base-name "
            "forward operator has only been exercised for AIFS/Aurora (real runs) and ACE2 "
            "(offline-validated, no GPU run yet) so far."
        )

    # aircraft (AIRCAR/AIRCFT): off by default, same backend guard pattern
    # as raobs/amv -- see integrate_aircraft_obs.md.
    if exp.get('aircraft_enabled', False) and backend not in AIRCRAFT_SUPPORTED_BACKENDS:
        raise ValueError(
            f"observations.aircraft.enabled is only supported for model_backend in "
            f"{AIRCRAFT_SUPPORTED_BACKENDS}, got {backend!r} -- the 't'/'u'/'v' base-name "
            "forward operator has only been exercised for raobs/amv on these backends so far."
        )

    # gpsro (GPS-RO bending angle): off by default, same backend guard
    # pattern as raobs/amv -- see integrate_gpsro.md.
    if exp.get('gpsro_enabled', False) and backend not in GPSRO_SUPPORTED_BACKENDS:
        raise ValueError(
            f"observations.gpsro.enabled is only supported for model_backend in "
            f"{GPSRO_SUPPORTED_BACKENDS}, got {backend!r} -- the 't'/'q'/'z' base-name "
            "forward operator has only been exercised for AIFS/Aurora so far."
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
    logger.info('   jc_ps_weight: ' + (str(exp['jc_ps_weight']) + ' hPa^-2 (x N_obs)'
                                        if exp.get('jc_ps_weight') else '(off)'))
    logger.info('   raobs_enabled: ' + (str(exp['raobs_variables']) if exp.get('raobs_enabled', False) else '(off, psobs only)'))
    logger.info('   amv_enabled: ' + ('True (SATWND, GSI-style box-thinned)' if exp.get('amv_enabled', False) else '(off)'))
    logger.info('   gpsro_enabled: ' + ('True (bending angle, 1D Abel operator)' if exp.get('gpsro_enabled', False) else '(off)'))
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
            # beyond them take the nearest entry; null falls back to the fraction
            table = exp.get('raobs_q_error_g_kg', RAOBS_Q_ERROR_G_KG)
            if table:
                tp = np.array(sorted(float(k) for k in table))
                tv = np.array([float(table[k]) for k in sorted(table, key=float)])
                lev = np.asarray(levels_hpa['q'], dtype=float)
                err = np.interp(np.log(lev), np.log(tp), tv) * 1.e-3
                raobs_traj['error_q_level'] = torch.as_tensor(err, dtype=torch.float32, device=device)
            bgc = exp.get('raobs_q_bg_check', RAOBS_Q_BG_CHECK)
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


# ---------------------------------------------------------------------------
# amv (satellite-derived wind) -- GSI-style box thinning
# ---------------------------------------------------------------------------
#
# Originally ported from the `long-window-4dvar` (JAX/NeuralGCM) reference
# repo's `observation_sources.py` ('upper_air', source='prepbufr',
# prepbufr_classes=['SATWND']) and `prepbufr_parquet.py`
# (`load_prepbufr_upper_air_points`/`aggregate_upper_air_superobs`), which
# group raw vectors sharing the SAME fixed interpolation stencil (same
# k-NN horizontal neighbors + same bracketing pressure-level pair) into one
# inverse-variance-weighted composite superob. That design matches the
# reference's REGULAR lat/lon grid, where "same stencil" is exactly one
# native grid cell -- but on this repo's irregular k-NN grid, "same exact
# ordered k=4-NN tuple" turns out to be a much finer partition (empirically
# ~5-30 km on AIFS's ~112 km-spacing O96 grid -- see integrate_amv_satwnd.md
# "GSI comparison"), so compression was minimal (~1.13x) and millions of
# near-raw-density, mutually correlated wind vectors entered the loss
# essentially independently, overwhelming the ps+raobs terms and (combined
# with no gross-error QC at all) measurably degrading forecast skill.
#
# Replaced with GSI's actual SATWND treatment, confirmed directly from
# `read_satwnd.f90`/`setupw.f90` (NOAA-EMC/GSI): a physical horizontal x
# vertical thinning MESH (`rmesh_km` x `pmesh_hpa`, independent of the
# model's native grid resolution, same as GSI's `convinfo` `rmesh`/`pmesh`
# columns), keeping exactly ONE observation per mesh cell -- the one with
# the lowest "badness" score (GSI's `crit1`, here `time_offset_hours/6 +
# 0.5`, tie-broken by source error since this repo's parquet archive
# doesn't carry the AMV-specific QI/EE fields GSI itself prefers) -- not an
# average. This is simpler (no in-group error-correlation bookkeeping) and
# structurally caps the assimilated density at one ob per mesh cell,
# independent of the model grid.


def _amv_horizontal_vertical_stencil(grid_interp, lon, lat, pressure_hpa, levels_hpa, k, device):
    """Per-point k-NN horizontal (grid_interp) + log-pressure vertical
    bracket stencil. `levels_hpa` need not be sorted. Returns
    (horizontal_idx (n,k) int64, horizontal_wts (n,k) float64,
    level_lo_idx (n,) int32, level_hi_idx (n,) int32, vertical_frac (n,)
    float64 in [0, 1], 0 at level_lo and 1 at level_hi). Pressures outside
    `levels_hpa`'s range are clipped to the nearest end (get_amv already
    intersects the read-time `pressure_range_hpa` filter with the model's
    own level range, so this should rarely bind)."""
    n = len(lon)
    if n == 0:
        return (
            np.zeros((0, k), dtype=np.int64), np.zeros((0, k), dtype=np.float64),
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )
    idx_t, wts_t = grid_interp.weights(np.asarray(lon), np.asarray(lat), k=k, device=device)
    horizontal_idx = idx_t.detach().cpu().numpy().astype(np.int64)
    horizontal_wts = wts_t.detach().cpu().numpy().astype(np.float64)

    levels_hpa = np.asarray(levels_hpa, dtype=np.float64)
    order = np.argsort(levels_hpa)
    sorted_levels = levels_hpa[order]
    pressure = np.clip(np.asarray(pressure_hpa, dtype=np.float64), sorted_levels[0], sorted_levels[-1])
    upper_position = np.searchsorted(sorted_levels, pressure, side='left')
    upper_position = np.clip(upper_position, 1, sorted_levels.size - 1)
    lower_position = upper_position - 1
    log_p = np.log(pressure)
    log_lo = np.log(sorted_levels[lower_position])
    log_hi = np.log(sorted_levels[upper_position])
    frac = np.where(log_hi > log_lo, (log_p - log_lo) / np.where(log_hi > log_lo, log_hi - log_lo, 1.), 0.)
    frac = np.clip(frac, 0., 1.)
    level_lo_idx = order[lower_position].astype(np.int32)
    level_hi_idx = order[upper_position].astype(np.int32)
    return horizontal_idx, horizontal_wts, level_lo_idx, level_hi_idx, frac


def _amv_thin_slot(
    lat, lon, pressure_hpa, components, error_std, report_type, time_offset_hours,
    horizontal_idx, horizontal_wts, level_lo_idx, level_hi_idx, vertical_frac,
    k, separate_report_types,
    rmesh_km, pmesh_hpa, error_floor, vertical_bin_mode='pmesh',
    satellite_id=None, separate_satellite_ids=False, extra_fields=None,
):
    """GSI-style box thinning of one verification slot's raw point
    observations -- shared by `get_amv` (`components={'u_component_of_wind':
    u, 'v_component_of_wind': v}`, `satellite_id` set) and `get_aircraft`
    (`components={'temperature': t}` or `{'u_component_of_wind': u,
    'v_component_of_wind': v}`, `satellite_id=None`, `extra_fields=
    {'aircraft_id': ...}` -- see integrate_aircraft_obs.md for why aircraft
    temperature and wind are thinned as two SEPARATE calls, not one
    combined 3-variable call: the archive reports them as mutually
    exclusive rows, so a single shared box would silently keep only a
    temperature OR a wind row per box). `components` values and
    `extra_fields` values are carried through and subset by the same
    selection as every other per-row field -- this function itself has no
    opinion on how many/which variables are attached to a selected raw
    observation.

    `satellite_id`/`separate_satellite_ids` are AMV-specific (SATWND's
    satellite identity): when `satellite_id` is None (the default), it is
    omitted from both the grouping key and the output -- used by any
    caller, like aircraft, with no comparable per-row identity to group by.

    Below, "raw AMV points" in the historical docstring text generalizes to
    "raw points" for any caller.
    keep exactly ONE raw wind vector per (horizontal rmesh_km bin x
    vertical bin [x report_type] [x satellite_id]) box -- the one
    with the lowest `crit1` "badness" score, not an average. See the
    module-level comment above and integrate_amv_satwnd.md's "GSI
    comparison" for why (ported from `read_satwnd.f90`'s own thinning).

    The horizontal bin is a reduced (near-equal-area) lat/lon mesh: latitude
    bands of height ~rmesh_km, longitude bins per band narrowed by
    cos(latitude) so bin width stays ~rmesh_km at every latitude -- GSI's
    own `make3grids` earth mesh does the equivalent. This is INDEPENDENT of
    the model's native grid, unlike the old same-k-NN-tuple grouping.

    `vertical_bin_mode` (default `'pmesh'`, GSI's own scheme): a fixed-width
    `pmesh_hpa`-hPa pressure bin, independent of the model's own vertical
    levels -- correct for GSI's own 64-150+ level host models, but on a
    model with few real vertical degrees of freedom (e.g. ACE2's 8 native
    layers) a uniform-width mesh keeps more vertically-independent-looking
    observations than the model can actually resolve, AND the single kept
    observation per bin (selected purely by time-closeness, not pressure)
    can land anywhere within that bin's span, arbitrarily far from any real
    model level -- see integrate_amv_satwnd.md "Vertical thinning tied to
    model levels" for the diagnosis (ACE2, 2026-10-21).
    `vertical_bin_mode='nearest_level'` bins by the observation's own
    nearest MODEL pressure level instead (`level_lo_idx`/`level_hi_idx`/
    `vertical_frac`, already computed by `_amv_horizontal_vertical_stencil`
    for the forward-operator stencil -- no new computation needed): the
    vertical bin count becomes exactly the model's own level count, and
    every kept observation is, by construction, closer to its assigned
    level than to any other (the Voronoi cell of that level in
    log-pressure) -- not just probably close, as a weighted/combined score
    would only encourage. `pmesh_hpa` is unused in this mode. Selection
    *within* a bin is still GSI's own `crit1` (time-closeness) either way --
    this only changes where bin boundaries fall, not how ties within a bin
    are broken.

    `crit1 = time_offset_hours/6 + 0.5` is GSI's own formula for every
    SATWND type (`read_satwnd.f90` line ~1214); GSI tightens this further
    for some types using the AMV-specific QI/expected-error fields, which
    this repo's parquet archive doesn't carry (see integrate_amv_satwnd.md)
    -- a small source-error tie-break (`+ 1e-3 * sigma`) substitutes for
    that, preferring the better-quality member when times are equal.

    The output values ARE the selected raw observation's own values and its
    own (horizontal k-NN x vertical log-p) stencil -- no averaging, so no
    in-group error-correlation combination is needed either."""
    extra_fields = extra_fields or {}
    count = int(np.asarray(lat).size)
    if count == 0:
        empty_dtypes = {
            'lat': np.float32, 'lon': np.float32, 'pressure_hpa': np.float32,
            'error_std': np.float32, 'report_type': np.int32, 'component_count': np.int32,
        }
        for name, arr in components.items():
            empty_dtypes[name] = np.asarray(arr).dtype if np.asarray(arr).size else np.float32
        for name, arr in extra_fields.items():
            empty_dtypes[name] = np.asarray(arr).dtype if np.asarray(arr).size else object
        if satellite_id is not None:
            empty_dtypes['satellite_id'] = np.int32
        result = {name: np.empty(0, dtype=dtype) for name, dtype in empty_dtypes.items()}
        result['node_point_idx'] = np.empty((0, 2 * k), dtype=np.int64)
        result['node_level_idx'] = np.empty((0, 2 * k), dtype=np.int32)
        result['node_weights'] = np.empty((0, 2 * k), dtype=np.float32)
        return result

    report_type = np.asarray(report_type, dtype=np.int32)
    if satellite_id is not None:
        satellite_id = np.asarray(satellite_id, dtype=np.int32)
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    pressure_arr = np.asarray(pressure_hpa, dtype=np.float64)

    sigma = np.asarray(error_std, dtype=np.float64)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.):
        raise ValueError('component error_std must be finite and positive.')
    sigma = np.maximum(float(error_floor), sigma)

    # Reduced horizontal mesh: lat bands of height ~rmesh_km, lon bins per
    # band narrowed by cos(lat) to keep near-equal-area cells.
    km_per_deg_lat = 111.19
    nlat_bins = max(1, int(round(180. * km_per_deg_lat / float(rmesh_km))))
    lat_band = np.clip(((lat_arr + 90.) / 180. * nlat_bins).astype(np.int64), 0, nlat_bins - 1)
    band_center_lat = (lat_band + 0.5) * (180. / nlat_bins) - 90.
    nlon_bins = np.maximum(1, np.round(
        360. * np.cos(np.radians(band_center_lat)) * km_per_deg_lat / float(rmesh_km)
    )).astype(np.int64)
    lon_bin = np.clip((lon_arr / 360. * nlon_bins).astype(np.int64), 0, nlon_bins - 1)
    if vertical_bin_mode == 'nearest_level':
        # Nearest model pressure level (by the SAME log-p bracket already
        # computed for the forward-operator stencil) as the vertical bin
        # key, instead of GSI's fixed-width pmesh_hpa mesh -- see the
        # docstring above.
        vertical_bin = np.where(
            np.asarray(vertical_frac) < 0.5,
            np.asarray(level_lo_idx), np.asarray(level_hi_idx),
        ).astype(np.int64)
    elif vertical_bin_mode == 'pmesh':
        vertical_bin = np.floor(pressure_arr / float(pmesh_hpa)).astype(np.int64)
    else:
        raise ValueError(f"vertical_bin_mode must be 'pmesh' or 'nearest_level', got {vertical_bin_mode!r}")

    key_parts = [lat_band[:, None], lon_bin[:, None], vertical_bin[:, None]]
    if separate_report_types:
        key_parts.append(report_type[:, None].astype(np.int64))
    if separate_satellite_ids and satellite_id is not None:
        key_parts.append(satellite_id[:, None].astype(np.int64))
    keys = np.concatenate(key_parts, axis=1)
    _, inverse, group_sizes = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    group_count = group_sizes.size

    timedif = np.asarray(time_offset_hours, dtype=np.float64)
    crit1 = timedif / 6. + 0.5 + 1.e-3 * sigma

    # Vectorized argmin-per-group: sort by (group, score), take each
    # group's first row in that order (same unique/searchsorted idiom used
    # elsewhere in this module).
    order = np.lexsort((crit1, inverse))
    sorted_inverse = inverse[order]
    first_pos = np.searchsorted(sorted_inverse, np.arange(group_count))
    best_idx = order[first_pos]

    per_point_node_weights = np.concatenate((
        horizontal_wts * (1. - vertical_frac)[:, None],
        horizontal_wts * vertical_frac[:, None],
    ), axis=1)

    order_out = np.argsort(best_idx)
    sel = best_idx[order_out]
    result = {
        'lat': lat_arr[sel].astype(np.float32),
        'lon': lon_arr[sel].astype(np.float32),
        'pressure_hpa': pressure_arr[sel].astype(np.float32),
        'error_std': sigma[sel].astype(np.float32),
        'report_type': report_type[sel],
        'component_count': group_sizes[order_out].astype(np.int32),
        'node_point_idx': np.tile(horizontal_idx[sel], (1, 2)).astype(np.int64),
        'node_level_idx': np.repeat(
            np.stack((level_lo_idx[sel], level_hi_idx[sel]), axis=1), k, axis=1
        ).astype(np.int32),
        'node_weights': per_point_node_weights[sel].astype(np.float32),
    }
    if satellite_id is not None:
        result['satellite_id'] = satellite_id[sel]
    for name, values in components.items():
        result[name] = np.asarray(values, dtype=np.float64)[sel].astype(np.float32)
    for name, values in extra_fields.items():
        result[name] = np.asarray(values, dtype=object)[sel]
    return result


def get_amv(exp, model, logger, grid_interp, psobs_traj):
    """Read quality-controlled satellite-derived wind (AMV) observations
    from the PREPBUFR parquet archive's 'SATWND' class and GSI-style box-
    thin them to one observation per mesh cell (see the module comment
    above). Returns `None` if `exp['amv_enabled']` is not True -- callers
    (`compute_optimal`, `compute_loss_4dvar`, `compute_ps_observation_hx`,
    `save_trajectory_diagnostics`) all accept `amv_traj=None` and skip the
    amv term entirely, so a config that never sets `amv_enabled`
    assimilates exactly as before.

    Piggybacks on psobs's obs-slot grid (same `step_lo`/`alpha` time
    bracketing), same as get_raobs -- real SATWND reports are available at
    any synoptic hour, so this needs no independent time grid.
    """
    if not exp.get('amv_enabled', False):
        return None
    from amv_parquet import DEFAULT_AMV_CLASSES, load_amv_parquet

    device = model.device
    dt_obs = exp['dt_obs']
    date = exp['date']
    vdate = copy.copy(date)
    verification_times = []
    for _ in range(len(psobs_traj['step_lo'])):
        verification_times.append(vdate)
        vdate = add_hours(vdate, dt_obs)
    n_obs = len(verification_times)

    u_levels = np.asarray(model.pressure_levels('u'), dtype=np.float64)
    v_levels = np.asarray(model.pressure_levels('v'), dtype=np.float64)
    if u_levels.shape != v_levels.shape or not np.allclose(u_levels, v_levels):
        raise ValueError("amv_enabled requires model 'u' and 'v' to share the same pressure levels.")

    default_pressure_range = (float(u_levels.max()), float(u_levels.min()))
    configured_range = exp.get('amv_pressure_range_hpa', default_pressure_range)
    pressure_min = max(float(min(configured_range)), float(u_levels.min()))
    pressure_max = min(float(max(configured_range)), float(u_levels.max()))
    if pressure_min >= pressure_max:
        raise ValueError('amv_pressure_range_hpa does not overlap the model pressure levels.')

    points = load_amv_parquet(
        exp.get('amv_path', exp['obspath']),
        verification_times,
        time_tolerance_hours=exp.get('amv_time_tolerance_hours', 1.),
        prepbufr_classes=exp.get('amv_prepbufr_classes', DEFAULT_AMV_CLASSES),
        report_types=exp.get('amv_report_types'),
        quality_mark_max=exp.get('amv_quality_mark_max', 3),
        pressure_range_hpa=(pressure_max, pressure_min),
        use_source_error=exp.get('amv_use_source_error', True),
        fallback_error_std_ms=exp.get('amv_fallback_error_std_ms', 7.),
        exclude_satellite_ids=exp.get('amv_exclude_satellite_ids', ()),
        logger=logger,
    )

    k = AMV_HORIZONTAL_K
    separate_report_types = exp.get('amv_separate_report_types', True)
    separate_satellite_ids = exp.get('amv_separate_satellite_ids', True)
    error_floor_ms = exp.get('amv_error_floor_ms', 4.)
    # GSI-style thinning mesh: rmesh_km (horizontal) x a vertical bin,
    # independent of the model's native grid -- see _amv_thin_slot and
    # integrate_amv_satwnd.md. Default 120 km matches the conventional-data
    # example in the GSI User's Guide (section 3.4.7).
    rmesh_km = float(exp.get('amv_rmesh_km', 120.))
    pmesh_hpa = float(exp.get('amv_pmesh_hpa', 50.))
    # vertical_bin_mode: 'pmesh' (default, GSI's own fixed-width pmesh_hpa
    # mesh -- 50 hPa is a starting point, not a confirmed operational SATWND
    # value) or 'nearest_level' (bins by the model's own nearest pressure
    # level instead -- for a model with few real vertical degrees of
    # freedom, e.g. ACE2's 8 native layers, this both caps the vertical bin
    # count at the model's own level count and guarantees the kept
    # observation is closer to its assigned level than to any other, by
    # construction -- see _amv_thin_slot's docstring and
    # integrate_amv_satwnd.md "Vertical thinning tied to model levels").
    vertical_bin_mode = exp.get('amv_vertical_bin_mode', 'pmesh')
    if vertical_bin_mode not in ('pmesh', 'nearest_level'):
        raise ValueError(
            f"amv_vertical_bin_mode must be 'pmesh' or 'nearest_level', got {vertical_bin_mode!r}"
        )

    thinned_by_slot = []
    for j in range(n_obs):
        horizontal_idx, horizontal_wts, level_lo_idx, level_hi_idx, vertical_frac = (
            _amv_horizontal_vertical_stencil(
                grid_interp, points['lon'][j], points['lat'][j], points['pressure_hpa'][j],
                u_levels, k, device,
            )
        )
        thinned_by_slot.append(_amv_thin_slot(
            points['lat'][j], points['lon'][j], points['pressure_hpa'][j],
            {
                'u_component_of_wind': points['u_component_of_wind'][j],
                'v_component_of_wind': points['v_component_of_wind'][j],
            },
            points['error_std'][j], points['report_type'][j],
            points['time_offset_hours'][j],
            horizontal_idx, horizontal_wts, level_lo_idx, level_hi_idx, vertical_frac,
            k, separate_report_types,
            rmesh_km, pmesh_hpa, error_floor_ms, vertical_bin_mode=vertical_bin_mode,
            satellite_id=points['satellite_id'][j], separate_satellite_ids=separate_satellite_ids,
        ))

    counts = [int(thinned_by_slot[j]['lat'].size) for j in range(n_obs)]
    capacity = int(exp.get('amv_capacity', max(1, max(counts) if counts else 1)))
    if counts and max(counts) > capacity:
        raise ValueError(f"amv thinned count {max(counts)} exceeds configured amv_capacity {capacity}")

    amv_traj = {
        'n_valid': torch.as_tensor(counts, dtype=torch.int32, device=device),
        'levels_hpa': torch.as_tensor(u_levels, dtype=torch.float32, device=device),
    }
    scalar_fields = {
        'lat': torch.float32, 'lon': torch.float32, 'pressure_hpa': torch.float32,
        'u_component_of_wind': torch.float32, 'v_component_of_wind': torch.float32,
        'error_std': torch.float32, 'report_type': torch.int32, 'satellite_id': torch.int32,
        'component_count': torch.int32,
    }
    for name, dtype in scalar_fields.items():
        padded = torch.zeros((n_obs, capacity), dtype=dtype, device=device)
        for j in range(n_obs):
            n = counts[j]
            if n:
                padded[j, :n] = torch.as_tensor(thinned_by_slot[j][name], device=device)
        amv_traj[name] = padded
    mask = torch.zeros((n_obs, capacity), dtype=torch.bool, device=device)
    for j in range(n_obs):
        mask[j, :counts[j]] = True
    amv_traj['mask'] = mask

    for name, dtype in (('node_point_idx', torch.int64), ('node_level_idx', torch.int64)):
        padded = torch.zeros((n_obs, capacity, 2 * k), dtype=dtype, device=device)
        for j in range(n_obs):
            n = counts[j]
            if n:
                padded[j, :n] = torch.as_tensor(thinned_by_slot[j][name], dtype=dtype, device=device)
        amv_traj[name] = padded
    padded_weights = torch.zeros((n_obs, capacity, 2 * k), dtype=torch.float32, device=device)
    for j in range(n_obs):
        n = counts[j]
        if n:
            padded_weights[j, :n] = torch.as_tensor(thinned_by_slot[j]['node_weights'], device=device)
    amv_traj['node_weights'] = padded_weights

    # Background gross check (GSI's setupw.f90, Stage C -- see
    # integrate_amv_satwnd.md "GSI comparison"): ratio = |O-B| /
    # clip(error, error_min_ms, error_max_ms) > background_check_stddev
    # rejects the observation. error_min_ms/error_max_ms default to no
    # clamping (plain |O-B|/error, same convention as raobs's q_bg_check
    # and gpsro's background_check_stddev) -- set both to adopt GSI's full
    # clamped-ratio form.
    bgc = exp.get('amv_background_check_stddev', 4.0)
    amv_traj['background_check_stddev'] = float(bgc) if bgc else None
    if exp.get('amv_error_min_ms') is not None:
        amv_traj['error_min_ms'] = float(exp['amv_error_min_ms'])
    if exp.get('amv_error_max_ms') is not None:
        amv_traj['error_max_ms'] = float(exp['amv_error_max_ms'])

    amv_traj['n_steps'] = psobs_traj['n_steps']
    amv_traj['step_lo'] = psobs_traj['step_lo']
    amv_traj['alpha'] = psobs_traj['alpha']
    return amv_traj


# ---------------------------------------------------------------------------
# aircraft (AIRCAR/AIRCFT) -- see integrate_aircraft_obs.md
# ---------------------------------------------------------------------------
#
# Structurally the same as amv (vectorized read, native-pressure point
# stencil, GSI box thinning, same `_amv_horizontal_vertical_stencil`/
# `_amv_thin_slot` machinery) EXCEPT the archive reports aircraft
# temperature and wind as mutually exclusive rows (confirmed directly:
# report_type 133 = temperature-only, 233 = wind-only, zero rows have
# both), so get_aircraft thins them as two INDEPENDENT sub-streams ('t',
# 'uv') -- a single shared thinning box keyed across all three variables
# would otherwise silently keep only a temperature OR a wind row per box.
# `aircraft_traj` is therefore `{'t': {...}, 'uv': {...}, <shared keys>}`,
# not one flat dict of padded fields the way amv_traj is.

_aircraft_bias_table_cache = {}


def _load_aircraft_temperature_bias_table(path):
    """Static pressure x region aircraft temperature bias table (K) --
    see build_aircraft_temperature_bias_table.py / integrate_aircraft_obs.md
    for what this is and why it's NOT per-tail. Cached in-process (read
    once per path, reused across every cycle/window in a run).

    Returns `(bin_hi, bin_lo, values)`: `bin_hi`/`bin_lo` are the pressure
    bin edges (hPa, high/surface first), and `values[region]` is a
    same-length array of mean O-B (NaN where the source archive had no
    data for that bin/region -- new combination, nothing to correct).
    """
    if path not in _aircraft_bias_table_cache:
        with open(path) as f:
            raw = json.load(f)
        edges = raw['pressure_bin_edges_hpa']
        bin_hi = np.asarray(edges[:-1], dtype=np.float64)
        bin_lo = np.asarray(edges[1:], dtype=np.float64)
        regions = ('NH', 'Tropics', 'SH')
        values = {r: np.full(len(bin_hi), np.nan) for r in regions}
        for i in range(len(bin_hi)):
            label = f'{int(bin_lo[i])}-{int(bin_hi[i])}'
            for r in regions:
                v = raw['table'][label][r]['mean_omb_k']
                if v is not None:
                    values[r][i] = v
        _aircraft_bias_table_cache[path] = (bin_hi, bin_lo, values)
    return _aircraft_bias_table_cache[path]


def _aircraft_temperature_bias_correction(pressure_hpa, lat, path):
    """Per-observation bias (K) to SUBTRACT from the raw temperature
    observation (`corrected = raw - bias`), looked up by (pressure bin,
    region) from the static table in `path`. 0 wherever the table has no
    data for that bin/region (outside the table's pressure range, or a
    region/bin combination the source archive didn't cover) -- an
    unknown correction is better left as no correction than guessed at.
    """
    bin_hi, bin_lo, values = _load_aircraft_temperature_bias_table(path)
    pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)
    region = np.where(lat > 20., 'NH', np.where(lat < -20., 'SH', 'Tropics'))
    bias = np.zeros_like(pressure_hpa)
    for i in range(len(bin_hi)):
        in_bin = (pressure_hpa <= bin_hi[i]) & (pressure_hpa > bin_lo[i])
        if not in_bin.any():
            continue
        for r in ('NH', 'Tropics', 'SH'):
            v = values[r][i]
            if np.isfinite(v):
                sel = in_bin & (region == r)
                bias[sel] = v
    return bias


def get_aircraft(exp, model, logger, grid_interp, psobs_traj):
    """Read quality-controlled aircraft (AIRCAR/AIRCFT) temperature and
    wind observations and GSI-style box-thin them, independently per
    sub-stream (see the module comment above). Returns `None` if
    `exp['aircraft_enabled']` is not True -- callers all accept
    `aircraft_traj=None` and skip the aircraft term entirely, so a config
    that never sets `aircraft_enabled` assimilates exactly as before.

    Piggybacks on psobs's obs-slot grid, same as get_raobs/get_amv.
    """
    if not exp.get('aircraft_enabled', False):
        return None
    from aircraft_parquet import DEFAULT_AIRCRAFT_CLASSES, load_aircraft_parquet

    device = model.device
    dt_obs = exp['dt_obs']
    date = exp['date']
    vdate = copy.copy(date)
    verification_times = []
    for _ in range(len(psobs_traj['step_lo'])):
        verification_times.append(vdate)
        vdate = add_hours(vdate, dt_obs)
    n_obs = len(verification_times)

    t_levels = np.asarray(model.pressure_levels('t'), dtype=np.float64)
    u_levels = np.asarray(model.pressure_levels('u'), dtype=np.float64)
    v_levels = np.asarray(model.pressure_levels('v'), dtype=np.float64)
    if u_levels.shape != v_levels.shape or not np.allclose(u_levels, v_levels):
        raise ValueError("aircraft_enabled requires model 'u' and 'v' to share the same pressure levels.")

    # Pressure range read: intersection of BOTH variable groups' model
    # level ranges (in practice identical sets for the backends this
    # supports, but computed safely in case they ever differ) -- same
    # pattern as get_amv, extended to cover 't' too.
    model_min = max(float(t_levels.min()), float(u_levels.min()))
    model_max = min(float(t_levels.max()), float(u_levels.max()))
    default_pressure_range = (model_max, model_min)
    configured_range = exp.get('aircraft_pressure_range_hpa', default_pressure_range)
    pressure_min = max(float(min(configured_range)), model_min)
    pressure_max = min(float(max(configured_range)), model_max)
    if pressure_min >= pressure_max:
        raise ValueError('aircraft_pressure_range_hpa does not overlap the model pressure levels.')

    points = load_aircraft_parquet(
        exp.get('aircraft_path', exp['obspath']),
        verification_times,
        time_tolerance_hours=exp.get('aircraft_time_tolerance_hours', 1.),
        prepbufr_classes=exp.get('aircraft_prepbufr_classes', DEFAULT_AIRCRAFT_CLASSES),
        report_types=exp.get('aircraft_report_types'),
        quality_mark_max=exp.get('aircraft_quality_mark_max', 3),
        pressure_range_hpa=(pressure_max, pressure_min),
        use_source_error=exp.get('aircraft_use_source_error', True),
        fallback_error_std_t_k=exp.get('aircraft_fallback_error_std_t_k', 1.5),
        fallback_error_std_ms=exp.get('aircraft_fallback_error_std_ms', 3.5),
        logger=logger,
    )

    # Static pressure x region temperature bias correction (NOT per-tail --
    # see build_aircraft_temperature_bias_table.py / integrate_aircraft_obs.md).
    # Applied here, on the raw 't' observation before thinning, so every
    # downstream consumer (thinning, the forward operator, the loss term,
    # the diagnostics writer) sees "the observation" as already corrected --
    # no other code path needs to know this happened. Wind ('uv') is left
    # untouched, same scope as GSI's own aircraft bias correction (and a
    # real-data check found no systematic per-tail wind bias to correct --
    # see integrate_aircraft_obs.md).
    if exp.get('aircraft_bias_correction_enabled', False):
        bias_table_path = exp['aircraft_bias_correction_table']
        for j in range(n_obs):
            if points['t']['temperature'][j].size:
                bias = _aircraft_temperature_bias_correction(
                    points['t']['pressure_hpa'][j], points['t']['lat'][j], bias_table_path,
                )
                points['t']['temperature'][j] = points['t']['temperature'][j] - bias

    k = AMV_HORIZONTAL_K
    separate_report_types = exp.get('aircraft_separate_report_types', True)
    error_floor_t_k = exp.get('aircraft_error_floor_t_k', 0.5)
    error_floor_ms = exp.get('aircraft_error_floor_ms', 2.0)
    # Same GSI-style thinning mesh for both sub-streams -- one thinning
    # policy, not two (see module comment above).
    rmesh_km = float(exp.get('aircraft_rmesh_km', 120.))
    pmesh_hpa = float(exp.get('aircraft_pmesh_hpa', 50.))
    vertical_bin_mode = exp.get('aircraft_vertical_bin_mode', 'pmesh')
    if vertical_bin_mode not in ('pmesh', 'nearest_level'):
        raise ValueError(
            f"aircraft_vertical_bin_mode must be 'pmesh' or 'nearest_level', got {vertical_bin_mode!r}"
        )

    def _thin_substream(sub_points, sub_levels, component_names, error_floor):
        thinned_by_slot = []
        for j in range(n_obs):
            horizontal_idx, horizontal_wts, level_lo_idx, level_hi_idx, vertical_frac = (
                _amv_horizontal_vertical_stencil(
                    grid_interp, sub_points['lon'][j], sub_points['lat'][j], sub_points['pressure_hpa'][j],
                    sub_levels, k, device,
                )
            )
            components = {name: sub_points[name][j] for name in component_names}
            thinned_by_slot.append(_amv_thin_slot(
                sub_points['lat'][j], sub_points['lon'][j], sub_points['pressure_hpa'][j],
                components, sub_points['error_std'][j], sub_points['report_type'][j],
                sub_points['time_offset_hours'][j],
                horizontal_idx, horizontal_wts, level_lo_idx, level_hi_idx, vertical_frac,
                k, separate_report_types,
                rmesh_km, pmesh_hpa, error_floor, vertical_bin_mode=vertical_bin_mode,
                extra_fields={'aircraft_id': sub_points['aircraft_id'][j]},
            ))
        return thinned_by_slot

    thinned_t = _thin_substream(points['t'], t_levels, ['temperature'], error_floor_t_k)
    thinned_uv = _thin_substream(
        points['uv'], u_levels, ['u_component_of_wind', 'v_component_of_wind'], error_floor_ms,
    )

    def _pad_substream(thinned_by_slot, levels_array, scalar_fields):
        counts = [int(thinned_by_slot[j]['lat'].size) for j in range(n_obs)]
        capacity = int(exp.get('aircraft_capacity', max(1, max(counts) if counts else 1)))
        if counts and max(counts) > capacity:
            raise ValueError(f"aircraft thinned count {max(counts)} exceeds configured aircraft_capacity {capacity}")
        traj = {
            'n_valid': torch.as_tensor(counts, dtype=torch.int32, device=device),
            'levels_hpa': torch.as_tensor(levels_array, dtype=torch.float32, device=device),
        }
        for name, dtype in scalar_fields.items():
            padded = torch.zeros((n_obs, capacity), dtype=dtype, device=device)
            for j in range(n_obs):
                n = counts[j]
                if n:
                    padded[j, :n] = torch.as_tensor(thinned_by_slot[j][name], device=device)
            traj[name] = padded
        mask = torch.zeros((n_obs, capacity), dtype=torch.bool, device=device)
        for j in range(n_obs):
            mask[j, :counts[j]] = True
        traj['mask'] = mask
        for name, dtype in (('node_point_idx', torch.int64), ('node_level_idx', torch.int64)):
            padded = torch.zeros((n_obs, capacity, 2 * k), dtype=dtype, device=device)
            for j in range(n_obs):
                n = counts[j]
                if n:
                    padded[j, :n] = torch.as_tensor(thinned_by_slot[j][name], dtype=dtype, device=device)
            traj[name] = padded
        padded_weights = torch.zeros((n_obs, capacity, 2 * k), dtype=torch.float32, device=device)
        for j in range(n_obs):
            n = counts[j]
            if n:
                padded_weights[j, :n] = torch.as_tensor(thinned_by_slot[j]['node_weights'], device=device)
        traj['node_weights'] = padded_weights
        # aircraft_id: diagnostic-only identity metadata (for future
        # per-aircraft bias-correction work -- not used anywhere in this
        # pass), kept as a plain numpy object array of strings, NOT a torch
        # tensor (it never participates in any differentiable computation).
        # '' pads unused slots.
        aircraft_id = np.full((n_obs, capacity), '', dtype=object)
        for j in range(n_obs):
            n = counts[j]
            if n:
                aircraft_id[j, :n] = thinned_by_slot[j]['aircraft_id']
        traj['aircraft_id'] = aircraft_id
        return traj

    aircraft_traj = {
        't': _pad_substream(thinned_t, t_levels, {
            'lat': torch.float32, 'lon': torch.float32, 'pressure_hpa': torch.float32,
            'temperature': torch.float32, 'error_std': torch.float32,
            'report_type': torch.int32, 'component_count': torch.int32,
        }),
        'uv': _pad_substream(thinned_uv, u_levels, {
            'lat': torch.float32, 'lon': torch.float32, 'pressure_hpa': torch.float32,
            'u_component_of_wind': torch.float32, 'v_component_of_wind': torch.float32,
            'error_std': torch.float32, 'report_type': torch.int32, 'component_count': torch.int32,
        }),
    }

    # Background gross check (same convention as amv -- see get_amv),
    # applied identically to both sub-streams; error floors/clamps are
    # per-sub-stream (temperature K vs wind m/s are different units).
    bgc = exp.get('aircraft_background_check_stddev', 4.0)
    aircraft_traj['background_check_stddev'] = float(bgc) if bgc else None
    if exp.get('aircraft_error_min_t_k') is not None:
        aircraft_traj['error_min_t_k'] = float(exp['aircraft_error_min_t_k'])
    if exp.get('aircraft_error_max_t_k') is not None:
        aircraft_traj['error_max_t_k'] = float(exp['aircraft_error_max_t_k'])
    if exp.get('aircraft_error_min_ms') is not None:
        aircraft_traj['error_min_ms'] = float(exp['aircraft_error_min_ms'])
    if exp.get('aircraft_error_max_ms') is not None:
        aircraft_traj['error_max_ms'] = float(exp['aircraft_error_max_ms'])

    aircraft_traj['n_steps'] = psobs_traj['n_steps']
    aircraft_traj['step_lo'] = psobs_traj['step_lo']
    aircraft_traj['alpha'] = psobs_traj['alpha']
    return aircraft_traj


def _compute_aircraft_observation_diagnostics_at_time(decoded, aircraft_traj, oind):
    """Aircraft H(x) for one verification-time slot -- temperature ('t')
    and wind ('uv') sub-streams evaluated independently, each against its
    own fixed 2*k-node stencil, same mechanism as
    `_compute_amv_observation_diagnostics_at_time` (duplicated rather than
    generalized -- short function, keeps amv's own validated version
    untouched; see integrate_aircraft_obs.md).

    Background gross check: same ported GSI form as amv, with separate
    (temperature K / wind m/s) clamp bounds, both evaluated at the CURRENT
    state every loss call.
    """
    results = {}
    background_check = aircraft_traj.get('background_check_stddev')

    sub_t = aircraft_traj['t']
    field_t = decoded['t']
    gathered_t = field_t[sub_t['node_level_idx'][oind], sub_t['node_point_idx'][oind]]
    model_obspace_t = (gathered_t * sub_t['node_weights'][oind]).sum(dim=-1)
    observation_t = sub_t['temperature'][oind]
    interpolation_failed_t = ~torch.isfinite(model_obspace_t)
    used_t = sub_t['mask'][oind] & ~interpolation_failed_t
    safe_model_obspace_t = torch.where(interpolation_failed_t, torch.zeros_like(model_obspace_t), model_obspace_t)
    innovation_t = observation_t - safe_model_obspace_t
    effective_error_t = sub_t['error_std'][oind]
    if background_check:
        qc_error_t = effective_error_t
        lo = aircraft_traj.get('error_min_t_k')
        hi = aircraft_traj.get('error_max_t_k')
        if lo is not None or hi is not None:
            qc_error_t = qc_error_t.clamp(min=lo if lo is not None else 0., max=hi if hi is not None else float('inf'))
        used_t = used_t & (innovation_t.abs() <= float(background_check) * qc_error_t).detach()
    results['t'] = (innovation_t, safe_model_obspace_t, effective_error_t, used_t)

    sub_uv = aircraft_traj['uv']
    mask_uv = sub_uv['mask'][oind]
    for obs_key, base in (('u_component_of_wind', 'u'), ('v_component_of_wind', 'v')):
        field = decoded[base]
        gathered = field[sub_uv['node_level_idx'][oind], sub_uv['node_point_idx'][oind]]
        model_obspace = (gathered * sub_uv['node_weights'][oind]).sum(dim=-1)
        observation = sub_uv[obs_key][oind]
        interpolation_failed = ~torch.isfinite(model_obspace)
        used = mask_uv & ~interpolation_failed
        safe_model_obspace = torch.where(interpolation_failed, torch.zeros_like(model_obspace), model_obspace)
        innovation = observation - safe_model_obspace
        effective_error = sub_uv['error_std'][oind]
        if background_check:
            qc_error = effective_error
            lo = aircraft_traj.get('error_min_ms')
            hi = aircraft_traj.get('error_max_ms')
            if lo is not None or hi is not None:
                qc_error = qc_error.clamp(min=lo if lo is not None else 0., max=hi if hi is not None else float('inf'))
            used = used & (innovation.abs() <= float(background_check) * qc_error).detach()
        results[base] = (innovation, safe_model_obspace, effective_error, used)
    return results


# ---------------------------------------------------------------------------
# gpsro (GPS radio-occultation bending angle)
# ---------------------------------------------------------------------------
#
# Ported from the `long-window-4dvar` (JAX/NeuralGCM) reference repo's
# `gpsro_parquet.py` (reader/QC/thinning/error model) and `gpsro_operator.py`
# (the differentiable 1D Abel-transform bending-angle forward operator,
# cross-validated directly against the reference -- see integrate_gpsro.md).
# Each profile's levels keep their OWN (possibly height-drifting) tangent-
# point lat/lon, so each (profile, level) gets its own horizontally
# interpolated model column -- not one column per profile.
#
# Deliberately simpler than the reference in two places (see
# integrate_gpsro.md for why): the background/sanity check here is a plain
# fixed-stddev threshold evaluated at the CURRENT state every loss call
# (same convention as raobs's `q_bg_check`), not the reference's frozen,
# quantile-scaled-by-lead-time scheme; and the reference's purely
# observation-side profile-shape QC (`gpsro_profile_quality_control`) is not
# ported at all -- the reference's own config defaults it OFF too.


def get_gpsro(exp, model, logger, grid_interp, psobs_traj):
    """Read quality-controlled, vertically thinned GPS-RO bending-angle
    profiles and build the fixed per-(profile, level) composite
    interpolation stencil the 1D Abel-transform operator needs. Returns
    `None` if `exp['gpsro_enabled']` is not True -- callers all accept
    `gpsro_traj=None` and skip the gpsro term entirely.

    Piggybacks on psobs's obs-slot grid (same `step_lo`/`alpha` time
    bracketing), same as get_raobs/get_amv.
    """
    if not exp.get('gpsro_enabled', False):
        return None
    from gpsro_parquet import load_gpsro_profiles

    device = model.device
    dt_obs = exp['dt_obs']
    date = exp['date']
    vdate = copy.copy(date)
    verification_times = []
    for _ in range(len(psobs_traj['step_lo'])):
        verification_times.append(vdate)
        vdate = add_hours(vdate, dt_obs)
    n_obs = len(verification_times)

    t_levels = np.asarray(model.pressure_levels('t'), dtype=np.float64)
    q_levels = np.asarray(model.pressure_levels('q'), dtype=np.float64)
    z_levels = np.asarray(model.pressure_levels('z'), dtype=np.float64)
    if not np.array_equal(t_levels, z_levels):
        raise ValueError("gpsro_enabled requires model 't' and 'z' to share the same pressure levels.")
    # AIFS/Aurora's 'q' omits the top couple of 't'/'z' levels (e.g. AIFS-
    # single-2.0: 10/50 hPa -- see integrate_prepbufr_obs.md) since
    # stratospheric humidity is negligible. Rather than require an exact
    # match (losing those levels -- and hence tangent-height support --
    # from the whole column), pad 'q' with zero (dry) at any 't'/'z' level
    # it doesn't have; this must be a strict SUBSET, in the same relative
    # order, or there's a real level mismatch the padding can't fix.
    q_present_idx = np.full(q_levels.size, -1, dtype=np.int64)
    for i, level in enumerate(q_levels):
        matches = np.flatnonzero(np.isclose(t_levels, level))
        if matches.size != 1:
            raise ValueError(
                "gpsro_enabled requires every model 'q' pressure level to be exactly one of "
                f"'t'/'z''s levels; {level!r} is not (t/z levels: {t_levels.tolist()!r})."
            )
        q_present_idx[i] = matches[0]
    if np.any(np.diff(q_present_idx) <= 0):
        raise ValueError(
            "gpsro_enabled requires model 'q' levels to appear in the same relative order as 't'/'z''s."
        )

    data = load_gpsro_profiles(
        exp['gpsro_path'], verification_times,
        time_tolerance_hours=exp.get('gpsro_time_tolerance_hours', 1.5),
        impact_height_range_km=exp.get('gpsro_impact_height_range_km', (6., 40.)),
        impact_height_spacing_m=exp.get('gpsro_impact_height_spacing_m', 1000.),
        maximum_bending_angle_rad=exp.get('gpsro_maximum_bending_angle_rad', .1),
        quality_flags_disqualifying_mask=exp.get('gpsro_quality_flags_disqualifying_mask', 0),
        minimum_percent_confidence=exp.get('gpsro_minimum_percent_confidence'),
        error_model=exp.get('gpsro_error_model'),
        logger=logger,
    )
    by_time = data['by_time']

    profile_counts = [len(profiles) for profiles in by_time]
    level_counts = [
        max((len(p['bending_angle_rad']) for p in profiles), default=0) for profiles in by_time
    ]
    profile_capacity = int(exp.get('gpsro_profile_capacity', max(1, max(profile_counts, default=1))))
    level_capacity = int(exp.get('gpsro_level_capacity', max(1, max(level_counts, default=1))))
    if max(profile_counts, default=0) > profile_capacity:
        raise ValueError(
            f"gpsro profile count {max(profile_counts)} exceeds configured gpsro_profile_capacity {profile_capacity}"
        )
    if max(level_counts, default=0) > level_capacity:
        raise ValueError(
            f"gpsro level count {max(level_counts)} exceeds configured gpsro_level_capacity {level_capacity}"
        )

    k = GPSRO_HORIZONTAL_K
    level_shape = (n_obs, profile_capacity, level_capacity)
    latitude = np.zeros(level_shape, dtype=np.float32)
    longitude = np.zeros(level_shape, dtype=np.float32)
    impact_height = np.full(level_shape, 1000. * min(exp.get('gpsro_impact_height_range_km', (6., 40.))), dtype=np.float32)
    bending_angle = np.zeros(level_shape, dtype=np.float32)
    error_std = np.ones(level_shape, dtype=np.float32)
    curvature = np.full(level_shape, 6371000., dtype=np.float32)
    undulation = np.zeros(level_shape, dtype=np.float32)
    mask = np.zeros(level_shape, dtype=bool)
    satellite_id = np.full((n_obs, profile_capacity), -1, dtype=np.int32)

    for j, profiles in enumerate(by_time):
        for p_index, profile in enumerate(profiles):
            n_levels = len(profile['bending_angle_rad'])
            sl = (j, p_index, slice(0, n_levels))
            latitude[sl] = profile['latitude_deg_north']
            longitude[sl] = np.mod(profile['longitude_deg_east'], 360.)
            impact_height[sl] = profile['impact_height_m']
            bending_angle[sl] = profile['bending_angle_rad']
            error_std[sl] = profile['error_std_rad']
            curvature[sl] = profile['earth_local_radius_curvature_m']
            undulation[sl] = profile['geoid_undulation_m']
            mask[(j, p_index, slice(0, n_levels))] = True
            satellite_id[j, p_index] = profile['satellite_id']

    interp_idx = torch.zeros((n_obs, profile_capacity, level_capacity, k), dtype=torch.long, device=device)
    interp_wts = torch.zeros((n_obs, profile_capacity, level_capacity, k), dtype=torch.float32, device=device)
    for j in range(n_obs):
        flat_lon = longitude[j].reshape(-1)
        flat_lat = latitude[j].reshape(-1)
        idx, wts = grid_interp.weights(flat_lon, flat_lat, k=k, device=device)
        interp_idx[j] = idx.reshape(profile_capacity, level_capacity, k)
        interp_wts[j] = wts.reshape(profile_capacity, level_capacity, k)

    gpsro_traj = {
        'n_valid': torch.as_tensor(profile_counts, dtype=torch.int32, device=device),
        'pressure_hpa': torch.as_tensor(t_levels, dtype=torch.float32, device=device),
        'n_t_levels': t_levels.size,
        'q_present_idx': torch.as_tensor(q_present_idx, dtype=torch.long, device=device),
        'latitude': torch.as_tensor(latitude, device=device),
        'longitude': torch.as_tensor(longitude, device=device),
        'impact_height_m': torch.as_tensor(impact_height, device=device),
        'bending_angle_rad': torch.as_tensor(bending_angle, device=device),
        'error_std_rad': torch.as_tensor(error_std, device=device),
        'earth_local_radius_curvature_m': torch.as_tensor(curvature, device=device),
        'geoid_undulation_m': torch.as_tensor(undulation, device=device),
        'mask': torch.as_tensor(mask, device=device),
        'satellite_id': torch.as_tensor(satellite_id, device=device),
        'interp_idx': interp_idx,
        'interp_wts': interp_wts,
        'maximum_simulated_bending_angle_rad': float(exp.get('gpsro_maximum_simulated_bending_angle_rad', .05)),
        # Background gross check (GSI's setupbend.f90, Stage C -- see
        # integrate_gpsro.md "GSI comparison"): ratio = |O-B| /
        # clip(error, error_min_rad, error_max_rad) > background_check_stddev
        # rejects the observation. error_min_rad/error_max_rad default to
        # no clamping (plain |O-B|/error, this repo's prior convention) --
        # set both to adopt GSI's full clamped-ratio form.
        'background_check_stddev': exp.get('gpsro_background_check_stddev', 4.),
        'error_min_rad': exp.get('gpsro_error_min_rad'),
        'error_max_rad': exp.get('gpsro_error_max_rad'),
        # Model-top extension + super-refraction screen, both ported from
        # GSI (`setupbend.f90`) -- see gpsro_operator.py's
        # bending_angle_column_validity_diagnostics docstring and
        # integrate_gpsro.md "GSI comparison". Default 10 km reclaims
        # AIFS-single-2.0's 31-40 km band (the default
        # impact_height_range_km upper bound) that was previously rejected
        # outright for falling above the model's own ~31 km top.
        'top_extrapolation_m': 1000. * float(exp.get('gpsro_top_extrapolation_km', 10.)),
        'critical_gradient_n_per_km': exp.get(
            'gpsro_critical_gradient_n_per_km', gpsro_operator.GSI_CRITICAL_REFRACTIVITY_GRADIENT_N_PER_KM,
        ),
        'sr_detection_fraction': exp.get(
            'gpsro_sr_detection_fraction', gpsro_operator.GSI_SUPER_REFRACTION_DETECTION_FRACTION,
        ),
        'sr_screen_height_m': 1000. * float(exp.get(
            'gpsro_sr_screen_height_km', gpsro_operator.GSI_SUPER_REFRACTION_SCREEN_HEIGHT_M / 1000.,
        )),
        'n_steps': psobs_traj['n_steps'],
        'step_lo': psobs_traj['step_lo'],
        'alpha': psobs_traj['alpha'],
    }
    return gpsro_traj


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
    operator, there is no orography QC and no elevation-based reduction --
    each variable is compared against the model's own native level directly.
    QC is the read-time QC already folded into `mask_<base>`, plus, for q
    only, an optional background check (`q_bg_check`: reject
    |O-B| > q_bg_check * error, evaluated at the current state).
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


def _compute_amv_observation_diagnostics_at_time(decoded, amv_traj, oind):
    """AMV H(x) for one verification-time slot.

    Each thinned observation's fixed 2*k-node stencil (`node_point_idx`/
    `node_level_idx`/`node_weights`, from get_amv/_amv_thin_slot) is
    applied directly to the CURRENT decoded u/v field -- no grid_interp
    call needed here, since the stored weights already fold in both the
    horizontal k-NN weights and the vertical log-pressure blend.

    Background gross check (GSI's `setupw.f90`, ported as
    `amv_background_check_stddev`/`amv_error_min_ms`/`amv_error_max_ms` --
    see get_amv and integrate_amv_satwnd.md "GSI comparison"): rejects an
    observation whose |O-B| exceeds `background_check_stddev * clip(error,
    error_min_ms, error_max_ms)`, evaluated at the CURRENT state on every
    loss call (same convention as raobs's `q_bg_check`/gpsro's
    `background_check_stddev` -- simpler than GSI's own frozen,
    once-before-optimization check).
    """
    node_point_idx = amv_traj['node_point_idx'][oind]   # (capacity, 2k)
    node_level_idx = amv_traj['node_level_idx'][oind]   # (capacity, 2k)
    node_weights = amv_traj['node_weights'][oind]       # (capacity, 2k)
    mask = amv_traj['mask'][oind]
    background_check = amv_traj.get('background_check_stddev')
    error_min_ms = amv_traj.get('error_min_ms')
    error_max_ms = amv_traj.get('error_max_ms')
    results = {}
    for obs_key, base in AMV_VARIABLE_MODEL_BASE.items():
        field = decoded[base]                             # (n_levels, n_grid_points)
        gathered = field[node_level_idx, node_point_idx]   # (capacity, 2k)
        model_obspace = (gathered * node_weights).sum(dim=-1)
        observation = amv_traj[obs_key][oind]
        interpolation_failed = ~torch.isfinite(model_obspace)
        used = mask & ~interpolation_failed
        safe_model_obspace = torch.where(interpolation_failed, torch.zeros_like(model_obspace), model_obspace)
        innovation = observation - safe_model_obspace
        effective_error = amv_traj['error_std'][oind]
        if background_check:
            qc_error = effective_error
            if error_min_ms is not None or error_max_ms is not None:
                lo = error_min_ms if error_min_ms is not None else 0.
                hi = error_max_ms if error_max_ms is not None else float('inf')
                qc_error = qc_error.clamp(min=lo, max=hi)
            used = used & (innovation.abs() <= float(background_check) * qc_error).detach()
        results[base] = (innovation, safe_model_obspace, effective_error, used)
    return results


def _compute_gpsro_observation_diagnostics_at_time(decoded, gpsro_traj, grid_interp, oind):
    """GPS-RO bending-angle H(x) for one verification-time slot.

    Each (profile, level) gets its OWN horizontally interpolated model
    column (`grid_interp.interp`'s idx/wts broadcast over whatever leading
    shape they're given -- no change needed there, see
    integrate_gpsro.md), then the full column is handed to the 1D
    Abel-transform operator (`gpsro_operator.simulate_bending_angles`).

    `used` additionally requires the model-column validity predicate
    (`bending_angle_columns_are_valid` -- thermodynamic sanity, a
    supported tangent point (optionally extrapolated above the model top,
    `top_extrapolation_m`), non-reversing refractional height, and a
    super-refraction screen -- all three ported from GSI's
    `setupbend.f90`, see get_gpsro/integrate_gpsro.md "GSI comparison")
    and a background gross check (GSI's own clamped-ratio form when
    `error_min_rad`/`error_max_rad` are set, a plain fixed-stddev
    multiplier otherwise -- evaluated at the CURRENT state every loss
    call, simpler than GSI's frozen, once-before-optimization version).
    """
    idx = gpsro_traj['interp_idx'][oind]     # (profile_capacity, level_capacity, k)
    wts = gpsro_traj['interp_wts'][oind]
    mask = gpsro_traj['mask'][oind]          # (profile_capacity, level_capacity)

    # 'q' may omit the top couple of 't'/'z' levels (e.g. AIFS-single-2.0:
    # 10/50 hPa, negligible stratospheric humidity -- see get_gpsro).
    # Scatter it onto the full 't'/'z' level axis, zero (dry) at any level
    # it doesn't have, so every base shares one consistent column before
    # interpolation.
    q_field = decoded['q']
    if q_field.shape[0] != gpsro_traj['n_t_levels']:
        q_field = torch.zeros(
            (gpsro_traj['n_t_levels'],) + q_field.shape[1:], dtype=q_field.dtype, device=q_field.device,
        )
        q_field[gpsro_traj['q_present_idx']] = decoded['q']
    fields = {'t': decoded['t'], 'q': q_field, 'z': decoded['z']}

    columns = {}
    for base in ('t', 'q', 'z'):
        # fields[base]: (n_model_levels, n_grid_points) -> (n_model_levels, profile_capacity, level_capacity)
        interpolated = grid_interp.interp(idx, wts, fields[base])
        columns[base] = interpolated.movedim(0, -1)   # -> (profile_capacity, level_capacity, n_model_levels)

    pressure_hpa = gpsro_traj['pressure_hpa']
    latitude = gpsro_traj['latitude'][oind]
    curvature = gpsro_traj['earth_local_radius_curvature_m'][oind]
    undulation = gpsro_traj['geoid_undulation_m'][oind]
    impact_height = gpsro_traj['impact_height_m'][oind]

    model_equivalent = gpsro_operator.simulate_bending_angles(
        columns['t'], columns['q'], columns['z'], pressure_hpa,
        latitude, curvature, undulation, impact_height, mask,
    )
    column_valid = gpsro_operator.bending_angle_columns_are_valid(
        columns['t'], columns['q'], columns['z'], pressure_hpa,
        latitude, curvature, undulation, impact_height,
        top_extrapolation_m=gpsro_traj.get('top_extrapolation_m', 0.),
        critical_gradient_n_per_km=gpsro_traj.get('critical_gradient_n_per_km'),
        sr_detection_fraction=gpsro_traj.get(
            'sr_detection_fraction', gpsro_operator.GSI_SUPER_REFRACTION_DETECTION_FRACTION,
        ),
        sr_screen_height_m=gpsro_traj.get(
            'sr_screen_height_m', gpsro_operator.GSI_SUPER_REFRACTION_SCREEN_HEIGHT_M,
        ),
    )
    observation = gpsro_traj['bending_angle_rad'][oind]
    effective_error = gpsro_traj['error_std_rad'][oind]
    maximum_angle = gpsro_traj['maximum_simulated_bending_angle_rad']

    interpolation_failed = ~torch.isfinite(model_equivalent)
    safe_model_equivalent = torch.where(interpolation_failed, torch.zeros_like(model_equivalent), model_equivalent)
    simulated_angle_ok = ~interpolation_failed & (safe_model_equivalent > 0.) & (safe_model_equivalent <= maximum_angle)
    innovation = observation - safe_model_equivalent
    used = mask & column_valid & simulated_angle_ok
    # Background gross check (GSI's setupbend.f90, Stage C -- see
    # get_gpsro and integrate_gpsro.md "GSI comparison"): ratio = |O-B| /
    # clip(error, error_min_rad, error_max_rad) > background_check_stddev
    # rejects the observation, evaluated at the CURRENT state every loss
    # call (same convention as raobs's q_bg_check/amv's own clamped check
    # -- simpler than GSI's frozen, once-before-optimization version).
    background_check = gpsro_traj.get('background_check_stddev')
    if background_check:
        error_min_rad = gpsro_traj.get('error_min_rad')
        error_max_rad = gpsro_traj.get('error_max_rad')
        qc_error = effective_error
        if error_min_rad is not None or error_max_rad is not None:
            lo = error_min_rad if error_min_rad is not None else 0.
            hi = error_max_rad if error_max_rad is not None else float('inf')
            qc_error = qc_error.clamp(min=lo, max=hi)
        used = used & (innovation.abs() <= float(background_check) * qc_error).detach()
    return innovation, safe_model_equivalent, effective_error, used


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
    # amv (SATWND, box-thinned) needs 'u'/'v' decoded the same time-
    # interpolated way, sharing psobs's obs-slot grid exactly like raobs
    # above (see get_amv).
    if exp.get('amv_enabled', False):
        names |= set(AMV_VARIABLE_MODEL_BASE.values())
    # gpsro needs t/q decoded the same time-interpolated way ('z' is
    # already always included above) -- see get_gpsro.
    if exp.get('gpsro_enabled', False):
        names |= {'t', 'q'}
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
                              interp_specs, step_start=0, raobs_traj=None, amv_traj=None,
                              aircraft_traj=None, gpsro_traj=None):
    """Evaluate a trajectory at every surface-pressure observation slot.
    Post-optimization diagnostics only -- no gradients anywhere here.

    Always returns six values (`obs_trajectory, model_trajectory,
    raobs_trajectory, amv_trajectory, aircraft_trajectory,
    gpsro_trajectory`); the last four are `None` when `raobs_traj`/
    `amv_traj`/`aircraft_traj`/`gpsro_traj` is `None`.
      - `model_trajectory`: the decoded model state at every 6h step, from
        `step_start` to `n_steps` inclusive (native rollout cadence).
      - `obs_trajectory`: H(x)/QC at every obs slot `j` with
        `step_lo[j] >= step_start`, using the same time-interpolation as the
        loss (`_interp_decoded` over `interp_specs`). Slots earlier than
        `step_start` are omitted -- the caller pads them.
      - `raobs_trajectory` (only when `raobs_traj` is not None): the same
        shape of thing, per raobs variable -- see `_record_obs`.
      - `amv_trajectory` (only when `amv_traj` is not None): the same shape
        of thing, per amv (u/v) variable -- see `_record_obs`.
      - `gpsro_trajectory` (only when `gpsro_traj` is not None): H(x)/
        column-validity fields on `(time, profile_slot, level_slot)` --
        see `_record_obs`.

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
    raobs_diagnostics = {base: {} for base in raobs_traj['variables']} if raobs_traj is not None else None
    amv_diagnostics = {base: {} for base in AMV_VARIABLE_MODEL_BASE.values()} if amv_traj is not None else None
    aircraft_diagnostics = {base: {} for base in AIRCRAFT_VARIABLE_MODEL_BASE.values()} if aircraft_traj is not None else None
    gpsro_diagnostics = {} if gpsro_traj is not None else None

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
        if amv_traj is not None:
            amv_results = _compute_amv_observation_diagnostics_at_time(decoded_j, amv_traj, j)
            for base, (innovation_a, model_equivalent_a, effective_error_a, used_a) in amv_results.items():
                for name, value in {
                    'model_equivalent': model_equivalent_a,
                    'effective_error': effective_error_a,
                    'used': used_a,
                }.items():
                    amv_diagnostics[base].setdefault(name, []).append(value.detach().cpu().numpy())
        if aircraft_traj is not None:
            aircraft_results = _compute_aircraft_observation_diagnostics_at_time(decoded_j, aircraft_traj, j)
            for base, (innovation_c, model_equivalent_c, effective_error_c, used_c) in aircraft_results.items():
                for name, value in {
                    'model_equivalent': model_equivalent_c,
                    'effective_error': effective_error_c,
                    'used': used_c,
                }.items():
                    aircraft_diagnostics[base].setdefault(name, []).append(value.detach().cpu().numpy())
        if gpsro_traj is not None:
            innovation_g, model_equivalent_g, effective_error_g, used_g = \
                _compute_gpsro_observation_diagnostics_at_time(decoded_j, gpsro_traj, grid_interp, j)
            for name, value in {
                'model_equivalent': model_equivalent_g,
                'effective_error': effective_error_g,
                'used': used_g,
            }.items():
                gpsro_diagnostics.setdefault(name, []).append(value.detach().cpu().numpy())

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
    raobs_trajectory = None
    if raobs_traj is not None:
        raobs_trajectory = {
            base: {name: np.stack(v, axis=0) for name, v in fields.items()}
            for base, fields in raobs_diagnostics.items()
        }
    amv_trajectory = None
    if amv_traj is not None:
        amv_trajectory = {
            base: {name: np.stack(v, axis=0) for name, v in fields.items()}
            for base, fields in amv_diagnostics.items()
        }
    aircraft_trajectory = None
    if aircraft_traj is not None:
        aircraft_trajectory = {
            base: {name: np.stack(v, axis=0) for name, v in fields.items()}
            for base, fields in aircraft_diagnostics.items()
        }
    gpsro_trajectory = None
    if gpsro_traj is not None:
        gpsro_trajectory = {name: np.stack(v, axis=0) for name, v in gpsro_diagnostics.items()}
    return obs_trajectory, model_trajectory, raobs_trajectory, amv_trajectory, aircraft_trajectory, gpsro_trajectory


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
                       keep_mask=None, ref_state1=None, jc_ref=None, raobs_traj=None, amv_traj=None,
                       aircraft_traj=None, gpsro_traj=None):
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

    `amv_traj` (from `get_amv`, present iff `exp['amv_enabled']` is True):
    satellite-derived wind (box-thinned SATWND) terms, added to `J` the same
    way, same obs-slot grid -- `None` (the default) leaves assimilation
    unchanged from before amv existed.

    `aircraft_traj` (from `get_aircraft`, present iff
    `exp['aircraft_enabled']` is True): aircraft (AIRCAR/AIRCFT) temperature
    and wind terms, added to `J` the same way, same obs-slot grid -- `None`
    (the default) leaves assimilation unchanged from before aircraft
    existed. See integrate_aircraft_obs.md.

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
    Jamv = [None] * n_obs if amv_traj is not None else None
    Jaircraft = [None] * n_obs if aircraft_traj is not None else None
    Jgpsro = [None] * n_obs if gpsro_traj is not None else None

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

    def _amv_term(j, decoded_j):
        results = _compute_amv_observation_diagnostics_at_time(decoded_j, amv_traj, j)
        Jt = 0.
        counts = {}
        for base, (innovation, model_equivalent, effective_error, used) in results.items():
            innov = torch.where(used, innovation, torch.zeros_like(innovation))
            err = torch.where(used, effective_error, torch.full_like(effective_error, 1.e10))
            Jt = Jt + ((innov / err) ** 2).sum()
            counts[base] = int(used.sum().item())
        if do_print:
            print(f"epoch {epoch}, oind {j} (t+{j * dt_obs}h): amv J {Jt.item() if torch.is_tensor(Jt) else Jt}, "
                  f"obs used {counts}")
        return Jt

    def _aircraft_term(j, decoded_j):
        results = _compute_aircraft_observation_diagnostics_at_time(decoded_j, aircraft_traj, j)
        Jt = 0.
        counts = {}
        for base, (innovation, model_equivalent, effective_error, used) in results.items():
            innov = torch.where(used, innovation, torch.zeros_like(innovation))
            err = torch.where(used, effective_error, torch.full_like(effective_error, 1.e10))
            Jt = Jt + ((innov / err) ** 2).sum()
            counts[base] = int(used.sum().item())
        if do_print:
            print(f"epoch {epoch}, oind {j} (t+{j * dt_obs}h): aircraft J {Jt.item() if torch.is_tensor(Jt) else Jt}, "
                  f"obs used {counts}")
        return Jt

    def _gpsro_term(j, decoded_j):
        innovation, model_equivalent, effective_error, used = \
            _compute_gpsro_observation_diagnostics_at_time(decoded_j, gpsro_traj, grid_interp, j)
        innov = torch.where(used, innovation, torch.zeros_like(innovation))
        err = torch.where(used, effective_error, torch.full_like(effective_error, 1.e10))
        Jt = ((innov / err) ** 2).sum()
        if do_print:
            print(f"epoch {epoch}, oind {j} (t+{j * dt_obs}h): gpsro J {Jt.item()}, "
                  f"{int(used.sum().item())} levels used (out of {int(gpsro_traj['n_valid'][j])} profiles)")
        return Jt

    jc_sp = {}
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
            if amv_traj is not None:
                Jamv[j] = _amv_term(j, decoded_j)
            if aircraft_traj is not None:
                Jaircraft[j] = _aircraft_term(j, decoded_j)
            if gpsro_traj is not None:
                Jgpsro[j] = _gpsro_term(j, decoded_j)
        decoded_prev = decoded_cur

    J = sum(Jol)
    if amv_traj is not None:
        J = J + sum(Jamv)
    if gpsro_traj is not None:
        J = J + sum(Jgpsro)
    if raobs_traj is not None:
        J = J + sum(Jraobs)
    if aircraft_traj is not None:
        J = J + sum(Jaircraft)
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


def compute_optimal(exp, model, input_encoded, verif_ic, psobs_traj, grid_interp, logger, raobs_traj=None, amv_traj=None, aircraft_traj=None, gpsro_traj=None):
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
        loss, all_loss = compute_loss_4dvar(model, increment, latent_scale, input_encoded, verif_ic, psobs_traj, epoch, exp, grid_interp, print_every, interp_specs, keep_mask, ref_state1, jc_ref, raobs_traj, amv_traj, aircraft_traj, gpsro_traj)

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


def save_trajectory_diagnostics(exp, model, input_encoded, analysis_state, verif_ic, psobs_traj, save_levs, grid_interp, logger, raobs_traj=None, amv_traj=None, aircraft_traj=None, gpsro_traj=None):
    interp_specs = _resolve_loss_interp_specs(model, exp)
    timestep_h = int(round(model.timestep.total_seconds() / 3600))
    dt_obs = exp['dt_obs']
    n_obs = int(psobs_traj['n_valid'].shape[0])
    step_lo = psobs_traj['step_lo'].tolist()

    background, initial_traj, raobs_background, amv_background, aircraft_background, gpsro_background = compute_ps_observation_hx(
        model, input_encoded, verif_ic, psobs_traj, exp, grid_interp, interp_specs, step_start=0,
        raobs_traj=raobs_traj, amv_traj=amv_traj, aircraft_traj=aircraft_traj, gpsro_traj=gpsro_traj,
    )

    # The latent analysis_state is valid at input_encoded.date + dt_verif
    # (see compute_optimal). compute_ps_observation_hx starts its trajectory
    # at that 6h-step index; obs slots earlier than it have no analysis value
    # and are NaN/0-padded here so the analysis arrays line up with the
    # full-window background arrays.
    shift_hours = (analysis_state.date - input_encoded.date).total_seconds() / 3600.0
    step_start = int(round(shift_hours / timestep_h))
    j_start = sum(1 for s in step_lo if s < step_start)
    analysis, final_traj, raobs_analysis, amv_analysis, aircraft_analysis, gpsro_analysis = compute_ps_observation_hx(
        model, analysis_state, verif_ic, psobs_traj, exp, grid_interp, interp_specs, step_start=step_start,
        raobs_traj=raobs_traj, amv_traj=amv_traj, aircraft_traj=aircraft_traj, gpsro_traj=gpsro_traj,
    )
    analysis['model_equivalent'] = _pad_leading(analysis['model_equivalent'], j_start, np.nan)
    analysis['used'] = _pad_leading(analysis['used'], j_start, False)
    analysis['qc_flag'] = _pad_leading(analysis['qc_flag'], j_start, 0)
    if raobs_traj is not None:
        for base in raobs_traj['variables']:
            raobs_analysis[base]['model_equivalent'] = _pad_leading(raobs_analysis[base]['model_equivalent'], j_start, np.nan)
            raobs_analysis[base]['used'] = _pad_leading(raobs_analysis[base]['used'], j_start, False)
    if amv_traj is not None:
        for base in AMV_VARIABLE_MODEL_BASE.values():
            amv_analysis[base]['model_equivalent'] = _pad_leading(amv_analysis[base]['model_equivalent'], j_start, np.nan)
            amv_analysis[base]['used'] = _pad_leading(amv_analysis[base]['used'], j_start, False)
    if aircraft_traj is not None:
        for base in AIRCRAFT_VARIABLE_MODEL_BASE.values():
            aircraft_analysis[base]['model_equivalent'] = _pad_leading(aircraft_analysis[base]['model_equivalent'], j_start, np.nan)
            aircraft_analysis[base]['used'] = _pad_leading(aircraft_analysis[base]['used'], j_start, False)
    if gpsro_traj is not None:
        gpsro_analysis['model_equivalent'] = _pad_leading(gpsro_analysis['model_equivalent'], j_start, np.nan)
        gpsro_analysis['used'] = _pad_leading(gpsro_analysis['used'], j_start, False)

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

    if amv_traj is not None:
        ofile_amv = (
            exp['path_output'] + exp['date'] + '_amv_diagnostics_' + exp['window_name'] + 'h'
            + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
        )
        save_amv_diagnostics(
            exp, amv_traj, amv_background, amv_analysis, valid_times, lead_hours, ofile_amv, logger,
        )

    if aircraft_traj is not None:
        ofile_aircraft = (
            exp['path_output'] + exp['date'] + '_aircraft_diagnostics_' + exp['window_name'] + 'h'
            + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
        )
        save_aircraft_diagnostics(
            exp, aircraft_traj, aircraft_background, aircraft_analysis, valid_times, lead_hours, ofile_aircraft, logger,
        )

    if gpsro_traj is not None:
        ofile_gpsro = (
            exp['path_output'] + exp['date'] + '_gpsro_diagnostics_' + exp['window_name'] + 'h'
            + exp['suffix'] + '_' + str(exp['max_epoch']) + 'it.nc'
        )
        save_gpsro_diagnostics(
            exp, gpsro_traj, gpsro_background, gpsro_analysis, valid_times, lead_hours, ofile_gpsro, logger,
        )

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


_AMV_BASE_LONG_NAME = {v: k for k, v in AMV_VARIABLE_MODEL_BASE.items()}
_AMV_BASE_UNITS = {'u': 'm s-1', 'v': 'm s-1'}


def save_amv_diagnostics(exp, amv_traj, background, analysis, valid_times, lead_hours, ofile, logger):
    """Save per-variable AMV (satellite-derived wind) O-B/O-A diagnostics,
    mirroring the raobs diagnostics netCDF -- but each `superob_slot` is one
    GSI-style box-thinned observation with its OWN pressure (no shared
    level coordinate -- see get_amv/_amv_thin_slot), recorded as a per-
    (time, superob_slot) field instead of a dimension coordinate.
    Dimension/variable names keep the `superob_*` spelling from the earlier
    composite-stencil-superobbing design (now a kept-single-ob design, GSI's
    `ithin`/`rmesh`/`pmesh` box thinning, not an average) for diagnostics
    continuity.
    """
    dt_obs = exp['dt_obs']
    n_obs, capacity = amv_traj['lat'].shape
    ds = xarray.Dataset(
        coords={
            'time': ('time', valid_times),
            'superob_slot': ('superob_slot', np.arange(capacity, dtype=np.int32)),
            'lead_time_hours': ('time', lead_hours),
        },
        attrs={
            'title': '4D-Var satellite-derived wind (AMV, SATWND, GSI-style box-thinned) observation diagnostics',
            'window_start': str(exp['date']),
            'window_length_hours': int(exp['n_verif'] * exp['dt_verif']),
            'dt_obs_hours': int(dt_obs),
            'innovation_convention': 'observation_minus_model',
            'forward_operator': 'fixed stencil (single kept observation) -- horizontal k-NN x log-pressure vertical bracket, see get_amv',
            'history': 'Created by long_window_4dvar.py',
        },
    )
    dims = ('time', 'superob_slot')
    mask = amv_traj['mask'].detach().cpu().numpy().astype(bool)
    for name, long_name, units in (
        ('lat', 'kept observation latitude', 'degrees_north'),
        ('lon', 'kept observation longitude', 'degrees_east'),
        ('pressure_hpa', 'kept observation pressure', 'hPa'),
        ('report_type', 'PREPBUFR report type (kept observation)', '1'),
        ('satellite_id', 'satellite id (kept observation)', '1'),
        ('component_count', 'raw wind vectors found in this thinning box (1 kept)', '1'),
    ):
        values = amv_traj[name].detach().cpu().numpy()
        values = np.where(mask, values, np.nan) if np.issubdtype(values.dtype, np.floating) else values
        ds['superob_'+name] = (dims, values)
        ds['superob_'+name].attrs.update({'long_name': long_name, 'units': units})

    for base in AMV_VARIABLE_MODEL_BASE.values():
        long_name = _AMV_BASE_LONG_NAME[base]
        units = _AMV_BASE_UNITS[base]
        observation = amv_traj[{'u': 'u_component_of_wind', 'v': 'v_component_of_wind'}[base]].detach().cpu().numpy().astype(np.float32)
        background_hx = background[base]['model_equivalent'].astype(np.float32)
        analysis_hx = analysis[base]['model_equivalent'].astype(np.float32)
        background_used = background[base]['used'].astype(bool)
        analysis_used = analysis[base]['used'].astype(bool)

        ds[base+'_observation'] = (dims, np.where(mask, observation, np.nan))
        ds[base+'_observation'].attrs.update({'long_name': long_name+' observation (kept observation, GSI-style box-thinned)', 'units': units})
        ds[base+'_background'] = (dims, np.where(mask, background_hx, np.nan))
        ds[base+'_background'].attrs.update({'long_name': long_name+' background model equivalent H(x_b)', 'units': units})
        ds[base+'_analysis'] = (dims, np.where(mask, analysis_hx, np.nan))
        ds[base+'_analysis'].attrs.update({'long_name': long_name+' analysis model equivalent H(x_a)', 'units': units})
        ds[base+'_omb'] = (dims, np.where(mask, observation - background_hx, np.nan))
        ds[base+'_omb'].attrs.update({'long_name': long_name+' observation minus background', 'units': units})
        ds[base+'_oma'] = (dims, np.where(mask, observation - analysis_hx, np.nan))
        ds[base+'_oma'].attrs.update({'long_name': long_name+' observation minus analysis', 'units': units})
        ds[base+'_used'] = (dims, background_used.astype(np.int8))
        ds[base+'_used'].attrs.update({'long_name': long_name+' observation used in the background loss', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})
        ds[base+'_analysis_used'] = (dims, analysis_used.astype(np.int8))
        ds[base+'_analysis_used'].attrs.update({'long_name': long_name+' observation used when reevaluated at the analysis', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})

    encoding = {name: {'zlib': True, 'complevel': 2} for name in ds.data_vars}
    ds.to_netcdf(ofile, encoding=encoding)
    logger.info('saved amv diagnostics: ' + ofile)
    return ofile


_AIRCRAFT_BASE_LONG_NAME = {'t': 'temperature', 'u': 'eastward wind', 'v': 'northward wind'}
_AIRCRAFT_BASE_UNITS = {'t': 'K', 'u': 'm s-1', 'v': 'm s-1'}
# which aircraft_traj sub-stream ('t' or 'uv') carries each base's kept
# observation/stencil fields -- see get_aircraft's module comment.
_AIRCRAFT_BASE_SUBSTREAM = {'t': 't', 'u': 'uv', 'v': 'uv'}


def save_aircraft_diagnostics(exp, aircraft_traj, background, analysis, valid_times, lead_hours, ofile, logger):
    """Save per-variable aircraft (AIRCAR/AIRCFT) O-B/O-A diagnostics,
    mirroring `save_amv_diagnostics` -- but with TWO independent slot
    dimensions (`t_slot`, `uv_slot`), one per thinned sub-stream (see
    get_aircraft's module comment for why temperature and wind are thinned
    independently), each carrying its own kept observation's lat/lon/
    pressure/report_type/aircraft_id/component_count.
    """
    dt_obs = exp['dt_obs']
    n_obs, t_capacity = aircraft_traj['t']['lat'].shape
    _, uv_capacity = aircraft_traj['uv']['lat'].shape
    ds = xarray.Dataset(
        coords={
            'time': ('time', valid_times),
            't_slot': ('t_slot', np.arange(t_capacity, dtype=np.int32)),
            'uv_slot': ('uv_slot', np.arange(uv_capacity, dtype=np.int32)),
            'lead_time_hours': ('time', lead_hours),
        },
        attrs={
            'title': '4D-Var aircraft (AIRCAR/AIRCFT, GSI-style box-thinned) observation diagnostics',
            'window_start': str(exp['date']),
            'window_length_hours': int(exp['n_verif'] * exp['dt_verif']),
            'dt_obs_hours': int(dt_obs),
            'innovation_convention': 'observation_minus_model',
            'forward_operator': 'fixed stencil (single kept observation) -- horizontal k-NN x log-pressure vertical bracket, see get_aircraft',
            'note': 'temperature (t_slot) and wind (uv_slot) are independently thinned sub-streams -- see get_aircraft',
            'history': 'Created by long_window_4dvar.py',
        },
    )
    for slot_name, sub_key in (('t_slot', 't'), ('uv_slot', 'uv')):
        dims = ('time', slot_name)
        sub = aircraft_traj[sub_key]
        mask = sub['mask'].detach().cpu().numpy().astype(bool)
        for name, long_name, units in (
            ('lat', 'kept observation latitude', 'degrees_north'),
            ('lon', 'kept observation longitude', 'degrees_east'),
            ('pressure_hpa', 'kept observation pressure', 'hPa'),
            ('report_type', 'PREPBUFR report type (kept observation)', '1'),
            ('component_count', 'raw reports found in this thinning box (1 kept)', '1'),
        ):
            values = sub[name].detach().cpu().numpy()
            values = np.where(mask, values, np.nan) if np.issubdtype(values.dtype, np.floating) else values
            ds[sub_key+'_'+name] = (dims, values)
            ds[sub_key+'_'+name].attrs.update({'long_name': long_name, 'units': units})
        aircraft_id = np.where(mask, sub['aircraft_id'], '')
        ds[sub_key+'_aircraft_id'] = (dims, aircraft_id.astype(str))
        ds[sub_key+'_aircraft_id'].attrs.update({
            'long_name': 'kept observation aircraft identifier (ACID) -- metadata only, not used for '
                          'thinning/grouping or any bias correction in this pass',
        })

    for base in AIRCRAFT_VARIABLE_MODEL_BASE.values():
        sub_key = _AIRCRAFT_BASE_SUBSTREAM[base]
        slot_name = sub_key + '_slot'
        dims = ('time', slot_name)
        sub = aircraft_traj[sub_key]
        mask = sub['mask'].detach().cpu().numpy().astype(bool)
        obs_key = {v: k for k, v in AIRCRAFT_VARIABLE_MODEL_BASE.items()}[base]
        long_name = _AIRCRAFT_BASE_LONG_NAME[base]
        units = _AIRCRAFT_BASE_UNITS[base]
        observation = sub[obs_key].detach().cpu().numpy().astype(np.float32)
        background_hx = background[base]['model_equivalent'].astype(np.float32)
        analysis_hx = analysis[base]['model_equivalent'].astype(np.float32)
        background_used = background[base]['used'].astype(bool)
        analysis_used = analysis[base]['used'].astype(bool)

        ds[base+'_observation'] = (dims, np.where(mask, observation, np.nan))
        ds[base+'_observation'].attrs.update({'long_name': long_name+' observation (kept observation, GSI-style box-thinned)', 'units': units})
        ds[base+'_background'] = (dims, np.where(mask, background_hx, np.nan))
        ds[base+'_background'].attrs.update({'long_name': long_name+' background model equivalent H(x_b)', 'units': units})
        ds[base+'_analysis'] = (dims, np.where(mask, analysis_hx, np.nan))
        ds[base+'_analysis'].attrs.update({'long_name': long_name+' analysis model equivalent H(x_a)', 'units': units})
        ds[base+'_omb'] = (dims, np.where(mask, observation - background_hx, np.nan))
        ds[base+'_omb'].attrs.update({'long_name': long_name+' observation minus background', 'units': units})
        ds[base+'_oma'] = (dims, np.where(mask, observation - analysis_hx, np.nan))
        ds[base+'_oma'].attrs.update({'long_name': long_name+' observation minus analysis', 'units': units})
        ds[base+'_used'] = (dims, background_used.astype(np.int8))
        ds[base+'_used'].attrs.update({'long_name': long_name+' observation used in the background loss', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})
        ds[base+'_analysis_used'] = (dims, analysis_used.astype(np.int8))
        ds[base+'_analysis_used'].attrs.update({'long_name': long_name+' observation used when reevaluated at the analysis', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})

    encoding = {name: {'zlib': True, 'complevel': 2} for name in ds.data_vars}
    ds.to_netcdf(ofile, encoding=encoding)
    logger.info('saved aircraft diagnostics: ' + ofile)
    return ofile


def save_gpsro_diagnostics(exp, gpsro_traj, background, analysis, valid_times, lead_hours, ofile, logger):
    """Save GPS-RO bending-angle O-B/O-A diagnostics on `(time, profile_slot,
    level_slot)` -- both are plain indices with no cross-time identity,
    same convention as psobs's `observation_slot`."""
    dt_obs = exp['dt_obs']
    n_obs, profile_capacity, level_capacity = gpsro_traj['mask'].shape
    ds = xarray.Dataset(
        coords={
            'time': ('time', valid_times),
            'profile_slot': ('profile_slot', np.arange(profile_capacity, dtype=np.int32)),
            'level_slot': ('level_slot', np.arange(level_capacity, dtype=np.int32)),
            'lead_time_hours': ('time', lead_hours),
        },
        attrs={
            'title': '4D-Var GPS radio-occultation bending-angle observation diagnostics',
            'window_start': str(exp['date']),
            'window_length_hours': int(exp['n_verif'] * exp['dt_verif']),
            'dt_obs_hours': int(dt_obs),
            'innovation_convention': 'observation_minus_model',
            'forward_operator': gpsro_operator.GPSRO_OPERATOR_VERSION+' (1D Abel transform)',
            'history': 'Created by long_window_4dvar.py',
        },
    )
    mask = gpsro_traj['mask'].detach().cpu().numpy().astype(bool)
    level_dims = ('time', 'profile_slot', 'level_slot')
    for name, dims, long_name, units in (
        ('latitude', ('time', 'profile_slot', 'level_slot'), 'tangent-point latitude', 'degrees_north'),
        ('longitude', ('time', 'profile_slot', 'level_slot'), 'tangent-point longitude', 'degrees_east'),
        ('impact_height_m', level_dims, 'impact height', 'm'),
        ('satellite_id', ('time', 'profile_slot'), 'satellite id', '1'),
    ):
        values = gpsro_traj[name].detach().cpu().numpy()
        ds['gpsro_'+name] = (dims, values)
        ds['gpsro_'+name].attrs.update({'long_name': long_name, 'units': units})

    observation = gpsro_traj['bending_angle_rad'].detach().cpu().numpy().astype(np.float32)
    background_hx = background['model_equivalent'].astype(np.float32)
    analysis_hx = analysis['model_equivalent'].astype(np.float32)
    background_used = background['used'].astype(bool)
    analysis_used = analysis['used'].astype(bool)

    ds['bending_angle_observation'] = (level_dims, np.where(mask, observation, np.nan))
    ds['bending_angle_observation'].attrs.update({'long_name': 'observed bending angle', 'units': 'rad'})
    ds['bending_angle_background'] = (level_dims, np.where(mask, background_hx, np.nan))
    ds['bending_angle_background'].attrs.update({'long_name': 'background model equivalent H(x_b)', 'units': 'rad'})
    ds['bending_angle_analysis'] = (level_dims, np.where(mask, analysis_hx, np.nan))
    ds['bending_angle_analysis'].attrs.update({'long_name': 'analysis model equivalent H(x_a)', 'units': 'rad'})
    ds['bending_angle_omb'] = (level_dims, np.where(mask, observation - background_hx, np.nan))
    ds['bending_angle_omb'].attrs.update({'long_name': 'observation minus background', 'units': 'rad'})
    ds['bending_angle_oma'] = (level_dims, np.where(mask, observation - analysis_hx, np.nan))
    ds['bending_angle_oma'].attrs.update({'long_name': 'observation minus analysis', 'units': 'rad'})
    ds['bending_angle_error_std'] = (level_dims, np.where(mask, gpsro_traj['error_std_rad'].detach().cpu().numpy(), np.nan))
    ds['bending_angle_error_std'].attrs.update({'long_name': 'assigned observation-error standard deviation', 'units': 'rad'})
    ds['bending_angle_used'] = (level_dims, background_used.astype(np.int8))
    ds['bending_angle_used'].attrs.update({'long_name': 'observation used in the background loss', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})
    ds['bending_angle_analysis_used'] = (level_dims, analysis_used.astype(np.int8))
    ds['bending_angle_analysis_used'].attrs.update({'long_name': 'observation used when reevaluated at the analysis', 'flag_values': np.array([0, 1], dtype=np.int8), 'flag_meanings': 'not_used used'})

    encoding = {name: {'zlib': True, 'complevel': 2} for name in ds.data_vars}
    ds.to_netcdf(ofile, encoding=encoding)
    logger.info('saved gpsro diagnostics: ' + ofile)
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
