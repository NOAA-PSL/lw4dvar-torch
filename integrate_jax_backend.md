# JAX backends: the torch/JAX bridge and the NGCM backend

Read this when working on `backends/jax_bridge.py`, `backends/hybrid_levels.py`
or `backends/ngcm/` (`model_backend: ngcm`), or when adding another JAX model.

## Why a bridge (2026-11-04)

The solver core only needs `LatentForecastModel.advance()` and
`decode_state()` to be differentiable in torch (`loss.backward()` in
`compute_optimal`). A JAX model meets that through
`jax_bridge.torch_function(fn)`, a `torch.autograd.Function`:

- forward: torch inputs -> JAX via DLPack, `fn` jitted, outputs -> torch
  (zero-copy);
- backward: recomputes `fn` under `jax.vjp` with the torch cotangents (the
  JAX analogue of `torch.utils.checkpoint`; only each call's inputs are
  saved);
- `fn(constants, *arrays)`: `constants` is a JAX pytree (forcings, the
  valid time) passed as a traced, non-differentiated argument, so per-call
  values do not recompile (closing over them would).

This keeps one solver (obs operators, Jo/Jb/Jc, QC, cycling) for torch and
JAX models, instead of the separate `lw4dvar-jax` codebase.

Pitfalls found on the H100:

- torch -> JAX must COPY into a JAX-owned buffer. Zero-copy inputs work in
  small functions but a full model step fails in XLA's CUDA-graph command
  buffers ("Failed to set memcpy d2d node params: CUDA_ERROR_INVALID_VALUE").
  `jax.dlpack.from_dlpack(x, copy=True)` is UNIMPLEMENTED through the GPU
  PJRT C API, so `to_jax` imports zero-copy, `jnp.copy`s and blocks (the
  torch source must stay alive until the copy has read it). JAX -> torch
  stays zero-copy.
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (run_ngcm.sh does): otherwise JAX
  reserves 75 % of the GPU and starves torch's caching allocator.
- torch and JAX run on different CUDA streams: the bridge synchronizes
  torch's stream before JAX reads and blocks on JAX results before torch
  reads (one sync per call).

`backends/test_jax_bridge.py`: gradcheck (float64) through a scanned toy
rollout with an increment on step 0, an unused output, a constant input,
torch ops around the call; constants do not retrace. Passes on CPU and H100.

## Environment: `ngcm-torch`

A venv layered on the `ngcm-sl` conda env (JAX 0.11.1 + ngcm-train's
dependencies, untouched) adding torch 2.7.1+cu128 (`pip install --no-deps`
plus filelock, jinja2, networkx, sympy, curand/cusparselt/cufile/nvtx).
torch runs on the base env's newer CUDA 12.9 / cuDNN 9.27 libraries (minor-
version compatible). torch resolves its CUDA libraries relative to its own
site-packages, so the base env's `nvidia/*` directories are symlinked into
the venv's `site-packages/nvidia/` (without that, `import torch` after
`import jax` fails with "libcudnn.so.9: cannot open shared object file").
`/scratch4/BMC/gsienkf/whitaker/conda/envs/ngcm-torch`; launcher
`run_ngcm.sh` (`sbatch run_ngcm.sh CONFIG.yml`, or any `.py` as first
argument). Note `run_ngcm.sh` (like run_ace2.sh) sets `#SBATCH -e`, so pass
`-e` too when overriding `-o`.

## The NGCM backend (`backends/ngcm/ngcm_model.py`)

ngcm-train's NeuralGCM-style TL63 semi-Lagrangian model on 32 ACE2 hybrid
layers (repo `jswhit/ngcm-train`), built and restored with
`ngcm_train.inference.load_model` from the run's checkpoint metadata
(`exp.ngcm_exp`, `ngcm_step`, `ngcm_data`, `ngcm_train_path`), run through
terrax's `InferenceModel` (assimilate / advance / observe).

- Packed state, channels-first `(1, 161, 128, 65)`: modal vorticity,
  divergence, temperature (32 each), ln ps (1), NODAL q and condensate (32
  each, zero-padded 64 -> 65 latitudes). The time is rebuilt from
  `ModelState.date` each call; forcings (SST, sea ice) are read per step as
  a 2-entry [t, t + 6 h] window from the store's surface.zarr (fixed
  structure, no retracing). ICs and the z500 truth come straight from the
  local zarr store -- no prefetch.
- Latent control: the packed state itself, `latent_increment / factor`
  added before the first step. `ngcm_state_scales` ({family: factor},
  lw4dvar-jax's `state_scales` convention: the optimizer works on factor *
  state). Default = the original NeuralGCM ps-obs runs translated:
  NeuralGCM nondimensionalizes with dinosaur's `DEFAULT_SCALE` (length =
  Earth radius, time = 1/(2 Omega) ~ 6857 s, 1 kg, 1 K); ngcm-train uses
  terrax's `SI_UNITS`, so vorticity and divergence (s^-1 here) get
  1/(2 Omega), temperature and ln ps 1, tracers 1e5. Background rms
  (l >= 1): vorticity 1.1e-6, divergence 3.1e-7, T 0.44, ln ps 3e-3,
  q 3.9e-3, condensate 3.4e-5.
- Families for `control_variables`: vorticity, divergence,
  air_temperature, log_surface_pressure, hybrid_specific_humidity,
  specific_total_condensate. Native names differ from the canonical
  pressure-level aliases on purpose.
- decode_state: `observe` gives nodal hybrid T, q, condensate, u, v and ps;
  'z'/'t'/'q'/'u'/'v' on 13 pressure levels come from
  `backends/hybrid_levels.py` (ACE2's hydrostatic scheme, backend-neutral;
  ACE2 still has its own copy); 'z500'; 't_lowest'/'q_lowest'/'p_lowest'
  for `ps_operator: ps_native` (the default, as for ACE2). Orography: the
  store's surface geopotential, T63-truncated (what the dycore sees, without
  the learned correction).

## Validation (`backends/ngcm/smoke_test_ngcm.py`, r72_top150 step 51660, 2021-01-01)

- Bridged per-step rollout vs plain JAX with the full forcing window: max
  |diff| 0.
- 32-layer hybrid -> pressure at t0 vs ERA5 pressure.zarr: z500 rms 18 m,
  bias +0.9 m (ACE2's 8 layers: -19 m bias); t850 0.62 K. +24 h: 36 m,
  1.21 K.
- H100: 0.066 s/step forward; forward+backward 0.96 s for 4 steps (backward
  compile 37 s once). CPU: 2.4 s/step forward.
- Gradient vs finite differences, per family, smooth directions at physical
  amplitudes, loss linear in the ps change: vorticity 5 %, T 11 %, ln ps
  10 %, q 1.5 %, divergence ~20 % (small signal); smaller perturbations hit
  the float32 rounding floor of a 4-step forecast. Condensate is ~2.7x off
  (non-smooth near zero) -- leave it out of `control_variables`, as the
  original runs did. A loss like mean((ps - 1e5)/100)^2 is useless for this
  check: float32 rounding of ~0.1-1 Pa moves it by O(1).

## Configs

- `config_test_ngcm.yml`: one 5-day window (ACE2 test analogue).
- `config_ngcm_4d_2015.yml`: the original NeuralGCM ps-obs runs
  (long-window-4dvar-psobs/config.yml): 4-day window, 3-hourly obs
  (dt_verif 6, dt_obs 3: the model step is 6 h), lr 1e-4 constant, weight
  decay 1e-3, 50 epochs, 100 cycles from 2015-01-01. Its `lam: 20`
  geopotential-tendency penalty has no counterpart (off).

## The official NeuralGCM backend (`backends/neuralgcm/`, `model_backend: neuralgcm`, 2026-11-05)

`neuralgcm.PressureLevelModel` checkpoints (e.g. `deterministic_2_8_deg.pkl`)
through the same bridge, to reproduce the original JAX ps-obs experiments
(long-window-4dvar-psobs) with this solver. Keys: `neuralgcm_checkpoint`,
`neuralgcm_input_dir` (the original runs' per-time ERA5 netCDFs,
YYYY-MM-DDTHH.nc, 6-hourly, which also give ICs, forcings and the z500 /
orography verification), `neuralgcm_state_scales`, `ngcm_train_path` (for
`ngcm_train.precision.use_highest_precision()`: dinosaur's default GPU
float32 dot is miscompiled with jax <= 0.11.1).

- 1 h step, so 3-hourly obs fall on model steps (dt_verif 3).
- Packed state (1, 390, 128, 65): every float leaf of the ModelState --
  `state` (193 channels: vorticity, divergence, temperature_variation,
  ln ps, cloud ice, cloud liquid, humidity; modal (127, 65)), `memory` (the
  previous state, same structure), diagnostics, and the two sim_time
  scalars in a last channel. The control (latent) is the 193 `state`
  channels; everything else is a `diagnostic_column`, so the control mask
  never resets the memory.
- `decode` and `advance` take SINGLE-TIME forcings (`data_from_xarray`;
  advance prepends its own time axis): passing a time series fails inside
  `jnp.interp`. The backend uses the 6-hourly file nearest each step, as
  `unroll` does (round half to even).
- No surface pressure from `decode`: `ps_operator: logpinterp` (the
  original's); `decode_state` adds 'sp' from the state's ln ps
  (`model_coords.horizontal.to_nodal`) for `jc_ps_weight`.
- `state_scales` default {tracer_specific_humidity: 1e5} (the original's,
  in its own units). Tracers not named there are FROZEN (increment scale
  0), like the original's `mask_gradient`.

Pitfalls found getting the original setup to descend:

- **`control_variables` does not freeze excluded variables.** It resets
  them to the background only AFTER the first model step, so their
  increments still get gradients through that step and AdamW moves them by
  ~lr per element. With cloud ice / liquid (rms ~2e-7) excluded that way,
  the 1e-5 first step was ~50x the cloud field and the cost ROSE (529k ->
  935k at epoch 2, non-finite gradient at epoch 3). Freezing them in the
  backend (as the original did) fixed it. Any backend whose latent is the
  state itself should freeze, not mask, variables it must not change.
- The original's optimizer is `optax.adamw` with
  `warmup_constant_schedule(lr/10 -> lr over 10 epochs)`: in this solver
  `warmup_steps: 10, start_factor: 0.1, end_factor: 1.0`.

Result (`config_neuralgcm_4d_2015.yml` with `n_init: 4`): first background
z500 rms 9.75 / 3.18 / 7.95 / 7.49 m (NH / tropics / SH / global) against the
original's 9.75 / 3.16 / 7.95 / 7.48; first-cycle cost 528716 -> 205575
(original 534874 -> 199026); ~5 min per cycle on an H100 (as the original).
Analyses are 3-7 m worse than the original's (global 16.9, 13.1, 12.5, 11.9
vs ~10.2, 9.0, 9.1, 9.1 m), narrowing each cycle. Known differences: the
original's `lam: 20` penalty on the first-step (1 h) geopotential tendency
at all levels (normalized per level by its spatial std, times N_obs) has no
counterpart here; and this solver reports the analysis at window start +
dt_verif (+3 h), the original at the window start. Next: whether
`jc_ps_weight` (sp-based) can do the job of `lam`.

Open: the `ngcm` backend still excludes condensate via `control_variables`
(its default 1e5 factor keeps condensate steps tiny, but it is not frozen).
