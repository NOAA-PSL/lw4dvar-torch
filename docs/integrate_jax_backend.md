# JAX backends: the torch/JAX bridge and the ngcm, neuralgcm and graphcast backends

Read this when working on `backends/jax_bridge.py`, `backends/hybrid_levels.py`,
`backends/ngcm/` (`model_backend: ngcm`), `backends/neuralgcm/`
(`neuralgcm`) or `backends/graphcast/` (`graphcast`), or when adding another
JAX model.

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
dt_verif (+3 h), the original at the window start.

`jc_ps_weight` sweep (2026-11-05, 4 cycles, `logs/neuralgcm_jc{30,300,3000}_*.out`).
Global z500 analysis rms, cycles 1-4:

- kappa 30: 9.43, 8.61, 8.28, 8.77 m. The first cycle's cost fell
  529k -> 216k, but cycles 3-4 ended above their epoch-1 cost, and
  cycle 3 kept the background.
- kappa 300 and 3000: the cost ROSE (to 1.9-2.6M at kappa 3000), and
  most cycles kept the zero increment (analysis = background).
  Backgrounds degraded each cycle (10.6 m by cycle 4 at kappa 3000).

The first-step sp change stayed at 0.03-0.05 hPa rms at every kappa.
The first AdamW step (about lr per element) creates that imbalance, and
at a constant lr the optimizer cannot remove it, so a stiffer penalty
only adds cost. Kappa 30 roughly reaches the original's `lam: 20`
analyses (~10.2, 9.0, 9.1, 9.1, verified at t0). Its optimization is
still poor, though.

Static-ensemble Jb and learning-rate sweep (2026-11-06..08; 120 cycles,
2015-01-01..01-30, `config_neuralgcm_4d_2015_t0inc_jb36m12_*_120.yml`,
untracked; logs `logs/neuralgcm_t0inc_jb36m12_*`). All runs use a t0
increment, Jc off, and an NMC Jb from 36 h − 12 h forecast differences
(`jb_samples: .../neuralgcm_2p8_nmc36m12_dec2020may2021_6h.pt`,
`jb_loc_km: 1500`). Z500 rms (m) within the 4-day window, averaged over
the 120 cycles (`diagnostics/z500err_window.py`), analysis at 0 h / 96 h:

| run | NH | Tropics | SH | Global |
|---|---|---|---|---|
| lr 1e-3 | 12.7 / 19.5 | 9.7 / 11.2 | 19.8 / 29.5 | 14.7 / 21.5 |
| lr 5e-3 | 12.0 / 17.4 | 9.2 / 10.5 | 17.0 / 25.3 | 13.1 / 18.8 |
| `jb_inflation: 4`, lr 2.5e-3 | 12.0 / 17.4 | 9.2 / 10.5 | 17.0 / 25.2 | 13.2 / 18.7 |
| lr 1e-2, `obs_density_box_deg: 10` | 12.2 / 16.7 | 8.2 / 9.5 | 15.7 / 19.6 | 12.5 / 15.9 |

- Inflating B 4x at half the lr gives the same result as lr 5e-3.
- lr 1e-2 with obs-density weighting is best, mostly in the SH and tropics
  (sparse or clustered obs), while the NH barely changes. The background
  also improves (global 96 h 20.4 → 17.2 m), so the gain carries from cycle
  to cycle. This run changed both lr and the weighting, so their separate
  effects are not known.
- Running: lr 5e-2 with the same settings (job 25823479).

Open: the `ngcm` backend still excludes condensate via `control_variables`
(its default 1e5 factor keeps condensate steps tiny, but it is not frozen).

## The GraphCast backend (`backends/graphcast/`, `model_backend: graphcast`, 2026-11-06)

GraphCast_small (DeepMind's `graphcast` package: 1 degree, 13 pressure
levels, 6 h step) through the same bridge. It is the model of lw4dvar-jax's
`graphcast` backend, run here with this solver.

Keys:

- `graphcast_checkpoint`: `graphcast_small.npz`.
- `graphcast_stats_dir`: `{mean,stddev,diffs_stddev}_by_level.nc`.
- `graphcast_input_dir`: lw4dvar-jax's per-time ERA5 netCDFs,
  YYYY-MM-DDTHH.nc, 6-hourly. These also give the ICs, the z500 truth
  and the orography.
- `graphcast_frozen`: variables held at the background. Default
  `[specific_humidity]`.

All three paths currently live under
`/scratch4/BMC/gsienkf/Jeffrey.Whitaker/long-window-4dvar-jswhit2/`
(`models/`, `stats/`, `input/2015-gc`, which covers 2014-12-29T18 to
2015-04-30T18).

### Environment: `graphcast-torch`

A venv on the `ngcm-sl` conda env (JAX 0.11.1), built with
`--system-site-packages`. `zz_ngcm_torch.pth` adds `ngcm-torch`'s
site-packages, so torch 2.7.1+cu128 (and its nvidia symlinks) is not
reinstalled. It adds, with `pip install --no-deps`:

- graphcast 0.2, from `/scratch4/BMC/gsienkf/Jeffrey.Whitaker/python/weathernext`;
- jraph, trimesh, rtree and dm-tree.

Path: `/scratch4/BMC/gsienkf/whitaker/conda/envs/graphcast-torch`.
Launcher: `run_graphcast.sh`, a copy of `run_ngcm.sh`. It runs
`sbatch run_graphcast.sh CONFIG.yml`, or any `.py` given as the first
argument.

### State and control

- **Packed state.** Channels-first `(1, 166, 181, 360)`.
  - The model takes two input times, t - 6 h and t. The 83 target
    channels at t come first, then the same 83 at t - 6 h.
  - The 83 channels are 5 surface variables plus 6 variables on 13
    levels. They are in task order, with levels running 50 -> 1000 hPa.
  - Latitudes run south to north and include the pole rows. The FCN3
    bilinear interpolator handles this grid.
- **Forcings and static fields.** The 5 forcings (TOA solar, year/day
  progress) are computed from the date with `graphcast.data_utils`, not
  read from files. They are passed as the bridge's non-differentiated
  `constants`, together with the static fields and the parameters.
  Closing over the parameters instead would embed them as compile-time
  constants.
- **Latent control.**
  - The latent is the t slice, and the analysis is valid at window start
    (`increment_at_start`).
  - Increment = latent x `stddev_by_level`. This is lw4dvar-jax's
    normalization, so learning rates carry over.
  - The t - 6 h slice stays at the background. It is not in the latent;
    it is a `diagnostic_column`.
  - Frozen variables get an increment scale of 0.
  - One difference from lw4dvar-jax: AdamW's weight decay pulls toward
    the background. lw4dvar-jax's offset normalization pulled toward the
    climatological mean.
- **decode_state.** It is pure torch, with no model call.
  - Level families are returned surface first, the order
    `get_surface_pressure` assumes.
  - It adds `z500`. Both the short names (z, t, q, u, v, w, t2m, msl,
    u10, v10, tp6) and the long names are accepted.
  - There is no surface pressure. Ps obs need `ps_operator: logpinterp`
    (validation enforces this), and `jc_ps_weight` uses `msl`.
- **Not supported:** `jb_samples`.

### Validation

`backends/graphcast/smoke_test_graphcast.py`, 2015-01-02T00, H100.

**Against lw4dvar-jax's own adapter.** Same input files, 2 steps.

- Max |diff| after step 1: z 27 m^2/s^2 (field rms 9.8e4), T 0.27 K,
  u 0.7 m/s, msl 29 Pa.
- This is bf16 noise: the differences are the same size under XLA's
  deterministic ops.

**Skill.** z500 rms vs ERA5: 1.9, 3.1, 3.6 and 4.4 m at +6, +12, +18 and
+24 h.

**Speed.** A 16-step (4-day) rollout takes 0.6 s forward and 1.7 s forward
plus backward. Peak torch memory is 0.3 GB.

**Run-to-run noise.** GraphCast's forward pass is not exactly repeatable on
the GPU. The model runs in bf16, and its GNN sums are scatter-adds whose
order varies.

- The same 24 h cost repeated 4 times spreads by 0.14 on a cost of about
  19.
- The same config's first cost differed by 0.15% between two runs (12713
  vs 12733).
- `XLA_FLAGS=--xla_gpu_deterministic_ops=true` removes the spread but is
  about 25x slower (1.7 s -> 45 s for 16 steps). Don't use it for runs.

**Gradient vs central finite differences.** The cost is the 24 h z500 error
against ERA5. Each test perturbs one family with a smooth bump of
amplitude 0.003-0.3 latent units (stddevs).

| family | relative difference |
|---|---|
| u | 0.4-3 % at 0.03-0.1, 3-6 % at 0.3 |
| msl | 1-13 % at 0.1-0.3 |
| T | 14-25 % at 0.03-0.1, ~40 % at 0.3 |
| z | 27-86 % at every amplitude |

- **Small perturbations.** Below about 0.03 stddevs the finite
  differences are meaningless, even with deterministic ops. GraphCast
  normalizes its inputs and then casts them to bf16, which resolves only
  about 0.004 normalized units, so the cost is a staircase at that scale.
- **z is strongly nonlinear.** Its finite difference grows about 9x when
  the amplitude triples from 0.1 to 0.3. The model reacts nonlinearly to
  geopotential increments that are not in balance with T.
- **The bridge is not at fault.** A layout or transpose error would break
  every family. Since u and msl agree, the mismatches come from bf16 and
  nonlinearity.
- **Open issue: the geopotential increment** (2026-11-07, see below).

### Solver test (`config_test_graphcast.yml`)

The setup of lw4dvar-jax's test: 2 cycles of a 12 h window from
2015-01-11T00, `nhrs_back: 0`, 3 epochs, lr 1e-3, weight decay 1e-4.

**Learning-rate schedule.** Use `start_factor: 1.0`. lw4dvar-jax's
warmup is `int(0.2 * max_epoch)` steps, which is 0 at 3 epochs, so its
first step runs at the full lr. With `warmup_steps: 1, start_factor: 0.1`
this solver's first step was 10x smaller.

Results:

| | this solver | lw4dvar-jax |
|---|---|---|
| obs used per slot | identical (9755 / 9783 / 10110) | |
| cycle 1 epoch-1 cost | 12713-12733 | 12694.5 |
| cycle 1 Jo at t+0 (no model step) | 4032.8 | 4028.6 |
| cycle 1 cost after 3 epochs | 11368 | 11807 |
| cycle 2 z500 rms after analysis (NH / tropics / SH / global) | 2.73 / 2.29 / 2.21 / 2.41 m | 2.74 / 2.31 / 2.11 / 2.40 m |

- The Jo difference at t+0 involves no model step, so part of the gap in
  the first cost comes from the ps operator and its interpolation.
- The cost falls faster here because the later steps are larger. This
  solver decays the lr with a cosine to `end_factor`; lw4dvar-jax decays
  from 1e-3 to about 3e-4 and then 1e-5.

### Open issue: the geopotential increment (2026-11-07, not yet tested)

**The problem.** GraphCast takes z as an input independent of T. The
control therefore lets z move freely on all 13 levels. Physically, z
follows hydrostatically from T and surface pressure. An increment that
moves z without a matching change in T is an input combination the model
never saw in training. The gradient check shows the model reacting to it
very nonlinearly, which likely makes the minimization harder.

**Option 1: freeze z.** Set `graphcast_frozen: [specific_humidity,
geopotential]`. The analysis then changes T, u, v, w, msl and the other
surface fields, and z changes only through the model from +6 h on. The
drawback is that `logpinterp` computes model surface pressure from the z
profile alone (`get_surface_pressure`). With z frozen, Jo at window start
no longer depends on the control. The window-start obs are then wasted,
and the other obs times are fitted only through GraphCast's response to
the T, wind and msl increments.

**Option 2 (preferred): derive z from T and msl.** Drop z from the
independent control and compute its increment hydrostatically inside
`apply_increment`:

- start with a surface term from the msl increment;
- integrate the T increment upward over the 13 pressure levels;
- optionally account for the humidity (virtual temperature) contribution,
  although q is frozen by default.

This keeps the input consistent with what the model expects and keeps the
window-start obs informative.

**How to test it.** Run `config_test_graphcast.yml` and a longer cycling
run with the current control and with option 2. Compare the cost descent,
the gradient check for z and T, and the z500 analysis and background
errors.
