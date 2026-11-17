# Tests

`make check` is the merge gate: it must pass locally before a PR is opened,
and CI (`.github/workflows/check.yml`) runs the same target on every pull
request and push to `main`. Don't skip or loosen a test to get it green.

```
python -m venv .venv-ci && .venv-ci/bin/pip install -r envs/requirements-ci.txt
make check PYTHON=.venv-ci/bin/python
```

The checks run on CPU, with no checkpoints, submodules, LFS content or
network access, in a few seconds. `make check` runs:

- `ruff check`: errors only (syntax errors, undefined names), not style.
- `pytest -m "not gpu and not ace2"`: everything under `tests/` except the
  mini-ACE2 run (below), which has its own target and CI job.

## Layout

- `tests/unit/`: pure functions of the solver core. These are
  `area_weights_from_lats`, the surface-pressure operators
  (`get_surface_pressure`, `preduce`), the horizontal interpolators
  (`backends/{fcn3,ace2,aifs}/*_grid.py`, and `get_grid_interpolator`'s
  choice among them), the obs-density Jo weights, the
  control-variable mask, the parquet ps-obs reader (`src/psobs_parquet.py`,
  `tests/unit/test_psobs_parquet.py`, against tiny hand-crafted archives, not
  `tests/data/mini_ace2`'s fixture), config validation (every tracked
  `config_test_*.yml` and `config.yml.template` must load) and the date
  helpers. Each test checks against an independent reference (exact
  spherical areas, an isothermal atmosphere, weights computed by hand, a
  known-by-construction parquet table), not a copy of the code under test.

- `tests/math/`: the math gates, in float64 on a toy model
  (`toy_model.py`: a `LatentForecastModel` on an 8 x 16 grid, nonlinear in
  the latent increment, with a pressure-level geopotential the logpinterp
  operator reads). `problem.py` builds one 24 h window of synthetic ps obs
  every 3 h (between model steps) and evaluates the solver's own
  `compute_loss_4dvar` with QC frozen. The gates (`gates.py`) each return
  an error measure:
  - `gradient_error`: the cost's gradient against central finite
    differences, for plain Jo, `checkpoint_stride` 0 / 2,
    `control_variables`, `jc_ps_weight`, `obs_density_box_deg`,
    `jc_z_weight` and all of them together (~1e-9 to 2e-8; tolerance 1e-6);
    `_jc_z_penalty`'s value is also checked against a hand-computed case;
    the same for a toy model with `increment_at_start` (NeuralGCM's path:
    increment and control mask at step 0, the penalties over steps 0 -> 1),
    whose Jz is also checked against the value from steps 0 and 1;
  - `taylor_order`: the first-order Taylor remainder falls as eps^2 (order
    1.99; must exceed 1.9);
  - checkpointing (`checkpoint_stride` 1, 2, 3) leaves cost and gradient
    unchanged to 1e-12;
  - `dot_product_error`: <A v, w> = <v, A^T w> for the toy model's
    tangent-linear / adjoint (3e-16) and for `latent_b._WeightedSampleSum`,
    the custom backward of the ensemble B (fp32 inside, ~1e-7).

  Each gate has a planted bug it must catch, with a wide margin: one
  term of `get_surface_pressure`'s interpolation weight detached (gradient
  error 0.26), a checkpoint wrapper that detaches its input (0.82), a
  gradient 1 % too large (Taylor order 0.76), `jc_z_weight`'s per-level
  std normalization detached (gradient error 0.32), an advection whose backward
  rolls the wrong way (dot-product error 0.33), and `_WeightedSampleSum`'s
  backward with its sign flipped. A new backend with a hand-written
  backward (a custom `autograd.Function` or VJP) should get the same
  dot-product test (`model_tangent_and_adjoint`).
  - `test_jax_bridge.py`: the same dot-product gate for
    `backends/jax_bridge.py`'s custom VJP, the backward every JAX backend
    (`ngcm`, `neuralgcm`, `graphcast`) relies on. A from `jax.jvp` on a
    small scanned, checkpointed rollout, A^T from torch autograd through
    the bridge, for all inputs and both outputs, one output unused (None
    cotangent) and constant inputs (~1e-15; tolerance 1e-12), float64 on
    CPU. Planted bugs: a backward that drops one output's cotangent (error
    0.90) and one that linearizes about the wrong point (0.64). Needs jax
    (in `envs/requirements-ci.txt`; skipped without it).
    `backends/test_jax_bridge.py` stays as the GPU check (zero-copy,
    streams).

- `tests/ace2/` (marker `ace2`, `make check-ace2`, CI job `ace2-mini`): the
  solver end to end with a real, tiny model -- the mini ACE2 (2.5 deg,
  16-layer CRTM+land variables, SFNO embed_dim 64) on CPU, one 24 h window of
  ps obs (synthetic parquet psobs, see "Fixtures" below), 50 epochs through
  `src/long_window_4dvar.py`. Checks that it runs, writes its outputs, that
  the cost falls by at least 40% (observed 55%), and that its outputs match
  the golden summary (below); not analysis quality (the model is far too
  small). About a minute and a half of compute, pinned to one thread. Needs
  fme and pyarrow: `pip install -r envs/requirements-ci-ace2.txt`. The run,
  its config and its summary are in `tests/ace2/mini_ace2.py`.

## Golden outputs

Frozen summaries of whole minimizations under `tests/data/golden/` (a few
KB each), so any change to the numbers -- a default, a QC threshold, the
optimizer or its schedule, a weighting, an operator -- fails a test even
when the gradients stay right. If the change is intended, regenerate in the
CI env and say in the PR what moved and why:

```
python tests/make_fixtures.py golden          # the toy-model files
python tests/make_fixtures.py golden --ace2   # also mini_ace2.json (needs fme)
```

A failure lists the values that moved, with golden and new values.

- `tests/golden/` (in `make check`): the solver's `compute_optimal` on the
  math gates' toy problem, float64, 20 epochs of AdamW with the warmup +
  cosine schedule, with QC active -- four obs carry gross errors the
  background check rejects, and the station-height check rejects a few
  more. Five configs (`toy_golden.CONFIGS`): defaults; obs-density weights
  plus `jc_ps_weight`; `control_variables` plus `checkpoint_stride: 2`;
  `ps_operator: ps`; `jc_z_weight`. Frozen: the cost at every epoch, Jo per obs slot at
  the first and last epoch, the increment's statistics and a sample of its
  values, the analysis' per-channel mean and std, and per-slot obs counts,
  mean and rms O-B and O-A. Deterministic, compared to 1e-8 relative. A
  planted change -- the learning rate 1% higher, `bg_check` 4 -> 3.5 -- must
  move the cost far outside that (`test_planted_change_is_caught`).
- `tests/data/golden/mini_ace2.json` (in `make check-ace2`): the mini-ACE2
  run's cost history, Jo per slot, obs diagnostics per slot (counts, QC
  flags, mean and rms O-B and O-A) and area-weighted mean and std of h500,
  TMP850 and PRESsfc in the analysis forecast. Float32, so tolerances are
  per quantity (`mini_ace2.tolerance`), set from running at 1 and at 4
  threads: everything from the background is bit-identical or within 1e-5,
  the cost agrees to ~5e-6 for the first epochs, and the minimization then
  amplifies fp32 rounding to ~2e-3 by epoch 50. So the background, QC and
  the first five epochs are checked tightly (1e-5 / 1e-4), later epochs
  and the analysis to 1e-2. Checked against planted changes: `bg_check`
  4 -> 3.5 moves 76 values (first-epoch Jo by 4-14%), `jc_ps_weight`
  10 -> 11 moves 43 (the cost by 0.06% at epoch 2 to 0.8% at epoch 5).

The config saved in each file is the test's input config. The solver
applies its defaults where it reads each key, so there is no resolved
config to freeze yet (rule 4's resolved-config diff needs that first).

## Fixtures

`tests/data/<name>/` holds small frozen inputs in plain git (about 1 MB per
file, 10 MB in total), rebuilt by `tests/make_fixtures.py <name>` on Ursa.
Anything bigger -- the mini-ACE2 checkpoint, 22 MB -- is a GitHub release
asset (`ci-fixtures-vN`) pinned by sha256 in `tests/data/<name>/manifest.json`.
Tests find it at `$LW4DVAR_MINI_ACE2_CKPT`, else in `$LW4DVAR_FIXTURE_CACHE`
(default `~/.cache/lw4dvar`), else download it there; CI caches it under its
sha256. A new checkpoint goes into a new release, never over an old asset,
so older commits keep testing against the file they pinned.

`data/mini_ace2/psobs/` is the one exception to "rebuilt on Ursa": it's a
synthetic normalized-PREPBUFR-parquet archive (the schema
`src/psobs_parquet.py`/`src/prepbufr_parquet_common.py` read -- see
`docs/integrate_multiple_parquetobs.md`), generated entirely in-process by
`tests/make_fixtures.py`'s `_write_psobs_parquet` (fixed seed, no real obs
archive needed), so it rebuilds anywhere `pip install pyarrow` reaches,
including a laptop.

## What CI does not cover

CI is CPU-only. bf16 autocast, `torch.compile`, the real backends and their
checkpoints, and any science result (O-B, forecast skill) are checked on the
GPU machines, not here. A green CI means the code and its math are
consistent; it says nothing about whether a change improves the analyses.
