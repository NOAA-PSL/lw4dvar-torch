# Tests

`make check` is the merge gate: it must pass locally before a PR is opened,
and CI (`.github/workflows/check.yml`) runs the same target on every pull
request and push to `main`. Don't skip or loosen a test to get it green.

```
python -m venv .venv-ci && .venv-ci/bin/pip install -r requirements-ci.txt
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
  control-variable mask, config validation (every tracked `config_test_*.yml`
  and `config.yml.template` must load) and the date helpers. Each test checks
  against an independent reference (exact spherical areas, an isothermal
  atmosphere, weights computed by hand), not a copy of the code under test.

- `tests/ace2/` (marker `ace2`, `make check-ace2`, CI job `ace2-mini`): the
  solver end to end with a real, tiny model -- the mini ACE2 (2.5 deg,
  16-layer CRTM+land variables, SFNO embed_dim 64) on CPU, one 24 h window of
  ps obs, 50 epochs through `long_window_4dvar.py`. Checks that it runs,
  writes its outputs, and that the cost falls by at least 40% (observed 59%);
  not analysis quality (the model is far too small). About a minute of
  compute. Needs fme: `pip install -r requirements-ci-ace2.txt`.

## Fixtures

`tests/data/<name>/` holds small frozen inputs in plain git (about 1 MB per
file, 10 MB in total), rebuilt by `tests/make_fixtures.py <name>` on Ursa.
Anything bigger -- the mini-ACE2 checkpoint, 22 MB -- is a GitHub release
asset (`ci-fixtures-vN`) pinned by sha256 in `tests/data/<name>/manifest.json`.
Tests find it at `$LW4DVAR_MINI_ACE2_CKPT`, else in `$LW4DVAR_FIXTURE_CACHE`
(default `~/.cache/lw4dvar`), else download it there; CI caches it under its
sha256. A new checkpoint goes into a new release, never over an old asset,
so older commits keep testing against the file they pinned.

Still to come (issue #20): a CPU toy backend with a one-window smoke run,
the math gates (gradients vs finite differences, Taylor and adjoint
dot-product tests, each with a planted bug), and golden outputs.

## What CI does not cover

CI is CPU-only. bf16 autocast, `torch.compile`, the real backends and their
checkpoints, and any science result (O-B, forecast skill) are checked on the
GPU machines, not here. A green CI means the code and its math are
consistent; it says nothing about whether a change improves the analyses.
