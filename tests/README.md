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
- `pytest -m "not gpu"`: everything under `tests/`.

## Layout

- `tests/unit/`: pure functions of the solver core. These are
  `area_weights_from_lats`, the surface-pressure operators
  (`get_surface_pressure`, `preduce`), the obs-density Jo weights, the
  control-variable mask, config validation (every tracked `config_test_*.yml`
  and `config.yml.template` must load) and the date helpers. Each test checks
  against an independent reference (exact spherical areas, an isothermal
  atmosphere, weights computed by hand), not a copy of the code under test.

Still to come (issue #20): a CPU toy backend with a one-window smoke run,
the math gates (gradients vs finite differences, Taylor and adjoint
dot-product tests, each with a planted bug), and golden outputs.

## What CI does not cover

CI is CPU-only. bf16 autocast, `torch.compile`, the real backends and their
checkpoints, and any science result (O-B, forecast skill) are checked on the
GPU machines, not here. A green CI means the code and its math are
consistent; it says nothing about whether a change improves the analyses.
