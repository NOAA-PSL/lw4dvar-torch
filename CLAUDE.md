# CLAUDE.md

Guidance for Claude Code in this repository. This file is loaded every
session, so it holds only what every task needs plus an index of task
documents; detailed history and findings live in those documents -- read the
relevant one before working on that area.

## What this repository is

`lw4dvar-torch`: a long-window 4D-Var data-assimilation solver for surface
pressure observations, with a latent-space control variable, running on
interchangeable ML forecast-model backends selected by `exp.model_backend`
in the config: `aifs` (AIFS-single 2.0 / 1.1), `fcn3` (FourCastNet3),
`aurora` (Aurora v1.5), `ace2` (ACE2-ERA5), `sfno` (SFNO-73ch-small). Shared driver/solver:
`long_window_4dvar.py`, `long_window_4dvar_utils.py`; backend interface:
`forecast_model.py`; backends: `backends/<name>/`; all config keys are
documented in `config.yml.template`.

## Rules that apply to every task

- **One conda env per backend, never merged** -- run each backend in its
  own env with its own launcher: `aifs2`/`aifs1` (`run_aifs.sh`), `fcstnet3`
  (`run_fcn3.sh`; also runs `sfno` via `run_sfno.sh`), `aurora` (`run_aurora.sh`),
  `ace2` (`run_ace2.sh`), all
  under `/scratch4/BMC/gsienkf/whitaker/conda/envs/`; `ace2ic` is a
  login-node-only env for ACE2 IC/verification fetching. Backend modules are
  imported lazily so a process never needs another backend's dependencies.
- **Compute (GPU) nodes have no internet.** Every IC, verification file,
  checkpoint and forcing file must be prefetched from a login node first
  (`backends/<name>/*_prefetch_*.py`, `backends/ace2/ace2_ic.py`).
- **Clone/checkout recipe is in README.md ("Getting the code")**: clone with
  `GIT_LFS_SKIP_SMUDGE=1 git clone --recursive`, then `git lfs pull` in the
  AIFS/FCN3 submodules and `backends/ace2/ace2_prefetch_checkpoint.py YEAR ...`
  for ACE2 -- the ACE2-ERA5 submodule is kept pointer-only (its HF repo is
  ~75 GB of LFS content).
- **Give every concurrent run its own `exp_name`** -- runs sharing one write
  identically named files into the same output directory and overwrite each
  other's trajectories (this happened; see the backends doc).
- **Area-weight grid statistics with `model.area_weights`**
  (`forecast_model.area_weights_from_lats`), not cos(lat) -- cos(lat) is
  wrong on AIFS's reduced grid.
- One-off diagnostic/probe scripts and configs are left untracked by
  convention; commit only maintained code and tools.

## Task documents

| Document | Read it when working on |
|---|---|
| `integrate_multiple_backends.md` | anything backend-specific: adding or modifying a backend, backend dispatch, conda envs, checkpoints/submodules, IC/verification fetching, known backend bugs and pitfalls, validation and tuning results (learning rates, window lengths, checkpointing, `torch.compile`), the balance penalty (`jc_ps_weight`), area weights, the z500 diagnostics |

Add new task documents (e.g. `adding_new_observations.md`) to this table,
with one line saying when to read them.
