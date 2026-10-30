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
  own env with its own launcher: `aifs1` (`run_aifs.sh`), `fcstnet3`
  (`run_fcn3.sh`; also runs `sfno` via `run_sfno.sh`), `aurora` (`run_aurora.sh`),
  all under `/scratch4/BMC/gsienkf/whitaker/conda/envs/`; `ace2ic` is a
  login-node-only env for ACE2 IC/verification fetching. Backend modules are
  imported lazily so a process never needs another backend's dependencies.
  Exceptions, both for the same reason -- a dependency gap in a shared,
  not-group-writable env this account can't fix in place:
  - `aifs2-1.0deg-v1.0` (a user-trained checkpoint, still `model_backend:
    aifs`) needs its own env (`lwaifs2`, `run_aifs2-1.0deg-v1.0.sh`) because
    of a `hydra-core` pin gap in the shared `aifs2` env -- see
    `integrate_aifs2-1deg-v1.0_backend.md`. The conda env is really keyed by
    *checkpoint*, not strictly by *backend*, whenever two checkpoints of the
    same backend need genuinely different pins.
  - `ace2` (`run_ace2*.sh`) uses its own user-owned env, `lwaifs2`-style named
    `lwace2` (`/scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/envs/lwace2`,
    NOT the shared `/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2`), built
    2026-10-21 from this repo's own `ace2-spec.txt`/`ace2-requirements.txt`
    (`pip check` clean) after the shared env was found missing `requests`
    (listed in `ace2-requirements.txt` but not actually installed live --
    see the `requests`/`pyarrow` entries below) and, independent of that,
    actively used for someone else's ACE2 training work -- see
    `integrate_multiple_backends.md`'s "Two envs, deliberately" section for
    the build recipe (torch+cu128 first, `torch_harmonics` built
    `--no-build-isolation --no-deps` against it, then `fme`/everything else
    with a constraints file pinning torch/torchvision/triton/
    torch_harmonics so pip doesn't silently replace them).
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
- **All observations (surface pressure, and radiosonde profiles when
  enabled) are read from the normalized PREPBUFR parquet archive, not the
  legacy `.txt` files** -- `get_psobs`/`get_raobs` unconditionally import
  `psobs_parquet.py`/`raobs_parquet.py`, which lazily import `pyarrow`.
  `exp.obspath`/`nobs_max` now live under `exp.observations.psobs.path`/
  `.nobs_max` in every config (see `config.yml.template` and
  `integrate_prepbufr_obs.md`). **`pyarrow` dependency, resolved
  2026-10-07**: the shared `aifs2`/`aifs1`/`fcstnet3`/`aurora`/`ace2` envs
  (under `/scratch4/BMC/gsienkf/whitaker/conda/envs/`, not group-writable by
  this account) didn't have `pyarrow` installed. Fixed with a single
  `pip install --user pyarrow` (all five share Python 3.12.14, so one
  install into this account's own `~/.local` site-packages covers all of
  them -- see `integrate_prepbufr_obs.md` for why this works and its
  limits). Each affected `<name>-requirements.txt` now lists
  `pyarrow==25.0.1` to match. **This fix is account-scoped**: a different
  user running these same shared envs still needs their own `--user`
  install (or `pyarrow` added to the shared env for real) until its owner
  does that.
- **Same gap, `requests`, found 2026-10-21, now moot for ACE2**:
  `model_backend: ace2`'s `import ace2_model` (needed for raobs/amv on ACE2
  too, see `integrate_prepbufr_obs.md` "raobs admitted for `model_backend:
  ace2`") failed with `ModuleNotFoundError: No module named 'requests'`,
  raised deep inside `fme`'s own import chain (`fme` -> `wandb` ->
  `wandb.analytics.opentelemetry` -> `import requests`), not from any code
  in this repo. `requests==2.34.2` IS listed in `ace2-requirements.txt` but
  wasn't actually installed in the live SHARED `ace2` env -- same class of
  drift as the `pyarrow` gap above. First patched with `pip install --user
  requests==2.34.2` (account-scoped, same mechanism as the `pyarrow` fix);
  superseded the same day by building a dedicated user-owned `ace2` env (see
  the "One conda env per backend" bullet above), which installs
  `ace2-requirements.txt` fresh and doesn't depend on the shared env's state
  at all. The `--user`-install workaround is still the right move for
  `aifs2`/`aifs1`/`fcstnet3`/`aurora`, which stayed on their shared envs.
- One-off diagnostic/probe scripts and configs are left untracked by
  convention; commit only maintained code and tools.

## Task documents

| Document | Read it when working on |
|---|---|
| `integrate_multiple_backends.md` | anything backend-specific: adding or modifying a backend, backend dispatch, conda envs, checkpoints/submodules, IC/verification fetching, known backend bugs and pitfalls, validation and tuning results (learning rates, window lengths, checkpointing, `torch.compile`), the balance penalty (`jc_ps_weight`), area weights, the z500 diagnostics |
| `integrate_aifs2-1deg-v1.0_backend.md` | the `aifs2-1.0deg-v1.0` checkpoint specifically: its files/config, why it needs its own `lwaifs2` conda env (a hydra-core pin gap in the shared `aifs2` env), its probed grid/variable-set/hidden-mesh identity, and what's not yet validated for it (no real GPU run yet) |
| `integrate_prepbufr_obs.md` | the PREPBUFR-parquet observation pipeline: the parquet-based psobs reader (replacing the legacy `.txt` files), the radiosonde (raobs) forward operator/QC/diagnostics, the `observations:` config schema, and the `pyarrow`-per-env gap above |
| `integrate_amv_satwnd.md` | the satellite-derived wind (AMV) observation stream: why it reads the PREPBUFR archive's `SATWND` class rather than the separate raw-BUFR GOES-16 `amv_parquet` archive (empty for 2015), the GSI-style box-thinning + clamped gross-check design (replaced the original composite-stencil superobbing after it was found to cause a real z500 regression), the `exp.observations.amv` config schema, and what's not yet done (only an 11-cycle real GPU run so far, no GOES-16 stream) |
| `integrate_aircraft_obs.md` | the aircraft (AIRCAR/AIRCFT) observation stream: why it's a new AMV-shaped stream rather than a raobs-shaped one, why temperature and wind are thinned as two independent sub-streams (the archive reports them as mutually exclusive rows), the generalized `_amv_thin_slot`, the `exp.observations.aircraft` config schema, the static (not per-tail -- per-tail turned out not to be buildable, see the doc) pressure x region temperature bias correction (`build_aircraft_temperature_bias_table.py`, `aircraft_temperature_bias_table.json`), the t1 (aircraft temperature off)/t2 (bias correction on) sensitivity tests now cycling on real GPUs (see `sensitiveTest.record`), and what's not yet done (full-month before/after comparison to confirm either sensitivity test actually closes the regression) |
| `integrate_gpsro.md` | the GPS radio-occultation (GPSRO) bending-angle observation stream: the canonical-profile reader/QC/thinning (kept at 1000m -- tried removing it to match GSI, but this repo's 14-level models need it unlike GSI's 60-150+ level ones), the differentiable 1D Abel-transform forward operator (`gpsro_operator.py`, ported from and cross-validated against the `greg-long-window-4dvar` reference repo's JAX implementation; also compared against GSI's own numerical-quadrature operator -- different method, not cross-validated against GSI directly), three QC pieces ported from GSI (clamped gross check, super-refraction screen, model-top extrapolation), the `exp.observations.gpsro` config schema, and what's not yet done (no GPU smoke test of the final combined state, no reviewed error profile) |

Add new task documents (e.g. `adding_new_observations.md`) to this table,
with one line saying when to read them.
