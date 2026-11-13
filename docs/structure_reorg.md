# Repo structure reorganization: a flat root into src/configs/envs/docs/slurm

Read this when working on: where a file should live in this repo, why
`src/`'s modules resolve `backends/` the way they do, why some
`diagnostics/*.py`/backend probe scripts compute two different "root"
paths, or anything that looks like a stale path reference to a file that
used to live at the repo root.

## Status: done, committed, merged with main's jax-backend work (2026-11-13)

The repo root held ~64 non-directory entries -- 20 env/requirement files, 16
SLURM/PBS job scripts, 10 config files, 4 long-form docs, and only 6 actual
Python source files, plus `backends/`, `diagnostics/`, `tests/`,
`test_scripts/` which were already their own directories. User-requested
cleanup, done in two passes: first the "pure pointer" categories (docs,
env files, configs -- nothing at runtime cared where these sat except as
string literals), then the Python source files and the job scripts,
scoped down from a broader "move `backends/`+`diagnostics/` too" option
that was rejected because `backends/` contains 3 git submodules and is
referenced by path in 66 other files -- a materially riskier class of
change than anything below. See "What was rejected" at the end.

## What changed

| Old (repo root) | New | Risk |
|---|---|---|
| `*.md` task docs (4) | `docs/` | None -- nothing at runtime reads these by path |
| `*-requirements.txt`, `*-spec.txt` (20) | `envs/` | Low -- only CI config and a few doc comments named them |
| `config.yml.template`, `config_test_*.yml` (10) | `configs/` | Low-medium -- every launcher script and `tests/unit/test_config.py`'s glob named them |
| `long_window_4dvar.py`, `long_window_4dvar_utils.py`, `forecast_model.py`, `latent_b.py`, `psobs_parquet.py`, `prepbufr_parquet_common.py` (6) | `src/` | Medium -- touches real import-resolution code, not just path strings (see below) |
| `run_*.sh`, `resubmit_ace2_mi300x.sh` (16) | `slurm/` | Low -- SLURM/PBS jobs keep the submitter's cwd, so internal relative paths needed no logic change, only the `src/` update above |

`backends/`, `diagnostics/`, `tests/`, `test_scripts/` are unchanged --
deliberately left alone, see "What was rejected".

### docs/, envs/, configs/: pure path updates

`git mv` into the new directory, then fix every reference: `CLAUDE.md`,
`.github/workflows/check.yml`, `Makefile`, `tests/README.md`,
`tests/unit/test_config.py`'s `TRACKED_CONFIGS` glob,
`tests/ace2/test_ace2_mini.py`'s config dict, every `run_*.sh`/
`test_scripts/*.sh` that named a config file, and the two moved
`envs/requirements-ci*.txt` files' own internal example commands. One
`config_test_ace2_polaris.yml` subtlety: its `obspath` pointed at a
placeholder Eagle path that doesn't exist yet (see
`docs/integrate_multiple_parquetobs.md`), left as a `TODO` comment, not
silently "fixed" to something that would look valid but isn't.

**Deliberately NOT touched**: bare filename mentions inside the long
narrative docs (`integrate_multiple_backends.md`, `running_at_alcf.md`,
`ace2-training.md` -- 25+ mentions combined, and dozens more scattered
through `backends/*.py` docstrings/comments). These are historical
decision-record prose, not copy-paste commands -- a reader hitting
`config_test_aifs.yml` in a 2026-era writeup finds it one directory over,
not broken. `docs/integrate_multiple_parquetobs.md` (authored this same
session, not "historical") got its internal references fixed in full,
since there was no reason not to.

### src/: the real risk -- this repo has no installed-package imports

Every module resolves its neighbors via Python's "the launched script's
own directory goes on `sys.path[0]`" behavior, not via `pip install -e .`
or any `PYTHONPATH` setup. Moving the six loose files together into
`src/` preserves this for free for imports *among themselves*
(`long_window_4dvar.py` still flatly `import long_window_4dvar_utils`,
etc.) and for anything invoked as `python src/long_window_4dvar.py ...`
(Python puts `src/` on `sys.path[0]`, so `backends/<name>/*.py`'s own
`import forecast_model`, reached later via `_ensure_backend_on_path`,
resolves fine too). But anything that computed "repo root" itself, to add
it to `sys.path` from a *different* starting point, needed a one-line fix
to add `src/` wherever that was reaching for `forecast_model.py`/
`long_window_4dvar_utils.py`/`latent_b.py` specifically (vs. `backends/`,
which did NOT move and keeps the old, unmodified path):

- **`src/long_window_4dvar_utils.py`**: `_ensure_backend_on_path`'s
  `_REPO_ROOT` now does one more `dirname()` (the file itself sits one
  level deeper); still used only for the `backends/<name>` join, which is
  correctly unaffected by `src/`.
- **`tests/conftest.py`**: inserts `REPO_ROOT/src` instead of `REPO_ROOT`.
- **6 of `diagnostics/*.py`** (`check_ens_b.py`, `check_latent_b_cov.py`,
  `ens_b_vertical.py`, `estimate_latent_b.py`, `z500err_window.py`, plus
  `build_latent_ens_b.py` needed no change at all): each independently
  computes "repo root" via its own `dirname(dirname(__file__))` to reach
  `latent_b.py`/`forecast_model.py`/`long_window_4dvar_utils.py` -- fixed
  to add `src/` to that specific insert. `z500err_window.py` is the
  trickiest case: it reuses the SAME `_REPO_ROOT` variable for two
  different purposes (a `backends/<name>` join, which must stay
  unmodified, and a `from forecast_model import ...`, which needed `src/`
  appended) -- fixed by appending `src/` only at the second call site, not
  by changing `_REPO_ROOT` itself.
- **13 backend scripts**, found via a full `sys.path.insert` sweep across
  `backends/`, not by guessing from filenames -- standalone probes/smoke
  tests/prefetch scripts that import a backend's own `*_model.py`
  (`aurora_model`, `aifs_ic`, etc.), which in turn does `import
  forecast_model` -- so the insert the PROBE script does is for that
  *transitive* import, not for anything the probe script itself names
  directly:
  `backends/aifs/aifs_prefetch_ic.py`,
  `backends/aurora/{profile_aurora_rollout,probe_aurora_nan,probe_aurora_checkpoint,probe_aurora_checkpoint_stride,aurora_prefetch_ic,probe_aurora_compile_confirm,probe_aurora_compile,smoke_test_aurora}.py`,
  `backends/fcn3/fcn3_prefetch_ic.py`,
  `backends/ace2/smoke_test_ace2.py`,
  `backends/sfno/{smoke_test_sfno,probe_sfno_bf16}.py`. Two sub-patterns:
  `os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))`
  (9 files) got `, 'src'` appended to the `os.path.join`; `sys.path.insert(0,
  ".")` (4 files, relying on cwd == repo root rather than `__file__`) became
  `sys.path.insert(0, "src")`. `backends/sfno/{sfno_ic,sfno_model}.py`'s
  OWN inserts (reaching into `backends/fcn3/` as a sibling backend dir) are
  unrelated and untouched.
- **The one real bug caught only by re-grepping after the fact**:
  `test_scripts/run_test_aifs1.sh` still invoked the un-prefixed
  `long_window_4dvar.py` -- missed on the first pass because it's a
  `test_scripts/*.sh` launcher, not a `diagnostics/*.py` or backend probe.
  Lesson: after this kind of move, grep for the bare old name again at the
  end, across every extension, not just the categories touched so far.

`CLAUDE.md`'s own description of the driver/solver/backend-interface
module paths, and the one `_REPO_ROOT` comment in
`src/long_window_4dvar_utils.py` itself, were updated to spell out this
`src/` vs `backends/` split so it doesn't have to be re-derived next time.

### slurm/: lower risk than src/, but not zero

SLURM (`sbatch`) and PBS (`qsub`) jobs run with the working directory set
to wherever the submit command was invoked from, NOT the script's own
directory -- confirmed by checking every moved script for an explicit
`cd`: none of the `run_*.sh` have one, so as long as the convention "submit
from the repo root" continues (now typing `sbatch slurm/run_aifs.sh`
instead of `sbatch run_aifs.sh`), every internal relative path
(`configs/config_test_aifs.yml`, and now `src/long_window_4dvar.py`)
resolves exactly as before. `resubmit_ace2_mi300x.sh` is the one exception
that matters: it explicitly `cd "$(dirname "$0")"` (so its lock/log/state
files sit next to the script, for its cron-based resubmission use case) --
this still works unchanged post-move since its `SCRIPT=` target
(`run_ace2_train_mi300x.sh`) is now a sibling in the same `slurm/`
directory it `cd`s into.

What needed fixing, beyond the `src/long_window_4dvar.py` path in the 8
scripts that invoke it directly: every literal `qsub`/`sbatch <script>.sh`
copy-paste example comment naming itself needed a `slurm/` prefix (these
are meant to be typed from the repo root); `test_scripts/README.md`'s
now-false claim that the production launchers "remain at the repo root";
and comment-only mentions in `backends/aurora/aurora_prefetch_ic.py` and
`backends/ace2/training/rocm_rank_env.sh`. Prose cross-references BETWEEN
moved scripts (e.g. "Ursa twin of run_ace2_train_polaris.sh") were left
bare -- both files are now siblings in `slurm/`, so the reference still
resolves the same way a reader would expect.

## What was rejected: moving `backends/`/`diagnostics/` into `src/` too

Considered and explicitly turned down. Moving them WITH the six loose
files (preserving relative structure) would have made the `_REPO_ROOT`-style
internal path logic above correct for free (no code changes needed, since
it's all relative) -- but `backends/` is referenced by path in 66 files
outside itself (README, CLAUDE.md, every doc, every launcher), and, the
actual blocker: `backends/{aifs,fcn3,ace2}/` contains 3 git submodules
(`.gitmodules` + physical checkouts). Relocating those means updating
`.gitmodules` and re-syncing every existing checkout's submodule state --
a meaningfully riskier class of operation than anything else in this
reorg, for a repo where other people/processes may have their own
checkouts. Scoped down to "6 loose files only, `backends/`/`diagnostics/`
untouched" instead.

## Follow-up: merging main's jax-backend work (2026-11-13)

This reorg landed on `add-parquet-ps-dev` as a single squashed commit
(`69ed2ce`, alongside the parquet ps-obs work -- see
`docs/integrate_multiple_parquetobs.md`), pushed, and opened as PR #26.
`main` had moved 11 commits ahead in the meantime (PR #22,
`jax-backend`: the `ngcm`/`neuralgcm`/`graphcast` backends, predating this
reorg). Merging `origin/main` in (`git merge-tree` dry run first, then a
real merge, commit `d8a1983`) found only 2 real conflicts, both narrow
textual overlaps on lines both branches happened to touch for unrelated
reasons -- not a logical clash -- and auto-merged `configs/config.yml.template`,
`diagnostics/z500err_window.py`, `tests/unit/test_config.py` cleanly on
its own:

- **`CLAUDE.md`**: 3 hunks (intro backend list, conda-env launcher bullet,
  task-docs table) -- this branch re-pathed the existing rows into
  `src/`/`configs/`/`docs/`/`slurm/` and added 2 new doc rows; `main` added
  the `ngcm`/`neuralgcm`/`graphcast` backends and 1 new doc row
  (`integrate_jax_backend.md`). Resolved by keeping both sides' additions
  and re-pathing `main`'s new doc row to `docs/integrate_jax_backend.md`
  for consistency with the rest of the table.
- **`src/long_window_4dvar_utils.py`**: 1 hunk, 1 line -- the
  `model_backend` validation error message, where this branch's
  `config.yml.template` -> `configs/config.yml.template` path fix and
  `main`'s expanded allowed-backend list landed on the same f-string line.
  (The `if backend not in (...)` tuple check itself, one line above,
  auto-merged with no conflict since only `main` touched it.)

**A real bug the merge surfaced, not just a conflict**: `main`'s
jax-backend work predates this reorg, so its 7 new root-level files landed
back at the flat repo root after the merge -- `config_neuralgcm_4d_2015.yml`,
`config_ngcm_4d_2015.yml`, `config_test_graphcast.yml`, `config_test_ngcm.yml`,
`integrate_jax_backend.md`, `run_graphcast.sh`, `run_ngcm.sh`. Git doesn't
flag this as a conflict (no shared history to diff against), so it's easy
to miss. Worse than cosmetic: `tests/unit/test_config.py`'s `TRACKED_CONFIGS`
glob only looks under `configs/config_test_*.yml`, so
`config_test_graphcast.yml`/`config_test_ngcm.yml` were silently excluded
from the "every tracked config must load" test -- and when actually
loaded (`.venv-ci` + `load_config()`), all 4 new configs failed outright:
they still used the pre-parquet-migration flat `obspath` key, which this
branch's `_resolve_observations_config` has no fallback for anymore (see
`docs/integrate_multiple_parquetobs.md`). Fixed in a follow-up commit
(`a8a3798`): `git mv` all 7 files into `configs/`/`docs/`/`slurm/`,
migrated all 4 configs to the `observations.psobs` schema (same pattern as
every other tracked config), and fixed `run_ngcm.sh`/`run_graphcast.sh`'s
own solver-invocation line (`long_window_4dvar.py` -> `src/long_window_4dvar.py`,
default config -> `configs/config_test_*.yml`) -- their narrative comments
(`sbatch run_ngcm.sh ...` usage examples in docstrings elsewhere) were left
bare, same "historical prose, not copy-paste commands" convention as
`integrate_multiple_backends.md` etc. above. `make check` went from 117 to
119 passed (the 2 newly-relocated configs are now actually exercised).

**Lesson for next time**: after merging in a sibling branch that predates
this kind of reorg, explicitly diff the new-file list against the merge
base (`git diff --diff-filter=A --name-only <merge-base> <other-branch>`)
and check which of those look like they belong in `configs/`/`docs/`/`envs/`/`slurm/`
by naming convention -- git's conflict detection has nothing to say about
files only one side added, so this class of issue is silent by
construction.

## Verification

Both stages (src/ move, then slurm/ move) were verified independently and
together, from a from-scratch `.venv-ci` (`rdhpcs-python/3.12` +
`envs/requirements-ci.txt`/`envs/requirements-ci-ace2.txt`, same as
`tests/README.md` documents): `ruff check .` clean, all unit tests pass
(112), and -- the test that actually exercises the `src/` + `configs/`
path resolution end to end, since it invokes `src/long_window_4dvar.py` as
a real subprocess the same way a `slurm/run_*.sh` job does --
`pytest -m ace2` (the mini-ACE2 solver smoke test) passes.
