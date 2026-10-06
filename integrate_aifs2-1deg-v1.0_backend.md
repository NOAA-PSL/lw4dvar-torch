# Adding `aifs2-1.0deg-v1.0`: a user-trained 1-degree AIFS-single-2.0 checkpoint (2026-10-02)

Read this when working on the `aifs2-1.0deg-v1.0` checkpoint specifically:
its files, its conda env, and why it needs a separate env at all. For
everything else AIFS-generic (the shared `aifs_model.py`/`aifs_ic.py`,
`model_backend: aifs` dispatch, the aifs1-vs-aifs2 API-compat shims), see
`integrate_multiple_backends.md`.

## What this is

User trained AIFS-single-2.0 at 1-degree resolution (checkpoint at
`cbc9c7b59f4e48b5ae3415a72b0a0e9f/inference-last.ckpt`, under
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/outputs/
pretraining/checkpoint/`) and asked for it to be wired into this repo's
long-window 4D-Var driver, reusing the existing `aifs` backend
(`model_backend: aifs`) and, if possible, the existing shared `aifs2` conda
env -- not a new backend, just a third checkpoint alongside the official
0.25deg AIFS-single-2.0 and AIFS-single-1.1.

## Checkpoint identity -- confirmed by directly probing it, not assumed

Loaded via `anemoi.inference.runners.simple.SimpleRunner` (CPU, login node,
fast metadata-only path) and then via the real `RunConfiguration.load()` +
`create_runner()` path `aifs_model.py` itself uses (CPU, slower -- builds the
full model/graph):

- `timestep`: 6:00:00 (matches both existing checkpoints' convention).
- `number_of_input_features`: 89 (vs. the official 0.25deg checkpoint's ~103+
  -- fewer because of the missing categories below).
- Native grid: 40320-point O96 reduced-Gaussian grid (lat range
  -89.28..89.28) -- confirmed via direct inspection, not inferred from the
  "1 degree" name. Coincidentally the exact same point count as the official
  0.25deg/N320 checkpoint's HIDDEN mesh (see `AIFSModel.latent_shape`'s
  docstring) -- this 1-degree checkpoint's full DATA grid is roughly what
  used to be only the 0.25deg checkpoint's internal encoder bottleneck
  resolution.
- **Zero wave variables (no `mwd`/`swh`) and no `snowc`/`sd` at all** --
  confirmed directly, the same situation as AIFS-single-1.1 (see
  `aifs_inference_v1p1.yaml`'s docstring in `integrate_multiple_backends.md`).
  **User confirmed (2026-10-02) this matches how the checkpoint was trained**
  -- not a bug in the export.
- Full 89-variable list (base families): `10u, 10v, 2d, 2t, lsm, msl, q
  (12 levels: 100-1000hPa, no 10/50hPa), sdor, skt, slor, sp, t (14 levels),
  tcw, u (14 levels), v (14 levels), z (single + 14 levels)`, plus the usual
  computed lat/lon/time trig forcings (`cos_latitude`, `sin_julian_day`, ...).
  No `sot`/`swvl1`/`swvl2` (volumetric soil moisture) either -- unlike the
  official 0.25deg checkpoint's optional `control_variables` example in
  `config.yml.template`, there is nothing soil-moisture-related to exclude
  here.
- `prognostic_variables`: 76 of the 89 (the rest are constant/computed
  forcings: `lsm`/`sdor`/`slor`/`z` + the trig/julian-day/insolation
  features).
- Full model load (`create_runner` -> `runner.model`, the exact path
  `AIFSModel.__init__` takes): `AnemoiModelEncProcDec`, 231,103,261
  parameters. `multi_step`: 2 (matches both existing checkpoints).
  `select_variables_and_masks` exists and works directly (the aifs2-generation
  API, not aifs1's older fallback) -- `computed+constant` vars/mask:
  `['cos_latitude', 'cos_longitude', 'sin_latitude', 'sin_longitude']` /
  `[5, 7, 25, 27]`. `runner.device` is already a real `torch.device` (aifs2
  API, not aifs1's plain str). `prognostic_input_mask`/`prognostic_output_mask`
  both length 76. `autocast`: `torch.float16` (same as both existing
  checkpoints).
- **Hidden mesh**: name `"hidden"`, **10944 nodes** x 1024 channels -- this is
  the shape `latent_scale`/the 4D-Var control variable must match for this
  checkpoint (`AIFSModel.latent_shape`), genuinely different from the
  official 0.25deg checkpoint's 40320 x 1024 (that checkpoint's data grid IS
  this checkpoint's hidden-mesh node count -- see above). **Not yet exercised
  by a real compute_optimal run** -- flagged here since a config carrying over
  a `latent_scale`/learn_rate tuned for the 40320-node case would not
  necessarily transfer to a ~3.7x-smaller latent space.
- **Conclusion: API-wise this checkpoint behaves exactly like the aifs2
  generation, not aifs1's older one** -- none of `aifs_model.py`'s 8
  aifs1-compat fallback branches (see `integrate_multiple_backends.md`'s
  "AIFS-single-1.1 API compatibility fixes" section) are exercised loading or
  describing this checkpoint. **No code changes were needed in
  `aifs_model.py`/`aifs_ic.py`/`aifs_grid.py` for this checkpoint** -- only a
  new anemoi-inference yaml (no wave/snow config, like v1.1's) and a new
  experiment config/env, all data not code.

## The real blocker: hydra-core, not anemoi -- and a permissions dead end

First attempt: load this checkpoint through `create_runner()` (the real path
`AIFSModel.__init__` uses) under the repo's existing shared `aifs2` conda env
(`/scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2`, pinned by
`aifs2-requirements.txt`/`aifs2-spec.txt`). **Failed**:

```
ModuleNotFoundError: No module named 'hydra._internal.target_policy'
```

(raised from inside `torch.load(self.checkpoint.path, ...)` in
`anemoi/inference/runner.py`'s `model` property -- a pickle-deserialization
failure, not an API-surface issue.)

- **Root-caused, not guessed**: the user's own working inference pipeline for
  this exact checkpoint (`aifs-single-mse-2.0-1deg/scripts/inference/
  submit_run_inference_1.00deg.slurm`) activates a *different* conda env,
  `lwaifs2` (`/scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/envs/
  lwaifs2`) -- NOT this repo's shared `aifs2` env. Compared the two directly:
  `lwaifs2` has `hydra-core==1.3.7`; the repo's `aifs2` env has
  `hydra-core==1.3.5`. **Every other package that matters is byte-identical**
  between `lwaifs2` and `aifs2-requirements.txt`: `torch==2.7.1+cu128`,
  `anemoi-inference==0.8.3`, `anemoi-models==0.9.3`,
  `anemoi-transform==0.1.16.post2`, `anemoi-utils==0.4.35.post3`,
  `torch-geometric==2.6.1`, the same pinned `flash_attn` wheel. (Full
  `pip freeze` diff: only `hydra-core` version, `packaging`'s install channel,
  and one harmless extra `pyarrow==25.0.1` differ -- see
  `aifs2-1.0deg-v1.0-requirements.txt`.)
- Confirmed `hydra._internal.target_policy` is real and present in
  `lwaifs2`'s hydra-core 1.3.7 install, absent from the repo `aifs2` env's
  1.3.5. Read the module directly: it's a genuine hydra 1.3.6/1.3.7 security
  hardening feature (sandboxes what `hydra.utils.instantiate`'s declarative
  `_target_` configs can resolve to/call, with an integrity-digest check) --
  **not something to paper over with a stub/shim module**. A fake stand-in
  class would either fail the real unpickling (wrong identity/fields) or,
  worse, defeat a real pickle-deserialization security control. The correct
  fix is a genuine hydra-core version, not a code workaround.
- **Confirmed full checkpoint load succeeds end-to-end under `lwaifs2`**:
  reran the exact same `RunConfiguration.load()` + `create_runner()` probe
  with `lwaifs2`'s python -- loaded cleanly (see checkpoint identity section
  above; all those numbers came from this run).
- **Could not fix this by upgrading the repo's shared `aifs2` env in place**:
  `/scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2` is owned by
  `Jeffrey.Whitaker`, group `gsienkf`, mode `drwxr-sr-x` -- **not
  group-writable** (confirmed directly: `touch` inside its `site-packages/`
  gets `Permission denied`). Bumping `hydra-core` there is someone else's
  call, not something this session can or should do unilaterally.

## Resolution (user's call, 2026-10-02)

Rather than clone/rebuild a whole separate conda env just to bump one patch
dependency, **reuse the user's own already-working `lwaifs2` env directly**
for this one checkpoint. User explicitly chose this over the alternatives
(asking Jeffrey to bump the shared env's hydra-core; cloning+patching a new
env) and asked for the per-backend-env documentation convention (a
`<name>-requirements.txt` + `<name>-spec.txt` pair) to be followed for it too,
labeled consistently with the checkpoint name (`aifs2-1.0deg-v1.0`) -- a
`-v1.1` label was a typo, corrected to `-v1.0` to match the checkpoint name
chosen earlier in this same conversation.

This means `aifs2-1.0deg-v1.0` has its own env, launcher, and requirements
docs, separate from the existing `aifs1`/`aifs2` pair -- CLAUDE.md's "one
conda env per backend, never merged" rule now has a partial exception: the
conda env is keyed by *checkpoint*, not strictly by *backend*, when two
checkpoints of the same `model_backend` need genuinely different pins.

## Files added

- `backends/aifs/aifs2-1.0deg-v1.0/aifs-single-mse-2.0-1.0deg-v1.0.ckpt` --
  **a symlink**, not a real file or git submodule (this checkpoint isn't from
  an HF repo, it's the user's own training output on scratch) -- points at
  `/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/
  outputs/pretraining/checkpoint/cbc9c7b59f4e48b5ae3415a72b0a0e9f/
  inference-last.ckpt`. The whole `backends/aifs/aifs2-1.0deg-v1.0/`
  directory is gitignored (not portable -- an absolute scratch path, unlike
  the submodule-vendored checkpoints).
- `backends/aifs/aifs_inference_aifs2-1.0deg-v1.0.yaml` -- new,
  minimal anemoi-inference config (patterned on `aifs_inference_v1p1.yaml`:
  no wave/snow `typed_variables`/pre/post-processors, since this checkpoint
  has none of those variables). `patch_metadata.dataset.constant_fields:
  [z, sdor, slor, lsm]` (no `wmb`, confirmed absent from this checkpoint's
  `variable_categories()`, matching v1.1's set, not v2.0's).
- `config_test_aifs2-1.0deg-v1.0.yml` -- new experiment config, modeled on
  `config_test_aifs1.yml`'s small smoke-test shape (not yet a tuned
  production window): `n_init: 1`, single 12h window (`n_verif: 2`,
  `dt_verif: 6`), `max_epoch: 5`, `learn_rate: 1.0e-2` (untuned -- this
  checkpoint's hidden-mesh is ~3.7x smaller than the official 2.0 checkpoint's,
  see "Hidden mesh" above, so its learn_rate/latent_scale sensitivity has not
  been separately validated the way aifs1 vs aifs2 needed different
  learn_rates). `control_variables: [u, v, t, z, q, sp, msl, 2t, 2d, 10u, 10v,
  tcw]` (same list as `config_test_aifs.yml`'s -- all resolve against this
  checkpoint's 89-variable set; `skt` is also available but not yet added).
  `ic_cache: './ic_cache_aifs2-1.0deg-v1.0/'` -- a fresh, separate cache
  directory (not reusing `ic_cache/` or `ic_cache_aifs1/`): this checkpoint's
  grid/variable set differs from both, and the ERA5 GRIB cache is keyed only
  by date, not by checkpoint (see `integrate_multiple_backends.md`'s aifs1
  "stale-cache gotcha" -- the same risk applies here).
- `run_aifs2-1.0deg-v1.0.sh` -- new SLURM launcher, same H100
  partition/account as `run_aifs.sh`, but activates
  `/scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwaifs2`
  (not the shared `aifs2` env) for the reason above.
- `run_prefetch_aifs2-1.0deg-v1.0.sh` -- new SLURM launcher for
  `backends/aifs/aifs_prefetch_ic.py` (unmodified -- already generic/
  config-driven, no checkpoint-specific code needed), submitted to
  `u1-service` (not the GPU partition) since this is a CDS/network fetch, not
  compute. Partition/account/qos (`u1-service`/`gsienkf`/`batch`, 24h hard
  MaxTime) copied from the user's own working precedent for this exact
  checkpoint: `aifs-single-mse-2.0-1deg/scripts/inference/
  submit_fetch_ics_1.00deg.slurm`, which fetches ICs for the same checkpoint
  for a different (non-4D-Var) inference pipeline. Symlinks
  `config_test_aifs2-1.0deg-v1.0.yml` to `config.yml` first (the fixed
  filename `load_config()` defaults to), matching every other run script's
  convention.
- `aifs2-1.0deg-v1.0-requirements.txt` / `aifs2-1.0deg-v1.0-spec.txt` --
  `pip freeze` / `conda list --explicit` of the `lwaifs2` env, matching this
  repo's existing per-env documentation convention
  (`aifs2-requirements.txt`/`aifs2-spec.txt`, etc.).
- `.gitignore`: added `backends/aifs/aifs2-1.0deg-v1.0/` (the checkpoint
  symlink dir) and `ic_cache_aifs2-1.0deg-v1.0/`; also added the
  pre-existing-but-previously-unlisted `ic_cache_aifs1/` while touching this
  block.

## Known gaps -- not yet done

- **No real GPU run yet.** Everything above was verified on a CPU login
  node (checkpoint metadata + full model instantiation only) -- no
  `long_window_4dvar.py` run, no real ERA5 IC fetch, no `compute_optimal`
  pass, no z500 diagnostic. `run_aifs2-1.0deg-v1.0.sh` /
  `config_test_aifs2-1.0deg-v1.0.yml` are ready to submit but unexercised.
- `backends/aifs/aifs_prefetch_ic.py` (via `run_prefetch_aifs2-1.0deg-v1.0.sh`,
  on `u1-service`, under `lwaifs2`) must warm `ic_cache_aifs2-1.0deg-v1.0/`
  before the first real `run_aifs2-1.0deg-v1.0.sh` job -- not yet submitted/
  run as of this writing.
- `learn_rate`/`latent_scale` are carried over from `config_test_aifs1.yml`
  as a starting point, not independently tuned for this checkpoint's smaller
  (10944-node) hidden mesh -- expect this needs its own sweep, the same way
  aifs1 needed a different `learn_rate` than aifs2 (see
  `integrate_multiple_backends.md`'s "aifs1 non-finite-gradient divergence"
  section) for an unrelated but structurally similar reason.
- Whether `hydra-core==1.3.7`'s new target-policy sandboxing changes any
  *runtime* (not load-time) behavior anemoi-inference relies on (e.g. if
  anemoi's own config resolution calls something now blocklisted) has not
  been checked beyond "the checkpoint loads and the model instantiates" --
  no actual `hydra.utils.instantiate()` call has been exercised yet through
  this repo's code paths (`aifs_model.py` never calls it directly; it's
  internal to `anemoi.inference.config.run.RunConfiguration.load()` and
  `create_runner()`, both of which ran successfully here).
- Whether the shared `aifs2` env could instead just have `hydra-core` bumped
  in place (the user's `lwaifs2` env is existence proof that 1.3.7 is
  compatible with the rest of the exact same pinned stack) was raised as an
  option but not pursued -- the env is owned by a different user
  (`Jeffrey.Whitaker`) and not writable by this account.
