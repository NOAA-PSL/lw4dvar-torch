# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this repository is

`lw4dvar-torch` is a **merge** of two sibling repos that each ported the same
long-window 4D-Var data-assimilation solver to a different forecast-model
backend:

- `/scratch4/BMC/gsienkf/Jeffrey.Whitaker/long-window-4dvar-aifsv2/` --
  ECMWF's AIFS Single v2 (PyTorch/anemoi).
- `/scratch4/BMC/gsienkf/Jeffrey.Whitaker/long-window-4dvar-fcstnetv3/` --
  NVIDIA's FourCastNet3 (PyTorch/makani).

Both repos' own CLAUDE.md files document the considerable backend-specific
work that went into each port (grid geometry, IC fetching, checkpoint
architecture quirks, memory/chunking fixes, etc.) -- that history is not
duplicated here. This repo's job is narrower: let a user pick **which**
backend runs a given experiment via one config key
(`exp.model_backend: aifs|fcn3`), instead of maintaining two nearly-identical
copies of the solver core in two directories.

The two source repos remain in active use (as of 2026-09-17, both have
in-flight tuning runs) and are not being retired yet -- this repo is
validated (see "Smoke test validation" below) but has not yet run a real,
production-length experiment for either backend. Treat the two source repos
as the historical reference implementations and this repo as where new
cross-backend work should land.

## Repository layout

```
forecast_model.py            # shared LatentForecastModel interface, unchanged
                              # across backends (confirmed identical modulo
                              # one type-annotation widening -- see below)
long_window_4dvar.py          # shared driver, dispatches on model_backend
long_window_4dvar_utils.py    # shared solver core, dispatches on model_backend
config.yml.template           # one template, backend-conditional sections
run_aifs.sh / run_fcn3.sh     # SLURM launchers -- differ only in which
                               # conda env they activate + job name/logs
config_test_aifs.yml          # small (12h window, 5 epoch) smoke-test configs
config_test_fcn3.yml          # for each backend -- see "Smoke test validation"

backends/aifs/
  aifs_model.py, aifs_grid.py, aifs_ic.py, aifs_inference.yaml
  aifs-single-2.0/            # git submodule -> huggingface.co/ecmwf/aifs-single-2.0
backends/fcn3/
  fcn3_model.py, fcn3_grid.py, fcn3_ic.py
  fourcastnet3/                # git submodule -> huggingface.co/nvidia/fourcastnet3
```

`long_window_4dvar_utils.py` is 1526 lines, `long_window_4dvar.py` is 247 --
both noticeably larger than either source repo's version, because they now
contain both backends' logic side by side rather than one.

## Conda environments -- NOT merged, and deliberately so

Four environments, one per checkpoint (not one per backend architecture --
`aifs1` and `aifs2` both drive AIFS-family checkpoints through the same
`aifs_model.py`, just at two different anemoi/torch-geometric pin sets):

```
aifs2:     /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2      (AIFS-single-2.0: anemoi 0.8-0.9.x / torch-geometric 2.6.1 / flash-attn)
aifs1:     /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs1      (AIFS-single-1.1: anemoi 0.5-0.6.x / torch-geometric 2.4.0 / flash-attn)
fcstnet3:  /scratch4/BMC/gsienkf/whitaker/conda/envs/fcstnet3   (makani / physicsnemo / torch-harmonics)
aurora:    /scratch4/BMC/gsienkf/whitaker/conda/envs/aurora     (microsoft-aurora / torch)
```

All four pin `torch==2.7.1+cu128` (same CUDA/ABI target), but their
dependency trees have never been jointly resolved (`anemoi-graphs` wants
`numpy<2`; whatever `makani`/`physicsnemo`/`aurora` want has not been
checked against that, or against each other, or against `aifs1`'s own
older anemoi pins). A real architecture decision was made here, not just
inherited: **do not attempt to merge these into one environment.** "Choose
the backend at runtime" means "choose it in `config.yml`, and run with the
matching env/launcher (`run_aifs.sh` vs `run_fcn3.sh` vs `run_aurora.sh`)"
-- not "one process can load any of them." Every place
`long_window_4dvar_utils.py` needs a backend-specific module
(`aifs_model`/`fcn3_model`/`aurora_model`/`aifs_grid`/`fcn3_grid`/
`aifs_ic`/`fcn3_ic`/`aurora_ic`) imports it *locally*, inside the
function/branch that needs it, behind a small
`_ensure_backend_on_path(backend)` helper that adds `backends/<backend>/`
to `sys.path` on demand. This means a process running under `aifs2` never
needs FCN3's (or Aurora's) dependencies importable at all, and vice versa
-- confirmed directly (not assumed): from the `aifs2` env, `import
fcn3_model` fails with `No module named 'fcn3_model'` unless the fcn3
branch has actually run first, and symmetrically for the other envs.
`aifs1` and `aifs2` share the SAME `aifs_model.py`/`aifs_grid.py`/
`aifs_ic.py` (an `AIFSModel(checkpoint_path=..., config_path=...)`
constructor argument selects which checkpoint/config to load, see
`config.yml`'s `path_model`/`model_name` keys) -- only the conda
environment and the checkpoint file itself differ between the two.

`run_aifs.sh` and `run_fcn3.sh` both explicitly `module load cuda/12.8.1`
(matching both envs' `torch==2.7.1+cu128` build) rather than relying on the
cluster's default module version, which has drifted before (see the
fcstnetv3 repo's CLAUDE.md for the incident this pin traces back to).

## Backend dispatch design

- **`exp['model_backend']`** (`'aifs'` or `'fcn3'`) is the one required new
  config key. `load_config()` raises if it's missing or not one of these two
  values -- there is no silent default.
- **`get_model` / `get_grid_interpolator` / `get_input` / `get_verif`** each
  branch on `exp['model_backend']` at the top and call into the matching
  `backends/<backend>/*_model.py` / `*_grid.py` / `*_ic.py` module. AIFS's
  irregular N320 octahedral grid always uses `aifs_grid.GridInterpolator`
  (k-d-tree/k-NN); FCN3's regular grid defaults to
  `fcn3_grid.BilinearGridInterpolator` (exact, ~50x faster -- see the
  fcstnetv3 repo's CLAUDE.md), with `grid_interp: 'kdtree'` as an opt-out.
- **Packed-state layout differs by backend** (AIFS: `(1, multi_step,
  n_points, n_vars)`, `n_vars` trailing; FCN3: `(1, n_vars, H, W)`, `n_vars`
  at dim 1) -- generalized via a new `model.state_layout` property
  (`'channels_last'` for AIFS, `'channels_first'` for FCN3) that
  `_apply_control_mask`/`_resolve_control_mask` branch on, instead of the
  FCN3-only hardcoding the pre-merge fcstnetv3 repo had. This is the one
  piece of genuinely new shared logic this merge introduced, not just a
  restored/ported branch.
- **`model.wrap_state(tensor, date)`**: small new factory method on both
  `AIFSModel`/`FCN3Model` so `compute_loss_4dvar`/`compute_optimal` never
  need to import/name `AIFSState`/`FCN3State` directly.
- **`AIFSModel._prime_noise()`**: added as a one-line no-op. FCN3's
  stochastic-noise re-priming calls (`compute_optimal`, 3 call sites -- see
  fcstnetv3's CLAUDE.md for why each one matters) now run unconditionally
  for both backends rather than needing a backend check at every call site.
- **`reset_skt_over_ocean`** (AIFS-only) and AIFS's native-`sp` `ps_operator:
  'ps'` path (deleted from the fcstnetv3-only copy this merge started from)
  are both restored verbatim from the AIFS repo. Neither needs a
  `model_backend` check at its call site -- `load_config()` already raises if
  either is set with `model_backend: fcn3`, so they simply never trigger for
  that backend.
- **`FCN3Model.pressure_levels`** gained the same `base="z"` default AIFS's
  version already had (a pure widening -- every current call site already
  passed `'z'` explicitly).

## Vendored checkpoints: git submodules, not symlinks (2026-09-17)

Originally bridged via symlinks straight into the two source repos' own
vendored checkpoint directories (fast to set up, but not standalone and not
what a fresh clone of this repo could reproduce). Replaced with real git
submodules at the user's request:

```
backends/aifs/aifs-single-2.0  -> https://huggingface.co/ecmwf/aifs-single-2.0/   @ 08286fc
backends/fcn3/fourcastnet3     -> https://huggingface.co/nvidia/fourcastnet3/     @ df5d8d0
backends/aifs/aifs-single-1.1  -> https://huggingface.co/ecmwf/aifs-single-1.1    @ 049b9ab
```

`aifs-single-1.1` (added 2026-09-20, see "`aifs1` environment" below) was
the simplest of the three to convert: the user had cloned it directly from
its real Hugging Face URL already (not copied from a sibling repo's own
vendored checkout the way the first two were), so `git submodule add
<the-same-URL> backends/aifs/aifs-single-1.1` just adopted the existing
local clone in place ("Adding existing repo... to the index") with no
`protocol.file.allow` workaround and no re-fetch needed -- confirmed via
`git ls-files --stage` showing the expected `160000` gitlink mode pointing
at the clone's actual `HEAD`, not the full ~1GB checkpoint tracked as
regular file content.

Both pinned at the exact commit the source repos' own vendored clones were
already checked out at -- this is a reproduction of the already-validated
checkpoint, not a new/different version.

**How this was done without re-downloading ~14GB of LFS weights already on
disk**: `git submodule add` was pointed at the *local* source repo's
checkout first (`git -c protocol.file.allow=always submodule add
<local-path> <dest>` -- plain local-path/`file://` clones are blocked by
this environment's git config by default, hence the one-off
`protocol.file.allow=always` override, scoped to that single command, not
persisted anywhere). Because git-lfs resolves a local-path remote's LFS
objects from that remote's own `.git/lfs/objects` cache directly, this
pulled the real 994MB/2.8GB checkpoint files in seconds rather than
re-fetching from Hugging Face. `.gitmodules`' URL was then rewritten to the
real `huggingface.co` URL and `git submodule sync` run to repoint each
submodule's `remote.origin.url` to match -- **without** re-fetching, so the
already-correct local content was left alone. A fresh clone of this repo by
someone else will correctly pull from the real Hugging Face URLs via `git
submodule update --init` (needs `git-lfs` installed; confirmed available
here as `git-lfs/3.6.1`).

Net effect: `git clone --recurse-submodules` (or `git submodule update
--init` after a plain clone) is now sufficient to get a fully working
checkout of this repo -- no more manual symlinking into a sibling repo that
may not exist on another machine.

## Smoke test validation (2026-09-17)

`config_test_aifs.yml` / `config_test_fcn3.yml` (12h window: `n_verif: 2`,
`dt_verif: 6`; `max_epoch: 5`) were run end-to-end on the H100, through this
repo's merged driver, against the real submodule-backed checkpoints (not the
old symlink bridge) -- both completed cleanly (`JOB COMPLETED`, all
diagnostic/forecast files saved, no errors):

- **FCN3** (job 21627000): `Jtot` 19704 -> 21098 -> 19331 -> 19548 -> 18041
  over 5 epochs (noisy but net downward -- same shape as the fcstnetv3
  repo's own small-window runs). `control_variables` resolved to all
  72/72 packed-state columns. z500 error (GL): bg 6.72 -> before 6.67 ->
  after 11.20 -- worse post-optimization, the same "untuned learn_rate on a
  toy window" story documented at length in the fcstnetv3 repo's CLAUDE.md,
  not a merge regression.
- **AIFS** (job 21627001): `Jtot` decreased smoothly and monotonically every
  epoch (29500 -> 29468 -> 29435 -> 29330 -> 29295), `control_variables`
  correctly resolved via AIFS's own naming convention (`2t`, `10u`, `sp`,
  ...) to 75/106 columns. z500 error (GL): bg 8.54 -> before 9.37 -> after
  9.35 -- essentially flat, expected for `lr=1e-4` over only 5 epochs.

Both configs were edited on disk (by the user, directly) while their jobs
sat pending in the SLURM queue -- `control_variables`, `checkpoint_stride:
0`, and the `oberrstart`/`oberrdeltaperday` values were all added/changed
after submission, so the runs above reflect the edited versions (SLURM jobs
read the config file at actual runtime, not at `sbatch` time). One value
(`oberrdeltaperday: -1` in the AIFS config, which per the template should be
a *positive* per-day ob-error growth rate) was flagged as likely backwards
and has since been corrected by the user directly.

**What this validates**: `model_backend` dispatch, the submodule checkpoints,
and the `state_layout`/`_apply_control_mask` generalization all work
end-to-end on real GPU hardware for both backends, not just CPU-only
construction. **What it does not validate**: production-length windows,
`restart: True` cycling, `n_init > 1` multi-cycle runs, or a tuned
`learn_rate` -- none of these have been exercised through this repo yet
(the user's own currently-running FCN3 learning-rate tuning experiment is in
the original fcstnetv3 repo, not here).

## Backend comparison: FCN3 vs AIFS on a matched 5-day window (2026-09-17)

The smoke-test configs above were extended by the user into a real,
controlled side-by-side comparison: `n_verif: 20, dt_verif: 6` (5-day
window), `max_epoch: 100`, `learn_rate: 0.0025`, both backends started from
the identical `nhrs_back=48` back-date (`2014-12-30T00`) and scored against
the same real ps observation files -- run directly from this repo via
`run_fcn3.sh`/`run_aifs.sh config_test_fcn3.yml`/`config_test_aifs.yml`
(job names `lw4dvar_fcn3`/`lw4dvar_aifs`). This is the first real use of the
merge for its intended purpose (a fair apples-to-apples cross-backend
comparison from one driver), not just a validation smoke test.

- **`checkpoint_stride` matters a lot at this window length**: the FCN3 run
  OOM'd at `checkpoint_stride: 0` (no checkpointing -- every one of the 20
  steps' activations held simultaneously) and again, only slightly less
  badly, at `checkpoint_stride: 10` (only 2 of 20 steps checkpointed).
  Both failures are expected, not bugs -- higher `checkpoint_stride` means
  checkpointing happens *less* often (`s % checkpoint_stride == 0`), so it
  trades AWAY memory savings, the opposite of what "increase the stride to
  fix an OOM" would suggest. `checkpoint_stride: 1` (checkpoint every
  step, the default) is what actually worked. Because both attempts
  reused the same config filename and this repo's `run_fcn3.sh` doesn't
  timestamp its `-o`/`-e` files, `lw4dvar_fcn3.out`/`.err` ended up with
  all four retries (`0`, `10`, `2`, `1`) concatenated in one file --
  harmless here since each attempt re-logs its own full config header, but
  worth using distinct `--output`/`--error` names (as the smoke tests
  did) for a run meant to be analyzed cleanly afterward.
- **Per-epoch timing, confirmed with real numbers**: **FCN3 ~94.2s/epoch
  vs. AIFS ~18.25s/epoch -- a 5.2x ratio**, both very consistent
  epoch-to-epoch (measured directly from log timestamps, not estimated).
  This closes out a "worth a profiling pass" note that had been sitting in
  the fcstnetv3 repo's own CLAUDE.md with no real numbers behind it, and
  lines up almost exactly with the per-step memory/timing sweep already
  documented there (~4.5-4.8s per additional FCN3 step for a checkpointed
  rollout at `atmo_chunk_size=2`: 20 steps x ~4.7s = ~94s). This is a real,
  structural cost of FCN3's per-step DISCO spherical-convolution
  encode/decode burst, not something tunable away in this repo's driver
  code, and not an artifact of the merge.
- **Loss magnitude differs too, and not uniformly across the window**:
  epoch-1 `Jtot` was ~18% higher for FCN3 (452874 vs. 384417) on this one
  IC date, but the per-lead-time breakdown is not "FCN3 uniformly worse":
  - The **background term (t+0h)** is ~50% higher for FCN3 (8483 vs. 5673)
    with nearly identical obs-used counts (8903/9112 vs. 8929/9112 -- not a
    QC-rejection artifact), suggesting FCN3's 48h forecast genuinely
    disagrees with real ps observations (via the shared `logpinterp`
    forward operator) more than AIFS's does, right at the window start.
  - FCN3 has a sharp, backend-specific spike at t+36h (28391 vs. AIFS's
    12428).
  - By t+120h the two are comparable, FCN3 even slightly lower (24300 vs.
    25548).
  - **This is a single-case (one IC date) comparison, not a statistical
    claim about general relative skill** -- worth rechecking across more
    dates before concluding much, but real and reproducible enough (same
    obs, same forward operator, matched everything else) to be worth
    tracking, not dismissing.
- **`learn_rate: 0.0025` diverged for AIFS on this window** ("blew up",
  the user's own words) -- resubmitted at `2.0e-3`, which matches the low
  end of the AIFS repo's own previously-logged `learn_rate` history
  (`1.e-4 -> 1.e-3 -> 2.e-3`) for windows of comparable length. FCN3's
  `checkpoint_stride: 1` run at the same `0.0025` has not shown divergence
  through epoch 2 (`Jtot` 452874 -> 439241, decreasing) -- the two
  backends' stable learning-rate ranges are not assumed to match just
  because this comparison uses one shared value; re-tune independently if
  either looks unstable over more epochs.
- **Final result, both runs completed (100 epochs, 5-day window)**: AIFS
  (`learn_rate: 2.0e-3`) genuinely improved z500 -- GL/SH/TR all came out
  *below* the raw background after optimization (GL: bg 8.54 -> after
  8.30), the first clearly positive ps-obs-only 4D-Var result seen across
  either repo this session (NH was the one region that got slightly worse,
  3.71 -> 4.15). FCN3 (`learn_rate: 2.5e-3`, `checkpoint_stride: 1`) did
  not -- z500 got worse across every region after optimization (GL: bg
  13.88 -> after 17.36), the same "untuned learn_rate doesn't reliably
  improve z500 yet" pattern already documented at length in the fcstnetv3
  repo's own CLAUDE.md, not a new finding by itself.
- **FCN3's background (t=0, pre-optimization) z500 error is also
  substantially higher than AIFS's on this case** (GL RMS 10.37 vs. 6.70 m)
  -- raised as a concern ("something wrong with the way FCN3 is being
  initialized from ERA5?") and checked directly rather than assumed either
  way:
  - **Grid alignment confirmed exact**: AIFS's saved background's
    `latitude`/`longitude` match the ERA5 GRIB truth's own coordinates to
    `0.0` max difference (verified via `cfgrib`); FCN3's flat-to-(721,1440)
    reshape checked out the same way against its own ERA5 netCDF cache.
    Not a regridding/point-ordering bug on either side.
  - **Bias/scatter decomposition, global area-weighted, in meters of z500
    height error**: AIFS bias=1.32, std=6.57, RMS=6.70. FCN3 bias=0.42,
    std=10.36, RMS=10.37. **FCN3's bias is smaller than AIFS's**, not
    larger -- if there were a units mistake, a mis-indexed pressure level,
    a wrong ERA5 field/date, or a bad interpolation, a large systematic
    bias would be the expected signature, and neither model shows one.
    The entire RMS gap is in the *scatter* (~58% higher std for FCN3),
    which is the signature of a real forecast-dispersion difference at a
    48h lead time on this specific case, not an IC-construction defect.
  - **Leading hypothesis, not yet confirmed**: FCN3 is a stochastic/
    ensemble diffusion model (per its own architecture notes above) that
    this pipeline deliberately runs fully deterministically (one fixed
    noise realization, by design). NVIDIA presumably trained/evaluated it
    as part of an ensemble system; forcing a single fixed noise draw could
    plausibly cost real skill in a way AIFS (genuinely deterministic) has
    no analog for. Not proven from one case/one seed -- worth checking
    whether the scatter changes with a different fixed `noise_seed`, or
    longer-term whether averaging a few fixed-noise realizations closes
    the gap, before treating this as settled.

## Third backend: Microsoft Aurora (validated end-to-end, 2026-09-18)

Adding a third backend, `aurora` (`AuroraV1p5`, `microsoft/aurora`'s
`aurora-0.25-v1.5.ckpt`), at the user's request -- same phased approach as
the AIFS/FCN3 merge, and (so far) noticeably smoother than either: no CUDA
extension build was needed, no torch.no_grad() workaround, no stochastic
noise handling.

- **New conda env** `.../envs/aurora`: `torch==2.7.1+cu128`,
  `torchvision==0.22.1+cu128`, `triton==3.3.1`, `microsoft-aurora==2.0.1`.
  Hit the same unpinned-torch resolver trap already documented for
  `torch-harmonics`: `pip install microsoft-aurora` silently pulled
  `torch==2.14.0+cu13` via `timm`->`torchvision`; fixed by
  force-reinstalling the pinned torch/torchvision/triton with `--no-deps`.
- **Checkpoint NOT vendored as a git submodule**, unlike AIFS/FCN3:
  `microsoft/aurora` on HF is a shared monorepo of many unrelated
  checkpoint variants (0.1, several 0.25 variants, wave, air-pollution,
  ensemble...) -- a submodule would force pulling everything (tens of GB).
  `backends/aurora/aurora_prefetch_checkpoint.py` fetches just the ~4.9GB
  needed (`aurora-0.25-v1.5.ckpt` + `-static.pickle`) via
  `huggingface_hub.hf_hub_download` (the same mechanism the `aurora`
  package itself uses internally), revision-pinned, into a gitignored
  local cache -- must run from a login node (compute nodes have no
  internet, same constraint as ERA5 IC fetching).
- **Architecture, confirmed by reading the real source and testing
  directly, not assumed**: `AuroraV1p5.forward()` has the same
  encoder -> backbone -> decoder split as AIFS/FCN3, but is simpler than
  both -- no hardcoded `torch.no_grad()` (AIFS's `predict_step` has one)
  and no stochastic noise at all (FCN3's diffusion-noise re-priming has no
  analog here; noise/ensemble is a wholly separate class+checkpoint,
  `AuroraV1p5Ensemble`, not used). Control injection uses a
  `register_forward_pre_hook` on `model.decoder` rather than manually
  unrolling `forward()` -- sufficient specifically because neither of the
  other two backends' reasons for a manual unroll apply here.
  `latent_shape` (backbone output == decoder input), determined
  empirically (`probe_aurora_latent.py`): `(259200, 1024)`, a flat
  Swin-transformer token sequence, a third distinct convention (AIFS:
  `(n_hidden, channels)`; FCN3: `(channels, H, W)`).
- **Needs two lagged time levels like AIFS** (`max_history_size=2`), not
  self-starting like FCN3 -- so `aurora_model.py`'s packed state follows
  AIFS's `(1, 2, n_points, n_vars)` channels-last convention exactly,
  reusing `state_layout='channels_last'` and `_apply_control_mask`'s
  existing branch with zero new shared-driver code.
- **Grid: 720x1440, not 721x1440** -- a real, non-obvious finding, not an
  assumption. Aurora's own `Batch.crop(patch_size=4)` silently drops
  ERA5's 721st latitude row (721 % 4 == 1) before every real forward call,
  so the model's own output is always 720 rows. A first attempt carrying
  a 721-row state failed with a shape mismatch when packing the model's
  720-row prediction back in; fixed by making this backend's native grid
  720x1440 throughout, matching exactly what the checkpoint computes
  (dropping the South Pole row) rather than silently carrying a stale,
  never-updated row. `fcn3_grid.BilinearGridInterpolator` needs no changes
  for this -- it derives `nlat`/`nlon` from the data itself.
- **Real, non-obvious fp16-autocast pitfall found and root-caused (not
  just patched around)**: `AuroraV1p5` runs encoder/backbone/decoder under
  `torch.autocast(dtype=torch.float16)` by default. Three probe attempts
  (raw zeros, realistic-but-spatially-uniform fields, then
  spatially-varying fields) all produced a NaN `increment.grad` with a
  naive loss computed on Aurora's raw UNNORMALISED output (physical units,
  e.g. `msl` ~1e5 Pa) -- ruled out degenerate input as the cause (spatial
  variation didn't fix it) before finding the real cause: gradients of
  that scale overflow fp16's dynamic range during the autocast-region
  backward. Re-normalising the prediction before computing the loss gave
  a finite gradient. Real lesson for any future loss/diagnostic code
  touching Aurora's decoded output: stay scale-aware (the real ps-obs loss
  already is, via innovation/oberr normalization, so this isn't expected
  to bite the actual solver -- but a naive diagnostic could reintroduce
  it).
- **26 surface variables** (richer than FCN3's 7 or AIFS's set), including
  a native `sp` (surface pressure) field unlike FCN3 (so AIFS's
  `ps_operator: 'ps'` might work for Aurora too, though unwired/untested),
  7 of which are output-only (`i10fg, blh, uvb_1h, ssrd_1h, ttr_1h,
  scaled_tp_1h, scaled_sf_1h` -- predicted but never present in real ERA5
  input; Aurora's own `_pre_encoder_hook` unconditionally zero-pads these
  before every encoder call, so the roll-forward step needs no special
  casing for them). `insolation` similarly needs no special roll-forward
  handling -- Aurora's own `_post_unnorm_hook` overwrites the model's
  predicted value with the true recomputed one after every step.
- **Validated** (`smoke_test_aurora.py`, same bar as AIFSModel/FCN3Model's
  own smoke tests): `decode_state` round-trips correctly; a real
  differentiable `advance()`+`backward()` gives a finite, nonzero
  gradient; AdamW measurably reduces a real (scale-aware) loss over 15
  epochs, with the same cold-start overshoot-then-partial-recovery shape
  already documented for FCN3's own untuned single-step smoke test -- a
  reassuring consistency signal, not a fluke pass. All checks pass.

**`reset_skt_over_ocean` support added** (2026-09-18, at the user's
observation that Aurora has both `skt` and `lsm`): `lsm` is now ALSO given
its own constant-in-time packed-state column (beyond the 91 real
surf/atmos columns), mirroring exactly how AIFS's own `lsm` is a genuine
packed-state column, so `reset_skt_over_ocean` (AIFS-only until now) works
unmodified for Aurora too. One real bug found and fixed along the way:
`lsm` briefly appeared in both `batch.surf_vars` and `batch.static_vars`
simultaneously (Aurora's own `patchembed.py` asserts these variable-name
sets never collide) -- fixed by excluding the extra static-passthrough
column from the `surf_vars` dict `_unpack` builds.

**`aurora_ic.py` written and verified against a real ERA5 fetch**
(2015-01-01T00, not just assumed from param tables): fetches the 18
real-fetchable surface variables (of Aurora's 26 -- 7 are output-only/
never fetched, `insolation` is computed analytically) plus the 5
pressure-level families, following `fcn3_ic.py`'s CDS/netCDF approach.
Two real differences from FCN3: Aurora needs two lagged time levels (like
AIFS), and needs its own fresh ERA5 orography fetch for `get_verif`'s
QC (like FCN3, unlike AIFS) -- Aurora's checkpoint-bundled static `z` is
exposed via `decode_state` but is NOT verified to equal real ERA5
orography, so the QC-critical path uses a real fetch instead of that
assumption. One real bug found this way: `ci` (`siconc`) comes back NaN
over land (ERA5's own convention) -- `np.nan_to_num` fixes it, matching
what the official aurora repo's own example notebook does defensively.

**Wired into the shared driver dispatch** (`get_model`/
`get_grid_interpolator`/`get_input`/`get_verif`/`load_config`, all now
have an `aurora` branch) and `config.yml.template` documents every
Aurora-specific key. `get_grid_interpolator` reuses
`fcn3_grid.BilinearGridInterpolator` directly for Aurora (same 0.25deg
grid, just 720 rows instead of 721 -- the interpolator derives
`nlat`/`nlon` from the data itself) -- the one deliberate exception to
"each backend's imports stay in its own directory," since
`fcn3_grid.py`'s own dependencies carry none of the cross-backend risk
that isolation exists to avoid. Verified: `AuroraModel` +
`BilinearGridInterpolator` construct correctly through the real driver
dispatch (CPU, from the `aurora` env); confirmed no regression for
FCN3/AIFS (also re-verified through the same shared dispatch code after
the edits).

**Future enhancement, explicitly deferred, not yet scoped in**: Aurora
supports `variable_lead_time`/`fine_lead_times` (see
`aurora/rollout.py`), which can produce real model predictions at
sub-6h lead times instead of the linear-interpolation approximation
(`_interp_decoded`) the shared driver currently uses for `dt_obs <
dt_verif` -- a genuine accuracy win over the AIFS repo's own documented
S2/tidal representativeness error. Confirmed by reading `rollout.py`
directly: all sub-lead-time predictions within one 6h window are
computed independently from the *same* previous-step state (not chained
autoregressively), each requiring a full separate encoder+backbone+
decoder forward pass -- a real compute-cost multiplier proportional to
how many sub-6h observation slots exist in the window, not a free
capability. Also requires restructuring `latent_increment` injection so
the control variable's effect is applied consistently across every
sub-lead-time call within the first main step, not just a small tweak.
Deliberately not implemented yet -- revisit once the basic Aurora
backend has a real end-to-end validation run.

**Real end-to-end validation run completed** (`config_test_aurora.yml`:
`n_init: 1`, single 12h window, `n_verif: 2`/`dt_verif: 6`, `max_epoch:
5`, real ERA5 IC + real ps observations). Took three real bug fixes to get
a clean run, each found by the run itself, not by inspection:

- `AuroraModel` was missing `_prime_noise()` (a no-op -- Aurora has no
  stochastic state) and `wrap_state()`, both required by
  `compute_optimal`/`compute_loss_4dvar` but never exercised by
  `smoke_test_aurora.py` (which calls `advance()`/`decode_state()`
  directly, not through the full solver loop). `__abstractmethods__ ==
  frozenset()` being empty didn't mean the interface was complete -- these
  are concrete AIFSModel/FCN3Model conveniences the shared driver actually
  calls, not `forecast_model.LatentForecastModel` abstract requirements.
- `get_verif` hit a stale-cache `KeyError` on `geopotential_at_surface` --
  the `2015-01-01T00` sfc file cached during earlier `aurora_ic.py`
  verification predated the orography-fetch fix. Fixed by deleting and
  re-fetching just that one file.
- **Real memory bug**: the run got through a full epoch 1 with a
  physically plausible loss (`Jtot=17084` on real data) before OOMing at
  92.86 GiB (of 93 GiB) on epoch 2. Root cause: `AuroraModel.__init__`
  never called `self.wrapper.configure_activation_checkpointing()` --
  Aurora's own docstring says this "is required in order to compute
  gradients without running out of memory." It wraps every individual
  Swin3D transformer block/encoder/decoder layer in its own activation-
  checkpoint boundary, complementary to (not a replacement for) the outer
  per-6h-step `torch.utils.checkpoint` this wrapper already does around
  `_advance_one_step`. Fixed with one line; re-validated via
  `smoke_test_aurora.py` (unaffected) before retrying the real run.

After those three fixes, the run completed cleanly end to end: real
ps observations (8934-9756 obs used per slot, out of 9112-9969 available
-- realistic counts, matching the other two backends' own validation
runs), all 5 epochs, `Jtot` trending down overall (17084 -> 16966 ->
18680 [bump] -> 14601 -> 12682), all diagnostic files saved, and real
finite z500 numbers: `bg(t0)` GL 3.74, `before` 4.05, `after` 4.45 -- z500
got slightly worse post-optimization, the same "untuned learn_rate on a
toy window" story already documented at length for AIFS's and FCN3's own
early validation runs, not a new or Aurora-specific problem. This
demonstrates the PIPELINE works end-to-end for Aurora, not that its
current (default, untuned) `learn_rate`/`max_epoch`/window-length produce
a good analysis -- same honest framing the other two backends' own
first validation runs used.

Prefetched ERA5 dates now cached in `ic_cache/`: `2014-12-29T18`,
`2014-12-30T00`/`T18`, `2014-12-31T00`, `2015-01-01T00`/`T06` -- enough
for either a 24h or 48h back-forecast IC at `sdate: 2015-01-01T00`.

**Not yet done**: a longer/multi-epoch tuning run (matching the FCN3/AIFS
`learn_rate` tuning history), `restart`/multi-cycle runs, and the
`fine_lead_times` enhancement above.

## Aurora 16-step divergence: root cause and fix (2026-09-19)

Following up on the 5-day/100-epoch single-cycle test requested above: that
run (48h back-forecast, `n_verif: 20`, `max_epoch: 100`) diverged --
starting at epoch 2, every observation past `t=0` was rejected by QC, and
the loss froze at exactly the background (t=0) term for every subsequent
epoch. Root-caused via systematic bisection and a targeted forward-only
probe, not guessed at.

- **Bisection** (`n_verif` in {4, 8, 12, 16}, `learn_rate=1e-4`, all other
  settings fixed): 4/8/12-step windows optimize cleanly; 16-step windows
  collapse **immediately** at the very first corrected observation
  (oind1, t+6h), not gradually.
- **Two magnitude-based mitigations both failed identically**, ruling out
  "the injected correction is too large" as the cause: a `latent_scale:
  100.0` config knob (divides the injected perturbation by 100x before
  adding it to the model, independent of AdamW's own per-element step
  normalization) produced the exact same frozen-loss signature. Disabling
  the outer per-step `torch.utils.checkpoint` wrapping entirely
  (`checkpoint_stride: 0`, ruling out an interaction with Aurora's own
  internal `configure_activation_checkpointing()`) also made no
  difference. Restricting `control_variables` to `[u, v, z, t, q, sp]`
  (confining the increment's *direct* effect at the injection step to only
  the core prognostic fields, per `_apply_control_mask` -- see
  `long_window_4dvar_utils.py`'s docstring) also made no difference --
  ruling out "the increment is corrupting an unclipped output-only
  diagnostic channel" as the mechanism.
- **Decisive probe** (`probe_aurora_nan.py`, forward-only, no gradients):
  loaded the actual "best" increment saved by the diverged run and found
  it was **already NaN** (`torch.load` -> `norm=nan, max_abs=nan`) --
  the corruption happened during the optimizer's OWN update, not from
  injecting a small-but-nonzero perturbation into an already-fragile
  forward pass. Injecting that NaN increment into a otherwise-clean
  background rollout (step 0 itself has zero NaN/Inf in any decoded
  field, at any field, real physical values throughout) immediately
  produces 100%-NaN decoded fields at step 1 -- every element of every
  variable, both core prognostic and the unclipped output-only surface
  diagnostics (`i10fg`, `blh`, `uvb_1h`, `ssrd_1h`, `ttr_1h`,
  `scaled_tp_1h`, `scaled_sf_1h`) -- consistent with a NaN value anywhere
  in a transformer's latent tokens propagating globally through
  attention's all-to-all mixing.
- **Root cause, confirmed by reading both the optimizer loop and the
  `aurora` package source**:
  - `compute_optimal`'s AdamW loop (`long_window_4dvar_utils.py`) checks
    `torch.isfinite(loss)` before `.backward()`, but had no equivalent
    check on `increment.grad` afterward. `torch.nn.utils.clip_grad_norm_`
    does **not** sanitize a NaN gradient -- a NaN element makes the
    computed norm NaN, so the rescale factor (`max_norm / (total_norm +
    eps)`) is also NaN, and NaN survives the "clip" unchanged.
    `optimizer.step()` then applies the NaN gradient, corrupting
    `increment` itself to NaN.
  - Because `increment` starts at all-zeros, this NaN gradient occurs
    **deterministically from epoch 1** for a 16-step window -- the same
    computation, run from the same starting point, fails the same way
    every time. This is exactly why every increment-*magnitude* lever
    (`latent_scale`, `control_variables`) failed identically: none of them
    touch the actual cause, which lives entirely upstream, in
    backpropagating through the model itself.
  - The corrupted (NaN) increment then gets misjudged as "new best":
    `_compute_ps_observation_diagnostics_at_time`'s `interpolation_failed
    = ~torch.isfinite(...)` check (correctly) marks every downstream obs
    as unusable rather than producing a NaN loss -- so the *loss* itself
    stays finite and looks artificially LOWER (only the t=0 background
    term remains) than the real, honest epoch-1 loss, and gets saved as
    the run's best result.
  - **Trigger, found in the installed `aurora` package's own source**
    (`aurora/model/aurora.py`): `AuroraV1p5.__init__` defaults to
    `autocast=True, autocast_dtype=torch.float16`, applied to its
    encoder, backbone, AND decoder (more aggressive than the base
    `Aurora` class, whose own default is `autocast_dtype=torch.bfloat16`
    and only autocasts the backbone). fp16's narrow dynamic range
    (~+-65504) is fine for a pure forward pass (confirmed: the
    zero-increment background rollout has zero NaN/Inf at any step,
    any field, any window length tested) but overflows when
    backpropagating gradients through a long (16-step) chained rollout --
    a known fp16-vs-bf16 tradeoff (fp16 has more mantissa precision but
    only fp16 exponent range; bf16 trades precision for fp32's exponent
    range), and precisely the kind of instability mixed-precision
    *training* guards against with a `GradScaler` -- a concern that
    doesn't arise for Aurora's typical inference-only use case (no
    backward pass at all), which is presumably why `AuroraV1p5` felt safe
    defaulting more aggressively into fp16 than its own base class.
- **Fix, two parts**:
  1. `backends/aurora/aurora_model.py`: `AuroraModel.__init__` gained an
     `autocast_dtype` parameter, defaulting to `torch.bfloat16` (was
     implicitly `torch.float16` via `AuroraV1p5()`'s own default), passed
     through to `AuroraV1p5(autocast_dtype=autocast_dtype)`. bf16 has
     fp32's exponent range, eliminating the overflow, at the same
     memory/speed class as fp16 (unlike falling back to full fp32
     autocast, which would cost real memory headroom this checkpoint
     doesn't have to spare -- see `configure_activation_checkpointing()`
     already being load-bearing just to fit in memory at all).
  2. `long_window_4dvar_utils.py`'s `compute_optimal`: added a symmetric
     `torch.isfinite(increment.grad).all()` check right after
     `loss.backward()`, alongside the existing `torch.isfinite(loss)`
     check before it -- skips that epoch's `optimizer.step()` (via
     `continue`, after `optimizer.zero_grad()`) rather than silently
     applying/saving a corrupted update, as a defense-in-depth backstop
     independent of whichever backend or dtype choice is in play.
- **Verified, not just reasoned about**: re-ran the exact 16-step
  bisection config (`config_aurora_bisect_16.yml`, `learn_rate=1e-4`,
  `max_epoch=3`) with only the `bfloat16` default changed. Epoch 2 (the
  epoch that previously collapsed to 0 obs used at every oind past t=0)
  now shows real, smoothly-varying obs-used counts at every oind (e.g.
  oind10: 9659 -> 9673 obs used) and per-oind `J` trending down
  individually (e.g. oind10: 14975.9 -> 14728.3). `Jtot` decreases
  monotonically across all 3 epochs (212710.1 -> 209731.9 -> 205469.9) --
  a genuine, healthy optimization trajectory, not a collapse. The
  gradient-finiteness guard was never triggered during this run (the
  primary bf16 fix was sufficient on its own), consistent with it being a
  backstop rather than the actual fix.
- Diagnostic configs from this investigation
  (`config_aurora_bisect_{4,8,12,16}.yml`, `_16_scale.yml`, `_16_nocp.yml`,
  `_16_ctrlvars.yml`, `_16_bf16` reuses the plain `_16.yml`) and
  `probe_aurora_nan.py` are left uncommitted/untracked for now, pending a
  decision on whether to keep any as regression checks.

## First real production-scale Aurora single-cycle run (2026-09-19)

With the bf16 fix in place, re-ran the real 5-day/100-epoch single-cycle
test this whole investigation was blocking (`config_aurora_5day_100it.yml`
-- 48h back-forecast, `n_verif: 20`/`dt_verif: 6` = a 5-day/21-slot window,
`learn_rate: 2.5e-3`, `max_epoch: 100`, job 21716154). Completed cleanly,
`sacct` exit code `0:0`, no errors, no collapse at any point.

- **Optimization**: `Jtot` decreased smoothly and substantially across all
  100 epochs -- 303113 -> 143514 (epoch 10) -> 98332 (epoch 20) -> 64169
  (epoch 40) -> 54856 (epoch 60) -> 50000 (epoch 80) -> 48292 (epoch 100),
  an 84% reduction overall, with the normal AdamW cold-start-then-flatten
  shape (occasional small uphill wobbles near convergence, never a
  collapse). Real, varying, sensible obs-used counts at every one of the
  21 six-hourly slots throughout the entire run (e.g. final epoch: 8922 -
  10211 obs used per slot out of 9112-10428 available).
- **z500 diagnostic** (NH/tropics/SH/global RMS error, m^2/s^2 / GRAV):
  ```
  bg(t0)  2015-01-01T00:  7.89  3.42  8.11  6.78
  before  2015-01-01T06:  9.00  3.07  9.59  7.74
  after   2015-01-01T06:  9.30  5.39 10.24  8.53
  ```
  The analysis is slightly WORSE than the uncorrected background in every
  region (global 7.74 -> 8.53) after this real, fully-converged 100-epoch
  optimization -- not a pipeline bug (the ps-obs cost function itself
  converged correctly and substantially), but the same "ps-obs-only 4D-Var
  doesn't automatically improve z500, especially with an untuned
  learn_rate" finding already documented at length for both AIFS's and
  FCN3's own early tuning passes (see FCN3's `config_test.yml` run above
  and the AIFS repo's own CLAUDE.md `learn_rate` history). This is the
  first real (non-toy, fully-converged) result for Aurora specifically --
  worth reporting honestly, not the pipeline's fault, but a real open
  tuning question, same as for the other two backends.
- All diagnostic files saved without error in
  `output/test_aurora_5d_48h_100it/`: `*_latent_increment_120h_100it.pt`
  (1.06GB -- matches the `(259200, 1024)` latent shape's byte count
  exactly), `*_loss_latent_120h_100it.json`,
  `*_observation_diagnostics_120h_100it.nc`,
  `*_control_forecast_120h_100it.nc`/`*_optimal_forecast_120h_100it.nc`
  (~4.3-4.6GB each -- the full 21-step decoded trajectory at Aurora's
  native 720x1440 resolution), `*_control_inputs_120h_100it.nc`/`.pkl` and
  `*_optimal_inputs_120h_100it.nc`/`.pkl`, `*_loss_120h_100it.txt`.
- **Not yet done**: `learn_rate`/`max_epoch`/window-length tuning for
  Aurora specifically (this run used the same untuned values carried over
  from the earlier smoke test), `restart`/multi-cycle runs, and a
  profiling pass to look for speedups (Aurora's ~100s/epoch for this
  20-step window is noticeably slower than AIFS's own ~18-25s/epoch for a
  matched window -- see "Known gaps" below for the planned optimization
  experiments).

## Login-node ERA5 prefetch scripts, and a sys.path bug they share (2026-09-19)

`aifs_prefetch_ic.py`/`fcn3_prefetch_ic.py` were copied in from their
single-backend source repos into `backends/{aifs,fcn3}/`; added
`backends/aurora/aurora_prefetch_ic.py` as the analogous third script
(same structure, differing only in the initial-IC fetch:
`aurora_ic.build_input_state`, which fetches TWO lagged dates the same way
`aifs_ic.fetch_era5_grib(..., lagged=True)` does, vs. FCN3's single
time-level fetch). Verified end-to-end against `config_test_aurora.yml`
(temporarily symlinked to `config.yml`, its own driver-default lookup
convention) -- correctly fetched the lagged initial-condition pair, the
verification state, and the z500-diagnostic-truth date, all served from
cache with no network hit.

**Found a real bug in this process, not specific to Aurora**: all three
scripts do `import long_window_4dvar_utils as utils` at the top, but
`long_window_4dvar_utils.py` lives at the repo ROOT while these scripts now
live in `backends/<name>/`. Running a script as `python backends/fcn3/
fcn3_prefetch_ic.py` sets `sys.path[0]` to the script's OWN directory
(`backends/fcn3/`) -- Python's standard behavior, independent of the
caller's current working directory -- so `import long_window_4dvar_utils`
fails with `ModuleNotFoundError`, confirmed by direct test. This didn't
exist in the original single-backend repos, where these scripts lived at
the repo root alongside `long_window_4dvar_utils.py` itself. Fixed in
`aurora_prefetch_ic.py`, and then applied the identical one-line fix to
`aifs_prefetch_ic.py`/`fcn3_prefetch_ic.py` at the user's request:
`sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
'..', '..')))` before the `long_window_4dvar_utils`/backend-module imports,
in all three scripts. **All three verified end-to-end** (real runs, not
just import checks): FCN3 (`config_test_fcn3.yml`, non-restart branch) and
Aurora (`config_test_aurora.yml`, non-restart branch) both ran cleanly
against the shared `ic_cache/`; AIFS (`config_test_aifs.yml`, which
happened to have `restart: True`) also ran cleanly, exercising the
restart-branch date-shift logic as a bonus, against its own
`long-window-4dvar-aifsv2/ic_cache/` (that config's own `ic_cache` setting,
unrelated to the sys.path fix).

## `aifs1` environment: AIFS-single-1.1 (2026-09-20)

User downloaded `ecmwf/aifs-single-1.1`'s HF repo into
`backends/aifs/aifs-single-1.1/` (alongside the existing
`aifs-single-2.0/`) and asked for a conda environment to run it, expecting
it to differ from `aifs2` -- confirmed correct before building anything,
not assumed.

- **Confirmed the pin gap first**: `aifs-single-1.1`'s own repo has no
  `pyproject.toml`/`uv.lock` (unlike `aifs-single-2.0`, whose lock file is
  what `aifs2`'s own pins were read from -- see "Conda environments"
  above) -- its `run_AIFS_v1.1.ipynb` notebook's commented `!pip install`
  cell is the most authoritative pin source available for this
  checkpoint: `anemoi-inference[huggingface]==0.6.3`,
  `anemoi-models==0.5.0`, `torch-geometric==2.4.0`,
  `earthkit-regrid==0.4.0`, `ecmwf-opendata`, `flash_attn` (unpinned).
  Compared directly against `aifs2`'s installed versions
  (`anemoi-inference==0.8.3`, `anemoi-models==0.9.3`,
  `torch-geometric==2.6.1`) -- different enough (especially
  `anemoi-models` 0.5.0 vs 0.9.3) to justify a separate environment before
  writing a single line of setup, not just following the user's hunch
  blindly.
- **Checked PyPI metadata before installing anything**, to catch a
  conflict before it happened rather than after: `anemoi-models==0.5.0`
  requires only `torch>=2.2` and `torch-geometric<2.5,>=2.3` -- both
  compatible with the already-established `torch==2.7.1+cu128` pin used
  by every other environment in this repo, no resolver trap expected.
- **Reused `aifs2`'s exact flash-attn wheel choice** rather than
  re-deriving it: `flash-attn==2.8.3`, prebuilt for cu12/torch2.7/cp312,
  from `cathalobrien/get-flash-attn`'s GitHub releases (found the exact
  asset via the public GitHub API, since `gh` wasn't authenticated in this
  session -- `flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-
  linux_x86_64.whl`). Installed with `--no-deps` (matching the discipline
  used everywhere else in this repo for pinned wheels) so it can't drag in
  a different torch.
- **Build order**: `conda create` (Python 3.12, matching `aifs2`) ->
  `torch==2.7.1+cu128` from the cu128 index -> the flash-attn wheel ->
  the notebook's pinned anemoi/earthkit stack in one `pip install` call.
  `pip check` clean, no conflicts, no manual pin-downs needed this time
  (unlike `aifs2`'s own `mir-python`/`numpy<2` incident -- that conflict
  was specifically with `anemoi-graphs`, which this inference-only
  environment doesn't need at all).
- **`anemoi-inference validate` reported version mismatches -- read
  carefully rather than treated as pass/fail**: missing
  `anemoi.datasets==0.5.21`/`anemoi.graphs==0.5.0` (training-only
  provenance, not inference dependencies) and `anemoi.models`/
  `anemoi.utils` version mismatches against the checkpoint's OWN recorded
  training environment (`anemoi.models==0.4.2.post64` with an
  **uncommitted** local patch, per the checkpoint's own metadata -- even
  EXACTLY matching versions couldn't fully reproduce the training
  environment, since it contained an uncommitted change). This looked
  more alarming on paper than `aifs2`'s own "harmless patch-version
  differences" validate result, so it was verified functionally rather
  than trusted at face value.
- **Real functional verification, not just `validate`'s metadata
  check**: loaded the actual checkpoint via
  `anemoi.inference.runners.simple.SimpleRunner` on CPU first (fast
  iteration) -- confirmed the real attributes `aifs_model.py` actually
  needs (`checkpoint.timestep` = 6:00:00, matching AIFS-2.0's convention;
  `checkpoint.number_of_input_features` = 103;
  `checkpoint.variable_to_input_tensor_index` has 103 real variable
  names; `checkpoint.latitudes`/`longitudes` are real N320-mesh-sized
  (542080-point) arrays; the model itself instantiates with 253M
  parameters) -- confirming the "missing modules" `validate` flagged are
  genuinely just training-time provenance, not things the checkpoint
  needs to load.
- **GPU verification, not just CPU**: a separate real check
  (`test_scripts/run_verify_aifs1_gpu.sh`) confirmed flash-attn's actual
  CUDA kernel produces finite, correctly-shaped output (not just that it
  imports without an ABI error -- the FCN3/torch-harmonics incident
  elsewhere in this repo's history is exactly the class of "imports fine,
  silently wrong at runtime" failure this guards against) and that the
  checkpoint loads cleanly with `device='cuda'` too.
- `pip freeze` saved to
  `/scratch4/BMC/gsienkf/whitaker/conda/envs/aifs1-requirements.lock.txt`,
  matching `aifs2`'s own convention.
- **Not yet done**: no `aifs_config`/`config.yml` entry wiring
  `path_model`/`model_name` at this checkpoint specifically has been
  written or run through the actual 4D-Var driver yet -- this was
  environment setup and checkpoint-loading verification only, not an
  end-to-end pipeline validation the way FCN3/Aurora's first real runs
  were. `aifs_inference.yaml` (the shared AIFS anemoi-inference config)
  has not been checked for anything v1.1-specific that might differ from
  v2.0's expectations.

## AIFS-single-1.1 API compatibility fixes and end-to-end validation (2026-09-20/21)

Followed up on the `aifs1` environment build above by actually running
`long_window_4dvar.py` against the real `aifs-single-1.1` checkpoint through
this repo's shared (AIFS-generation-agnostic) `aifs_model.py`/`aifs_ic.py`.
Found and fixed 8 real API incompatibilities between `anemoi-models`/
`anemoi-inference` 0.5.0/0.6.3 (`aifs1`, this checkpoint's own pin) and
0.9.3/0.8.3 (`aifs2`, `aifs-single-2.0`'s pin, what these files were
originally written and validated against) -- **every fix verified by direct
comparison of both versions' real behavior/source, not guessed**, and all
kept as `try/except` compatibility shims so `aifs_model.py`/`aifs_ic.py`
remain the SAME shared files serving both checkpoint generations (the
existing `model_backend: aifs` dispatch already treats them as one backend;
`path_model`/`model_name`/`aifs_config` alone select the checkpoint).

- **First, definitively answered whether this whole exercise was even
  necessary**: could `aifs2`'s newer environment just load the v1.1
  checkpoint directly? **No** -- confirmed via direct attempt: `torch.load()`
  fails at the pickle-deserialization level with `ModuleNotFoundError: No
  module named 'torch_geometric.nn.conv.utils.inspector'`, a
  `torch_geometric` internal module-layout change between the pinned 2.4.0
  (aifs1) and 2.6.1 (aifs2) that breaks unpickling a v1.1-era saved graph
  object outright -- not an API-surface difference fixable with a shim, a
  hard checkpoint-format incompatibility. Settled the question before
  spending any further effort assuming the answer either way.
- **The 8 real incompatibilities found and fixed** (all in `aifs_model.py`
  unless noted):
  1. `anemoi.models.distributed.shapes.get_shard_shapes` doesn't exist in
     aifs1's pin -- renamed from `get_shape_shards` sometime after 0.5.0.
     Confirmed byte-identical implementation/signature under both names
     (a pure rename) -- `try: from ... import get_shard_shapes / except
     ImportError: from ... import get_shape_shards as get_shard_shapes`.
  2. `Checkpoint.select_variables_and_masks` doesn't exist on aifs1's older
     `Checkpoint` class. New `_select_variables_and_masks(checkpoint,
     include)` helper falls back to deriving the same result from
     `checkpoint.variable_categories()` +
     `checkpoint.variable_to_input_tensor_index` -- verified byte-identical
     output against the real method on `aifs-single-2.0`
     (`vars=['cos_latitude', 'cos_longitude', 'sin_latitude',
     'sin_longitude']`, `mask=[6, 8, 35, 37]`) before trusting it as a
     genuine fallback rather than a guess. Only supports the one
     `include=['computed+constant']` call this codebase actually uses --
     raises `NotImplementedError` for anything else rather than silently
     doing the wrong thing.
  3. `runner.device` is a plain `str` (`'cpu'`/`'cuda'`) on aifs1's older
     `anemoi-inference`, not already a `torch.device` like on aifs2 --
     broke `self.device.type` elsewhere in the file. Fixed with
     `self._device = torch.device(self.runner.device)` --
     `torch.device(...)` is idempotent on an existing `torch.device`, so
     this normalizes both versions without a branch.
  4. `AnemoiModelEncProcDec.forward()` doesn't accept a `grid_shard_shapes`
     kwarg on aifs1's older model class -- `_predict_step_with_grad` wraps
     the call in `try: ...forward(xin, model_comm_group=None,
     grid_shard_shapes=None) / except TypeError: ...forward(xin,
     model_comm_group=None)`.
  5. `_assemble_input` returns a 2-tuple `(x_data_latent, x_skip)` on aifs1
     vs. a 3-tuple `(x_data_latent, x_skip, shard_shapes_data)` on aifs2 --
     new `_assemble_input_compat(m, x, batch_size)` helper try/excepts the
     3-arg call, and on `TypeError` calls the 2-arg form and derives the
     third element itself via `get_shard_shapes(x_data_latent, 0, None)`.
  6. `_run_mapper` doesn't accept `x_src_is_sharded`/`x_dst_is_sharded`/
     `keep_x_dst_sharded` kwargs on aifs1's older signature -- new
     `_run_mapper_compat(...)` try/excepts the full-kwarg call down to the
     older, shorter one. Both of these compat helpers are used inside
     `_predict_step_with_grad_latent` (the manually-unrolled encoder ->
     `+ latent_increment` -> processor -> decoder sequence that's the core
     4D-Var control-injection point) in place of the direct calls -- this
     was fixed PROACTIVELY, before ever hitting a live traceback for it,
     by directly comparing both anemoi-models versions' real `forward()`
     source ahead of time, since the same class of signature drift found
     in `_assemble_input`/`_run_mapper` for the encoder/decoder was
     expected to (and did) apply here too.
  7. (`aifs_ic.py`) `DefaultRunner` has no `.variables` attribute on aifs1's
     older `anemoi-inference` -- new `_retrieved_prognostic_variables`/
     `_retrieved_constant_forcings_variables` helpers fall back to
     `runner.checkpoint.prognostic_variables` and a `variable_categories()`
     - derived equivalent (verified exact match,
     `['lsm', 'sdor', 'slor', 'wmb', 'z']`, against the real newer API on
     aifs2) respectively.
  8. (`aifs_ic.py`) `create_input(runner, config, variables=..., purpose=...)`
     and `GribFileInput(runner, path=..., variables=...)` don't accept a
     `variables` kwarg on aifs1's older `anemoi-inference` -- `_build_input`
     now returns `(input_object, variables)` and try/excepts down to the
     no-`variables` call; `build_input_state` similarly try/excepts down to
     one unfiltered `GribFileInput` read of the whole GRIB file (safe here
     specifically because `fetch_era5_grib` already writes prognostic +
     constant-forcing fields into the SAME file, so one unfiltered read
     already returns the fully-combined state on the older API -- no
     second read/`_combine_states` call is needed in that branch).
- **New minimal `aifs_inference_v1p1.yaml`** (v1.1-specific anemoi-inference
  config, alongside the existing shared `aifs_inference.yaml`): v1.1's
  checkpoint has zero wave variables and no `snowc`/`sd` at all (confirmed
  directly), so the existing config's v2.0-specific wave `typed_variables`/
  `pre_processors`/`post_processors` fail Pydantic validation
  (`Extra inputs are not permitted`, referencing `mwd`) when pointed at
  v1.1 -- needed its own stripped-down config rather than a conditional
  branch in one shared YAML.
- **Stale-cache gotcha, found and fixed**: `ic_cache_aifs/` (the existing
  cache dir used for `aifs-single-2.0` runs) is keyed only by date, not by
  checkpoint -- reusing it for a v1.1 fetch silently served a cached GRIB
  file that actually contained v2.0-specific wave variables (confirmed via
  direct eccodes inspection: 192 messages including `mwd`, `swh`, `wmb`,
  `cdww`, `mwp`). Fixed by using a separate `ic_cache_aifs1/` directory for
  all v1.1 fetches (`config_test_aifs1.yml`'s `ic_cache` key) and
  re-fetching fresh from CDS into it.
- **End-to-end validation, real GPU job 21837119**
  (`test_scripts/run_test_aifs1.sh`, `config_test_aifs1.yml`: `n_init: 1`,
  `restart: False`, single window, `n_verif: 2`/`dt_verif: 6` = a 12h
  window, `max_epoch: 5`, real `psobs` files, `ic_cache_aifs1/` pre-warmed
  from the login node so the compute-node run needed no network access at
  all). **Completed cleanly, exit code 0, ~40s of actual compute** (most of
  the ~14 minutes of wall clock the job took was SLURM queue wait, not
  runtime):
  - Checkpoint loaded, background built from a real ERA5 48h forecast,
    real ps observations read at all 3 slots (8932/9112, 9479-9485/9691,
    9559-9737/9969 -- large, realistic QC-passing counts).
  - 5-epoch AdamW optimization ran to completion: `Jtot` 23114 (epoch 1) ->
    19967 (epoch 2, new best) -> 20339 -> 28899 -> 32846 -- the epoch
    3-5 uphill trend is the same untuned-fixed-`lr`-overshoot-past-the-
    optimum shape this repo's CLAUDE.md already documents repeatedly for
    other backends' first low-epoch-count smoke tests, not a new failure
    mode.
  - All diagnostic files saved without error (`*_latent_increment_*.pt`,
    `*_observation_diagnostics_*.nc`, `*_control_forecast_*.nc`,
    `*_optimal_forecast_*.nc`).
  - z500 diagnostic printed real, finite numbers at every region: `bg(t0)`
    GL 9.68, `before` GL 10.59, `after` GL 10.26 -- a small improvement
    from the best-epoch (epoch 2) increment, unlike FCN3's own first smoke
    test (which got worse) -- a genuinely different, encouraging result,
    though still just a 3-obs-slot/5-epoch toy window, not yet a tuned
    real experiment.
  - This confirms every one of the 8 fixes above works correctly together
    on real GPU hardware with the real checkpoint, including the
    `flash_attn` CUDA kernel and the manually-unrolled latent-increment
    injection path in `_predict_step_with_grad_latent` -- the decisive
    validation this whole fix chain was working toward.
- **Also prefetched, not yet used**: `ic_cache_aifs1/` additionally holds
  2015-01-01T12, 2015-01-02T00, and 2015-01-02T12 (fetched in preparation
  for a future multi-cycle test) -- see "Known gaps" below, `n_init > 1`
  for aifs1 specifically is still unexercised.

## aifs1 non-finite-gradient divergence: root-caused as a pure learn_rate issue, not window-length or a bug (2026-09-21)

First real production-scale comparison run (`config_aifs1_12h_100it_6day.yml`
vs. `config_aifs2_12h_100it_6day.yml`, both 5-day/24-step windows, 2 cycles,
100 epochs, otherwise-identical configs, jobs 21875528/21875530) surfaced a
real, reproducible difference: aifs2 completed cycle 1 cleanly (Jtot 136567,
z500 improved 8.54->8.43) and reached epoch 28/100 of cycle 2 before hitting
the SLURM wall-time limit -- healthy throughout, zero non-finite-gradient
issues. aifs1 diverged partway through cycle 1 every time it was tried.

- **First made the existing skip-and-continue guard fatal.** `compute_optimal`'s
  `torch.isfinite(increment.grad).all()` check (added for the earlier Aurora
  divergence, see below) previously warned and skipped the optimizer step,
  then silently continued -- observed here burning through dozens of
  further epochs with zero progress before exiting 0 as if the run had
  succeeded, since every occurrence of this guard firing in this codebase's
  history turned out to be a persistent, deterministic condition rather
  than a one-off transient blip. Changed to `raise RuntimeError(...)`
  instead (`long_window_4dvar_utils.py`, commit 507f367) -- makes a stalled
  optimization visible (nonzero exit, full traceback) rather than masking
  it.
- **Reproduced at 3 different learn_rates, ruling out a specific bad lr
  value**: 2.5e-3 (original config, failed epoch 18), 1.5e-3 (failed epoch
  24 on a fresh-code run after an earlier stale-code run of the same lr
  masked the fatal guard by having started 50s before the fix was
  committed -- a real gotcha: a long-running job launched just before a
  code fix lands keeps running the OLD in-memory module, so always confirm
  a job actually started AFTER a relevant commit before trusting its
  behavior as validating that commit), and 1e-3 (failed epoch 32). All
  three failures were on the otherwise-identical 5-day/24-step/
  `checkpoint_stride: 0` window aifs2 handles cleanly.
- **Ruled out a precision/autocast mismatch** (the natural first hypothesis,
  given the near-identical shape to the Aurora 16-step divergence below):
  checked `checkpoint.precision` directly on both real checkpoints (CPU,
  `SimpleRunner`) -- both report `"16-mixed"`, resolving to identical
  `torch.float16` autocast. Not a precision difference between the two
  checkpoints at all.
- **Added an opt-in `autocast_dtype` override to `AIFSModel.__init__`**
  (`aifs_model.py`, commit 9d0b0ca; wired through `get_model()` as
  `exp['aifs_autocast_dtype']`, string or `torch.dtype`, defaults to the
  existing checkpoint-derived behavior so aifs2 is unaffected) to test
  forcing `bfloat16` -- the exact fix that resolved Aurora's own
  identically-shaped divergence. **Tested and it did NOT fix it**: still
  failed, at epoch 21 (delayed slightly from fp16's epoch 18 at the same
  `learn_rate: 2.5e-3`, but same failure mode -- a sudden loss jump at
  epoch 17, 340955->534954, worse than the epoch-1 starting loss, followed
  by 3 more elevated-but-finite epochs before finally going non-finite at
  21). This is a genuinely different mechanism from Aurora's fp16-overflow
  root cause, not a variant of the same bug. (Also fixed a stale, now-known-
  incorrect comment in `AIFSModel.__init__` claiming the rollout "already
  runs under a bf16 torch.autocast" -- it's fp16 by default, on both
  checkpoints.)
- **Ruled out a bug in the aifs1 compat shims**: compared
  `_assemble_input_compat`/`_run_mapper_compat` line-by-line against both
  anemoi-models versions' real `forward()` source (0.5.0 vs. 0.9.3,
  `encoder_processor_decoder.py`) -- the older-API fallback paths exactly
  replicate the real call sequence (2-tuple `_assemble_input`,
  separately-computed `get_shape_shards`, no sharding kwargs on
  `_run_mapper`). No discrepancy found. Also confirmed the two configs are
  otherwise identical (`diff` showed only checkpoint path/`ic_cache`/
  `learn_rate` differ).
- **Bisected window length** (`config_aifs1_bisect_{4,8,12,16}.yml`,
  `run_aifs1_bisect_{4,8,12,16}.sh` -- same methodology that found the
  real root cause of the Aurora 16-step divergence below: `n_init: 1`,
  `cycle: False`, `learn_rate: 1e-4` deliberately low to isolate window
  length as the variable, `max_epoch: 40`, default fp16 autocast): **all
  four lengths (4/8/12/16 steps) completed all 40 epochs cleanly, zero
  non-finite gradients.** This looked at first like a length-dependent
  threshold between 16 and 24 steps -- but critically, none of the 3
  learn_rates that failed at n_verif=24 (1e-3, 1.5e-3, 2.5e-3) had ever
  been tested at 1e-4, so a clean bisection result alone couldn't
  distinguish "shorter windows are stable" from "1e-4 is just low enough
  to be stable at any length."
- **Decisive confirmatory test** (`config_aifs1_bisect_24.yml`,
  `run_aifs1_bisect_24.sh`: n_verif=24, the ORIGINAL full window, at the
  SAME `learn_rate: 1e-4` as the successful bisection runs): **completed
  all 40 epochs cleanly**, sailing straight through the epoch 18-34 range
  that killed every higher-`learn_rate` attempt. **Conclusion: this is a
  pure `learn_rate`-tuning issue specific to aifs-single-1.1, not a
  window-length-dependent structural instability, not a compat-shim bug,
  and not a precision issue.** aifs-single-1.1 simply requires a
  substantially lower `learn_rate` than aifs-single-2.0 to stay
  numerically stable over this window/config -- somewhere between 1e-4
  (confirmed stable) and 1e-3 (fails ~epoch 32). Mechanism not further
  investigated (older/smaller checkpoint, presumably less well-conditioned
  gradients through a long chained rollout -- plausible but not verified
  beyond ruling out the alternatives above).
- **Caveat**: 1e-4 alone is not a great production setting as-is -- over
  40 epochs at n_verif=24 loss only moved 514103 -> 488774 (much smaller
  improvement than the higher-`learn_rate` runs achieved before
  diverging, e.g. 2.5e-3 had already reached ~300k by epoch 16) -- it
  would need either many more epochs or an intermediate `learn_rate` that
  is both stable and reasonably fast, not yet found. **Not yet resolved**:
  what the fastest stable `learn_rate` actually is (user is running
  further sweeps as of this writing) -- the 2-cycle/100-epoch aifs1 vs.
  aifs2 comparison run has not yet been resubmitted with a working
  `learn_rate`.
- Diagnostic configs/scripts from this investigation
  (`config_aifs1_bisect_{4,8,12,16,24}.yml`,
  `run_aifs1_bisect_{4,8,12,16,24}.sh`) are left uncommitted/untracked,
  matching this repo's existing convention for one-off diagnostic configs
  (see the Aurora bisection configs' own note above).

## Known gaps / next steps

- **`compile_wrapper: True` crashes on the second cycle of a multi-cycle
  (`n_init > 1`) run -- a real bug, not yet root-caused (2026-09-20).**
  First real multi-cycle Aurora experiment (`config_aurora_5day_50it.yml`,
  `n_init: 12`, `restart: False`, `compile_wrapper: True`, job 21781560)
  crashed during cycle 2's `.backward()` (cycle 1 completed all 50 epochs
  cleanly; cycle 2 got to epoch ~14-15) with:
  ```
  RuntimeError: CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error
  ```
  raised from inside a torch-inductor-generated kernel
  (`torch_inductor/.../c3rj5....py:20074, buf911.copy_(...)`), not from
  plain Python/PyTorch code. Every `compile_wrapper` validation so far
  (the 1.78x speedup confirmation, the full `checkpoint_stride` sweep) was
  SINGLE-cycle only -- `get_model()` (and hence the one-time
  `torch.compile(self.wrapper)` call) happens ONCE, outside
  `long_window_4dvar.py`'s `for k in range(n_init)` loop, so a multi-cycle
  run reuses the SAME compiled model instance across every cycle's
  independent `compute_optimal` call (fresh `increment` tensor, fresh
  optimizer, fresh backward graph each time) -- this specific reuse
  pattern was never exercised before this run.
  - **Points away from a simple node hardware fault**: ran on `u22g14`,
    the same node that had already completed several other
    `compile_wrapper` jobs successfully earlier the same day (the
    checkpoint_stride sweep, the compile-vs-uncompiled confirmation).
    Not conclusive on its own (hardware faults can be intermittent), but
    doesn't fit a simple "this node is broken" explanation either.
  - **Leading hypothesis, not yet verified**: something about
    torch.compile/inductor's cached kernels or captured buffer references
    from cycle 1 becomes invalid once cycle 2 allocates fresh tensors
    (`increment = torch.zeros(...)`, a new `AdamW` optimizer, etc.) --
    consistent with the crash occurring right as a SECOND independent
    optimization begins against the same compiled graph, not at any point
    within cycle 1's own 50 epochs.
  - **Workaround for now**: disable `compile_wrapper` for multi-cycle
    (`n_init > 1`) runs -- user resubmitted `config_aurora_5day_50it.yml`
    with `compile_wrapper: False` to unblock the actual science experiment
    while this is investigated separately. Uncompiled multi-cycle Aurora
    behavior is itself still unexercised before this (see the pre-existing
    "`restart: True` cycling and `n_init > 1` multi-cycle runs are
    unexercised" gap below), so this resubmission is also the first real
    test of that, independent of the compile question.
  - **Not yet done**: root-cause the actual mechanism (does disabling
    compile only for cycles after the first avoid it? does a fresh
    `torch.compile()` call per cycle avoid it at the cost of paying the
    compilation tax every cycle instead of once? is this a known
    upstream `torch`/`inductor` issue at this PyTorch version worth
    checking against public bug trackers?) -- deferred, to revisit.

- **Aurora per-epoch runtime optimization (in progress, 2026-09-19)**:
  Aurora's ~100s/epoch for the 20-step/5-day window above is ~4-5x AIFS's
  own ~18-25s/epoch for a matched window -- a similar ratio to FCN3's own
  already-investigated (and found largely structural) 5.2x gap. Status of
  each candidate:
  - **`use_fp16_safe_attention=False` -- TESTED, REJECTED (real crash, not
    just a non-improvement).** The hypothesis (disabling `AuroraV1p5`'s
    default fp16-overflow-safe manual attention fallback in favor of
    PyTorch's fast fused/flash `scaled_dot_product_attention`, now that
    `autocast_dtype=bfloat16` makes the fp16-only clamp a no-op) looked
    sound on paper but is **not safe on this stack**: setting it produces
    a deterministic `CUDA error: an illegal memory access was encountered`
    during `.backward()`, reproduced independently on two separate GPUs at
    `steps=4` (not a flaky node -- both `profile_aurora_rollout.py` and
    `probe_aurora_checkpoint.py` hit it identically). Reverting only this
    flag back to `True` (keeping `autocast_dtype=bfloat16` and
    `cudnn.benchmark=True`) fixed it immediately (`probe_aurora_checkpoint.py
    --steps 4`: `OK peak=39.05GiB time=53.73s`). Most likely cause: Swin3D's
    unusual windowed-attention tensor shapes hit an edge case in the
    installed PyTorch/CUDA build's fused-attention backward kernel,
    independent of dtype -- this is a real kernel-compatibility bug, not a
    numerical-safety tradeoff. **Left at the upstream default (`True`) in
    `aurora_model.py`.** Do not re-attempt without first confirming a
    working fused SDPA backward for this exact shape/PyTorch/CUDA
    combination.
  - **`torch.backends.cudnn.benchmark = True`** -- applied in
    `AuroraModel.__init__` (device=='cuda' branch, alongside the existing
    `allow_tf32` settings). Confirmed NOT implicated in the crash above
    (the fix-isolation test kept this `True` and succeeded), so it stays
    on. Not yet separately verified to give a measurable speedup on its
    own (Aurora's Swin3D architecture is attention/matmul-heavy, less
    conv-bound than FCN3's DISCO kernels, so the benefit may be small) --
    a real profiling run will show whether it matters.
  - **Profiling -- DONE, points at kernel-launch overhead, not raw
    compute.** (`torch.profiler` on a 4-step differentiable rollout via
    `profile_aurora_rollout.py`, same methodology as FCN3's
    `profile_fcn3_rollout.py`.) Self CUDA time total: 17.43s of GPU-busy
    time against ~50s of wall clock for the same 4 steps (see the
    checkpoint-redundancy table above) -- a large gap, and the profiler's
    own CPU-side numbers point at why: **"Command Buffer Full" (a
    GPU-command-queue-backpressure marker, not a real kernel) consumes
    31.6% of total CPU time (9.9s)**, meaning the CPU spends a third of
    its time simply blocked waiting for queue space to submit more work --
    the classic symptom of issuing too many small kernel launches rather
    than being compute-bound. Consistent with this: `aten::copy_` alone
    (30502 calls, just for 4 steps) is the single largest NAMED op by self
    CUDA time (27.4%), and `aten::roll` (Swin's window-shift for windowed
    attention, pure data movement, zero FLOPs) is another 7.8% at 4650
    calls -- copy/roll/reshape/clone/dtype-cast bookkeeping together are a
    large fraction of total GPU time, not the matmul/attention compute
    itself (`aten::addmm` -- real Linear-layer compute -- is "only" 17.0%,
    `aten::bmm` -- attention's core QK^T/AV matmuls -- is 9.6%).
    - **This session's own coarse auto-categorization undercounted
      attention specifically** -- worth noting as a methodology gap, not
      just a result: it bucketed by op-name substrings including
      `"attention"`/`"sdpa"`/`"flash"`, but Aurora runs with
      `use_fp16_safe_attention=True` (see above -- the fused/flash kernel
      crashes on this stack), so attention is computed via the manual
      fallback (`fp16_safe_scaled_dot_product_attention`), which
      decomposes into plain `bmm`/`mul`/`clamp`/elementwise ops with no
      "attention" in their names -- these landed in the auto-categorizer's
      57.6% "other" bucket along with backward-pass ops
      (`ClampBackward1`/`MulBackward0`) and the `full_rollout_fwd`
      record_function marker's own 21.2%-self-CUDA entry (itself likely a
      profiler accounting artifact of the region boundary, not a discrete
      kernel -- not meaningful as a standalone cost). Don't trust the
      auto-categorizer's percentages as attention-vs-everything-else; read
      the raw top-25-by-name table instead, as done above.
    - **Practical implication**: this reframes `torch.compile` from a
      speculative stretch goal to the best-targeted remaining lever --
      kernel fusion (fewer, larger kernel launches) directly attacks the
      "Command Buffer Full" stall and the copy/roll/reshape overhead in a
      way none of the flag-toggle fixes above can. Still genuinely risky
      given the confirmed fused-SDPA-kernel incompatibility on this exact
      PyTorch/CUDA stack (`use_fp16_safe_attention=False` section above) --
      a compiled graph recognizing the same attention pattern could
      plausibly try to lower to a similarly problematic kernel, though
      compiling the already-in-use manual fallback path (rather than the
      native SDPA call) is a meaningfully different and probably safer
      starting point.
    - **Attempted 2026-09-19, real and confirmed working: a genuine
      ~1.87x steady-state speedup, correctness-verified, at a real
      one-time compilation cost.** `torch.compile(model.wrapper)`
      (`probe_aurora_compile.py`, staged: forward-only -> +backward ->
      +outer checkpoint -> +4 steps, each independently try/excepted so
      one stage's failure wouldn't block learning from others) ran
      cleanly through every stage -- no crash, unlike the native-SDPA
      attempt, consistent with compiling the manual attention fallback
      path (already in use, since `use_fp16_safe_attention=True`) being a
      safer target than the native kernel would have been.
      - **Correctness**: `increment.grad.norm()` matched EXACTLY between
        compiled and uncompiled runs at the same setting (3.061e+07 both,
        steps=1) -- compilation changes speed, not the numerical result.
      - **Clean apples-to-apples confirmation**
        (`probe_aurora_compile_confirm.py`, same script/loaded-model
        instance for both arms, avoiding the pitfall below): at the
        production-matching setting (steps=4, outer checkpoint on),
        uncompiled steady-state (calls 2-3 of 3, after the cold-start
        effect) = **17.85s**, compiled steady-state = **9.55s** -- a real
        **1.87x** speedup. (An earlier same-day cross-script comparison
        against `probe_aurora_checkpoint.py`'s single-cold-call number,
        58.32s, suggested a much larger ~6x speedup -- that number is
        WRONG, an artifact of comparing a cold single call in one script
        against a warmed-up call in another; flagged as unconfirmed and
        deliberately not reported as fact before this same-script
        re-test caught it. Even the UNCOMPILED baseline shows a large
        cold-start effect on its own: 58.81s (call 1) -> 17.79s (call 2)
        -> 17.90s (call 3), matching that old 58.32s number almost
        exactly and confirming it was a cold-start artifact, not a real
        steady-state number, all along.)
      - **Compilation itself is expensive and its cost is NOT fixed** --
        136s (steps=1, no checkpoint) vs. 180s (steps=4, with checkpoint,
        different model instance/call history) vs. 272s (steps=4, with
        checkpoint, on a model instance that had already run 3 UNCOMPILED
        calls at that same setting first) for what should nominally be
        compiling the same single-6h-step forward graph each time
        (`self.wrapper.forward` is called once per step inside
        `advance()`'s loop, and `torch.compile(model.wrapper)` compiles
        that one call, not the whole multi-step loop) -- the exact cause
        of this scaling isn't nailed down (candidates: the dynamic
        `register_forward_pre_hook`/`remove()` pattern this wrapper uses
        for latent injection, done fresh on every call, interacting with
        dynamo's guards; and/or `torch.utils.checkpoint`'s backward-time
        recompute being traced/compiled somewhat independently per
        checkpoint region) -- but the practical number to plan around is
        "a few hundred seconds per distinct (steps, checkpoint-on/off,
        prior-call-history) combination encountered," not a fixed cost.
      - **A real experiment pays this cost more than once, not just at
        startup**: `long_window_4dvar_utils.py`/`long_window_4dvar.py`
        call `model.advance()` at several different `steps` values across
        one run (the main optimization loop's `n_steps`; `ref_state1`'s
        `steps=1`; `compute_ps_observation_hx`'s `steps_per_verif`;
        `make_forecasts`'s `steps_per_output`; the driver's own
        `bg_shifted`/restart-forecast advances) -- each distinct `steps`
        value (and use_checkpoint on/off) is a separate thing dynamo may
        need to compile, so a full experiment's real one-time compile tax
        is the sum across however many distinct combinations it actually
        exercises, not a single flat cost paid once.
      - **Net estimate for a real 100-epoch/20-step run** (extrapolating,
        not yet measured directly at that exact scale): a ~1.87x
        steady-state speedup on the dominant main-loop cost (currently
        ~100s/epoch) would bring it to ~54s/epoch; even after several
        hundred seconds of one-time compilation tax across the various
        call sites above, total wall-clock for a full 100-epoch run should
        drop meaningfully (rough math: ~10000s uncompiled -> ~5500-6500s
        compiled including compile overhead) -- a genuine, worthwhile
        reduction, though it does not on its own close the full ~4-5x gap
        to AIFS's ~18-25s/epoch (a compiled Aurora epoch would still be
        roughly ~2.5x AIFS's cost).
      - **Wired in and validated end-to-end at full production scale,
        2026-09-19 -- real, confirmed ~1.78x total wall-clock speedup.**
        `AuroraModel.__init__` gained the recommended opt-in
        `compile_wrapper: bool = False` constructor parameter (wraps
        `self.wrapper` in `torch.compile()` right after it's fully
        constructed/moved/frozen/checkpoint-configured); `get_model`
        reads it from `exp.get('compile_wrapper', False)`;
        `config.yml.template` documents it (Aurora-only, off by default).
        Re-ran the EXACT same 5-day/48h-back/100-epoch single-cycle
        config already validated uncompiled (`config_aurora_5day_100it.
        yml`, job 21716154, "First real production-scale Aurora
        single-cycle run" above), as a new config
        (`config_aurora_5day_100it_compile.yml`, only difference:
        `compile_wrapper: True`, separate output dir) rather than reusing
        or modifying the original -- job 21716154's underlying config had
        since been changed by the user for other work (`restart: True`,
        pointing at that run's own output), so a clean comparison needed
        its own fixed config matching the ORIGINAL validated settings
        exactly (recovered from that run's own saved
        `config.yml.20260919_011648` snapshot).
        - **Total wall-clock: 1:34:19 (job 21762536) vs. 2:47:47
          (uncompiled baseline, job 21716154) -- a real 1.78x end-to-end
          speedup**, including the one-time compilation cost -- close to,
          slightly under, the 1.87x steady-state-only figure above, as
          expected once compile overhead is folded in.
        - **Correctness confirmed at full scale, not just the earlier
          short probe**: final `Jtot` 49076.2 (compiled) vs. 48292.5
          (uncompiled) -- a ~1.6% difference, and the z500 before/after
          diagnostic tells the identical qualitative story at both
          (global RMS error worse after this untuned optimization: 8.78
          vs. 8.53). Both differences are consistent with the same class
          of GPU kernel-selection floating-point nondeterminism this
          repo's CLAUDE.md already documents extensively for bf16 ops
          elsewhere (kernel fusion changes floating-point operation
          order, shifting results at the last few bits, compounding over
          a 100-epoch/20-step differentiable rollout) -- not a real
          divergence or a correctness regression.
        - **The `torch._dynamo` `recompile_limit(8)` warning observed
          during this run is a real, partial coverage gap, not a
          failure**: several of Aurora's own internal per-variable-name
          normalization functions (`normalise_surf_var`,
          `normalise_atmos_var` in the installed `aurora` package) branch
          on the variable's name as a Python string (`name == '100u'`,
          `name == 'v'`, ...) -- with 70+ named fields, dynamo's default
          8-recompile budget per function is exceeded, so those specific
          functions fall back to eager execution rather than staying
          compiled (logged as a warning, not an error -- the run
          completes correctly regardless, per the correctness check
          above). This likely explains some of the compile-time
          variability observed in the earlier probes (different subsets
          of these per-name branches get hit depending on which physical
          variables/call order a given run's `steps`/prior-call-history
          happens to exercise first). Not fixed here (would mean patching
          the installed `aurora` package or raising
          `torch._dynamo.config.recompile_limit`, either a bigger change
          than this optimization pass warrants) -- flagged as a genuine,
          if modest, amount of speedup left on the table by this gap.
        - Config kept (`config_aurora_5day_100it_compile.yml`) as the
          reference for reproducing this comparison.
      - **Combining `compile_wrapper` with `checkpoint_stride: 0` (the
        outer-checkpoint-disabled setting below) OOMs at the real 20-step
        production window -- tested directly, not assumed.**
        `probe_aurora_checkpoint.py` gained `--compile`/`--n_calls` flags
        for exactly this combination; at `steps=20,
        outer_checkpoint=False, compile=True` the very first call OOM'd
        after 218.51s (`run_probe_checkpoint0_compile.sh`). Consistent
        with the uncompiled finding below (disabling the outer checkpoint
        alone already leaves only ~11 GiB of headroom at 20 steps,
        66->82 GiB) -- compilation's own memory overhead (kernel fusion
        can increase peak activation memory even as it reduces launch
        count) pushes this specific combination over the ~93 GiB H100
        ceiling. **Do not combine these two settings at the full
        production window length** -- `compile_wrapper: True` is safe and
        validated with the DEFAULT `checkpoint_stride: 1`; disabling the
        outer checkpoint is a separate, narrower opt-in (steps <=16, see
        below) that should not be combined with compilation at 20 steps
        without first re-checking memory at whatever shorter window it's
        used for.
      - **Full `checkpoint_stride` sweep (2026-09-19), and a sharper,
        more surprising version of the finding above: with `compile_
        wrapper: True`, ANY `checkpoint_stride` other than the default
        `1` OOMs at the real 20-step window -- not just the fully-disabled
        extreme.** `backends/aurora/probe_aurora_checkpoint_stride.py`
        (new -- replicates `compute_loss_4dvar`'s exact per-step
        `use_ckpt = (checkpoint_stride > 0) and (s % checkpoint_stride ==
        0)` logic, since `AuroraModel.advance()` itself only takes one
        `use_checkpoint` flag applied to every step of however many
        `steps` are requested in one call; the actual per-step
        alternation lives in the driver's own loop, not the model
        wrapper) swept `checkpoint_stride` in {1, 2, 4, 5, 10, 20,
        disabled} x {uncompiled, compiled} at `steps=20`:

        | stride | steps checkpointed | uncompiled peak/time | compiled |
        |---|---|---|---|
        | 1 (default) | 20/20 | 66.12 GiB / 135.5s | **works** (1.78x, validated above) |
        | 2 | 10/20 | 74.83 GiB / 127.1s | **OOM** (212.95s) |
        | 4 | 5/20 | 79.08 GiB / 122.7s | **OOM** (97.2s) |
        | 5 | 4/20 | 79.93 GiB / 121.7s | **OOM** (97.6s) |
        | 10 | 2/20 | 81.62 GiB / 123.6s | **OOM** (96.7s) |
        | 20 | 1/20 | 82.48 GiB / 119.0s | **OOM** (96.7s) |
        | disabled | 0/20 | 81.76 GiB / 117.2s | **OOM** (218.5s) |

        - **Uncompiled**: confirms the earlier stride=1/disabled
          extremes' linear-interpolation estimate was quantitatively
          accurate (stride=2 predicted ~74 GiB/~126s from the two
          endpoints; measured 74.83 GiB/127.1s) but the true shape is NOT
          linear -- it's front-loaded/diminishing-returns (most of the
          66->82 GiB memory increase and 135.5->117.2s speedup is already
          captured by stride=2, with stride=4 through disabled clustering
          close together near the disabled extreme). All fit comfortably
          under the ~93 GiB ceiling at every stride tested.
        - **Compiled**: the surprising part. Given stride=2 alone
          (uncompiled) leaves a comfortable ~18 GiB of margin (74.83 GiB
          of 93), the working hypothesis going in was that SOME
          intermediate stride might have enough headroom to survive
          compilation's added memory cost. It doesn't -- every stride
          tested above 1 OOMs, including stride=2. This means
          compilation's own memory overhead at this window length is
          large enough that it isn't a matter of finding the right
          partial-checkpointing tradeoff -- **`checkpoint_stride: 1`
          (full per-step checkpointing) is a hard requirement for using
          `compile_wrapper: True` at the 20-step production window**, not
          a spectrum to tune. (The two OOM-timing clusters -- stride=2 at
          212.95s vs. stride>=4/disabled all around ~97-99s -- suggest
          stride=2's extra checkpointed steps let it get further into the
          computation, likely into backward's recompute phase, before
          exhausting memory, while the others fail earlier; not
          independently confirmed, a plausible reading of the timing
          pattern rather than a traced root cause.)
        - **Practical takeaway**: don't try to combine `compile_wrapper`
          with a relaxed `checkpoint_stride` at the full production
          window length -- use the validated combination
          (`compile_wrapper: True` + default `checkpoint_stride: 1`,
          1.78x real speedup) or, if the extra ~13% speedup from relaxing
          `checkpoint_stride` matters more than compilation for some
          run, use it uncompiled instead. Untested: whether a SHORTER
          window (<=16 steps, where the uncompiled sweep already showed
          more memory margin at every stride) would let some intermediate
          stride survive compilation -- plausible given the mechanism,
          not measured.
  - **Outer per-step `torch.utils.checkpoint` redundancy -- TESTED, a real
    tradeoff, not a clear win.** (`probe_aurora_checkpoint.py`/
    `run_probe_aurora_checkpoint.sh`, sweeping `outer_checkpoint` on/off at
    steps in {4, 8, 16, 20}, one-shot differentiable advance+backward per
    trial, fresh process each.) Disabling the wrapper's outer per-step
    checkpoint (relying solely on Aurora's own internal per-Swin3D-block
    checkpointing) gives a consistent ~13-14% speedup at every window
    length tested, but at a real, growing memory cost:

    | steps | outer ON (peak/time) | outer OFF (peak/time) |
    |---|---|---|
    | 4  | 39.05 GiB / 58.3s  | 41.10 GiB / 50.1s  |
    | 8  | 45.81 GiB / 74.4s  | 51.27 GiB / 66.7s  |
    | 16 | 59.35 GiB / 115.5s | 71.60 GiB / 100.5s |
    | 20 | 66.12 GiB / 135.5s | 81.76 GiB / 117.2s |

    At the real 20-step production window, disabling it drops headroom
    before the H100's ~93 GiB ceiling from ~27 GiB to ~11 GiB -- a much
    thinner margin, and one that would erode further at longer windows
    (24-28 steps, as flagged for the other two backends' own "known gaps").
    **Recommendation: keep `checkpoint_stride: 1` (the current default) for
    production robustness**; `checkpoint_stride: 0` is a real, available
    opt-in for shorter windows (<=16 steps, where memory stays comfortably
    under ~72 GiB) where the ~13% speedup is worth the tighter margin --
    not flipped as the new default here, since the memory cost is real and
    the speedup alone doesn't come close to closing the ~4-5x gap to AIFS.
  Honest caveat, unchanged: Aurora is a full-resolution (721x1440, ~2x
  AIFS's mesh point count) transformer, while AIFS is a sparser graph-based
  encoder/processor/decoder -- some of the gap is likely structural rather
  than tuning, the same conclusion already reached for FCN3's own
  DISCO-conv cost. Treat "close most of the gap" as the realistic goal.
- No production-length (multi-day, many-epoch) run of AIFS or FCN3 through
  this repo yet -- only the two 12h/5-epoch smoke tests above (Aurora now
  has one, see "First real production-scale Aurora single-cycle run"
  above).
- `restart: True` cycling and `n_init > 1` are unexercised here (they were
  validated in each source repo separately, but the merge's dispatch logic
  around them has not been re-checked).
- The two source repos (`long-window-4dvar-aifsv2`,
  `long-window-4dvar-fcstnetv3`) are still where active tuning work is
  happening (as of 2026-09-17) and have not been archived/retired -- don't
  assume this repo is the only place experiments are running.
- **aifs1's fastest STABLE `learn_rate` is not yet known.** See "aifs1
  non-finite-gradient divergence" above: confirmed 1e-4 is stable but slow
  (loss 514103->488774 over 40 epochs at n_verif=24) and 1e-3/1.5e-3/2.5e-3
  all diverge (non-finite gradient, epoch 18-32) on the real 5-day/24-step
  production window -- the actual fastest-safe value between 1e-4 and 1e-3
  is unresolved, sweeps in progress. The 2-cycle/100-epoch aifs1 vs. aifs2
  comparison run has not been resubmitted with a working `learn_rate` yet.
  `restart: True` cycling and `n_init > 1` multi-cycle runs are also still
  unexercised for aifs1 specifically (`ic_cache_aifs1/` already has 3 extra
  prefetched dates ready for this).
- Whether the latent increment could ever be injected at `t=0` instead of
  `t+timestep` (a NeuralGCM-style design, raised because FCN3 is
  self-starting/single-time-level unlike AIFS) was investigated empirically
  in the fcstnetv3 repo and found not viable with the current FCN3
  checkpoint (`decode()` was never trained downstream of anything but the
  processor step; skipping it gives a badly damped/biased reconstruction,
  ~7-9x worse than a real 6h forecast step). See that repo's CLAUDE.md
  "Latent-increment injection timing" section for the full writeup and
  `probe_t0_correction.py` for the measurement -- both backends here still
  use the `t+timestep` injection point, unchanged.
