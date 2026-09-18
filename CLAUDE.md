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

Two environments, unchanged from the source repos:

```
aifs2:     /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2      (anemoi / torch-geometric / flash-attn)
fcstnet3:  /scratch4/BMC/gsienkf/whitaker/conda/envs/fcstnet3   (makani / physicsnemo / torch-harmonics)
```

Both pin `torch==2.7.1+cu128` (same CUDA/ABI target), but their dependency
trees have never been jointly resolved (`anemoi-graphs` wants `numpy<2`;
whatever `makani`/`physicsnemo` want has not been checked against that). A
real architecture decision was made here, not just inherited: **do not
attempt to merge these into one environment.** "Choose the backend at
runtime" means "choose it in `config.yml`, and run with the matching
env/launcher (`run_aifs.sh` vs `run_fcn3.sh`)" -- not "one process can load
either." Every place `long_window_4dvar_utils.py` needs a backend-specific
module (`aifs_model`/`fcn3_model`/`aifs_grid`/`fcn3_grid`/`aifs_ic`/`fcn3_ic`)
imports it *locally*, inside the function/branch that needs it, behind a
small `_ensure_backend_on_path(backend)` helper that adds
`backends/<backend>/` to `sys.path` on demand. This means a process running
under `aifs2` never needs FCN3's dependencies importable at all, and vice
versa -- confirmed directly (not assumed): from the `aifs2` env,
`import fcn3_model` fails with `No module named 'fcn3_model'` unless the
fcn3 branch has actually run first, and symmetrically for `fcstnet3`.

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
```

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

## Third backend: Microsoft Aurora (in progress, 2026-09-18)

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

**Not yet done**: `aurora_ic.py` (real ERA5 IC fetching -- 26 surface vars
including several accumulated/scaled fields, richer than either existing
backend's IC fetcher), wiring `model_backend: aurora` into
`long_window_4dvar_utils.py`'s dispatch (`get_model`/`get_grid_interpolator`/
`get_input`/`get_verif`), `config.yml.template` documentation, and any real
(non-synthetic-IC) end-to-end validation run.

## Known gaps / next steps

- No production-length (multi-day, many-epoch) run of either backend
  through this repo yet -- only the two 12h/5-epoch smoke tests above.
- `restart: True` cycling and `n_init > 1` are unexercised here (they were
  validated in each source repo separately, but the merge's dispatch logic
  around them has not been re-checked).
- The two source repos (`long-window-4dvar-aifsv2`,
  `long-window-4dvar-fcstnetv3`) are still where active tuning work is
  happening (as of 2026-09-17) and have not been archived/retired -- don't
  assume this repo is the only place experiments are running.
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
