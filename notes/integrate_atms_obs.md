# Assimilating ATMS radiances with the radianced two-head decoder

Read this when working on: ATMS observations in lw4dvar-torch, the ATMS
observation operator (`obs_atms_operator.py`) or its bundle, the
`exp.observations.atms` config block, or the emissivity control variable `z`.

Status: plan, 2026-10-16. Branch `feature/lwatms`, based on
`origin/add-raobs-parquet` (`26e332c`), which has the 1-degree O96 AIFS
checkpoint and the `exp.observations` config schema.

## Division of work

The science plan -- operator, cost function, channel sets, decoder work
(R1-R4), solver work (L1-L5) and experiments (E0-E5) -- is
`docs/plan_twohead_4dvar.md` in NOAA-PSL/radianced
(`/scratch4/BMC/gsienkf/Sergey.Frolov/atms`), with the evidence in
`docs/jacobian_notes.md` there. This note covers how that work enters this
repository.

| repo | owns |
|---|---|
| radianced | decoder training (R1), obs files (R3), offline checks on AIFS forecasts (R4), exporting the bundle (R2) |
| lw4dvar-torch | a copied, pure-torch operator file, the ATMS obs reader, the Jo/Jz terms, the `z` control, diagnostics |

Decisions (2026-10-16):

- Base branch: `add-raobs-parquet`, so ATMS is a third obs type in the same
  `exp.observations` schema as psobs and raobs.
- The operator is a **single file copied** from radianced
  (`atmsdec/operator.py` -> `obs_atms_operator.py` here), not an installed
  package. It must depend only on torch. The bundle `atms_twohead.pt` holds
  weights, normalization, input names in AIFS naming, emissivity PCs `U`,
  the `sigma_o` table, channel sets and a version tag; the file checks the
  tag on load. Re-copy the file whenever the bundle format changes.
- The obs files from R3 carry everything the operator needs that is not in
  the AIFS state (ERA5 emissivity precursors, scan geometry, surface class,
  y - BC, clw_obs), so runs never open the ERA5 zarrs.

## What this branch has and lacks

- 1-degree checkpoint: `config_test_aifs2-1.0deg-v1.0*.yml`, env `lwaifs2`,
  `run_aifs2-1.0deg-v1.0.sh`; details in
  `integrate_aifs2-1deg-v1.0_backend.md`. The checkpoint directory
  `backends/aifs/aifs2-1.0deg-v1.0/` is not tracked and does not exist in
  this clone yet; link it from Bo Huang's copy before running.
- Checkpoints (all under Bo Huang's `aifs-single-mse-2.0-1deg/outputs/
  pretraining/checkpoint/`, read from each checkpoint's metadata 2026-10-16):
  `1166d230.../inference-last.ckpt` (pretraining, 66 epochs; linked here as
  `aifs-single-mse-2.0-1.0deg-v1.0-pretraining.ckpt`, used by the ps+raobs
  configs), `cbc9c7b5.../inference-last.ckpt` (fine-tuning), and
  `a10c7d4f...-2epoches`. All three have the same 89 model inputs: t/u/v/z
  at 14 levels 10-1000 hPa; q at 12 levels 100-1000 hPa; sp, msl, skt, 2t,
  2d, 10u, 10v, tcw; static lsm, z, sdor, slor. Training data: ERA5
  `era5-o96-1979-2023-6h-v8.zarr` joined with
  `era5-o96-10hpa-1979-2023-6h-v1.zarr`, q_10 dropped -- the same zarrs
  radianced uses.
- **q levels differ from the radianced plan.** The data has q_50 but the
  model does not use it; the plan's first-test table assumes q at 50-1000
  hPa. R1 must train on exactly the 12 q levels 100-1000 hPa.
- psobs is mandatory: `observations.psobs.path` is required, `get_psobs`
  reads every slot from the PREPBUFR parquet archive (`psobs_parquet.py`),
  and `get_raobs` borrows psobs's slot grid (`step_lo`, `alpha`, `n_steps`).
  **Decision 2026-10-16: psobs stays always on** -- every ATMS experiment
  also assimilates ps, and ATMS borrows psobs's slot grid exactly as raobs
  does. So the ps obs archive must cover every ATMS test period (2022 for
  E1-E5): the rebuilt archive at
  `/scratch4/BMC/gsienkf/Sergey.Frolov/obs/prepbufr_to_parquet` holds only
  2015-01-01..11 so far.
- Raobs is the template to follow: `get_raobs` -> `raobs_traj`,
  `_compute_raobs_observation_diagnostics_at_time`, the `raobs_traj=`
  argument of `compute_loss_4dvar` / `compute_optimal`, its variables folded
  into `_resolve_loss_interp_specs`, its own trajectory in the diagnostics.

## Running on MI300X (2026-10-16)

flash-attn has no ROCm build in our envs, so `exp.aifs_attention` swaps the
processor's sliding-window attention (window 512, 16 heads, head_dim 64,
10944 nodes) for an exact equivalent, `backends/aifs/aifs_attention.py`.
Env: `/scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm` (torch
2.7.1+rocm6.3). u1-mi300x takes only QoS `gpuwf`.

One layer, forward + backward, fp16 (`bench_attention.py`, untracked):

| implementation | MI300X ms | H100 ms | GB |
|---|---|---|---|
| block, 5-D layout (first version) | 11.5 | 15.6 | 4.9 |
| block, blocks folded into batch (`block_sliding_window` now) | 3.4 | 4.3 | 0.43 |
| flex_attention (`flex_sliding_window`) | 2.0 | 1.35 | 0.18 |
| flash-attn | -- | 1.09 | 0.23 |

The 5-D layout left SDPA only its MATH kernel. All agree with full masked
fp32 attention to ~1e-3 in fp16 (same as flash-attn) and ~2e-6 in fp32.

Cycle 1 of the ps+raobs baseline (100 epochs; Jtot epoch 100; global z500
error after):

| run | attention | checkpoint_stride | min/cycle | Jtot | z500 after |
|---|---|---|---|---|---|
| H100 (Bo, two runs) | flash | 0 | 5.9 | 1.579-1.609M | 7.71-7.75 |
| H100 | flash | 0 | 5.9 | 1.584M | 7.67 |
| H100 | block 5-D | 1 | 19 | 1.578M | 7.80 |
| MI300X | block 5-D | 1 | ~20 | 1.551M | 7.78 |
| H100 | flex | 0 | 5.4 | 1.562M | 7.79 |
| MI300X | flex | 0 | 5.7 | 1.586M | 7.72 |

**All runs on this branch use `aifs_attention: 'flex_sliding_window'`
with `checkpoint_stride: 0`, on H100 and MI300X alike (decided
2026-10-16)**, so results don't depend on the GPU type. New configs must set
both. The first call compiles for ~10 s. Run-to-run scatter (fp16, no
randomness in the driver) is ~2-5% in Jtot, so these all agree.

## Decoder and 2015 check (2026-10-16)

radianced hand-off `docs/handoff_lw4dvar_twohead.md` (read it): the decoder is
`outputs/bo_twohead_s1` (accepted, no retraining), the test period is 2015
(S-NPP only; ps obs exist), E1 is 2015-01-01 to 01-06. Its 62 inputs include
q at 50 hPa, which AIFS predicts as a *diagnostic* output only: lw4dvar's
packed state holds the 89 inputs, so `q_50` is not available.

Operator bundle (R2, radianced `atmsdec/operator.py` +
`scripts/export_operator.py`):
`outputs/bo_twohead_s1/atms_twohead_bo_s1.pt`, equal to `load_run` + forward
to 0 K; gradients finite. Given q on 100-1000 hPa only, the operator fills
q_50 with its training mean.

Offline check on all Jan-Feb 2015 ERA5 pairs (radianced job 23880727,
`outputs/bo_twohead_s1/fit_w17_2015*.json`), clear-sky ocean, forward
(z = 0), RMSE in K:

| ch | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| sigma_o | 0.70 | 0.37 | 0.20 | 0.21 | 0.29 | 0.49 | 0.62 |
| Sep 2022 holdout | 0.33 | 0.20 | 0.18 | 0.21 | 0.31 | 0.54 | 0.59 |
| 2015, q_50 as in ERA5 | 0.36 | 0.20 | 0.20 | 0.35 | 0.63 | 1.20 | 1.29 |
| 2015, q_50 = mean, Obs_Time = 0 (as the 4D-Var feeds it) | 0.37 | 0.21 | 0.19 | 0.31 | 0.54 | 1.05 | 1.12 |

- Channels 5-7 transfer to 2015 unchanged. 8-11 degrade ~1.5-2x.
- Bias of O - H(ERA5), 2015 clear ocean (sample of 30 cycles), ch 8 / 9:
  -0.17 / -0.32 K globally, -0.26 / -0.51 K in 30S-30N, ~0 in the
  extratropics. A tropical lower-stratosphere year shift (QBO phase,
  stratospheric water vapour), not a constant offset.
- The decoder is sensitive to q at 50-150 hPa far beyond physics (one
  natural sd of q_100 moves ch 9 by 0.09 K; +1e-7 kg/kg of q_50 moves
  window channels ~0.7 K): it uses stratospheric q as a proxy. Fixing q_50
  at the 2022 mean *lowers* the 2015 error. Consequence for the 4D-Var: the
  Jacobian will put spurious increments into q at 100-200 hPa; consider
  detaching q above ~250 hPa in the ATMS operator inputs.

## Implementation and E0 (2026-10-16)

- Obs files (R3): `/scratch4/BMC/gsienkf/Sergey.Frolov/obs/atms_2015/<YYYYMMDDHH>.parquet`,
  236 cycles Jan-Feb 2015, 2.47M obs, 20% `withheld` (seed = cycle), from
  radianced `scripts/build_lw4dvar_obs.py`.
- lw4dvar: `obs_atms_operator.py` (copy of radianced `atmsdec/operator.py`);
  `exp.observations.atms` (documented in `config.yml.template`);
  `get_atmsobs` (psobs's slot grid, read-time clear-ocean screen),
  `_compute_atms_observation_diagnostics_at_time` (H(x), all-ocean
  neighbours, per-channel background check, q detached above 250 hPa),
  `_atms_term` in `compute_loss_4dvar`, `save_atms_diagnostics`
  (`<date>_atms_diagnostics_*.nc`, assimilated + monitored channels).
- Smoke test (job 23883475, ps + ATMS, 1 cycle, 3 epochs): runs end to end;
  ~2300-3200 screened obs per 6-h slot, ~80% used (the rest withheld);
  epoch cost 3.5 s vs 3.4 s for ps alone on MI300X.
- E0 (job 23883547, `e0_atms_check.py`, untracked): on the ERA5 state at
  2015-01-01T00, lw4dvar's H(x) vs the decoder run offline on radianced's
  pairs: mean |diff| <= 0.0005 K, max <= 0.007 K for channels 5-9 (up to
  0.18 K for the excluded 12-15) -- interpolation-code differences, not the
  1e-4 K of identical inputs; O-H RMSE identical to 1e-3 K. Jo_atms
  finite-difference vs autograd along a random t/q/skt/2t/tcw direction:
  relative difference 9e-4 (eps 1e-2), 1.3e-3 (eps 3e-2). Passed.
- E1 (2026-10-16): 10 cycles from 2015-01-01T00, flex attention, MI300X,
  configs `config_lwatms_e1_*.yml`, summary `e1_summary.py` (untracked).
  No ATMS-only arm: psobs is always on. Arms: ps only (23883589); ps + ATMS
  ch 5-7 assimilated, 8-11 monitored (23883590); ps assimilated, ATMS 5-11
  monitored only (23883598, same analysis as ps only, a second realization).
  All ran clean, ~6 min/cycle, no slowdown from ATMS.

  | cycle 10 (2015-01-05T12) | ps only | ps, ATMS monitored | ps + ATMS 5-7 |
  |---|---|---|---|
  | z500 error after (m) | 9.80 | 9.90 | 7.99 |
  | ps O-B (hPa, used) | 1.090 | 1.105 | 1.088 |
  | withheld O-B ch 5 / 6 / 7 (K) | -- | 0.55 / 0.32 / 0.24 | 0.40 / 0.22 / 0.15 |
  | withheld O-B ch 8 / 9 / 10 / 11 (K, never assimilated) | -- | 0.34 / 0.59 / 1.07 / 1.37 | 0.27 / 0.53 / 1.03 / 1.33 |

  - z500 error: the ATMS arm is below both ps arms from cycle 3 on, and the
    gap grows to ~1.9 m (cycle 10 after: 7.99 vs 9.80 / 9.90; cycle 7:
    8.25 vs 9.59 / 9.99). The two ps-only realizations differ by <= 0.4 m.
  - ATMS O-A < O-B for the assimilated channels every cycle (ch 5 cycle 1:
    0.59 -> 0.42 K). Withheld obs track used obs to ~0.02 K, so no
    overfitting.
  - Background fit to independent obs: withheld O-B for ch 5-7 drops ~30%
    and stays there (cycles 2-10); in the ps-only analysis it slowly grows
    (ch 5 0.50 -> 0.53). The monitored ch 8-11 also improve (ch 8 0.34 ->
    0.27), i.e. the analysis increment is consistent with the upper-air
    channels the decoder never fit against.
  - ps fit unchanged (O-B 1.09 hPa in all arms): ATMS doesn't fight psobs.
  - Caveats: one 5-day period, one realization per arm; the hardware
    spread of the S0b baseline (below) is ~1.6 m at cycle 6, larger than the
    ps-only pair here; z500 is verified against ERA5, which itself
    assimilated these ATMS obs and is the decoder's training state, so
    the z500 gain is not fully independent. The withheld-obs result is
    the cleaner evidence. `sigma_o` is the decoder's, not yet tuned.
- S0b baseline (23880473, ps + raobs, MI300X flex) vs Bo's H100 flash run
  (23860081): cycle 1 matches (z500 after 7.76 vs 7.67 m; the 1-cycle H100
  flex run gave 7.79), then the two trajectories drift apart (cycle 6:
  11.08 vs 9.37; cycle 10: 9.78 vs 8.85) with the same final Jtot (~1.08e6).
  Read as chaotic divergence from non-deterministic kernels, not a bug, but
  it sets the noise floor for single-realization z500 comparisons.
- Desroziers on E1 (ps + ATMS, cycles 2-10, used obs, slot 0 excluded: it
  has no analysis value, same as psobs): sqrt(<O-B * O-A>) = 0.37 / 0.21 /
  0.15 K for ch 5 / 6 / 7, against the bundle's sigma_o 0.70 / 0.37 / 0.20
  -- and rms(O-B) itself (0.39 / 0.22 / 0.16) is below the bundle's sigma_o,
  so the bundle values are too large and ATMS is under-weighted. With R too
  large O-A stays near O-B, so this first estimate is still biased high;
  iterate. Monitored channels: mean O-B -0.14 / -0.32 / -0.69 / -0.81 K for
  ch 8-11 (the 2015 tropical bias of the decoder check). New key
  `observations.atms.sigma_o: {channel: K}` overrides the bundle per channel.
- Follow-up runs (2026-10-16, all MI300X flex, 10 cycles):
  E1b `lwatms_e1b_ps_atms_dz` (23892243): Jan, ps + ATMS with the
  Desroziers sigma_o. Second period, 2015-02-01T00: `lwatms_e1feb_ps`
  (23893622), `lwatms_e1feb_ps_atms` (23893623, bundle sigma_o),
  `lwatms_e1feb_ps_atms_dz` (23893624). For Feb: PREPBUFR for 2015-02-01..11
  downloaded and converted into the parquet archive (conversion job
  23892768); the lagged IC `era5_20150130T00_lagged.grib` was built by
  concatenating Bo's cached single-date 2015-01-29T18 and 01-30T00 GRIBs
  (checked: identical fields; message order doesn't matter to the reader) in
  a separate cache, `/scratch4/BMC/gsienkf/Sergey.Frolov/ic_cache/ic_cache_aifs2-1.0deg-v1.0/`
  (symlinks to Bo's files plus the new file).
- Results (all ran clean, ~6 min/cycle). z500 error after analysis (m),
  mean over cycles 3-10 / cycle 10; ps O-B (hPa) and withheld O-B (K) at
  cycle 10:

  | arm | z500 3-10 | z500 c10 | ps O-B | withheld ch 5 / 6 / 7 | withheld ch 8 / 9 |
  |---|---|---|---|---|---|
  | Jan ps only | 9.40 | 9.80 | 1.090 | -- | -- |
  | Jan ps + ATMS, bundle sigma_o | 8.27 | 7.99 | 1.088 | 0.40 / 0.22 / 0.15 | 0.27 / 0.53 |
  | Jan ps + ATMS, Desroziers | 8.42 | 8.05 | 1.122 | 0.35 / 0.20 / 0.15 | 0.27 / 0.52 |
  | Feb ps only | 8.07 | 8.42 | 1.155 | -- | -- |
  | Feb ps + ATMS, bundle sigma_o | 7.37 | 7.47 | 1.155 | 0.38 / 0.21 / 0.15 | 0.26 / 0.52 |
  | Feb ps + ATMS, Desroziers | 7.57 | 7.70 | 1.178 | 0.34 / 0.19 / 0.15 | 0.25 / 0.51 |

  - The ATMS gain reproduces in February: z500 -0.7 m (mean over cycles
    3-10), -1.0 m at cycle 10; ATMS below ps-only from cycle 1 on.
  - Desroziers sigma_o (about half the bundle's for ch 5-6): fit to
    withheld ATMS improves a little (ch 5 -0.04 K, ch 6 -0.02 K, ch 8-9
    -0.01 K), but z500 is 0.15-0.2 m worse than with the bundle sigma_o in
    both months, and the ps O-B gets worse by 0.02-0.03 hPa in both months
    (the bundle arms leave ps unchanged). Read: the diagonal-R Desroziers
    estimate is too tight -- ATMS is not thinned, neighbouring FOVs and
    the decoder's own errors are correlated, so their effective weight
    must be lower than 1/sigma^2 per ob. Keep the bundle sigma_o (or
    something between) for now; don't iterate Desroziers further without
    thinning or an error-correlation inflation.

## Known mismatches to handle

- Spatial interpolation: the decoder was trained with k-NN onto O96
  (`atmsdec/era5.py` `GridLocator`); lw4dvar's AIFS default is `bilinear` on
  the reduced Gaussian grid. E0's offline-vs-solver match must use the same
  scheme on both sides. Decide whether R1 trains with bilinear instead.
- Time: ATMS obs are spread over +/-3 h of each cycle. With `dt_obs: 6` each
  ob is matched to the nearest 6-h step. Later, use `dt_obs` 1-2 h (supported,
  with time interpolation of the decoded fields).
- Training/use mismatch (ERA5 analyses vs AIFS forecasts): measured by R4.

## Steps

| step | what | done when |
|---|---|---|
| S0 | Branch on `add-raobs-parquet`; this note; link the 1-degree checkpoint; prefetch 2022 ICs and verification on a login node | checkpoint loads in `lwaifs2`; IC cache has the test window |
| S0b | Baseline before any ATMS change: 10 cycles of ps + raobs from 2015-01-01T00, pretraining checkpoint, flex attention, MI300X (`config_lwatms_ps_raobs_10cyc_amd.yml`, `run_lwatms_ps_raobs_10cyc_amd.sh`, untracked; job 23880473). Bo's identical 100-cycle run took ~6 min/cycle on one H100 | Done 2026-10-16: runs clean; cycle 1 matches Bo's H100 run, later cycles drift by up to 1.7 m z500 (see "Implementation and E0") |
| S1 | Dropped 2026-10-16: psobs stays mandatory, ATMS borrows its slot grid like raobs | -- |
| S2 | Done 2026-10-16 (see "Implementation and E0"). Plumbing without `z` (L1, L2, L4): `observations.atms` config keys, `get_atmsobs`, `_compute_atms_observation_at_time`, Jo_atms in the loss, operator fields added to the decoded set. Start with a stub bundle exported from radianced `outputs/twohead_F4_s1` plus an input-name adapter, so this does not wait on R1 | runs end to end on `--qos=debug` |
| S3 | Done 2026-10-16 with the bo_twohead_s1 bundle. E0 with the stub, then with the R1 bundle | H(x_b) matches offline <= 1e-4 K; finite-difference vs autograd gradients agree < 1e-3 |
| S4 | Done 2026-10-16 as E1 on 2015-01-01..06, S-NPP, ch 5-7 assimilated, 8-11 monitored (see "Implementation and E0"): withheld O-B ch 5-7 down ~30%, z500 error down ~1.9 m by cycle 10. Was: channels 5-9, NOAA-20, 2022-09-10 to 09-15 | pass criteria in the radianced plan |
| S5 | Emissivity control (L3): `z` as a second optimizer parameter group with its own learning rate, Jz, surface-class mask; `z` is window-local, not carried across cycles. Then E2-E4 | as in the radianced plan |
| S6 | Diagnostics (L5): O-B / O-A per channel and scene for assimilated, emissivity, monitor and withheld S-NPP obs; document the new keys in `config.yml.template` | |

R1 (retrain on the 12 q levels, test month held out) runs in radianced in
parallel with S1-S2. The two lines meet at S3.

## Open questions

- Spatial interpolation (leaning, 2026-10-16): the goal is not to blend land
  and sea at coasts. Plain k-NN (`GridLocator`, k=4 inverse distance) does
  not achieve that -- it ignores lsm, like bilinear. Plan: (a) R3 obs files
  store each ob's neighbour indices and weights, computed by the same
  radianced code as the training pairs; lw4dvar only gathers with them, so
  training and solver match exactly and `grid_interp` does not apply to
  ATMS; (b) E1 screens to all-ocean neighbourhoods (every neighbour
  lsm < 0.01); (c) before E2, R1 switches surface fields (skt, 2t, 2d,
  10u, 10v) to surface-class-matched neighbours, falling back to all
  neighbours when none match; profiles stay plain k-NN.
- Per-slot obs cap (~20k) and `obs_density_box_deg`: shared with psobs, or a
  separate setting under `observations.atms`?
- When `add-raobs-parquet` is merged into `main` or changes, rebase
  `feature/lwatms` onto it.

## Optimizer checkpoints and the window-length test (2026-10-16)

- `exp.opt_checkpoint_every: N` saves `<date>_optstate_<window>h<suffix>_e<epoch>.pt`
  every N epochs and at the end: current and best increment, AdamW state,
  scheduler, history, and the background state. `exp.opt_resume: <file>`
  continues it up to this run's `max_epoch` (same max_epoch: exact
  schedule; larger: constant lr = the saved run's last lr, or
  `opt_resume_learn_rate`). Documented in `config.yml.template`.
- The background must come from the file: the 48-h spin-up from ERA5
  (`nhrs_back`) is not bit-reproducible on GPU -- Jo of the uncorrected
  background ranges 6462-6507 over seven processes started from the same
  ERA5 state, and two uninterrupted 6-epoch runs end 2.4% apart. Resuming on
  a rebuilt background gave a 2% jump at the first resumed epoch; with the
  saved one (`load_optstate_background`, called from the driver) the resumed
  losses match the uninterrupted run to 0.2% (jobs 23899539, 23899765,
  23900053; configs `config_lwatms_resumetest_*.yml`, untracked).
- Window-length test: first cycle 2015-01-01T00 only, 300 epochs, 4/5/7/10-day
  windows, ps only vs ps + ATMS (`config_lwatms_win<d>d_<arm>.yml`, jobs
  23900209-23900216), checkpoints every 100 epochs; scored for z500, q850 and
  250 hPa wind speed by `window_figures.py` (untracked).
- Paper-style figures: `atms_figures.py`, `atms_figures_vars.py` (untracked)
  -> `figures/atms_e1/`; deck "ATMS in long-window 4D-Var" (claude.ai artifact).
- Window-length results (all 8 ran clean; 15 / 18 / 25 / 36 min for 4 / 5 / 7 / 10 d).
  Figures `figures/window_length/winfig1_<region>.png`, `winfig2_loss.png`,
  numbers `window_scores.csv`. Analysis error 24 h into the window, global,
  z500 (m) ps -> ps + ATMS: 4 d 7.14 -> 7.21, 5 d 7.74 -> 7.67, 7 d 10.03 ->
  9.09, 10 d 10.64 -> 10.47; window-mean z500 8.30 -> 7.66, 8.98 -> 8.52,
  12.98 -> 11.69, 21.66 -> 18.07. q850 and 250 hPa wind lower with ATMS at
  every window. Single cycle from a cold start (48-h AIFS spin-up from ERA5,
  background z500 19-44 m), so this is about the fit, not the cycled
  analysis. Loss: 4-5 d nearly flat by 300 epochs (Jtot -1 to -2% over the
  last 50); 7 d and 10 d still falling (ATMS Jo visibly), with the cosine
  schedule already down to lr 2e-4 -- candidates for `opt_resume` from the
  `_e300` checkpoints.
- Resumed to 600 epochs at lr 1e-3 (7 d and 10 d, jobs 23914064-67; figures
  `figures/window_length_600/`): window-mean z500 -1.2 to -1.7 m, day-1 z500
  -0.3 to -0.8 m; ATMS keeps its lead; 7 d ps + ATMS converged, 10 d still
  falling. Still worse at day 1 than 4-5 d windows. ps O-A by window day:
  the 10 d analysis barely fits day 0 (1.25-1.29 hPa vs background 1.3; 4 d:
  0.97) -- the strong-constraint limit over 10 days.
- Parked (2026-10-17, separate thread from ATMS/raob q): weak-constraint
  test -- extra latent model-error increments every 48 h with their own
  penalty; 10 d, first cycle, ps and ps + ATMS; success = day-0 fit back to
  ~1.0 hPa without losing the late-window fit.

### Raob q error model (2026-10-17)

10 cycles from 2015-01-01, our runs on MI300X with flex attention, scored with `compare_obs_10cyc.py` (output in `figures/raob_qerr/`). Global analysis RMSE vs ERA5, mean over the window:

| run | z500 (m) | q850 (g/kg) | ws250 (m/s) |
|---|---|---|---|
| ps only (Bo) | 10.05 | 1.377 | 4.36 |
| ps + raobs (Bo) | 9.99 | 1.455 | 3.87 |
| ps + raobs, our replica of Bo's config | 10.82 | 1.445 | 3.95 |
| ps + raobs t/u/v only, no q | 7.92 | 1.336 | 3.21 |
| ps + raobs, per-level q error table + 4σ bg check | 7.93 | 1.318 | 3.21 |
| ps + ATMS ch 5-7 | 8.98 | 1.207 | 3.72 |

- The 20%-of-|obs| q error degraded every variable, not just q: replacing it lowers z500 RMSE by 27% and ws250 RMSE by 19%.
- The q table removes the dry bias. B-ERA5 at 850 hPa goes from -0.17 to +0.02 g/kg (`raob_q_bias.py`, cycles 2-10), and the O-B mean is flat across lead days.
- The q table is only marginally better than dropping q entirely.

### New ATMS runs, first 10 cycles (2026-10-17)

Global analysis RMSE vs ERA5, mean over the window. From `compare_obs_10cyc.py figures/atms_new_10cyc ...`.

| run | z500 (m) | q850 (g/kg) | ws250 (m/s) |
|---|---|---|---|
| ps only (Bo) | 10.05 | 1.377 | 4.36 |
| ps + ATMS 5-7 (E1) | 8.98 | 1.207 | 3.72 |
| ps + ATMS 5-7 (3-month run, E1 config) | 9.36 | 1.202 | 3.70 |
| + humidity channels 18-22 | 8.85 | 1.210 | 3.38 |
| all scenes, no z | 8.34 | 1.241 | 3.62 |
| all scenes + emissivity z | 7.62 | 1.228 | 3.55 |

- Run-to-run noise is about 0.4 m on z500: the 3-month run and E1 share a config.
- Humidity channels: ws250 improves by about 9% everywhere, and z500 improves in the tropics. q850 is unchanged.
- All scenes: SH z500 drops from 11.1 to 8.7 m (no z) and to 8.1 m (with z). z also helps NH z500 (10.7 to 9.6 m). q850 is slightly worse than with ocean only.

### ATMS runs, first 60 cycles (2015-01-01 .. 01-30T12), 2026-10-17

Global analysis RMSE vs ERA5, mean over the window. From `NCYC=60 compare_obs_10cyc.py figures/atms_1mo ...`.

| run | z500 (m) | q850 (g/kg) | ws250 (m/s) |
|---|---|---|---|
| ps only (Bo) | 12.15 | 1.550 | 4.90 |
| ps + raobs (Bo, old q error) | 9.92 | 1.544 | 3.74 |
| ps + ATMS 5-7 (3-month run) | 10.14 | 1.231 | 3.77 |
| + humidity ch 18-22 | 9.81 | 1.225 | 3.40 |
| all scenes, no z | 9.14 | 1.280 | 3.71 |
| all scenes + emissivity z | 8.68 | 1.265 | 3.65 |

- The 10-cycle ranking holds over the month.
- Humidity channels: ws250 improves by 10%, tropical z500 drops from 4.6 to 3.7 m, and NH q850 drops from 0.786 to 0.757.
- All scenes: SH z500 drops from 13.6 to 10.4 m (with z). Tropical z500 and q850 get worse.
- ps-only error grows over the month (z500 10.0 to 12.2 m vs the 10-cycle mean). ATMS holds q850 at about 1.23.

### ATMS ob-error model: scene and symmetric cloud (2026-10-17)

The fit script is `atms_cloud_error.py`: 60 cycles of the Jan-Apr 2015 ERA5 pairs, decoder run as in the 4D-Var. Figure and table are in `figures/atms_oberr/`.

- Symmetric cloud C = (clw(y_o) + clw(H(x)))/2, GSI ch 1-2 retrieval. It matches GSI clw_obs (corr 1.000 on raw obs).
- O - H(ERA5) spread vs C (Geer & Bauer ramp, ratio to clear sky): ch5 6.3, ch6 8.0, ch7 3.0, ch18 14.9, ch19 10.1, ch20 6.5, ch21 3.3, ch22 2.2.
- Scene ratios (land/snow/seaice): ch5 2.6/4.4/2.1, ch18 2.2/3.3/1.6.
- Cloudy ch18-20 have a cold bias: -2 to -5 K at C 0.3-0.7. The decoder has no cloud input.
- The bundle sigma_o is about 2x the clear-ocean spread (ch5 0.70 vs 0.36 K, ch18 3.08 vs 1.39 K). The base sigma is left unchanged.

Implemented as `observations.atms.oberr_model: 'scene_cloud'`, with `oberr_table` and `cloud_max` (default 0.5 kg/m2), plus a new scene `ocean_cloudy`:
- sigma = sigma_o * r, with r >= 1.
- r is set from x_b at a slot's first evaluation, then held fixed.
- The default 'none' leaves the code's behaviour unchanged.

Smoke test (1 cycle, all scenes, ch 5-7 + 18-22, z on):
- std(omb/sigma) is 0.5-0.8 for ch5 and ch18 in every scene and cloud bin.
- 1.4% of ocean obs exceed cloud_max.

Tests, 60 cycles each (2026-10-17):
- `1mo_oberr_a`: 3mo_ps_atms + ocean_cloudy + scene_cloud.
- `1mo_oberr_b`: 1mo_e2_z + the same.
- `1mo_oberr_c`: 1mo_ps_atms_h2o + the same.

### Forecast test protocol (2026-10-17)

`forecast_skill.py CONFIG LABEL era5|RUN_DIR START END` runs daily 00Z 5-day forecasts:
- From a run: the analysis of the window ending at T, i.e. `<T-120h>_optimal_inputs` (valid at window start + 6 h) advanced 114 h, then ocean skt reset from ERA5.
- From ERA5: ERA5 at T, lagged pair, no spin-up. Lagged GRIBs for 01-06..06-30 were built in my ic_cache by concatenating the cached single-date files.

Scoring every 6 h:
- vs ERA5: area-weighted RMSE and bias.
- vs obs: PREPBUFR ps and raob t/u/v/q through the solver's operators. Read-time and orography screen only, no background check; the same obs for every source.
- Lead 0 of a run's forecast is in-sample.

Output is `output/fcst_skill/<label>/<T>.nc`. `fcst_skill_plot.py OUT LABEL=legend ...` averages over the common dates. Config: `config_fcst_skill.yml`; launcher: `run_fcst_skill.sh`.

Check: lead-0 z500 RMSE 14.03 m vs 13.94 m for the 4D-Var analysis trajectory at the window end (h2o run, cycle 1).

Scope:
- ERA5: 2015-01-06..06-30.
- Runs: ps (Bo), ps + raobs with fixed q (6-month run), ps + ATMS 5-7 + 18-22 (h2o), first month only (01-06..02-04, 30 forecasts).

## Window channels on the state over clear ocean (2026-10-20)

- **New key** `observations.atms.window_state_scenes`, e.g. `['ocean_clear']`.
  - Window-channel obs (1-4, 16, 17) in these scenes see x and z both attached. Elsewhere the window
    channels still move z only.
  - Needs `emis_control: 'per_obs'`. Ocean obs also need symmetric clw <= `cloud_max`.
  - Over open ocean z = 0 and skt is reset from ERA5, so the channels act on q, tcw and surface wind.
- **Runs:** 10 cycles from 2015-01-01T00 on the arm A config, scored against A's first 10 cycles.
  - `config_lwatms_winoc.yml`, job 24130645.
  - `config_lwatms_winoc_s2.yml`, job 24130646, with window-channel sigma_o doubled.

  | analysis RMSE vs ERA5, window mean | A | winoc | winoc_s2 |
  |---|---|---|---|
  | q850 Global (g/kg) | 1.171 | **1.032** | 1.074 |
  | q850 Tropics | 1.492 | **1.299** | 1.364 |
  | q850 SH | 1.119 | **0.981** | 1.012 |
  | q850 NH | 0.786 | **0.729** | 0.752 |
  | z500 Global (m) | 5.97 | 6.07 | 6.02 |
  | ws250 Global (m/s) | 2.80 | 2.82 | 2.78 |

  | withheld-obs O-B (K), cycles 2-10, leads > 0 | A | winoc | winoc_s2 |
  |---|---|---|---|
  | ch 1-4, 16, 17, clear ocean | 3.04 | **2.45** | 2.69 |
  | ch 5-7, clear ocean | 0.306 | **0.275** | 0.285 |
  | ch 18-22, clear ocean | 1.885 | 1.848 | 1.844 |
  | land / snow / ice, any channel group | | within 0.03 of A | |

  - q850 improves by 7-13% in every region. The window-end gain is the same size.
  - z500 and ws250 are unchanged within the 0.4 m noise. SH z500 is +0.3 m.
  - The independent obs agree: withheld window channels improve 20%, and ch 5-7 over clear
    ocean 10%.
  - Doubling sigma_o keeps about half the gain, so the decoder's sigma_o is about right.
  - Next:
    - a month-long run of A + winoc (candidate new golden standard);
    - cloudy ocean, after extending `oberr_table` to the window channels;
    - land, after the window-constant gridded z (plan E4).

## Arm A vs the UFS replay's GSI, same obs (2026-10-20)

- **Inputs:**
  - `atms_vs_replay_oma.py` (untracked), job 24130458;
  - arm A cycles 2-60 matched to the replay's S-NPP `diag_atms_npp_{ges,anl}` on (lat, lon, y ch5);
  - the anl files are in `/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/raw_anl/`;
  - output `figures/atms_vs_replay/accept_a_c2_60.csv`, plus a per-ob parquet of 3.6 GB.
- **Method:** RMS over obs used by both, all leads > 0.
- **Comparability caveat:** GSI's O-A is CRTM on the replay analysis (C384, 127 levels), with a bias
  correction fitted to its own CRTM. Ours is the decoder on a 13-level 1° AIFS analysis.

  | clear ocean (K) | GSI O-F (6 h) | GSI O-A | our O-B | our O-A |
  |---|---|---|---|---|
  | ch 1 / 3 / 16 (window) | 2.28 / 1.83 / 3.11 | 1.68 / 1.58 / 2.50 | 3.65 / 2.07 / 4.19 | 3.55 / 2.02 / 4.11 |
  | ch 5 / 6 / 7 | 0.35 / 0.14 / 0.11 | 0.30 / 0.13 / 0.10 | 0.42 / 0.22 / 0.17 | 0.41 / 0.21 / 0.16 |
  | ch 8 / 9 / 10 / 11 (monitored) | 0.11 / 0.12 / 0.16 / 0.19 | 0.10 / 0.11 / 0.15 / 0.18 | 0.29 / 0.54 / 1.02 / 1.21 | same |
  | ch 18 / 20 / 22 | 1.03 / 1.08 / 1.24 | 0.86 / 0.81 / 0.95 | 1.57 / 1.66 / 1.84 | 1.45 / 1.50 / 1.69 |

  - Our O-A is 1.5-2x GSI's in the temperature channels and 1.7-1.8x in the humidity channels.
    The monitored ch 9-11 are 5-7x worse.
  - The humidity channels are at the decoder's own error floor (ch 18-22 test RMSE about 1.8 K on
    clear ocean, `docs/jacobian_notes.md` in radianced). Most of that gap is forward-operator
    error, not analysis error. Ch 9-11 are the 13-level model and its 10 hPa-class top.
  - Our O-A degrades through the window, e.g. ch 20 clear ocean 1.20 K at 0-24 h vs 1.78 K at
    96-120 h. Weak constraint targets this.
  - Over land, snow and ice, our window-channel O-A (1.0-1.7 K for ch 1-4) beats GSI's
    (1.0-1.9 K), because z is fitted to those same obs. GSI uses only about 350k of them, against
    our 2.5M over clear ocean.

### All-sky ocean test: `winoc_w` and `winoc_cld` (2026-10-20)

10 cycles from 2015-01-01T00, scored like `winoc` (`figures/allsky_10cyc/`). `winoc_w` = clear
ocean on the state with the new window-channel errors (`oberr_table_w.json`); `winoc_cld` = clear
and cloudy ocean (`window_state_scenes: [ocean_clear, ocean_cloudy]`), same table.

| q850 RMSE vs ERA5 (g/kg), window mean | Global | NH | Tropics | SH |
|---|---|---|---|---|
| A | 1.171 | 0.786 | 1.492 | 1.119 |
| winoc (bundle sigma_o) | 1.032 | 0.729 | 1.299 | 0.981 |
| winoc_w (new table, clear) | 1.042 | 0.731 | 1.315 | 0.990 |
| winoc_cld (new table, clear + cloudy) | **1.012** | **0.718** | **1.281** | **0.951** |

z500 and ws250 are neutral (all within 0.1 m and 0.03 m/s of each other globally).

Withheld obs, ocean, leads > 0, rms (mean) in K:

| run | window O-B clear | window O-B cloudy | ch 5-7 O-B cloudy | ch 18-22 O-B cloudy |
|---|---|---|---|---|
| A | 3.46 (-0.80) | 5.94 (+0.99) | 0.558 | 3.35 |
| winoc | 2.85 (-0.51) | 5.83 (+1.15) | 0.556 | 3.34 |
| winoc_w | 2.97 (-0.59) | 5.83 (+1.06) | 0.554 | 3.32 |
| winoc_cld | 2.95 (-0.64) | **5.67 (+0.95)** | **0.546** | **3.31** |

- Adding cloudy ocean helps everywhere it is measured and shows no moist bias: cloudy-ocean O-B
  mean stays positive (obs warmer than model, i.e. missing cloud emission) and slightly smaller.
- The new error table alone (`winoc_w` vs `winoc`) is slightly worse on clear ocean than the
  bundle sigma_o; its clear-sky floor is tighter. `winoc_cld` gets the best of both.
- Cloudy-ocean O-A barely improves on O-B (5.67 -> 5.31): the state can't represent the cloud,
  which is the case for a per-ob cloud control or condensate in the model
  (`eagle/long-window/HANDOFF.md`).

### Month runs: forecast skill of the window-channel arms (2026-10-21)

30 daily 5-day forecasts, 2015-01-06 .. 02-04 (`figures/fcst_skill_allsky/`), arm A and WC24
q075 each as clear-ocean ATMS (the earlier runs), + window channels on the state over clear
ocean (winoc), + clear and cloudy ocean (wincld).

| | A | A winoc | A wincld | WC | WC winoc | WC wincld |
|---|---|---|---|---|---|---|
| q850 vs ERA5, 0 / 24 h | 1.24 / 1.35 | 1.10 / 1.26 | **1.09 / 1.25** | 1.13 / 1.24 | 0.98 / 1.13 | **0.96 / 1.11** |
| z500 vs ERA5, 24 / 120 h | 14.0 / 54.3 | 14.1 / 52.8 | **13.8 / 52.9** | 11.5 / 46.3 | 11.5 / **45.4** | **11.4** / 46.6 |
| raob q850, 24 h | 1.11 | 1.10 | **1.09** | 1.05 | 1.03 | **1.03** |
| raob uv250 speed, 120 h | 10.24 | 10.01 | **9.94** | 9.20 | 8.95 | **8.94** |

- The window channels give a 12-15% q850 analysis gain that carries to about 24 h and is gone by 120 h.
  All-sky (wincld) is slightly better than winoc for moisture.
- Mass and wind are neutral to slightly better; WC stays ahead of A by much more than any
  ATMS variant moves either.

_Run outputs of the superseded experiments (3mo_ps_atms, early 1mo ATMS variants, wcsweep/wc*_a, wc24_q025, win4d-10d, prior/cprior, early 10-cycle raob tests, aborted 6mo non-tsens runs) were deleted on 2026-10-21; their scores remain in this note, figures/ and output/forecast_skill/._
