# The 37-level AIFS models (A1, A2) in lw4dvar: plan and status

Source: `/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/long-window/HANDOFF_A1_A2.md` (2026-10-22).
A1 = Bo's AIFS 1 deg setup retrained on all 37 ERA5 levels, model top 1 hPa; A2 = 32 levels, top
10 hPa. Same O96 grid, o48 x 1024 hidden mesh, 6 h step, two input times, so the latent control
and the solver carry over unchanged. Baseline to beat: `lwatms_1mo_wc24_q075_corrhum` (Bo's
checkpoint, 14 levels, top 10 hPa; `notes/obs_error_tuning.md`, "Best experiment").

## What A1 is (probed from `inference-anemoi-by_epoch-epoch_020-step_082446.ckpt`, 2026-10-22)

- 188 prognostic: t, u, v, z on 37 levels (1-1000 hPa), q on 70-1000 hPa (28 levels), 10u, 10v,
  2d, 2t, msl, skt, sp, tcw, and new **sd, siconc, swvl1, stl1**.
- 48 diagnostic: w on all 37 levels, q on 1-50 hPa (9), cp, tp.
- 19 forcings, new: cvh, cvl, lai_hv, lai_lv, slt, fsr.
- Input times [0, 1] -> output 2 (multi_step 2, as Bo's).
- `GlobalDryMassFixer` is the last output layer (sp shifted so global dry mass is conserved).
- Metadata layout is the new multi-dataset one (`metadata_inference.data.*`, `dataset.data.*`).
- Needs the `anemoi-dev` env (anemoi-models 0.19 fork, anemoi-inference 0.12.1.dev, torch 2.10,
  flash_attn 2.8.3); `lwaifs2` can't unpickle it. Don't install into it (shared with training).
- Training still running (epoch ~20 of 66, done ~2026-10-23 10:00); use epoch 20 for wiring,
  `epoch_065-step_260000` for every scored experiment.

The ATMS decoder trained on A1's levels: `atms/outputs/aifs37_a1_twohead_s{1,2}/`, 142 inputs
(t and q on all 37 levels, `above-k` masks, Bo's 12 surface/static fields + the 10 new ones, the
obs geometry). Offline sigma_o (2022 holdout), K:

| ch | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 18 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 (top 1 hPa) | 0.64 | 0.34 | 0.18 | 0.16 | 0.21 | 0.29 | 0.30 | 0.34 | 0.42 | 0.52 | 0.84 | 2.96 | 1.71 |
| A2 (top 10 hPa) | 0.67 | 0.35 | 0.18 | 0.17 | 0.21 | 0.29 | 0.31 | 0.41 | 0.80 | 1.42 | 1.92 | 2.98 | 1.72 |
| A0 (14 levels) | 0.68 | 0.36 | 0.21 | 0.22 | 0.30 | 0.47 | 0.58 | 0.72 | 0.91 | 1.57 | 2.12 | 3.04 | 1.82 |

So A1 makes the stratospheric channels usable: 12-14 at 0.34-0.52 K vs 0.7-1.6 K before, and
8-11 30-50% tighter than with 14 levels. Ch 15 (peak ~2 hPa, weighting function above the top)
stays a monitor channel. Caveat from Bo's decoder: ch 8-9 were 1.5-2x worse on 2015 than on the
training-year holdout, so the 2015 error must be measured before any of 8-14 is assimilated.

## Phase 0: wiring (no science; 1-cycle smokes only)

0a. **Backend.** `run_aifs37.sh` (H100, anemoi-dev, flash attention) and
    `backends/aifs/aifs_inference_aifs37-a1.yaml`. Port `aifs_model.py`'s compat shims
    (`_assemble_input`, `_run_mapper`, `forward`, `RunConfiguration`, the new metadata layout) to
    anemoi-models 0.19 without breaking the pinned envs. Record the probed identity in
    `integrate_multiple_backends.md`.
0b. **ICs and truth from the training zarr, not CDS.** 2015 is inside 1979-2023, so read ICs and
    ERA5 verification from `era5_o96_37lev/.../era5-o96-37lev-...-s4.zarr` + the stl1 zarr
    (anemoi-inference `dataset` input or a small reader). Same grid and normalization the model
    was trained on; no GRIB regridding, no login-node fetch. Own cache: `ic_cache_aifs37/`.
0c. **State plumbing.** Levels are already read from the model (`pressure_levels`); check
    everything else that names fields: `control_variables` (w and upper q are diagnostic; leave
    them out), `save_levs` (add 1, 10, 50 hPa), `reset_skt_ocean`, `detach_q_above_hpa`, the
    raob operator over 37 levels, and the ATMS field map (new prognostic sd, siconc, swvl1, stl1;
    the 6 new forcings come from the input state, not `decode_state`).
0d. **ATMS bundle.** Generalize `export_mw_bundle.py` (levels and surface names from the run's
    meta/config instead of the 14-level constants), extend `obs_atms_operator._MODEL_SURFACE`,
    export `aifs37_a1_twohead_s1` and check it against radianced to <= 1e-3 K.
0e. **Acceptance.**
    (1) 5-day forecasts from ERA5, 30 daily, Jan 2015, scored with `forecast_skill.py`: A1 day-5 z500
        within 10% of Bo's 27.4 m, and t50 / z10 sensible.
    (2) A 1-cycle ps-only DA smoke: finite gradients, Jo falls, run time and memory on one H100
        (37 levels roughly triples the state; `checkpoint_stride` if 80 GB is short).
0f. **2015 decoder error for A1.** Build 2015 37-level ERA5-ATMS pairs (radianced; `era5_pre37`
    has only 2022). Then `atms_decoder_error.py` with the A1 decoder gives the 2015 sigma and the
    inter-channel correlation for ch 1-15 and 18-22, i.e. the A1 versions of `decoder_era5.json`
    and `oberr_table_w.json`.

## Phase 1: the model alone (same observations as the baseline)

Every arm below is run with the final A1 checkpoint and scored against its Bo-checkpoint twin on
window RMSE vs ERA5 (`compare_obs_10cyc.py`), fresh O-B, and, for month runs, 30 daily forecasts.

| id | arm | length | question | prediction |
|---|---|---|---|---|
| M1 | ps only, A1 vs Bo's ps-only | 10 cycles | does the mass fixer stop sp drift (lw4dvar-torch#12)? | global-mean sp bias drift over the window at least halved; z500 neutral |
| M3 | best config ported (corrhum: WC24 q075, ATMS arm A + window ch over ocean, decoder sigma ch 18-22, corr R) with the A1 decoder; ATMS sigma and corr from 0f | 10 cycles, then 1 month | is A1 + its decoder at least as good as Bo's best? | analysis neutral to better in the troposphere, clearly better above 100 hPa; forecasts within +-2% at 500 hPa |

M3's month run is the new baseline for Phase 2. If M3 is worse than Bo's best in the troposphere,
stop and diagnose (model skill from 0e, decoder transfer from 0f) before adding channels.

## Phase 2: stratospheric channels (A1 only)

ATMS ch 8-14 peak at about 250 down to 5 hPa; with a 1 hPa top A1 now holds their whole
weighting functions (ch 15 excepted).

| id | arm (on top of M3) | length | prediction |
|---|---|---|---|
| S0 | ch 8-15 monitored only (already the default for 8-11; add 12-15) | from M3's month run | O-B mean and std per channel and scene; the tropical bias seen with Bo's decoder (up to -0.8 K in ch 8-11) checked with A1's |
| S1 | + assimilate ch 8-11, sigma from 0f, corr R block with ch 5-7 | 10 cycles | t/z/wind 250-50 hPa vs ERA5 and raobs clearly better (5-15%); 500 hPa neutral |
| S2 | + ch 12-14 | 10 cycles | 30-5 hPa t and z much better; raob 50-10 hPa t better; troposphere neutral |
| S3 | S1/S2 winner, with a static per-channel bias from S0 (only if mean O-B > 0.3 sigma) | 10 cycles | removes any degradation S1/S2 show in the tropics |
| S4 | winner | 1 month + 30 forecasts; 3 months if it wins | stratospheric forecast skill better at all leads; tropospheric day-5 neutral to better |

Channel rules carried over from Bo's: only obs above the surface-sensitive scenes' cloud limit,
the same scene/cloud sigma ramp (refit for 8-14 from 0f), Huber c 1.5. The upper channels don't see
the surface, so they need neither the emissivity control nor the scene ratios; they apply in every
scene.

Extra verification needed for Phase 2: t, z and wind vs ERA5 at 50, 10 and 1 hPa in
`compare_obs_10cyc.py` and `forecast_skill.py`, and raob t/wind at 100-10 hPa.

## Optional controls

- **A2** (top 10 hPa) on M3 and S1: separates "more levels" from "higher top". S2 should help A1
  much more than A2. Approved by the user (2026-10-22); A2's final checkpoint (epoch 65) is
  already done, and its decoder is `aifs37_a2_twohead_s{1,2}`.
- **A0r** (14 levels, new recipe, retraining) on M3: separates the recipe from the levels.

## Cost

10-cycle arms ~1.5-3 H100 h each, depending on 0e's run-time measurement. Month runs ~8-12 H100 h
plus about 100 GB each. Phase 0 and the 10-cycle tests go ahead; month runs of a winner per the
standing agreement; 3-month runs and A2/A0r need the user's OK.

## Status

- 2026-10-22: plan written; A1 metadata probed (above). User approved Phase 0, Phase 1 without
  M2 (M1, M3), Phase 2, and the A2 control. Phase 0a started.
- 2026-10-22, Phase 0 wiring:
  - `backends/aifs/aifs37_model.py` (`AIFS37Model(AIFSModel)`), `aifs_inference_aifs37.yaml`
    (`input: empty`; 0.12 has no `opendata`), `run_aifs37.sh` (anemoi-dev), config key
    `exp.aifs37: True`. aifs_model.py untouched. The 0.19 model is multi-dataset (`data`); the
    latent increment is added by a forward hook on `latent_aggregator` (set and cleared inside the
    checkpointed step, so the backward recompute sees it); the input roll is written out-of-place.
  - Probed: 207 input features, 188 prognostic, 236 outputs, hidden mesh 10944 nodes, autocast
    bf16 (from the checkpoint). Dynamic forcings: computed (julian day, local time, insolation)
    and **loaded every step** (cvh, cvl, lai_hv, lai_lv, slt, fsr), read from the zarr.
  - ICs and truth: `ERA5Zarr` joins Bo's v8 zarr (surface/forcing selection as in A1's
    training.yaml), the 37-level zarr and the stl1 zarr; grid identical to the checkpoint's.
    `get_input`/`get_verif` use `model.build_input_state` / `model.read_single_date_fields`
    when the model has them.
  - ATMS: `obs_atms_operator` now takes the surface inputs a bundle lists (`surface_inputs`,
    + sd, siconc, swvl1, stl1, cvh, cvl, lai_hv, lai_lv, slt, fsr) and mean-fills any q levels
    above the model's top q level; `export_mw_bundle.py` reads levels from the decoder's inputs
    (14 or 37 levels). Re-exported Bo's bundle and exported `aifs37_a1_twohead_s1`
    (`atms_twohead_a1_s1.pt`): both 0 K vs radianced, gradients finite.
  - **The full A1 decoder needs q at 1-50 hPa, which A1 only has as a diagnostic.** Filling those 9
    levels with the decoder mean moves Tb (2022 pairs, 83k obs) by ch 1-4: 2.3/2.6/1.4/0.9 K rms,
    ch 16/17: 2.4/1.6, ch 18-20: 0.9/0.6/0.4, ch 5: 0.4, ch 6-14 <= 0.18 K: the decoder uses
    stratospheric q as a proxy. So q70 decoders were trained (radianced
    `configs/era5_aifs37_{a1,a2}q70_twohead.yaml`, q on 70-1000 hPa only; jobs 24382067/8).
  - Wiring test (predict_step match, 5-day rollout vs ERA5, latent-gradient FD check, memory):
    job 24380725.
- 2026-10-22 (later):
  - A1 1-cycle ps-only DA smoke (24382946, epoch 20): 8 min on one H100, ~2.7 s/iteration, Jo
    falls; NH z500 before/after 15.08 -> 14.43 m. A1 q70 decoder (24382067) done: 200 epochs,
    val y 0.275, flat at the end. A2 q70 decoder (24382068) running.
  - **A2 ps-only control (user, 2026-10-22):** A2 is final (epoch 65,
    `aifs37-a2/.../4c9f9efb22fd4fdf960fb51876945282/`), so it runs now as the A2 counterpart of M1:
    wiring test 24405219, 1-cycle smoke 24405220 (`config_a37_a2_smoke_ps.yml`), then 10 cycles
    24405221 (`config_a37_a2_ps.yml`, exp `a37_a2_ps`) vs Bo's ps-only. Same data join as A1.
  - **Correlated R and sigma must be recomputed per decoder (user):** `decoder_era5.json` and
    `oberr_table_w.json` are from Bo's 14-level decoder. 0f now covers A1 and A2: 2015 37-level
    pairs `atms/data/era5_pre37_2015` (Jan-Apr 7, as `era5_pre_2015`; job 24406117), then
    `atms_decoder_error.py` with each q70 bundle.
  - A2 1-cycle smoke (24405220) passed: 9.6 min, Jo 360k -> 130k in 100 iterations, NH z500
    before/after 9.91 -> 9.03 m (A1 epoch 20: 15.08 -> 14.43). A2 wiring test (24405219): 187/187
    columns rolled, predict_step diff 0, day-5 z500 20.1 m (one forecast), fwd+bwd 11.3 s, 7.2 GiB;
    FD ratio 0.67 / 0.56 at eps 1 / 0.3 (A1 within 15%) -- watch, not a blocker.
    10 cycles 24405221 running.
  - 0f for A1 done (24412434, 4 min on u1-compute): `atms_decoder_error.py` now takes PRE /
    BUNDLE / QFILL / STRAT env; A1 q70 bundle `atms/outputs/aifs37_a1q70_twohead_s1/
    atms_twohead_a1q70_s1.pt` (0 K vs radianced). `figures/obs_error/decoder_era5_a1q70.json`
    (120 cycles Jan-Apr 2015, ch 1-22). std(y - H(ERA5)), clear ocean, K:

    | ch | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 18 | 20 | 22 |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | Bo 14-level | 0.31 | 0.20 | 0.18 | - | - | - | - | - | - | - | - | 1.02 | 1.26 | 1.55 |
    | A1 q70 | 0.29 | 0.18 | 0.14 | 0.16 | 0.24 | 0.37 | 0.36 | 0.44 | 0.53 | 0.63 | 1.00 | 0.94 | 1.07 | 1.30 |
    | A1 2022 sigma_o | 0.64 | 0.34 | 0.18 | 0.16 | 0.21 | 0.29 | 0.30 | 0.34 | 0.42 | 0.52 | 0.84 | 2.96 | | 1.71 |

    (2022 sigma_o: the offline holdout table above, not per scene.) A1 is 5-20% tighter than Bo's decoder on every assimilated channel and scene.
    Ch 8-14 on 2015 are only 1.0-1.3x the 2022 holdout (Bo's ch 8-9 were 1.5-2x), and these include
    ERA5's own error, so 8-14 are usable. Inter-channel corr of 5-7, 18-22 is unchanged (max |diff|
    0.03 clear ocean, 0.12 land). Ch 8-14 correlate with their neighbours 0.4-0.8 (7-8 0.59, 8-9
    0.75, 9-10 0.78, 12-13 0.70), so Phase 2 needs them in the correlated-R block with 5-7.
    Still to do: A1's `oberr_table_w.json` (scene/cloud ratios) and the A2 versions.
  - **A2 ps-only, 10 cycles (24405221, 53 min) vs Bo's ps-only** (`figures/a37_a2_ps_10cyc/`,
    analysis window mean vs ERA5): z500 Global / NH / SH / Tropics 9.61 / 10.04 / 12.55 / 4.24 m vs
    10.05 / 10.76 / 12.89 / 4.57 (-4 / -7 / -3 / -7%); t500 -6%, ws250 -5%, q850 -3% globally.
    Background (dashed) better by about the same. A2 ps-only matches Bo's ps + raobs on z500 and
    t500 NH, not on NH winds. Truth is Bo's ERA5 GRIB cache, ICs the training zarr (~0.1 K apart).
  - M1 (A1 final `epoch_066-step_260000`, ps only, 10 cycles): 24452842, `config_a37_a1_ps.yml`.
  - Exporter / operator: decoders that take a subset of the 37 levels (A2: k 5-36) now export and
    assemble by feature index (`_t_ids`, `_q_ids`); Bo, A1 q70 and A2 q70 bundles all 0 K vs
    radianced. A2 q70 bundle `atms/outputs/aifs37_a2q70_twohead_s1/atms_twohead_a2q70_s1.pt`.
    A1 scene/cloud table 24453006 (`figures/atms_oberr/oberr_table_w_a1q70.json`); A2 decoder error
    + table 24453047. `atms_cloud_error.py` takes PRE / BUNDLE / QFILL / STRAT like
    `atms_decoder_error.py`.

## Week-long chain plan (approved 2026-10-23; launched 14:47 UTC, user away ~1 week)

All on MI300X (`-A gsienkf -q gpuwf`; H100 queue estimated new jobs to start 2026-10-26). A1/A2 run
in `anemoi-rocm` with `aifs_attention: flex_sliding_window` (MI300X wiring test 24456804: A2
day-5 z500 20.3 m vs 20.1 m on H100, FD as on H100). Bo's model on MI300X: platform noise <= 0.4%.
Our jobs are kept off e02g03 (`ExcNodeList`) so A0r's 8-GPU segments always find a free node.

Mechanics (`a37tmp/`, untracked): `submit_week.sh` -> `chain_job.sh ENV BASE LAST CHUNK` x N, linked
afterany; `chain_next.py` resumes from the newest complete analysis in rpath (`*_optimal_inputs_*.pkl`
+ `*_optimal_forecast_*.nc`), writes `<rpath>/chain/chunk_<date>_n<k>.yml`, prints DONE at the last
date and STOP after two jobs without progress (`<rpath>/chain/state.json`). A timeout or crash loses
only the partial cycle. Job IDs: `a37tmp/week_jobs.txt`. Logs: `logs/wk_<run>_<job>.out`,
`logs/fc_<run>_<job>.out`.

| chain | config | cycles (last analysis) | jobs |
|---|---|---|---|
| A1-M3 | `config_a37_a1_m3.yml` (S0: ch 8-15 monitored) | 324 (06-11T12) | 24457019-33 |
| A2-M3 | `config_a37_a2_m3.yml` | 324 | 24457034-48 |
| A1-S1 (+ ch 8-11) | `config_a37_a1_s1.yml` | 180 (03-31T12) | 24457049-58 |
| A1-S2 (+ ch 8-14) | `config_a37_a1_s2.yml` | 180 | 24457059-68 |
| A1 ps only | `config_a37_a1_ps_long.yml` | 324 | 24457069-77 |
| A2 ps only | `config_a37_a2_ps_long.yml` | 324 | 24457078-86 |
| Bo corrhum + window corr R | `config_lwatms_3mo_wc24_q075_corrhum_wcorr.yml` | 180 | 24457087-95 |
| Bo corrhum (3-month, H100) | existing c10-c12 | 180 | 24404950-52 |

S1/S2: sigma ch 8-14 = A1 2015 decoder std (0.16-0.63 K), in the one correlated-R block with 5-7,
18-22; no bias correction (S3 skipped), no cloud ramp for 10-14 (not in the table).
First cycles at 14:54 UTC: all iterating, M3 ~10 min/cycle on MI300X.

Scoring: `wk_score` 24457096 (`a37tmp/score_week.sh`, after all chains): `figures/week_180_best`,
`figures/week_324_ps`, `figures/week_324_best` (window RMSE vs ERA5). Forecast skill
(`a37tmp/submit_week_fcst.sh`, `forecast_skill.py` now runs A1/A2 via `config_fcst_skill_a37_{a1,a2}.yml`;
~1 min per forecast): ERA5-initialized A1/A2 (24457457-58) and Bo ERA5 / ps_bo / ps_raob_bo extended
to 06-16 (24457459) now; each run's daily forecasts 01-06 .. 06-16 (324) or .. 04-05 (180) after its
chain (24457460-66); figures `figures/week_fcst_{180,324_ps,324_m3}` (24457467).

### Part 2 (2026-10-23 15:07 UTC): A1 restarted on epoch 53; corrhum moved to MI300X

**A1's final checkpoint is degraded: use epoch 53.** A1 training (mlflow run 102eab98...) went unstable
after step ~213k: grad norm 0.0008 -> 0.004 (217k) -> 0.19 (221k) -> 0.92 (225k), val loss 0.000859 (epochs
51-53) -> 0.00164 to the end. Forecast test on MI300X (`a37tmp/test_a37.py`, one 5-day forecast from
2015-01-01): epoch 66 day-5 z500 34.4 m, z10 158 m, and its latent gradient fails the FD check (predicted
-1.8e4, actual +5); epoch 65 the same; **epoch 53: 18.5 m, z10 77 m, FD -152 vs -115** (A2 epoch 65: 20.3 m,
51 m). The first epoch-66 cycles made the analysis worse than the background (ps only, tropics 5.6 -> 7.1 m).
A2's training is stable to the end (val 0.000957). The A1 chains, their forecasts, scoring and plots
(24457019-77, 24457457/60/62-64, 24457096, 24457467) and the queued H100 jobs (24404950-52 corrhum c10-c12,
24453460/1 M3 smokes, 24452842 M1) were cancelled (user OK); the epoch-66 outputs `output/a37_a1_*` hold
only partial first cycles.

Resubmitted (`a37tmp/submit_week2.sh`, all MI300X, `-x e02g03`):

| chain | config | cycles | jobs |
|---|---|---|---|
| A1e53-M3 | `config_a37_a1e53_m3.yml` | 324 | 24458700-14 |
| A1e53-S1 | `config_a37_a1e53_s1.yml` | 180 | 24458715-24 |
| A1e53-S2 | `config_a37_a1e53_s2.yml` | 180 | 24458725-34 |
| A1e53 ps only | `config_a37_a1e53_ps_long.yml` | 324 | 24458735-43 |
| Bo corrhum (continued from 03-09T00 on MI300X) | `config_lwatms_1mo_wc24_q075_corrhum.yml` | 324 | 24458744-53 |

Unchanged from part 1: A2-M3, A2 ps only, Bo corrhum + window corr R, A2 / Bo baseline forecasts.
Forecasts: A1e53 from ERA5 24458754; per run after its chain 24458755-59 (corrhum to 06-16). Scoring
24458760 (after all chains), forecast plots 24458761 (after all forecast jobs).
Bo corrhum is H100 for cycles 1-119 and MI300X from 120 (platform noise <= 0.4%).

**Multi-GPU sharding study (A1, MI300X, e02g03; `a37tmp/bench_shard.py`):** anemoi model sharding
(state replicated, hidden mesh split; our flex kernel works under the head all-to-all) is slower at
every N: 4D-Var iteration (20-step fwd+bwd, checkpointed) 3.86 / 4.30 / 4.92 / 5.28 s on 1 / 2 / 4 / 8
GPUs, rollout 0.85 / 1.00 / 1.17 / 1.24 s, memory 7.5 -> 4.7 GiB; one step matches to 1.2-1.6e-4 rel
(bf16). A1 is too small (~7 GiB, <0.1 s per step) for the per-layer all-to-alls to pay off. A cycle
is ~6 s per iteration against 3.9 s for the model, so the obs side / Python overhead (~1/3) and
compile or graph capture of the step are the better targets. Gradient check on epoch 53: 24458262.

### Speed study (2026-10-23, MI300X e02g03, A1 epoch 53)

- Profile of one A1-M3 cycle (`a37tmp/profile_cycle.py`, 20 iterations, synchronizing timers): 5.3 s per
  iteration; backward 65%, model forward ~27%, ATMS operator forward 7%, ps + raobs + decode + sigma
  < 1%. The obs operators are not the bottleneck; the model step is ~85-90%.
- 20-step fwd+bwd without checkpointing (as the configs run, `checkpoint_stride: 0`): 2.8-2.9 s, 6.6 GiB.
- `torch.compile` (`a37tmp/bench_compile.py`, one process per variant): processor -10%, whole model -11%
  fwd+bwd (rollout -5%); 35 s compile. Encoder/decoder are ModuleDicts (not compiled by this wrapper).
  The compiled day-5 cost J differs from the uncompiled one by 4-7% (372.7 vs 347.8 / 356.5); the
  uncompiled step is bit-reproducible across processes, so compile changes the numerics. Not adopted.
- Processes sharing one GPU (same benchmark): 1 / 2 / 4 per GPU -> fwd+bwd 2.81 / 4.37 / 9.1 s each,
  throughput 1.0 / 1.29 / 1.23x. Two chains per GPU is a cheap ~30% throughput gain for parallel
  experiments (each run 55% slower); it doesn't shorten one run's wall time.

### Parallel-in-time (sub-window) 4D-Var: speed proof of concept (2026-10-23, `a37tmp/pit_poc.py`)

One 4-day window, A1 epoch 53, MI300X e02g03, latent controls only, proxy cost = fit to ERA5 z500 / t850 /
u250 every 6 h (obs operators are < 10% of an iteration). Sequential = today's forcing form (latent
increment on step 1 + latent jumps at 24 / 48 / 72 h, one 16-step rollout, 1 GPU). Parallel = 4 ranks x
24-h sub-windows, each from the background trajectory's state at its start with its own latent control on
its first step, + 1 overlap step; continuity cost ||s_{k+1,1} - M(s_{k,4})||^2 (normalized state, weight 1)
with end states / gradients exchanged point-to-point. Re-shoot every R iterations = sequential no-grad
rerun of the window with all controls, sub-windows restarted from it (after a reset the continuity term
is a penalty on each latent jump's effect, i.e. a model-error term). Adam lr 2e-3, wd 1e-3, 100 iterations.

| | s / iter | final J (Jo + Jc) | J of the controls rerun sequentially |
|---|---|---|---|
| sequential, 1 GPU | 2.27 | 12.9 | - |
| parallel, no re-shoot (50 it) | 0.72 | 38.5 + 5.1 | 108 |
| parallel, re-shoot every 5 | 0.73 (0.84 with re-shoots) | 9.6 + 1.3 | 9.96 |
| parallel, re-shoot every 10 | 0.71 (0.76) | 9.7 + 1.3 | 10.6 |

Without re-shooting the speed-up is eaten by slow (Jacobi-like) propagation between sub-windows; with
it, per-iteration convergence matches or beats the sequential form, so time to the sequential run's
100-iteration cost is ~63 s vs 227 s (~3.5x). Caveats: proxy cost, one window, continuity weight not
tuned, not yet with real obs / the solver's schedule.

### PoC2: parallel-in-time Adam with real obs (2026-10-23, one 4-day cycle, A1 epoch 53, MI300X)

A1-M3 obs (ps, raobs, ATMS arm A with corr R and per-obs emissivity), WC24 jumps at 24/48/72 h, 200
iterations, 2015-01-01T00. Sequential: `a37tmp/config_poc2_seq.yml` (24464281, production code, 1 GPU).
Parallel: `a37tmp/pit_dev/` (the opt-in `exp.pit` mode, NOT in production -- reverted from
`long_window_4dvar*.py` at the user's request; run with `PYTHONPATH=a37tmp/pit_dev`), 4 ranks = 4 x 24-h
sub-windows, rank k owns jump k (rank 0 the t0 increment), gradients all-reduced, re-shoot every 5
iterations, continuity weight 0 or 1e4 (`a37tmp/config_poc2_pit_{w0,w1e4}.yml`, 24465146). Epoch-1 cost
identical (3,207,444 seq vs 3,207,443.75 parallel).

| | s / iter | J at 25 / 50 / 100 / 200 it (M) | wall time to 200 it | best controls, sequential form |
|---|---|---|---|---|
| sequential | 3.17 | 1.816 / 1.452 / 1.226 / 1.108 | 634 s | - |
| parallel, weight 0 | 0.99 | 2.185 / 1.681 / 1.310 / 1.144 | 197 s | 1.157 M |
| parallel, weight 1e4 | 0.99 | 2.192 / 1.730 / 1.436 / 1.199 | 197 s | 1.189 M |

3.2x per iteration; per iteration the parallel form converges more slowly, so the gain in time to a
given cost is ~2.2-2.4x (sequential needs ~425 s for the parallel run's 1.157 M, 474 s for 1.14 M).
The continuity term hurts (weight 1e4 worse than 0): coupling through the re-shoot is enough.
Not done: analysis quality vs ERA5 / obs of the parallel solution, more iterations, cycling.

### Incremental PoC: Gauss-Newton + CG (`a37tmp/inc_poc.py`, proxy problem)

Forward-mode TL (`torch.func.jvp`) fails: anemoi's custom autograd Functions lack `setup_context`.
Finite-difference TL (central, 2 rollouts) + exact adjoint (vjp on the retained outer-loop graph): data J
82.1 -> 43.6 / 35.6 / 33.5 after 1 / 2 / 3 outer loops x 15 CG (97 rollouts, 142 s); CG residual only
0.6-0.9 (stalls: the FD-TL is noisy in bf16 and not the exact transpose of the adjoint). Adam on the same
problem: 9.96 at 50 it (121 s), 6.43 at 100 it (238 s). The FD fallback is not viable; a fair test needs
an exact TL (forward-mode support in anemoi-models' autograd Functions) or an fp32 TL.

### Part 3 (2026-10-23 ~17:30 UTC): 4-day chains, sequential 100 vs parallel-in-time 200 (user's choice)

A1 epoch 53, A1-M3 obs, WC24 (jumps 24/48/72 h), 4-day window, 60 cycles 2015-01-01T00 .. 01-30T12,
MI300X `-x e02g03`. Sequential 100 iterations: `a37tmp/config_wk4_seq100.yml`, `chain_job.sh` (production
code), 30 cycles per job, 24466710-13. Parallel in time, 200 iterations, re-shoot every 5, continuity
weight 0: `a37tmp/config_wk4_pit200.yml`, `a37tmp/pit_chain_job.sh` (torchrun 4 GPUs, `a37tmp/pit_dev`,
one cycle per job since the dev driver is single-cycle), 66 jobs 24466714-79. Both resume via
`chain_next.py` (files `96h_100it` / `96h_200it`). Convergence of the single-cycle test:
`figures/pit_poc/convergence_4day.png`. Score both (window RMSE vs ERA5, fresh O-B, and vs the 5-day
A1e53-M3) when back; the first parallel restart (cycle 2) is the first test of restart under pit.

**Result, wall time (2026-11-01, scores pending: `figures/pit_vs_seq`, job 25192974):** seq-100 6.0 h of
job time for 60 cycles (~6.0 min/cycle, 1 GPU, 6 GPU-h); PIT-200 5.0 h (~5.0 min/cycle, 4 GPUs, 20 GPU-h).
**To do: PIT's scaling can be improved by not reloading the model every cycle.** The dev driver
(`a37tmp/pit_dev`) is single-cycle, so each cycle is a new torchrun job that restarts the processes and
reloads the model and obs. Example job 24466730: 74 s from start to epoch 1, then ~203 s for 199 iterations
(~1.0 s/it). About 25% of every PIT cycle is startup, not counting the queue gaps between chained jobs.
Making the pit driver cycle in one process (as the sequential driver does, 30 cycles per job) would remove most of that.

**Result, analysis quality (2026-11-01, `figures/pit_vs_seq`, `diagnostics/analysis_error_ts.py`, 60 cycles):
PIT-200 is clearly worse than seq-100.** Mean RMSE vs ERA5 at 24 h, seq-100 / PIT-200 / 5-day A1e53-M3 (same
cycles): global z500 6.11 / 8.14 / 6.32 m, q850 0.905 / 1.032 / 0.915 g/kg, ws250 2.35 / 3.13 / 2.41 m/s; at 90 h
z500 6.81 / 8.14 / 6.57 m. The gap is there from cycle 1 (global z500 ~7.6 vs ~6.1 m), even though PoC2's
cost was within ~4% of sequential, and the tropics are not degraded. Not yet determined: whether the
PIT analysis trajectory is written correctly (e.g. from a sequential rerun of the final controls vs stitched
sub-windows), or the solution itself is worse. Check the first cycle's written trajectory before trusting this.

**Why PIT-200 is worse (2026-11-01, cycle 1, 2015-01-01T00):** the written trajectory is honest (a sequential
no-grad rerun of the best controls, as `compute_optimal` does for every run). Cycle 1, three runs from the same
background and config: seq-100 (`a37_wk4_seq100`), seq-200 (PoC2 `a37_poc2_seq`), PIT-200:

| | J obs (seq. form) | latent rms t0 / 24 / 48 / 72 h | global z500 RMSE vs ERA5 at 6 / 24 / 48 / 72 / 90 h (m) |
|---|---|---|---|
| seq-100 | 1.328 M | 0.038 / 0.029 / 0.033 / 0.038 | 7.27 / 6.90 / 6.01 / 5.62 / 5.92 |
| seq-200 | 1.108 M | 0.056 / 0.043 / 0.050 / 0.062 | 6.83 / 6.16 / 5.59 / 5.51 / 5.88 |
| PIT-200 | 1.157 M | **0.119 / 0.076 / 0.071 / 0.067** | 6.88 / 7.72 / 7.66 / 7.79 / 6.94 |

More iterations help in the sequential form (seq-200 beats seq-100 everywhere), so the PIT loss isn't from
iterating longer. PIT fits the obs nearly as well as seq-200 but with controls 1.5-2x larger, and its error
is worst mid-window, after the jumps. Cause: with continuity weight 0 and the re-shoot under `no_grad`
(`pit_reshoot`), **no gradient crosses a sub-window boundary**. The t0 increment sees only day-1 obs, and jump k
only day-k obs. The cycle becomes four coupled-in-forward-only 1-day 4D-Vars. Nothing penalizes a control for
what it does to later days, so the controls grow and fit obs locally with offsetting increments and jumps.
Only AdamW weight decay limits them (no `control_prior` in this config). The obs cost is similar, but the
state between and away from obs is worse. Fix to test: carry the adjoint across boundaries (rank k returns
d J_k / d s_k(start) to rank k-1 each iteration, i.e. a backward sweep through the re-shoot), or a control
prior; PIT is not usable as is.

### Drift over long cycling: A0 vs A1 vs A2 (2026-11-01, `figures/a012_drift`, job 25192967)

`diagnostics/analysis_error_ts.py`, 24 h and 114 h into the window, ps-only (A0 = Bo's run) and M3 (A0 =
corrhum). Global z500 RMSE vs ERA5 at 24 h, mean over cycle ranges (m):

| cycles from | A0 ps | A0 M3 | A1 ps | A1 M3 | A2 ps | A2 M3 |
|---|---|---|---|---|---|---|
| 01-01 (1-4) | 8.9 | 6.4 | 8.7 | 6.7 | 8.4 | **6.0** |
| 01-11 (11-20) | 11.4 | **4.9** | 13.6 | 6.3 | 11.7 | 5.3 |
| 01-21 (21-40) | 13.3 | 7.4 | 16.4 | 6.9 | 14.8 | **6.0** |
| 01-31 (41-60) | 14.3 | 9.3 | 20.8 | 8.9 | 41.0 | 13.5 |
| 02-20 (61-100) | 15.6 | 8.7 | 31.8 | 13.4 | 189 | 73.7 |
| 04-01 (181-324) | 25.2 | 11.4 | 63.8 | 17.4 | 271 (to 04-07) | 99.4 |

- A2 is as good as or better than A0 for about the first 5 weeks of cycling with M3 obs, then breaks down within
  about 2 weeks: > 1.2x A0 from 02-09 (M3) and 01-30 (ps only). M3 saturates near 100 m by April; ps only
  grows to ~270 m and then fails (non-finite gradient, cycle 195).
- A1 degrades steadily, not abruptly: M3 7 -> 19 m, ps only 9 -> 95 m by June; its tropics go bad from about April.
- A0 drifts too under ps only (9 -> 33 m), slowly; M3 stays bounded (5-14 m).
- More obs slow every model's drift. The ranking at cycle 1 does not predict the ranking after a month.
  Motivates the no-cycling (PITA) test with A2.

### Cheap iteration study (2026-11-01, submitted; jobs in `a37tmp/iter_study/jobs.txt`)

Question: is 100 iterations enough, and does going further approach ERA5 or overfit? Single cold-start 5-day
windows (no cycling, first guess 24 h ERA5 forecast, `nhrs_back: 24`), A0 (corrhum config) and A2 (A2-M3 config,
same obs and WC24), dates 2015-01-15T00 and 03-15T00, 100 / 200 / 400 / 800 iterations, each its own run (own
cosine schedule). 16 one-GPU MI300X jobs (`gsienkf`/`gpuwf`), 25196706-21, `a37tmp/iter_study/make_and_submit.sh`,
outputs `output/its_<model>_<mmdd>_<n>/`. Score: analysis RMSE vs ERA5 across the window vs iteration count; error
still falling = under-converged, error rising while J falls = overfitting. Feeds the PITA (parallel-in-time
analysis: independent cold-start windows, no cycling) protocol's iteration count. User: don't rely on two
workers per GPU; e02g03 is free again (no longer reserved for A0r).

**PIT on hold (user, 2026-11-01).** The PIT run was not saddle point: it was Adam on the nonlinear cost with
forward-only re-shoots (no backward coupling). Saddle-point 4D-Var (Fisher & Gurol 2017) would supply the missing
backward coupling but needs an exact TL (blocked: no forward mode in anemoi's autograd Functions; FD-TL stalls CG
in bf16). If revisited, try first: pass d J_k / d s_k(start) from rank k to rank k-1 each iteration (lagged
adjoint coupling, no TL needed, ~1 GPU-h one-cycle test in `a37tmp/pit_dev`); then a double-backward TL check
for saddle point. Revisit only after the iteration study and PITA, which parallelizes across windows instead.

### PITA design (user, 2026-11-01): independent cold-start windows, no cycling

Context: the G1 gate ("do we have to cycle?") in the plan doc "Beating ERA5: global 25 km and CONUS 3 km plans"
(https://claude.ai/artifact/HadNSjKw5WBama6BSea62g). Decisions:
- A0 (corrhum config) and A2 (A2-M3 config): same obs and WC24. No sonde denial inside the window; no ensembles.
- First guess: ERA5 forecast aged 0 / 12 / 24 / 48 h (`nhrs_back`), a stand-in for HealDA.
- Scattered dates: 10 window starts every 15 days, 01-10 .. 05-25 (00Z), so the window end T falls 01-15 .. 05-30.
- Iterations: set from the iteration study.
- Scores:
  - Analysis vs ERA5 across the window (`analysis_error_ts.py`; age 0 is not meaningful against ERA5).
  - 5-day forecasts from the window end T vs ERA5 and raobs (`diagnostics/forecast_skill.py RUN_DIR`,
    START = END = T).
  - References already on disk, daily 01-06 .. 06-16: `output/forecast_skill/{wc24_q075_corrhum, a37_a2_m3,
    era5, a37_a2_era5}`.

**Iteration study, preliminary (2026-11-01, 100/200/400 done, 800 running; `figures/iter_study/rmse_vs_iterations.png`).**
Final J falls steadily (A0 Jan 1.46 / 1.27 / 1.09 M), but the analysis error vs ERA5 depends on where in the window it's scored:
- At 24 h, error keeps falling to 400 (z500 A0 Mar 4.70 / 4.23 / 4.07 m).
- At 72 h, z500 is best at 200 and turns up at 400 in all 4 runs; q850 and ws250 are flat after 200.
- At 114 h (window end), all variables are worse at 400 than at 200 in nearly every run (z500 A0 Jan
  5.60 -> 6.31 m, q850 +3-8%).

This is the overfitting signature: past ~200 iterations the extra cost reduction is fit to obs, not truth,
where the window has the least constraint (the end, only past obs plus the last 24 h jump). PITA's forecasts
start from the window end, so 200 is the working choice there.

### PITA run (2026-11-01, user OK)

200 iterations; 40 window starts every 4 days 2015-01-02 .. 06-07 (00Z) with a 24 h ERA5-forecast first guess, and
every 4th date (10) also with 0 / 12 / 48 h. A0 and A2: 140 windows, about 54 GPU-h, one window per 1-GPU MI300X job
(`a37tmp/pita/make_and_submit.sh`, ONLY / SKIP regex filters; job list `a37tmp/pita/jobs.txt`; outputs
`output/pita_<model>_<mmdd>_fg<age>/`). Smoke test first (fg0 = nhrs_back 0, never run before): 25200261-62.
Statistics: paired per-date scatter of day-5 z500 RMSE between near-identical runs is +-8% (corrhum vs wcorr,
lag-1 autocorrelation 0.0-0.16). With 40 dates, a difference of about 3.5% (+-8%) to 5% (+-12%) is detected
80% of the time. Measure the real PITA-vs-cycled scatter after this batch, and add dates if the result is borderline.
- Smoke test 2026-11-01: A2 fg0 (25200262, 01-02) iterates fine. A0 fg0 (25200261) failed: its lagged IC wasn't cached, so the
  job tried to reach CDS (no CDS credentials on this account; earthkit prompted, EOFError). **A cached lagged IC
  `era5_<T>_lagged.grib` is byte-identical to `cat era5_<T-6h>.grib era5_<T>.grib`** (checked on 3 dates), so 24 missing
  lagged ICs were built that way. Dec 30-31 2014 singles aren't cached, so the grid moved to 2015-01-04 .. 06-09
  (window ends 01-09 .. 06-14). A0 smoke rerun on 01-04 fg0: 25200703. The A2 01-02 window is outside the grid
  (smoke only).
- Full batch submitted 2026-11-01: 139 windows 25201051-25201189 (+ smoke 25200703 = a0_0104_fg0); a37tmp/pita/jobs.txt.

**Iteration study, complete (800 added, 2026-11-01).** The final cost J keeps falling (A0 Jan 1.46 / 1.27 / 1.09 / 0.93 M),
but 800 iterations is worse than 200 at every lead and variable in all 4 runs. At 24 h, z500 is best at 400 (A0 Mar 4.70 /
4.23 / 4.07 / 4.26 m). At 72 h it is best at 200 (A0 Jan 4.68 / 4.47 / 4.60 / 5.42). At 114 h it is best at 100-200 (A0 Jan
5.66 / 5.60 / 6.31 / 7.01; q850 0.99 / 0.97 / 1.00 / 1.07). With no control prior, 200 iterations is the practical
optimum. Going further needs regularization (a `control_prior` or B/Q weights), not more iterations.
Tool note: `diagnostics/analysis_error_ts.py` silently drops runs beyond the 14th (`zip(args.runs, COLORS)`).

**To try later (user: key finding, 2026-11-01): lowering the cost further needs regularization, not more iterations.**
These configs have no prior on the controls. A `control_prior` (`notes/weak_constraint_4dvar.md`) or properly weighted
B and Q would stop the controls from fitting obs noise. That has to come before any faster optimizer, such as the plan
doc's M1 meta-optimizer, can help. Test: rerun the iteration study (2 dates, 100-800 iterations) with a control prior;
pass = RMSE vs ERA5 keeps falling as J falls, and 800 is no worse than 200 at the window end.

### PITA results (2026-11-01, 40 dates x A0/A2, 200 its, first guess = 24 h ERA5 forecast)
Scripts: `a37tmp/pita/score_analyses.py` (-> `output/pita_scores/analysis_scores.csv`), `paired_analysis.py`,
`submit_fcst.sh` (forecast jobs 25211327-34, `output/forecast_skill/pita_*`), `paired_fcst.py`. Paired by date, global.
- Analysis vs ERA5, z500 at the window end (120 h): A0 PITA 6.6 m vs cycled corrhum 11.4 m (-40 +-5%, PITA better on 40/40
  dates); A2 PITA 5.8 m vs cycled A2-M3 78.7 m. A2 PITA is slightly better than A0 PITA (5.8 vs 6.6 m).
- First-guess age (10 dates): matters at 24 h (fg0 3.2, fg48 6.4 m for A0), not at 120 h (6.5-6.6 m for every age).
  Even a 48 h-old ERA5 forecast beats the cycled background.
- 5-day forecasts from the window end, A0: PITA beats cycled at every lead vs ERA5 (z500 120 h 46.1 vs 53.3 m, -12 +-5%)
  and vs raobs (T -14%, wind -8.5%, q -4% at 120 h). PITA is still worse than ERA5-initialized forecasts:
  +18% raob T, +20% raob wind, +62% z500-vs-ERA5 at 120 h; already +2-3% vs raobs at lead 0.
- A2 is the same picture (vs ERA5-init: +19% raob T, +20% wind at 120 h).
- Caveats: cycled ran 100 its, PITA 200 (the iteration study says that is only a few % at the window end). The PITA
  first guess carries ERA5 information, which flatters the vs-ERA5 scores, but the raob scores agree.
- Statistics: per-date sd of the paired difference is 14-16% (z500 analysis, A0), and the CIs are +-5%, so 40 dates is
  plenty for these effects.
- Conclusion (G1): we do not need to cycle; cycling as configured is the problem (drift). But a 5-day window started
  from a 24 h ERA5 forecast does not reach ERA5 quality at its end. "Improve on ERA5" has not happened yet.
- In-window fit to raobs (`a37tmp/pita/raob_window.py`, solver raobs_diagnostics, 00/12Z, 1000-50 hPa): A0 independent
  windows fit T at ~1.1 K (= 12 h ERA5 forecast O-F) falling to the ERA5 analysis level (1.0 K) by day 4-5; wind and q
  late in the window fit tighter than the ERA5 analysis. Cycled A0 T 1.4 K. With the day-5 forecasts ~20% worse than
  ERA5's, the late-window fit looks like overfitting (see the regularization item).
- First-guess age vs forecasts (10 dates): A0 120 h raob T 2.52-2.57 K and wind 8.39-8.54 m/s for ages 0/12/24/48 h,
  vs 2.26 K / 7.22 m/s from ERA5. Even fg0 (ERA5 itself) gives the same forecast. The window limits us, not the first guess.
- Deck: https://claude.ai/artifact/WZq7g4TJ7phaUAc4C1e6Wq (figures `a37tmp/pita/deck_figs.py` -> `figures/pita/deck_*.png`).

### IASI in PITA A2 (2026-11-02, 40 dates, `a37tmp/pita/make_iasi.py`, jobs in `a37tmp/pita/jobs_iasi.txt`)

Setup: A2 PITA fg24 window + `observations.iasi`, two-head IASI decoder fine-tuned on torch-crtm Jacobians
(`atms/outputs/iasi_a2q70_twohead_s1_jac10`, J-skill 0.42 -> 0.996), open ocean clear (GSI clear-sky convention, about
8k profiles per 6 h used), sigma_o = std(y - H(ERA5)) per channel, correlated R `figures/obs_error/iasi_decoder_era5_a2q70_jac10.json`
with `corr_r_masked`, cond 50 (shrinks the correlations x0.73 for 58 channels, x0.46 for 157). Channel sets: c157 (all
candidates) and c58 (greedy DFS top 58 = 90% of the candidates' DFS on A2's levels, `a37tmp/pita/iasi_channels_a2.json`).

Paired vs the same windows without IASI (mean rel diff, dates better of 40), c58:
- In the window vs ERA5 at 120 h: q850 -4.8% (40), ws250 -4.3% (38), z500 +1.2% (19); z500 Tropics +11% at 60 h (5).
  c157 worse than c58 everywhere (z500 +6%).
- 5-day forecasts from the window end: the gain does not survive. vs ERA5 at init q850 -4.9%, t850 -3.0%; at 72 h
  q850 -0.7 +- 0.7%, at 120 h every variable within +-2.5% (CI covers 0). vs raobs: +0.8-1.3% at 0 h (IASI competes with
  the raob fit), neutral from 24 h. Still far from ERA5-initialized (raob t 120 h 2.38 vs 2.00 K).
- Desroziers (4 Jan): IASI sigma_o right on average (ratio 0.99) but upper-T CO2 channels (positions 7-45) 1.13-1.34
  (over-weighted), channels 162-175 0.76-0.85. Next if pursued: per-channel Desroziers sigma_o, c58 only.

**c58d (Desroziers sigma_o, 2026-11-02).** The c58 rerun with sigma_o set to the pooled Desroziers sigma (ratios 0.78-1.18 vs the ERA5 sigma) is in-window neutral to slightly worse than c58.
In-window vs no-IASI A2 (40 dates, fg24, c58 -> c58d): q850 at 120 h -4.8 -> -4.9%; ws250 -4.3 -> -4.0%; tropical z500 at 60 h +10.5 -> +12.6%; global z500 at 60 h +1.6 -> +2.5%.
Raob O-A rises slightly (T +1.9 -> +2.1%, q +2.6 -> +3.1%, uv +2.9 -> +3.4%).
Retuning the diagonal sigma does not fix the tropical z500 loss, so that loss is not an ob-weight problem. Forecasts 25277758-61.
Forecasts, c58d vs c58 (40 dates, paired, global; negative means c58d is better): z500 vs ERA5 -1.4 +- 1.3% at 24 h, -0.3 +- 2.7% at 120 h.
Everything else is within +-0.5%, except the 0 h raob fit, which is 0.1-0.3% worse.
Desroziers sigma is forecast-neutral. IASI remains neutral at 120 h either way.
Decision (2026-11-02): keep the original sigma_o, std(y - H(ERA5)). c58 is the IASI configuration going forward. c58d is not adopted.

**Cycled A2-M3 with the Jacobian-tuned ATMS decoder (`a37_a2_m3_jac`, 2026-11-02).** The run cycled 2015-01-01 to 2015-03-01 as planned.
Its in-window analyses at 00 UTC were scored vs ERA5 against `a37_a2_m3` on the same 60 dates (paired; scores in `output/pita_scores/cycled_jac_scores.json`).
Global z500 is -8.6 +- 0.9% at 6 h and 66 h (better on 59/60 and 60/60 dates) and -7.2% at 120 h. Global ws250 is -2 to -3%; tropical z500 is -3%. q850 is neutral.
This is the opposite of cold-start PITA, where jac was worse (t850 +5.9%). Jacobian tuning pays off once the background is cycled.
Forecasts from the cycled jac analyses have not been run.

### Scoring truth: each model's own training ERA5 (decision 2026-11-03)

There are two O96 ERA5 sets:
- **v8** zarr: A0's training data, from MARS O96. The `ic_cache` GRIBs are bit-identical to it.
- **37-level** zarr: A1/A2's training data and initial conditions, built from the GCS 1-degree ERA5.

They differ by about 2.4 m z500, 0.23 K t850, 0.6 m/s u250 and 0.07 g/kg q850 rms. The difference is the same in 2015 and 2022, and the two are about equally rough at small scales.

**Rule:** score each model against its own training ERA5.
- `diagnostics/physical_scorecard.py --truth v8|era5_37` (required).
- `analysis_error_ts.score_cycle(..., cache_dir='zarr:era5_37')`.
- `forecast_skill.py` already does this, through the `aifs37` model's `read_single_date_fields`.

**Effect on earlier results:**
- In-window A2 analysis scores made before this date used v8 and need re-scoring (`a37tmp/pita/rescore_a2.py` -> `output/pita_scores/*_e37.json`). That includes the cycled jac −8.6% and PITA A2 / a2jac / IASI.
- On five cycled dates, jac vs base z500 at 66 h moved from −6% (v8) to −2% (37-level).
- A0 scores and all forecast scores are unaffected.
- A0-vs-A2 comparisons against ERA5 have no neutral truth: prefer raob or other obs-space scores.
