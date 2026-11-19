# Fine-tuning A2 for a better-behaved adjoint (plan, 2026-11-07)

Status: PLAN ONLY. Nothing has been trained or submitted. The user named two ways: (1) multistep rollout training,
(2) a penalty on a poor adjoint. This doc maps both, the evidence behind them, how we evaluate, and the smallest
first experiment.

## 1. Why: the defect we are trying to fix

From the adjoint-quality tests (notes/plan_20261105.md "ADJOINT QUALITY RESULT", figures/adjoint_quality/Q1-Q5,
a37tmp/adjoint/adjoint_quality.py; 4 twin windows of 2022):

- **What is fine.**
  - The gradient is accurate and deterministic. cos(bf16, fp32) is 0.88-0.97 and repeated runs are bit-identical.
  - It is smooth for small moves.
  - Realistic increments grow at a weather-like rate (2.5x at 48 h, 6-7x at 120 h). There is no damping null space.
- **What is wrong.**
  - A2 is hugely sensitive to small-scale, unbalanced input perturbations.
  - The leading singular values of the 6h -> 48h map are 100-420, with a flat spectrum (still 60-90 at rank 24).
  - Those directions are >85% at scales below 400 km. Their input side is geopotential changed without matching
    temperature (70%), or T + msl. Their output side is upper-tropospheric v and q.
  - Steepest descent points into these directions. The linear regime along -g ends at about 3% of a golden-sized step.
- **Supporting evidence from the analysis session** (a37tmp/pita2022/ic_attrib.py, figures/ic_attrib22).
  - Golden's window-end analysis has mass close to ERA5 (z is 1% of the normalized difference).
  - Wind (57%) and q (34%) carry most of the difference.
  - Wind at 250-400 hPa drives 52-62% of the 6-48 h forecast gap, in small-scale patches along the storm tracks.

Interpretation: ERA5 states always have hydrostatic and near-geostrophic balance. A2 never saw an input that breaks
balance, so its response to one is arbitrary and large. 4D-Var's gradient finds exactly those directions. The target
is therefore NOT "learn to backpropagate". It is: **respond weakly to off-manifold (unbalanced, small-scale) input
perturbations, while keeping the realistic growth of balanced, synoptic ones.**

## 2. How A2 was trained (what we would fine-tune)

Sources:
- `/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/long-window/EAGLE/eagle-aifs37-a2.yaml` (training block, around line
  790-860)
- `.../run/aifs37-a2/training/outputs/2026-10-2*/*/.hydra/config.yaml` and `train.log`

Same tree as `/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/run/aifs37-a2`.

- **Code and environment.** anemoi-training 0.17.0.post36, editable from
  `/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/anemoi-core/training`. Environment `anemoi-rocm`
  (EAGLE conda), on 1 node with 8x MI300X (`--gres=gpu:mi300x:8`, account gsienkf, qos gpuwf), DDP with 1 GPU per
  model.
- **Model.** AnemoiModelEncProcDec: hidden mesh o48 (10944 nodes), 1024 channels, transformer processor (16 layers,
  16 heads, sliding window 512, flex attention). Skip connection from the last input step. multistep_input 2.
  6-h step.
- **Task and data.**
  - Forecaster, `rollout: {start: 1, epoch_increment: 0, maximum: 1}`. **Single-step only: there was never a
    rollout or fine-tuning stage.**
  - Train 1979-2021, validate 2022. The validation year is our clean PITA test year, so fine-tuning must keep 2022
    out.
  - Data is the join of `era5-o96-1979-2023-6h-v8.zarr` (surface and forcings, Bo.Huang) and
    `era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr` (37 levels).
- **Loss.** MSE with scalers: pressure_level ThicknessVariableLevelScaler (y_intercept 0.2, slope 0.001, floor 0.1,
  total 6.5), general_variable (z 12, t 6, sp 10, u 0.8, v 0.5, q 0.6, w 0.001, 10u/10v/2d 0.5, tp 0.025,
  cp 0.0025), node_weights, nan_mask, time_steps.
  - No noise or augmentation of any kind.
  - Because of the weights, z and t dominate the loss, and the loss sees only the output, never the consistency of
    the inputs.
- **Optimizer.**
  - AdamW, betas (0.9, 0.95), lr 6.25e-5, cosine schedule to 3e-7 with 1000 warmup steps.
  - 260000 steps (65 epochs of 4000), batch 1 per GPU with accum 2: an effective batch of 16 samples.
  - bf16-mixed, gradient clip by value 32.
- **Cost.** About 39 h on one 8x MI300X node (four chained 12-h jobs, 2026-10-20 23:53 to 2026-10-22 14:55), so
  about 310 MI300X GPU-hours. That is about 0.27 s per single-step forward+backward sample per GPU.
- **Checkpoint state.** Only the inference checkpoint `inference-anemoi-by_epoch-epoch_065-step_260000.ckpt`
  survives (a pickled model, with no Lightning `last.ckpt` and no optimizer state). So we cannot resume, only
  warm-start the weights.
  - anemoi-training's `load_weights_only` and `transfer_learning_loading` (`train/train.py:377`,
    `utils/checkpoint.py:110`) expect a Lightning checkpoint (`state_dict` + `hyper_parameters.data_indices`).
  - A small shim that wraps the inference model's state_dict into that format would be needed. Not yet tested.

## 3. Way 1: multistep rollout fine-tuning

- **What.** Train on the loss summed over R autoregressive steps (anemoi `task.rollout`; e.g. start 2, +1 per short
  epoch, up to 4 = 24 h; later 8 = 48 h or 12 = 72 h). Our 4D-Var window is 20 steps (120 h), but long rollouts are
  expensive and blur more.
- **Why it could change the Jacobian.**
  - The loss at step k depends on the Jacobian of steps 1..k-1 acting on the model's own errors, which are
    small-scale and partly unbalanced.
  - Rollout training therefore penalizes amplifying those error modes, so the model learns to damp perturbations
    that look like its own errors.
  - It does not target "z without t" specifically, unless A2's own errors contain it.
  - A2 does output z and t independently, so its errors may contain hydrostatic inconsistency. A cheap pre-check
    (below) measures that.
- **Literature.** GraphCast (Lam et al. 2023) fine-tuned with a 2 -> 12 step curriculum; AIFS (Lang et al. 2024)
  uses rollout fine-tuning to 72 h. Both report better medium-range RMSE and smoother (blurrier) fields at long
  leads. MSE rollout training pushes toward the conditional mean, so small-scale variance falls with lead.
- **Cost.**
  - About R times the single-step cost, plus about 30% activation-checkpoint recompute. R = 4 is about 1.4 s per
    sample per GPU.
  - Memory with per-step checkpointing is fine on MI300X (192 GB). Our 4D-Var already backpropagates 20 steps on one
    GPU.
- **Risks.**
  - (i) Blurring, which damps the realistic growth we want to keep (golden increments grow 2.5x at 48 h). If growth
    collapses, 4D-Var loses reach to the window start: the excess null space we tested for and did not find.
  - (ii) It may not touch the z/t-inconsistent directions at all.
  - (iii) The latent space shifts, so latent-space tools need regenerating (section 7).

## 4. Way 2: a penalty on poor adjoint behavior

The common design choice: penalize the response only to **off-manifold perturbation families**, and only the
**excess** over a reference growth. Balanced synoptic growth then stays free. Perturbation families v ~ P:

- **P_hyd (hydrostatic inconsistency).** A z perturbation with no matching t, or a t perturbation with z unchanged.
  This is the leading SV input (70% geopotential).
- **P_ss (small scale).** Gaussian noise high-passed below about 400 km (residual after the `smooth()` operator in
  adjoint_quality.py), on t, z, u, v, q, sp/msl.
- **P_geo (geostrophic imbalance).** A wind perturbation without the matching mass field, and the reverse.

Sizes: about the background-error size per variable (e.g. 0.5 K, 5 m, 1 m/s, 0.3 g/kg), applied to the last input
time level only, as 4D-Var's increment is.

Options, with cost per training step (base = one single-step forward+backward):

| option | what it targets | cost | risk to physical growth |
|---|---|---|---|
| **N: denoising augmentation.** Input x + v (v ~ P), target unchanged y | teaches f(x+v) ≈ f(x): contracts off-manifold input | about 1x (free) | low for small v; if v is too big or too broad, it smooths everything (blur). Sanchez-Gonzalez et al. 2020 used input noise for rollout stability |
| **J: finite-difference Jacobian hinge.** L_J = E_v [max(0, \|f(x+εv) − f(x)\| / (ε\|v\|) − γ)^2], v ~ P, over 1 step (or K = 2-4 steps) | the off-manifold one-step (K-step) amplification directly | about 2x (1 step) to 2K x | low with a hinge: only excess over γ is penalized and only on P. γ from the reference growth below |
| **H: Hutchinson Frobenius** ‖J P^{1/2}‖_F^2 via random projections (Hoffman et al. 2019) | average off-manifold sensitivity | about 2x per probe (FD JVP), or 3x with double backprop | penalizes all P-directions, including any legitimate small-scale growth; a hinge is harder |
| **S: leading-SV penalty.** Power iteration on the K-step map | the worst direction (what Q4 measures) | k JVP+VJP per step, about 10-20x | the most targeted but expensive; the vector is not persistent across samples |
| **B: balance-consistency output penalty.** Hydrostatic residual of output z vs t | output consistency | about 0 | doesn't change the sensitivity directly; weak lever |

- **Reference growth γ.** Per 6-h step about 1.2-1.4, from published error-doubling of 1.5-2 days for
  analysis-size perturbations. Check it against A2's own balanced growth: the golden increment grows 1.7x in 24 h
  (about 1.14 per step).
  - The measured leading SVs (100-420 over 7 steps) imply about 2-2.4 per step.
  - So γ = 1.3 per step leaves balanced growth untouched, while penalizing the off-manifold modes.
- **Preferred.** N (free, simple) and J (targeted, cheap at K = 1). H and S are for later if N/J leave a residual.
  B is a side constraint only.
- **Evaluation tools already exist.** a37tmp/adjoint/adjoint_quality.py:
  - `sv_run` gives the singular values and composition;
  - Part A gives the Taylor ratio and gradient cosines;
  - Part B gives survival.

  The single-observation test (a37tmp/adjoint/single_ob.py, being run by another thread) gives balance metrics:
  wind/geostrophic ratio, small-scale fraction. Rerun all of these with the fine-tuned checkpoint (the backend takes
  `model_name` / `path_model` from the config).

## 5. Evaluation ladder (cheapest first; pre-registered gates)

**(a) Adjoint metrics.** 4 twin dates (0104, 0422, 0707, 1023), about 7 min per date: adjoint_quality.py, plus
single_ob.py when available.
- Gate PASS:
  - leading 6->48 h SV ≥ 5x lower than A2 (A2: 100-420);
  - the steepest-descent Taylor ratio stays ≥ 0.7 out to ≥ 0.1 |golden| (A2: about 0.03);
  - golden-increment survival at 48 h within 0.7-1.3x of A2's (2.5-2.7x), i.e. balanced growth kept;
  - single-ob wind/geostrophic ratio improves toward 0.5-1.
- Predictions:
  - N: SV 3-10x lower; survival -10%.
  - J: SV 5-20x lower; survival -5%.
  - R (rollout 4): SV 1.5-3x lower; survival -20 to -30% (blur).
  - C0 (control): no change (±10%).

**(b) Forecast skill vs ERA5 on 2022** (held out). Daily forecasts to 240 h with diagnostics/forecast_skill.py
(config like config_fcst_skill_a37_a2.yml, with the new checkpoint).
- Gate PASS:
  - z500, t850, ws250, q850 RMSE within +2% of A2 at 24-120 h (C0 sets the noise floor);
  - kinetic-energy spectrum at 120 h: small-scale energy (scales < 500 km) not down more than 20% vs A2 (the blur
    check).
- Predictions:
  - R: RMSE -1 to -3% at 72-240 h, small-scale energy down 20-40% (may fail the blur gate).
  - N/J: RMSE within ±1%, spectra within 10%.

**(c) 4D-Var benefit.**
1. The twin OSSE arms without B on 10 dates (configs from a37tmp/adjoint/make_osse.py): ot_s, ot_s600, ot_s24.
   - Gate: start ratio at 6 h ≤ 0.95 (A2: 0.99); ot_s600 no worse than ot_s (A2: 1.07); ot_s24 ≤ 1.1 (A2: 1.49).
2. Then 10 real-obs PITA windows with the golden config (fg12, 200 iterations).
   - Gate: the +7-8% raob gap vs ERA5 at 6 h roughly halves; 120 h no worse.
   - Note: golden's own background and IC are produced by A2, so rerun the fg12 backgrounds with the new model.

## 6. Recommended first experiment

**Step 0 (no training, about 30 min of 1 GPU).** Measure A2's one-step and 4-step amplification for each family
P_hyd / P_ss / P_geo, and for "A2's own 6-h error" (forecast minus ERA5, rescaled). Use the FD machinery in
adjoint_quality.py.
- This sets ε, the sizes and γ.
- It also tells us whether rollout (which only sees the model's own errors) can reach the bad directions at all. If
  A2's own errors amplify only about 1.3x per step while P_hyd amplifies 2-3x, Way 1 cannot be expected to fix it.

**Step 1: four short fine-tunes from epoch_065, identical data, schedule and seed.** Data 1979-2021, 2022 excluded.
- Arms:
  - **C0 continue-training:** rollout 1, plain MSE. Separates the effect of fine-tuning itself.
  - **R rollout:** curriculum 2 -> 4 steps.
  - **N denoise:** rollout 1, with P_hyd + P_ss + P_geo augmentation on 50% of samples.
  - **J hinge:** rollout 1, with the FD Jacobian hinge on P, λ chosen so L_J is about 10% of the MSE at the start.
- Each arm: 4000 optimizer steps (effective batch 16, as in the original). lr warmup 200 steps to 5e-6, then cosine
  to 3e-7. Checkpoints at 1000 / 2000 / 4000 to see the trend in the SV metric.
- Cost on one 8x MI300X node, using 0.27 s per sample per GPU:

  | arm | wall time | MI300X GPU-hours |
  |---|---|---|
  | C0 | 0.6 h | 5 |
  | N | 0.7 h | 6 |
  | J | 1.3 h | 10 |
  | R | 3 h | 24 |
  | total | about 5.5-6 h of node time | about 45 |

- Then ladder (a) on all 4 arms × 3 checkpoints: 12 × 4 dates × 7 min, 1 GPU each, about 6 GPU-hours. Then (b)
  and (c) only for the arms that pass (a).
- **Implementation choice.** A lean trainer in this repo, e.g. `a37tmp/finetune/finetune_a2.py`, then a tracked
  tool if it works.
  - It reuses the aifs37 backend's differentiable step (`_advance_one_step`, forcings, pre/post-processors) and reads
    the same two zarrs through anemoi-datasets `open_dataset` with the inference join config.
  - It replicates the MSE scalers above, and runs DDP via torchrun on 8 GPUs.
  - Why not anemoi-training: N and J need custom loss hooks, and warm-starting from an inference checkpoint needs a
    shim anyway. One code path for all four arms keeps the comparison fair.
  - C0 doubles as the check that the trainer reproduces A2's loss. Its validation MSE at step 0 must match A2's
    single-step MSE on 2022 within about 1%.
  - Alternative: anemoi-training for C0 and R (config-only) with a converted checkpoint, plus a Forecaster subclass
    for N and J.

## 7. Side effects and dependencies

- **Latent-space tools become stale** after any fine-tune, because the latent is the processor input:
  - latent B/Q EOFs (output/latent_eof_a2_*.npz);
  - output/colearn/jump_sigma.pt;
  - any saved latent increments.

  B and Q are parked, so this matters little now.
- **Radiance decoders.** The ATMS/AMSU-A/IASI two-head decoders read physical fields (t, q, surface; see
  `ATMS_MODEL_FIELDS` in long_window_4dvar_utils.py), so they are not tied to A2's latent. They would see slightly
  different forecast fields: recheck O-B bias, no retrain expected.
- **Golden baselines would need rerunning** with the fine-tuned model for any production comparison: fg12 backgrounds
  and the 40 windows.

## 8. Open questions for the user

1. **GPU budget.** About 6 node-hours of one 8x MI300X node for step 1, plus about 10 GPU-hours of evaluation. The
   original training used account gsienkf / qos gpuwf; is that the right account for this?
2. **Trainer.** A lean custom trainer here (recommended), or extend the EAGLE/anemoi-core training stack? Is anyone
   else using that fork?
3. **Production.** If a fine-tune passes, does "A2-ft" replace A2 for PITA? That means re-running the golden 2022
   baseline and the fg12 backgrounds.
4. **Physics track.** Should the later physics co-learning (diurnal cycle, convection, from obs) be built on the
   fine-tuned model? The adjoint regularizer would then stay in that loss.

## Prior evidence: DA-side tendency penalties (added 2026-11-07)
- Hakim, Whitaker, Huang, Frolov, arXiv:2608.11515 (NeuralGCM, ps-only long-window 4DVar): Jt penalizes first-step
  tendencies (alpha_t 20); "greatly reduces the amplitude of these rapidly decaying features, but does not eliminate
  them"; alpha_t 200 removes the early z500 drop but raises error across the window. Suggests (untested) overfitting at
  the window end -- now seen directly by the E-12h attribution (lwanalysis): window-end differences project onto growing
  directions. Moist gradients were zeroed.
- This repo's `jc_ps_weight` (integrate_multiple_backends.md, ACE2): first-step ps-tendency increment cut >100x, NH better,
  SH mixed. Not used in golden A2 runs. User: tendency penalties "didn't seem to work with AI models".
- Why, given our A2 results: in a physical-dycore model imbalance is shed as fast-DECAYING transients, which a first-step
  tendency penalty sees. In A2 the unbalanced small-scale directions GROW 100-420x over 48 h and turn into upper-level
  wind/q errors; their first-step tendency need not be large, so Jt/Jc cannot target them. This favours MODEL-side arms
  (N denoising, J excess growth over 48 h, not one step) over any DA-side tendency penalty; arm J must measure growth over
  several steps, not first-step tendency.
- Literature: Tian, Holdaway, Kleist (GRL 2026; arXiv:2411.14677): GraphCast/NeuralGCM TLM/adjoint show unphysical,
  noisy, localized sensitivities vs MPAS-A.

## Stage 0: perturbation-family amplification (started 2026-11-07)
Tool: `python -m diagnostics.adjoint_quality.aq families` (diagnostics/adjoint_quality/families.py); 4 twin dates
(0104 0422 0707 1023), fp32, perturbation at the 6-h state (last time level), common size 0.02 (normalised
all-prognostic norm) and 0.005 (linearity check), growth = |response| / |perturbation| at 6 / 24 / 48 h.
Predictions (written before the runs):
- hyd_z_small: largest, 48-h growth > 20; hyd_z_large 5-20; hyd_t_large ~5; geo_wind_large 3-10; small_all 5-20.
- Balanced: nmc_full / nmc_large 2-3 (like the golden increment's 2.5x).
- A2's own 6-h error: a2err_full ~2-3, a2err_small 2-4, i.e. rollout training (which only sees A2's own errors) would
  barely touch the bad directions.
- Decision rule: denoising family = the families with 48-h growth > 3x the balanced growth; J gamma = the balanced
  24-h growth (nmc_large) x 1.5; if a2err growth is close to nmc, expect arm R to do little for the adjoint.
Trainer: finetune/finetune_a2.py (+ run_finetune.sh), single MI300X per arm, accum 8, 4000 steps, train 1979-2020,
validate 2021, 2022 untouched; checkpoints evaluated via exp.model_weights (get_model).

### Stage 0 result (2026-11-07; jobs 25751737-40, smoke 25751386; figures/aq/a2_families/F1_family_growth.png)
Growth at 6 / 24 / 48 h after the perturbation (mean of 4 dates; size 0.02; the 0.005 size agrees within ~10%):
| family | 6 h | 24 h | 48 h |
|---|---|---|---|
| z only, large (hydrostatic imbalance) | 2.11 | 3.65 | 5.6 |
| z only, small | 2.16 | 2.65 | 3.2 |
| t only, large (hydrostatic imbalance) | 1.34 | 2.27 | 3.3 |
| balanced lagged-forecast difference, large / all scales | 1.10 / 1.04 | 1.47 / 1.32 | 1.91 / 1.82 |
| A2's own 6-h error (full / small scales) | 0.70 / 0.65 | 0.70 / 0.64 | 0.94 / 0.89 |
| wind only, large (geostrophic imbalance) | 0.69 | 0.69 | 0.79 |
| small-scale noise, all variables | 0.69 | 0.73 | 0.90 |
Against the predictions: the ORDER was right for z (largest) but the "small-scale" and "geostrophic" predictions were
wrong: A2 DAMPS small-scale noise and wind-only perturbations. The amplified kind is mass-field inconsistency: z
without matching t (or t without z), at any scale, which doubles in the first 6 h and turns into q and v (output 40-50%
q, 25-35% v in the normalised norm). A2's own forecast errors decay like the damped families: rollout training (arm R)
never sees the amplified directions, so R is expected to do little for the adjoint (kept as an arm for the record).
Random draws of these families grow 3-6x; the singular vectors (100-420x) are the worst cases within them.
Settings chosen for stage 1:
- N: noise families hyd_z_large, hyd_z_small, hyd_t_large (not small_all / geo_wind: already damped, adding them only
  risks blur); size 0.02 (about a third of A2's own 6-h error, 0.06), probability 0.5.
- J: same families, K = 4 steps (24 h), eps 0.01, gamma = 2.2 (1.5x the balanced 24-h growth of 1.47): the hinge is
  active for z-large (3.65) and z-small (2.65), at the edge for t-large (2.27); lambda auto (10% of the MSE).
Trainer smoke tests (25751712-15) pass: our training step equals the backend's advance exactly (max abs diff 0.0);
C0 ~0.3 s per sample. model_weights loading checked (25751744).

## Analysis-error noise model (2026-11-07; finetune/analysis_noise.py, a37tmp/finetune_noise/)
ERA5 ensemble hard to get -> synthetic noise with the statistics of OUR analysis errors, for fine-tuning A2 on imperfect
initial states. Fit on 2015 PITA (pita_a2iasi_c58_*_fg24, 40 windows): d = analysis - ERA5 (training zarr) on A2's
normalised prognostic state, at the window start (+6 h, optimal_inputs) and window end (wcend). 2022 golden
(pita22_best) = held-out check only. Generator: global column EOFs of d; per-EOF horizontal correlation by n passes of a
13-neighbour mean on the grid (n fitted to the real coefficient fields' retained-variance curve); per-EOF amplitude by
10-deg latitude band. Also "resample" mode (real 2015 d, random longitude shift and sign).
Predictions (written before the job):
- variance shares (start, 2015): wind 50-60%, q 25-40%, t 5-10%, z <3% (as lwanalysis 2022); window end larger by
  ~1.2-1.5x in wind/t; 2022 within 20% of 2015 per variable.
- small-scale (<400 km) share 30-50% for wind/q, <20% for z.
- synthetic (gen) reproduces the variance profile within 10% and the retained-variance curve within 0.05.
- growth test (A2 48 h from ERA5+noise vs ERA5 control, relative to forecasts from the real analyses): gen mode grows
  0.6-0.9x as much as the real analysis error (random phases miss flow-dependence); resample mode 0.8-1.0x.

## Stage 1, revised design (2026-11-07, after the hydrostatic check)
The 4D-Var increments are NOT hydrostatically worse than the true error (output/adjoint_hb.json, figures/hbal/HB1),
so the z-t-targeted arms J and R (and N) are dropped. Target: robustness to realistic analysis errors. Noise:
finetune/analysis_noise.py (2015 PITA analysis - ERA5; 2022 held out), gated on its validation (job 25761146, figure N3).
Arms (finetune/finetune_a2.py, same schedule: 4000 steps, accum 8, lr 5e-6 cosine, ckpts 1k/2k/4k):
- C0: continue training unchanged (control). Launched first.
- NA: ERA5 + analysis noise on both input time levels (same draw), half the samples clean; target ERA5 +6 h; normal loss.
- NR: the same noise, rollout 2 -> 3 -> 4 against ERA5 targets.
- H (optional): C0 + hydrostatic-residual output loss, only if output/adjoint_hb_era5.json (job 25760434) shows A2's
  forecasts violate hydrostatics well above ERA5's floor.
Robustness metric added to validation: the same 2021 forecasts started from ERA5 + fixed analysis-noise draws
(val_noisy_mse_6h / 24h); for C0 (started before the final noise statistics) it is computed afterwards on its checkpoints.
Predictions (before the runs):
- C0: clean validation MSE within +-1% of step 0; adjoint battery unchanged (+-10%); 2022 skill from ERA5 +-1%.
- NA: noisy-start 24-h validation MSE -5 to -15% vs C0; clean MSE +0 to +2% (6 h), within +-1% at 24 h;
  adjoint battery: leading 48-h singular value 1.3-3x lower, Taylor linear range along -g 1.5-3x longer, golden
  survival at 48 h -5 to -15%; 2022 skill from ERA5 within +1%; small-scale energy at 120 h -5 to -15%.
- NR: noisy-start 24-h MSE -10 to -25% vs C0; 2022 skill from ERA5 -1 to -3% at 72-240 h; small-scale energy at 120 h
  -15 to -35% (may fail the blur gate); singular values 2-5x lower; golden survival -10 to -30%.
- PITA (arms passing (a) and (b)): the 6-h raob gap vs ERA5 (+7-8%) shrinks by 1-3 points for NA, 2-4 for NR;
  120 h no worse.
- Arm H skipped (2026-11-07): output/adjoint_hb_era5.json -- A2 forecast hydrostatic residual 1.37 / 1.42 / 1.79 m at 6 / 24 / 120 h vs ERA5 floor 1.35 m; gate not met. Arms: C0, NA, NR.
### Result (2026-11-07; smoke 25760578, full 25761146 H100; output/finetune/noise_full, figures/finetune_noise/N1-N3)
- Real analysis - ERA5 (normalised): 2015 vs held-out 2022 agree closely (shares wind 0.56/0.58, q 0.35/0.33, t 0.06/0.05,
  z 0.007/0.008 at the window end; start has more q 0.42, wind 0.51, rms 0.135 vs 0.163 at the end). Small-scale share
  z 0.10, t ~0.35, wind ~0.33, q ~0.55. Adjacent-level corr t 0.54, u 0.69, q 0.53 (fall to ~0 at 6 levels). Hydrostatic
  imbalance fraction 0.57-0.65 (same as A2-vs-A2 differences). Corr of the error at t-6 h and t only 0.25-0.28.
- gen mode: variance, shares, vertical corr and imbalance match (within 1-2%); horizontal scale WRONG -- too smooth for
  t/u/q (small-scale share 0.23/0.18/0.34 vs 0.37/0.32/0.56) and too rough for z500 (one smoothing per global EOF mixes
  variables; 1 full pass is the smallest non-zero step). resample mode matches every diagnostic incl. scales.
- Growth (40 windows of 2022, |fcst - ERA5-started fcst|, normalised): real analysis 1.00/0.96/0.99/1.21 at 0/6/24/48 h;
  resample 1.00/0.89/0.92/1.15; gen 1.00/0.93/0.88/1.00 (lag-correlated gen same). Error vs ERA5 at 48 h: real 0.250,
  resample 0.243, gen 0.228, ERA5 control 0.187; wind250 5.96 / 5.69 / 5.14 / 4.08 m/s; z500 15.4 / 13.8 / 10.7 / 7.6 m.
- vs predictions: shares as predicted; end larger than start (1.2x) as predicted; gen within 10% in variance yes but not
  in scale; growth gen 0.83x of real at 48 h (predicted 0.6-0.9x), resample 0.95x (predicted 0.8-1.0x).
- Verdict: train with mode='resample' (real 2015 window-end errors, rotated/sign-flipped; 40 fields -> limited
  diversity, add more 2015 windows if needed). gen needs per-variable (not per-global-EOF) and fractional smoothing.
- Noise validation (job 25761146, output/finetune/noise_full/noise_growth.json, 40 windows of 2022): growth of the
  forecast difference vs the clean forecast, 0 -> 48 h. Real PITA analyses: norm 1.21, z500 2.17, wind250 1.49,
  q700 1.15. Generator mode 'gen': 1.00 / 1.33 / 1.11 / 1.09, and z500 first DROPS to 0.66 at 6 h -> FAILS the gate
  (A2 sheds the synthetic noise). Mode 'resample' (real 2015 analysis-minus-ERA5 fields, longitude-rotated, random sign):
  1.15 / 2.03 / 1.35 / 1.15 -> passes within ~10%. NA and NR therefore train with --noise_mode resample. The in-training
  noisy validation stays on 'gen' for every arm (C0 was already running with it); the decisive robustness check is
  re-run offline on all checkpoints with 'resample'.
- Launched 2026-11-07 (H100, anemoi-dev): C0 25762660 (output/finetune/C0); NA 25765595 (output/finetune/NA2) and
  NR 25765596 (output/finetune/NR2) with noise_mode resample_aug: per draw a random pool (start/end), two real 2015
  fields each with random longitude shift and sign, mixed cos(th) f1 + sin(th) f2, amplitude x U(0.5, 1.5); older
  input level lag_corr 0.27; half the samples clean. Smoke 25762918 passed. Superseded/queued jobs put on HOLD (not
  cancelled): 25761293 (C0, MI300X), 25762890/91 (NA/NR plain resample, MI300X), 25762897/98 (NA/NR plain resample,
  H100). In-training noisy validation uses gen-mode draws for every arm; the decisive robustness check is offline.

## User notes 2026-11-07 (late)
- radianced tried input-noise training on a similar problem and it "didn't pan out" -> low prior on NA/NR; they also risk
  teaching A2 to DAMP analysis-like perturbations (TL -> 0), i.e. a worse adjoint. The balanced-growth gate (0.7-1.3x) checks this.
- Physical-adjoint supervision needs pairs + their physical evolution: GEFS reforecast members / GFS NMC pairs (ufs2arco
  supports GFS and GEFS); HealDA - ERA5 = independent analysis-error perturbations (no evolution target). HealDA analyses
  and the 14->32-level vertical upsampler exist (healda/vert_upsampler/upsampler.pt). Being scoped; also ERA5 EDA without CDS keys.

## GEFS pilot (2026-11-08): is A2's tangent-linear physical for GEFS-like perturbations?
Data: operational GEFS v12 (s3://noaa-gefs-pds) via ufs2arco 0.19.1 aws_gefs_archive (EAGLE `data` env, u1-service),
Jan 2023 00Z, members 0-10, leads 0/6 and 24/30 h, 26 levels 10-1000 hPa, 0.5 deg; converted to A2's O96 packed state
(a37tmp/gefs/). Test: for member-minus-control pairs, A2's evolution of the difference (A2(x_m) - A2(x_c)) over 6 h and
24 h vs GEFS's own member-minus-control at the later lead; plus an fp32 finite-difference TL check on the same pairs.
Predictions (parent): 6-h pattern correlation 0.6-0.8, norm ratio 0.8-1.3; largest mismatch in q, upper-level wind and
small scales. Fork adds: 24-h correlation lower (0.4-0.6); TL (small-eps) vs nonlinear differences agree to >0.95 at 6 h.
- Stage-2 jobs (2026-11-07/08, H100): C0 finished training (25762660, 2 h 42 min).
  (a) adjoint battery with CALIBRATED step sizes (no golden increment: the fine-tunes have their own latent), the same
  for the A2 baseline so the comparison is like for like: A2 25769469-72, C0 25769552-55, NA 25770695-98, NR 25770700-03
  (output/aq/ft_{A2,C0,NA2,NR2}); configs a37tmp/finetune_eval/.
  (b) finetune/eval_skill.py (2022, 30 inits every 12 days, 120 h; RMSE z500/t850/ws250/q850 vs ERA5, small-scale
  KE ratio at 250 hPa, and forecasts from ERA5 + fixed real analysis-error draws): A2 25770690, C0 25770691,
  NA 25770694, NR 25770699 (output/finetune/eval/). NA/NR jobs depend afterok on their training.
- Stage-2 results so far (4 dates for (a), 30 inits of 2022 for (b); vs A2; summary output/finetune/eval/stage2_summary.json,
  script a37tmp/finetune_eval/summarize.py):
  - C0 (control): adjoint metrics within 3% (sv1 -0.3%, random amplification -0.8%), skill/robustness/small-scale KE
    within 0.4% -> fine-tuning itself is neutral, as predicted.
  - NA: (a) sv1 -7.5%, sv3 -18%, random 48-h amplification -20%, Taylor linear range along -g 0.01 -> 0.02 of the
    calibrated step, golden-like survival -8% at 48 h: a modest gain, below the predicted 1.3-3x.
    (b) FAILS: clean-start RMSE +11% z500 (24 and 120 h), +7-10% ws250, +6-8% t850, +5-7% q850. Robustness: from
    ERA5 + real analysis error, -12% z500 / -6% ws250 / -7% t850 at 24 h but only ~-1% by 120 h. Small-scale KE at
    120 h +8% (no blur). Reading: trained on half-noisy inputs it can't tell clean from noisy, so it "corrects" clean
    states too; the robustness gain fades by day 5. Gate (c) not run for NA.
GEFS pilot results (2026-11-08): download 25774716 (u1-service, 32 tasks, 8.8 min, 2046 samples, 361 GB uncompressed zarr v3;
zarr-v2 copy 25775378 for the A2 env, another 361 GB); smokes 25770681 (timeout: ufs2arco xr.merge bug, patched copy in
a37tmp/gefs/pylib), 25774457, 25774717; GPU smokes 25774760 (zarr-format clash), 25775362; evaluation 25775619-22 (H100, ~14 min each).
31 days x 10 member-minus-control pairs, fp32 no TF32. All-variable normalised (tot / large >400 km / small):
  6 h from lead 6:   corr 0.73 (0.86 / 0.63), |A2|/|GEFS| 0.76 (0.86 / 0.68); persistence corr 0.25; growth GEFS 1.15 vs A2 0.87
  6 h from lead 24:  corr 0.82 (0.93 / 0.72), ratio 0.86; growth GEFS 1.10 vs A2 0.94
  24 h from lead 6:  corr 0.51 (0.64 / 0.38), ratio 0.65; growth GEFS 1.56 vs A2 1.02 (z500 2.21 vs 1.60)
  A2 TL (eps 0.01) vs nonlinear A2: corr 0.92 (6 h), 0.87 (6 h from 24), 0.75 (24 h); TL/NL norm 1.03-1.13
  Worst: t850, 2t, stratosphere (100-50 hPa), q100; best: z (0.90 at 6 h).
Verdict vs predictions: corr 0.6-0.8 YES (0.73); ratio 0.8-1.3 NO (0.76: A2 under-grows); q/upper/small-scale mismatch YES;
24-h corr 0.4-0.6 YES (0.51); TL-NL >0.95 NO (0.92). Caveat: GEFS v12 has stochastic physics (SPPT/SKEB), so part of GEFS's
growth and decorrelation is stochastic forcing A2 cannot reproduce -- the correlation ceiling is < 1 and GEFS growth is an
upper bound. Figures figures/gefs_pilot/G1-G3; merged numbers output/gefs_pilot/pairs_merged.json.

## Pair-trained fine-tune: arms P and PU (2026-11-08, predictions before any run)
Data: operational GEFS v12, 00Z+12Z, control + 10 members, leads 0-30 h (all 6-h leads: a pair needs 3 consecutive leads,
input [L-6, L] -> target L+6, so f000/f006 -> f012 and f018/f024 -> f030), stored on A2's O96 grid and its 32 levels in
10-1000 hPa (t, u, v, q, z, w + sp, 2t, 2d, 10u, 10v) as float16 anomalies from ERA5 at the valid time. Months limited by
the ERA5 base (training zarr ends 2023-12): Oct 2020 - Dec 2021 and 2023 (never 2022). Held out for evaluation: 2023-01
(pilot) and 2023-07.
Arms (finetune/finetune_a2.py, same schedule as C0; outputs output/finetune/P, PU):
  P  = ERA5 loss + pair loss || [A2(x_m) - A2(x_c)] - [G_m - G_c](t+6h) ||^2 (output-normalised, same weights), weighted to
       ~20% of the total at the start.
  PU = P + growth penalty on the step-0 unphysical families (hyd_z_large/small, hyd_t_large, small_all), 24-h growth above
       1.5x the balanced growth (gamma from the J arm, 2.2).
Predictions (vs C0, held-out 2023-07 pair test, 6 h from lead 6; pilot C0-equivalent A2: corr 0.73, ratio 0.76):
  P:  pair corr 0.73 -> 0.80-0.85, ratio 0.76 -> 0.9-1.0, 24-h growth closer to GEFS (1.0 -> 1.2-1.4); adjoint battery:
      leading singular values unchanged or slightly UP (more growth), Taylor range unchanged; 2022 skill within +1%.
  PU: pair metrics as P; leading singular values down >=3x (unbalanced families capped) while balanced growth is kept
      (0.7-1.3x of P); 2022 skill within +1.5%.
  PITA (only arms passing the gates): 6-h raob gap +7-8% -> +5-6% for PU; P neutral.
  - NR (training 25765596, 7 h 31 min; evals 25770699-703; figures/finetune/FT1_skill.png, FT2_adjoint.png):
    (a) sv1 -38% (1.6x lower; range 128-203 vs 177-431), sv3 -38%, random 48-h amplification -44%, Taylor linear
    range along -g 3x (0.01 -> 0.03), steepest-descent increment growth -7% at 48 h (balanced growth kept).
    Below the 5x gate and the predicted 2-5x, but the clearest adjoint gain of any arm.
    (b) clean-start RMSE: z500 +8% at 6 h, +2.8% at 24 h, -0.8% at 120 h; ws250 -4%, t850 -2.6%, q850 -8.4% at
    120 h. From ERA5 + real analysis error: -9 to -13% at every lead to 120 h (NA's gain fades by 120 h).
    Small-scale KE at 120 h -19.7% (blur gate -20%: at the edge). Step-2000 checkpoint (25793738) is worse on clean
    skill (z500 +7% at 120 h) and less blurred (-6%): the 4-step phase both fixes skill and adds blur.
  - Gate verdict: no arm passes (a) as written (5x). NA fails (b). NR passes (b) at 120 h but is marginal at 6-24 h
    z500 (+3 to +8%) and at the blur limit. Gate (c) (twin OSSE + PITA) not launched: needs a decision.

### PT arm (2026-11-09): exact tangent-linear pair loss
PT = ERA5 loss + lam_p L_TL + lam_J L_J^TL, with L_TL = || M'(x_c)(x_m - x_c) - dG ||^2 (same weights, normaliser and
area weights as the P pair loss; M = the 6-h step, inputs at leads L-6, L) and L_J^TL = relu(||M'(x) v|| / ||v|| - gamma)^2
on the PU families (K = 1; --j_gamma 2.2 was calibrated on the 24-h growth, check the 6-h G in the log before a full
run). lam_p / lam_J use the P / PU auto-scaling. M' is a forward-mode JVP (pair_loss.jvp_step), so d/dtheta is
forward-over-reverse. What it needed: (1) block sliding-window attention instead of flex_attention (flex has no jvp rule;
same function), (2) identity jvp rules on anemoi's single-process `*ParallelSection` Functions, (3) a dual-aware
activation checkpoint (pair_loss._dual_checkpoint: the recompute in backward has no dual level, so the tangent is passed
explicitly), (4) fp32/TF32 off, math sdpa, save_on_cpu for the PT terms (--pt_offload). Needs
ANEMOI_INFERENCE_GRAPHTRANSFORMER_ATTENTION_BACKEND=pyg. Checks: a37tmp/pt_check.py (JVP vs central FD, weight-gradient
finite and non-zero, memory/time), a37tmp/pt_unit.py (checkpoint wrapper == plain). Results: see the status log.

### Gate (c) for NR (final checkpoint output/finetune/NR2/step_4000.pt), approved 2026-11-08 -- predictions first
1. Twin OSSE without B, 10 ot_s dates, strong constraint, AdamW 200 it, same synthetic obs; truth = NR's own twin
   (pita22nr_ot_s_*), plus the ERA5-truth arm (pita22nr_oe_s_*, model error included). Compared with A2's ot_s / oe_s.
   Predictions: ot_s 6-h analysis/background ratio 0.94-0.98 (A2 0.99); 24 h 0.78-0.84 (A2 0.82); 120 h 0.30-0.36
   (A2 0.33). Cosine of the 6-h increment with the true error 0.25-0.32 (A2 0.23); small-scale fraction 0.42-0.48
   (A2 0.50). oe_s 6-h ratio 1.00-1.04 (A2 1.05), 120 h 0.52-0.56 (A2 0.55).
2. 10 golden-config 2022 PITA windows with NR as the model (pita22nr_<mmdd>, fg12 from NR, real obs incl. ATMS/IASI).
   Scored vs golden (A2) on the same 10 windows, and each against its own model's ERA5-init forecast
   (nr_era5_22 for NR, a2_era5_22 for A2). Predictions: the 6-h raob gap to the own-model ERA5-init forecast
   shrinks from +7-8% to +4-6%; the 120-h raob gap from about +21% to +15-19%; NR-PITA vs golden on raob/ps forecasts
   -1 to -4% at 24-120 h (NR's own ERA5-init skill is about equal to A2's at 120 h, -1 to -3% better for wind/q);
   AMSU-A in window within +-1% of golden, -1 to -3% at 120-240 h.

## Learning to assimilate (2026-11-08; finetune/learn_assim.py, launcher a37tmp/l2a/run_l2a.sh)
User: "our goal is to train a good adjoint ... it needs to learn in a realistic setting ... teach the adjoint to be good
at recovering past ERA5 from future knowledge of ERA5" (a warm-PITA-like setting).
- Inner loop = a mini strong-constraint 4D-Var through A2 with the production control (latent increment u on step 1):
  background x_b = ERA5(t0) + resampled 2015 analysis error (AugResample, lag corr 0.27); obs = ERA5 outputs at
  t0+12..t0+24 h (the FUTURE only; the 6-h injection state is not observed), dense or a random --obs_frac of points,
  optional noise; J = A2's weighted output MSE on the masked points; K differentiable steps
  u <- u - s_k g/rms(g) (normalised gradient, learnable s_k, init 0.02 = rms of a golden increment).
- Outer loss: MSE_fcst (clean ERA5 -> +6 h, full weight, keeps skill) + beta * [E(x_1(u_K)) - E(x_1(0))] / E(x_1(0)).detach(),
  E = MSE vs ERA5 at the 6-h injection time. Denoising forward improvements lower both terms equally, so only a better
  gradient direction (adjoint) pays. beta auto: the L2A term = 30% of the total at step 0. Double backward (create_graph)
  through A2 needs the block sliding-window attention with SDPA forced to the math kernel during the inner loop.
- Arms: LA1 K=1, dense obs 12-24 h (pilot); LA2 K=2-4, sparse obs (5-10%), noise; LG gradient alignment
  1 - cos(-dJ/dx_b, x_true - x_b) w.r.t. the physical input state (one double backward, cheapest).
- Cheapest gate (held-out 2021, 12 samples, same backgrounds/obs for every model): r1_best = best-step one-step ratio
  E(x_1(-s g))/E(x_1(0)) over s in {0.005..0.08}; cos1 = cosine of that 6-h increment with the true 6-h error
  (normalised prognostic); rK, cosK after K_eval=4 normalised steps.
- Predictions (pilot LA1, 1500 outer steps x accum 4): A2 r1_best ~0.97-0.99, cos1 ~0.1-0.2 (dense future obs, 1 step);
  LA1 lowers r1_best by 0.01-0.03 and raises cos1 by 0.05-0.1; C0 = A2 within noise; clean 6-h val MSE within +1%.
- Later gates (if the pilot moves): twin/oe start ratio & increment cosine (A2: 0.99 / 0.23), adjoint battery and GEFS
  pair growth (keep physical growth), 2022 skill from ERA5 within +2%, then PITA windows.
