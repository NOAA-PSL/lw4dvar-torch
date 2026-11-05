# Weak-constraint 4D-Var and estimating B and Q

Goal: better forecasts *beyond* the window. A strong-constraint (SC) 5-day window starts its
forecasts from the window end, the worst-fitted point of its trajectory. For accept arm A
(`config_lwatms_1mo_accept_a.yml`, the reference run) z500 vs ERA5 is about 5.2 m at +24 h
into the window and 8.2 m at the end (+120 h). Model-error jumps let the trajectory stay close
to the obs through the window end.

## What exists (uncommitted, `feature/lwatms`)

- **`exp.weak_constraint`** (`long_window_4dvar_utils.py`):
  - `every_hours: 24` injects a latent increment η_k on the step h → h+6 at h = 24, 48, 72, 96.
    It is injected the same way as the window-start increment δ, and honors
    `control_variables`.
  - Helpers: `_resolve_weak_constraint`, `_wc_step`, `advance_with_jumps`.
  - The jumps are an extra AdamW group with `learn_rate_factor` × lr and their own
    `weight_decay`.
  - The analysis trajectory, the cycling advance and the window-end state all include the
    jumps. The end state is saved as `*_wcend_inputs_*.pkl`, and `forecast_skill.py` prefers it.
- **Cascade proxy, no code changes**: re-solve short windows from A's trajectory.
  - `config_lwatms_wc1_a.yml`: 1 day, 96–120 h.
  - `config_lwatms_wc2_a.yml`: 2 days, 72–120 h.
  - `config_lwatms_wc21_a.yml`: 2 days, then 1 day; the last day is solved 3 times.
  - Run with `run_lwatms_wc1_a.sh` and `run_lwatms_wc21_a.sh`.
- **Sizing B and Q in control space** with `fit_latent_to_era5.py`:
  - For each sample it fits the one-step latent δ to ERA5; each fit removes 95–96% of the
    one-step misfit.
  - **B ≈ 0.24 latent RMS**: 20S–90S 0.23, tropics 0.27, 20N–90N 0.21; nearly isotropic across
    channels.
  - **Q24 ≈ 0.185**; Q/B ≈ 0.77.
  - `estimate_latent_errors.py` (encoder differences) is the wrong metric. The decoder adds
    its output onto the input's last level, so the increment acts on the 6-h tendency, not
    the state.

## Results so far

- **Cascade forecasts**: 15 dates, Jan 6–20; the extension to 30 dates is in
  `figures/forecast_skill_cascade30`. z500 RMSE vs ERA5 at 120 h:

  | init from | z500 at 120 h (m) |
  |---|---|
  | A | 54.6 |
  | A + 1-day | 45.5 |
  | A + 2-day | 43.6 |
  | A + 2-day + 1-day | 42.0 |
  | ERA5 | 28.9 |

  About half of the gap to ERA5 is closed, and every variable and lead improves.
- **WC 1-month runs**, `config_lwatms_1mo_wc24_q{075,025}.yml`:
  - First window: global z500 at the window end is 9.85 m (A), 7.81 m (q025), 7.17 m (q075).
  - Forecasts, 30 dates (Jan 6 – Feb 4), z500 global at 120 h: A 54.3, q025 48.1, q075 46.3,
    cascade 2-day + 1-day 43.2, ERA5 27.4 m (`figures/forecast_skill_cascade30`; the
    observing-system ladder is in `figures/forecast_skill_summary`).
  - Weak-constraint cycling beats A at every lead and for every variable. It is still about
    3 m behind the cascade at 120 h; in the NH the gap is about 3.5 m.

- **control_prior smoke test** (`run_lwatms_prior_smoke.sh`, first cycle 2015-01-01T00,
  σ_b 0.24, σ_q 0.185, lr 8.3e-3 shared by both groups):

  | arm | loss | Jb | Jq | Jo | δ RMS | jump RMS at 24 / 48 / 72 / 96 h |
  |---|---|---|---|---|---|---|
  | A (weight decay) | 1.93 M | — | — | 1.93 M | 0.0195 | — |
  | SC + prior, 100 it | 2.12 M | 122k | — | 2.00 M | — | — |
  | q075 (weight decay) | 1.57 M | — | — | 1.57 M | — | 0.023 / 0.027 / 0.032 / 0.038 |
  | WC + prior, 100 it | 1.96 M | 77k | 96k | 1.79 M | 0.020 | 0.011 / 0.009 / 0.008 / 0.005 |
  | WC + prior, 300 it | 1.88 M | 103k | 90k | 1.69 M | 0.023 | 0.011 / 0.009 / 0.007 / 0.005 |

  - The prior binds: under it, the jumps come out half the size of q075's, and they shrink
    with lead instead of growing.
  - Under a diagonal σ_q = 0.185 prior, q075's jumps would cost Σv² ≈ 4 M, more than its whole
    Jo, so the obs cannot "pay" for them.
  - J is still falling between 100 and 300 iterations (about 4%), so the solution is not
    converged (step 2).
  - Likely cause: a diagonal (white) latent prior charges a smooth, coherent jump the same as
    noise of the same norm. Real B and Q are spatially and channel-correlated, so a prior
    built from the ERA5-fit samples (EOFs / a low-rank plus diagonal form) would make
    realistic jumps cheap. This moves step 4 (structure) ahead of the σ tuning.

## Problem: the B/Q "prior" is not a prior

With AdamW the penalty is weight decay. Adam's step is about lr per element, so lr ×
iterations sets the size of δ and η, not the decay. As a result:

- the jumps come out at RMS 0.02–0.04, while ERA5 says the model error is about 0.185;
- "Q/B 0.75" is a learning-rate ratio, not a variance ratio;
- 100 iterations with a cosine schedule are an early-stopping regularizer. The solution is
  not the minimum of a stated cost.

So σ_b and σ_q cannot be estimated by likelihood until the cost carries them explicitly.

## Sampling B and Q from ERA5 fits: size is ill-posed, structure is not

The one-step inversion onto ERA5 (`fit_latent_to_era5.py`) is ill-conditioned: the last few
percent of the misfit cost a much larger δ. One date (2015-01-07), B / Q24:

| fit | misfit removed | δ RMS B | δ RMS Q24 |
|---|---|---|---|
| Adam, lr 0.002, 500 it | 93% | 0.188 | 0.148 |
| Adam, lr 0.01, 200 it (used for σ_b = 0.24) | 96% | 0.238 | 0.186 |
| L-BFGS 60 it (`--optimizer lbfgs`, scaled control) | 97% | 0.359 | 0.279 |
| L-BFGS 150 it | 99% | 0.484 | 0.404 |

- The overall size of B and Q depends on where the fit stops; Q/B stays about 0.78–0.84.
- The structure is robust:
  - the leading 50–200 channel EOFs overlap 0.76–0.81 between optimizers (random: 0.05–0.20);
  - about 124 EOFs carry half the variance;
  - B and Q24 share most of it (overlap 0.74).
- Spatial correlation of the EOF amplitudes is short (about 0.1 by 300–400 km; mesh spacing is
  about 110 km), except the leading mode (0.3 at 1000 km).
- Hence: take the structure from samples, and estimate only the scale (σ_b, σ_q) by likelihood
  against the obs (step 3).
- `control_prior.b_cov` / `q_cov` take `latent_eof.py` output, normalized to unit mean
  variance, so σ stays the per-element RMS.
- Samples: `run_fit_latent_samples.sh` (Jan 7–31, L-BFGS 60, saved to
  `output/latent_samples_accept_a_lbfgs60`). The cached lagged ERA5 starts Jan 6.
- `latent_eof.py --holdout N` scores held-out samples by Gaussian log likelihood under the
  diagonal, channel-C and per-band channel-C priors.
- **25 dates (Jan 7–31), L-BFGS 60** → `output/latent_eof_{B,Q24}_accept_a.npz`:
  - Per-element RMS: B 0.36, Q24 0.27 (Q/B 0.76).
  - Half the variance takes 132 EOFs for B and 156 for Q24.
  - Spatial correlation:
    - B's leading mode is large-scale: 0.41 at 1000 km, still 0.29 at 10,000 km;
    - B's other modes, and all of Q24's, fall to about 0.1 by 300–600 km.
  - Held-out log likelihood (20 training / 5 test dates), nats per element:

    | prior | B | Q24 |
    |---|---|---|
    | diagonal | −0.413 | −0.143 |
    | channel C | −0.145 (+0.27) | 0.091 (+0.23) |
    | channel C per latitude band | −0.089 (+0.32) | 0.145 (+0.29) |

  - Channel correlation is most of the gain; latitude bands add about 0.05.
  - λmax/λmin ≈ 6e8: the near-null channel directions get almost no prior variance.
  - Test run: `run_lwatms_cprior_smoke.sh` (first cycle, SC and WC, σ_b 0.24, σ_q 0.185, the
    same σ as the diagonal-prior arms).

## Plan

1. **Explicit Gaussian prior (`exp.control_prior`)**: optimize unit-variance controls.
   - Substitution: δ = σ_b·u, η_k = σ_q·v_k (σ in injected latent units, as from
     `fit_latent_to_era5.py`).
   - Cost: J = Jo + Σu² + Σv² (same factor-2 convention as Jo = Σ r²); no weight decay.
   - With one lr for both groups, the physical step ratio is σ_q/σ_b. Log Jb, Jq, Jo separately.
   - Verify:
     - 1 cycle × 100 iterations reproduces A's loss when lr_u ≈ lr/σ_b (δ dynamics unchanged);
     - Jb is a sensible share of J;
     - a longer run (300 iterations) shows whether the prior binds before the iterations run out.
2. **Convergence**: a likelihood is only meaningful at (near) the minimum. Find the iteration
   count and lr schedule at which J, δ and η stop changing. If 100 iterations is far from the
   minimum, the estimates in step 3 measure the optimizer, not σ.
3. **Estimate σ_b, σ_q (and σ_o) from the obs**, cheapest first:
   - a. **χ² / Desroziers consistency**: with J = Σr² + Σu² + Σv², E[J_min] ≈ p, the number of
     used obs (quadratic-region obs only; Huber terms are not Gaussian). Per-term rescaling uses
     the Desroziers ratios.
   - b. **Predictive (held-out) log likelihood**:
     - score the next cycle's O−B under N(0, R) and pick σ_q (and σ_b) to maximize it;
     - sweep 0.5×, 1× and 2× the ERA5 values over about 10 cycles;
     - closest to the forecast goal; costs one short run per value.
   - c. **Full marginal likelihood** (Dee 1995): −log p(d) = ½ dᵀS⁻¹d + ½ log det S, with
     S = H(MBMᵀ + Q)Hᵀ + R through the window. Needs randomized log-det / trace estimates
     in a 40320 × 1024 control space. Do this only if a and b disagree.
4. **Structure**: latitude-dependent σ (tropics vs extratropics differ by about 30% in B) and
   a per-channel σ, if the likelihood supports it.
5. **Cycling**: the best explicit-prior WC configuration over a month, with the forecast-skill
   comparison against A and the cascade.

## Scoring σ, warm-started jumps and the sweep (2026-10-19)

- **The next-cycle background is the wrong score for σ under WC.** It is a free run from the
  window start with no jumps, so it degrades after 48 h (sp Jo per slot at 120 h: A 8.9k,
  q075 15.4k), while the WC analysis fit stays flat (5.6k vs A's 6.8k). The held-out score is
  forecasts from the window end against unseen obs: the 6–24 h leads of `forecast_skill.py`.
- **`weak_constraint.warm_start`** (needs `control_prior`):
  - the jumps' prior mean and first guess are the previous cycle's unit controls v, moved to
    the same valid time and linearly interpolated (12 h cycle, 24 h jumps:
    new 24 h = ½(old 24 + old 48), …, new 96 h = ½ old 96, with 0 one spacing past the last);
  - Jq = Σ(v − v_prev)², with the same σ_q;
  - the controls are in `exp['wc_state']` and in `*_wc_forcing_*.pt` (`controls`), so a
    restart picks them up;
  - the saved background trajectory includes the prior jumps, so O−B scores the real first
    guess.
- **Sweep** (`config_lwatms_wcsweep_*.yml`, 10 cycles from 2015-01-01T00, channel prior,
  lr = 2e-3/σ_b so the physical increment step is A's), jobs 24093286–91:

  | arm | σ_b / σ_q | iterations | warm start |
  |---|---|---|---|
  | sx1 | 0.24 / 0.185 | 100 | no |
  | sx15 | 0.36 / 0.278 | 100 | no |
  | sx2 | 0.48 / 0.37 | 100 | no |
  | sx15_200 | 0.36 / 0.278 | 200 | no |
  | ws_sx1 | 0.24 / 0.185 | 100 | yes |
  | ws_sx15 | 0.36 / 0.278 | 100 | yes |
  | diag_sx1 (diagonal prior, job 24093309) | 0.24 / 0.185 | 100 | no |

- **Sweep scores** (2026-10-20, `forecast_skill.py`, job 24117496, labels `sw_<arm>`).
  - Inits: 5 daily 00Z dates, Jan 6–10, the window ends of cycles 1, 3, 5, 7 and 9.
  - Scores: Global RMS(O − F) pooled over 6–24 h and over day 5 (114–120 h), plus z500 RMSE vs ERA5.
  - A and q075 (`lwatms_1mo_*`) are scored on the same dates.

  | run | ps 6–24 h (hPa) | raob t | raob q | raob uv | ps day 5 | raob uv day 5 | z500 6–24 h (m) | z500 day 5 |
  |---|---|---|---|---|---|---|---|---|
  | A (strong) | 1.507 | 1.554 | 0.783 | 5.300 | 4.726 | 10.58 | 10.99 | 47.4 |
  | q075 (no explicit prior) | 1.253 | 1.387 | 0.746 | 4.776 | 3.806 | 9.22 | 8.54 | 39.8 |
  | sx1 | 1.306 | 1.443 | 0.759 | 4.916 | 4.316 | 9.60 | 9.19 | 42.9 |
  | sx15 | 1.265 | 1.422 | 0.752 | 4.836 | 4.057 | 9.34 | 9.01 | 42.8 |
  | sx2 | 1.280 | 1.417 | 0.748 | 4.861 | 3.987 | 9.55 | 8.98 | 42.3 |
  | sx15_200 | 1.232 | 1.392 | 0.745 | 4.781 | 4.054 | 9.42 | 8.55 | 40.2 |
  | ws_sx1 | 1.290 | 1.434 | 0.759 | 4.907 | 3.760 | 9.23 | 9.19 | 42.5 |
  | ws_sx15 | 1.321 | 1.410 | 0.754 | 4.845 | 4.000 | 9.24 | 8.93 | 41.2 |
  | diag_sx1 | 1.311 | 1.475 | 0.766 | 5.010 | 3.906 | 9.74 | 9.71 | 43.6 |

  - Every WC arm beats A: 6–24 h ps by 12–18%, day-5 z500 by 8–15%.
  - The ERA5 σ is too tight. 1.5–2× σ beats 1×, and the gap between them is near the noise.
  - Iterations matter more than σ. sx15 at 200 iterations matches q075, and 100 iterations does
    not converge with the explicit prior. This is the open convergence check (item 2).
  - Warm start does not help at 6–24 h but helps day 5 (ws_sx1 ps 3.76 vs 4.32). One sample is
    too few to call it.
  - The channel prior beats the diagonal one (z500 6–24 h 9.19 vs 9.71 m).
  - Caveat: 5 dates and one realization each, early in cycling. Analysis run-to-run noise is
    about 0.4 m z500.
  - Decision (2026-10-20): the explicit prior is not worth its cost.
    - At 200 iterations, twice the GPU time, it only ties q075 (100 iterations, no explicit
      prior). At 100 iterations it is worse.
    - q075 stays the WC baseline.
    - The month-long sx15 + warm start run at 200 iterations is dropped.
    - The Q question moves to the noise-space control of a stochastic AIFS, in the eagle
      `long-window` training plan.

## Status

- [x] weak_constraint implementation, smoke test, 1-month q075/q025 runs
- [x] cascade proxy, 15 dates (30-date extension running)
- [x] 1. control_prior: implemented; smoke test done (results above)
- [ ] 2. convergence check
- [ ] 3a/3b/3c
- [ ] 4, 5

_Run outputs of the superseded experiments (3mo_ps_atms, early 1mo ATMS variants, wcsweep/wc*_a, wc24_q025, win4d-10d, prior/cprior, early 10-cycle raob tests, aborted 6mo non-tsens runs) were deleted on 2026-10-21; their scores remain in this note, figures/ and output/forecast_skill/._
