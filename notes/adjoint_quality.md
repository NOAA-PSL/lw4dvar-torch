# Adjoint-quality battery (`diagnostics/adjoint_quality/`)

A repeatable check of how well a differentiable forecast model's gradients serve latent-space 4D-Var. Built from the
2026-11-07 A2 investigation (notes/plan_20261105.md "ADJOINT QUALITY" and "SINGLE-OB BALANCE TEST"); run it on every
new model or fine-tune (e.g. the arms of notes/adjoint_finetune_plan.md) and compare against the A2 reference.

## What each part asks

All parts use the twin setting of the OSSE arms: background x0 = the config's first guess (`utils.get_input`, e.g.
A2's fg12 forecast); truth = the model's own forecast from the analysis at the window start (`get_input` with
`nhrs_back=0`, as `make_synthetic_obs` builds it), so the model is perfect and the gradient exact for it. The control
u is the latent increment injected on the first step, exactly as 4D-Var does. The norm everywhere is the normalised
prognostic norm: area-weighted (`model.area_weights`) mean over points and prognostic variables of
(difference x normaliser scale)^2. Cost J_L(u) = 0.5 x that norm^2 of (x_L(u) - truth_L), L = 24 h and 120 h.

Step sizes are relative to a reference latent norm |u_ref|: the norm of a 4D-Var increment for this model and date
(`--golden`, A2: the twin PITA run), or, without one, the norm at which the steepest-descent direction's 6-h response
equals A2's golden 6-h response (0.040 normalised units; `--target_resp6`).

| Part | Question | How | Output |
|---|---|---|---|
| `gradient` | Is the gradient noisy? Does reduced precision corrupt it? | cosines of fp32 vs bf16/fp16 gradients and of repeated gradients; gradient after moving u by 1e-3, 1e-2, 1e-1 \|u_ref\| (and at the golden increment) | `cos`, `stability` |
| `gradient` | How far does the gradient predict J? | Taylor test: (J(eps d) - J(0)) / (eps g.d) for d = steepest descent (fp32, reduced), random, golden direction, eps = 1e-4 .. 1 \|u_ref\|, fp32 forward | `taylor` (Q1) |
| `survival` | Does the model damp or amplify increments? | nonlinear fp32 runs of golden / real-obs / random / steepest-descent perturbations to 120 h; size relative to 6 h, split at ~400 km (kNN smoother) | `survival` (Q3) |
| `survival` | How much can it amplify a 6-h perturbation by 48 h? | randomized SVD (rank 24, oversampled, one power iteration) of the 6 h -> 48 h map in the normalised norm; JVP by central finite difference (fp32), VJP by autograd; also with z excluded; linearity and dot-product checks | `svd_48h_all/noz` (Q4) |
| `survival` | How much of the latent space does nothing? | 6-h response of random vs golden vs steepest-descent latent directions at equal norm | `response6h_at_equal_latent_norm` |
| `single_ob` | Does one mass ob give a balanced increment? | one z500 +30 m ob at 6 h (a) or 24 h (b), or ps -2 hPa at 6 h (c), at 50S 90E and 45N 180E; solvers G (raw adjoint direction, line search), BG (class-B-filtered adjoint, needs `--b_cov`), W (AdamW, config's golden settings, 200 it); 500-hPa wind increment vs the geostrophic wind of the z500 increment within 2000 km, small-scale and far-field (> 3000 km) fractions, vertical profiles | S1-S5, `single_ob_metrics.json` |

## Running it

Backends: the AIFS family (`model_backend: aifs`: A1 / A2 via `aifs37: true`, AIFS-single 1.x / 2.x) goes through
`adapter.AIFSAdapter`. fcn3 / sfno / ace2 / aurora raise `NotImplementedError` naming what an adapter needs (packed
layout, per-variable scale, precision path); add a subclass in `adapter.py`.

Precision: "fp32" disables every `torch.autocast` region (backend-agnostic) and, by default, TF32 (`--tf32` keeps it).
The reduced precision is the model's own autocast dtype (A2: bf16); models without one skip the precision comparison.

GPU (one MI300X, `anemoi-rocm` for A1/A2; submit from the repo root; `AQ_ENV` selects another env, see the header of
`run_aq.sh` for AIFS-single on H100):

```
G='output/pita22_ot_s_{mmdd}/*_latent_increment_120h_200it.pt'      # optional golden increments ({mmdd}, {date})
C='output/pita22_cv_s_{mmdd}/*_latent_increment_120h_200it.pt'
sbatch diagnostics/adjoint_quality/run_aq.sh all --config CFG --out output/aq/MODEL --golden "$G" --golden_cv "$C" [--smoke]
sbatch diagnostics/adjoint_quality/run_aq.sh single_ob --config CFG --out output/aq/MODEL_sob --loc sh nh \
       [--b_cov output/latent_eof_a2_q24_classes.npz] [--smoke]
```

CFG is any 4D-Var config for the model (the window's `sdate` is the date; A2 used a37tmp/adjoint/config_pita22_ot_s_MMDD.yml
for 0104, 0422, 0707, 1023). One JSON per date: `{tag}_{mmdd}.json` (parts merge into it), plus `_maps.npz` and
`_sv1_*.npz`; `meta` records schema, git hash, config, model id, precision, TF32 and all settings. Settings: defaults in
`battery.DEFAULTS`, `--settings YAML` and flags override. `--smoke` runs tiny settings end to end.

CPU (login node, `anemoi-dev` env for cartopy):

```
python -m diagnostics.adjoint_quality.aq plot --results output/aq/MODEL --single_ob output/aq/MODEL_sob --figs figures/aq/MODEL --label MODEL
python -m diagnostics.adjoint_quality.aq compare --models MODEL=output/aq/MODEL [OTHER=DIR ...] --figs figures/aq/compare
```

`compare` overlays the A2 reference bundle (`reference/a2_e065_pita22.json`, the 4 dates + single-ob metrics) unless
`--no_reference`; Q1-Q4 and S3 show the median and range over dates per model. `bundle` makes a new reference file.

Runtime (one date): A2 on MI300X `all` ~10 min (true fp32; ~7.5 min with `--tf32`), `single_ob` ~4 min per location
(3 obs x 3 solvers); AIFS-single 1-degree on H100 `all --smoke` ~6 min after a ~2-min model load.

Verified (2026-11-07): A2 0104 rerun with the tool (jobs 25750736 fp32, 25750737 `--tf32`, 25750738 single ob; smoke
25750262-63) reproduces the a37tmp results -- `--tf32` to 3 digits (singular values 238.7 / 217.3 / 210.2 vs 238.7 /
217.5 / 210.4; Taylor, survival, cosines identical), single-ob metrics identical to 2 digits. AIFS-single 1-degree
(aifs2-1.0deg-v1.0, lwaifs2 env, H100; a37tmp/adjoint/config_aq_aifs1deg_0104.yml) runs end to end in smoke mode
(25750281-82) through the calibrated |u_ref| path.

## Reading the figures

- **Q1 Taylor**: 1 = the gradient predicts the change of J exactly. With TF32 (the a37tmp runs, `--tf32`) eps < 2e-3
  is round-off and shaded; in true fp32 the ratio is ~1 down to 1e-4. The step at
  which the steepest-descent curve leaves ~1 is the usable step of a first-order optimizer.
- **Q2 cosines**: precision and repeat cosines near 1 = deterministic, precision-robust gradient; the stability
  points show how fast the gradient turns as u moves.
- **Q3 survival**: > 1 grows, < 1 damped; solid large scale, dotted small scale. Physical analysis-size errors grow
  (doubling ~1.5-2 days); strong damping would mean an excess null space, explosive growth of the steepest-descent
  perturbation means over-sensitivity.
- **Q4 singular values**: amplification of the most sensitive 6-h perturbations by 48 h; dotted lines = typical
  random ~400-km perturbations. The JSON `leading` entries give their variable / level make-up and small-scale share.
- **Q5 maps**: what the gradient asks for (steepest descent) vs what 4D-Var did (golden), t850 at 6 h.
- **S1/S2**: black = actual 500-hPa wind increment, red = geostrophic wind of the z500 increment (same arrow scale).
- **S3**: size ratio and pattern correlation (grey band: the background's own fields, the diagnostic's ceiling),
  small-scale and far-field fractions per solver.
- **S4**: global z500 increment from one ob (W vs G). **S5**: vertical profile at the ob (sign flips = unbalanced).

## A2 reference (epoch 65, PITA 2022 twin; jobs 25746139-42 and 25746804-07)

- Gradient: cos(bf16, fp32) 0.88-0.97; repeats identical; stability cos 0.97-1.00 up to 0.1 |golden|, but 0.06-0.33 at
  the golden increment itself. Precision and noise are not the problem.
- Taylor: along the golden direction 0.9-1.0 for 1e-2..0.3 |golden|; along steepest descent 0.6-0.9 at 1e-2, ~0.4 at
  0.03 and <= 0 from 0.1 |golden| -- the linear range is ~3% of a golden step. The golden increment is nearly
  orthogonal to the first gradient (cos -0.05..-0.16).
- Latent null space: a random latent direction's 6-h response is 15-18% of golden's at equal norm; steepest descent 1.6-1.9x.
- Survival: golden increments grow 1.7x by 24 h, 2.5-2.7x by 48 h, 5.8-7.5x by 120 h, including upscale growth; the
  steepest-descent perturbation grows fastest (z500 6-8x by 48 h). No damping.
- Singular values 6 h -> 48 h: 100-420 (all variables), 100-250 (no z), flat (60-90 at rank 24); random ~400-km
  perturbations amplify 4.5x (2.3x without z). Leading vectors > 85% below 400 km; input geopotential or T + msl
  (unbalanced), output upper-tropospheric v and q. FD linearity cos 0.78-0.94, dot test within 2-25%.
- Single ob (z500 +30 m at 6 h): wind / geostrophic ratio 0.35-0.60 (G), 0.52-0.66 (W), 0.54-0.89 (BG); pattern
  correlation 0.38-0.78 (G), 0.44-0.56 (W), 0.39-0.52 (BG), against 0.88-0.99 / 0.96-0.98 for the background itself.
  t, u, v flip sign every 100-200 hPa at the ob. AdamW spreads one ob globally: 54-71% of the z500 and 85-92% of the
  wind increment energy lies beyond 3000 km (zonal bands; G 8-45%). By 24 h the increments are near-balanced
  (ratio 0.7-0.95) but only 1-16 m of the 30 m remains at the ob.
- Reading: A2's adjoint is accurate and smooth, and A2 does not damp; it is hugely over-sensitive to unbalanced,
  small-scale inputs, which makes J badly curved along the gradient.

## Caveats

TF32: the backends switch TF32 matmuls on at load, so the a37tmp "fp32" runs (and the reference bundle) had
TF32. It explains the noisy small-eps Taylor points and the FD-JVP linearity of 0.78-0.94; in true fp32 (the tool's
default) linearity is 0.997-1.000 and the Taylor ratio is ~1 at small eps, with every conclusion unchanged (0104:
singular values 240.7 / 223.7 / 211.2, linear range along -g still ~3%, golden survival 2.66x at 48 h).
Twin only (perfect model); the normalised norm weights every level and z / q like the other variables; no physical
model's singular values for comparison; single-ob metrics use a 1-degree regrid; the calibrated |u_ref| (no golden run) makes
step sizes comparable across models only approximately.

## Files

`aq.py` (CLI), `battery.py` (gradient, survival), `single_ob.py`, `adapter.py` (backend adapter), `plots.py`
(Q1-Q5, S1-S5, compare), `run_aq.sh` (launcher), `reference/a2_e065_pita22.json` (A2 reference bundle). The original
one-off scripts stay in a37tmp/adjoint/.
