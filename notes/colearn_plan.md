# Co-learning a model correction with the PITA initial conditions (CL1)

Status 2026-11-06: step 0 passed, step A failed its gate twice (see Results); R1 not run. Evidence and running log: `notes/plan_20261105.md`
("Co-learning", "the three diagnostics", "strong results complete").

## Why

- Strong-constraint PITA 2022 (40 windows) is worse than weak (golden) from the window start
  (analysis vs ERA5 at 6 h: z500 +6%, q850 +7%) to day 5 (O-A raob t +18%, uv +27%), and its
  forecasts from the window end are worse by 6-10%. Not an optimizer limit (400 it: -2% O-A, plateau).
  Longer strong windows give better ICs (24 h +70% ... 120 h +6% vs golden), so obs do reach the
  start; the jumps of the weak run absorb model error that strong pushes into the IC.
- A2 has no systematic error vs ERA5 at 6-24 h (CV explained variance ~0), but drifts by day 5:
  upper-troposphere cooling (t300 -0.13/-0.15 K ocean/land), low-level moistening (q925 land
  +0.26 g/kg, tcw +0.25/+0.51 kg/m2), confirmed by raobs, ATMS and IASI water-vapour channels
  (output/colearn/bg_drift.txt, raob_drift.txt). Signature of too-weak moist convection.
- Literature: ERA5 itself is colder and moister than tropical sondes at 550-800 hPa and below
  950 hPa (Virman et al. 2021, Tellus A 73, doi 10.1080/16000870.2021.1929752), overproduces light
  precipitation with a too-early land diurnal peak, and has warm-night / cold-day 2 m biases. So a
  correction learned against ERA5 (step A) fixes A2's drift only; ERA5's own errors need obs (step B).

## The correction g_theta

- Physical space, applied after every 6-h step: x_{k+1} = M(x_k) + g_theta(x_k). Zero-initialised
  last layer, so g = 0 reproduces today's runs bit for bit.
- Column model, weights shared over all columns (like a physics parameterisation; small capacity so it
  cannot memorise a window): inputs per column t, q, u, v at 1000-200 hPa (A2 has no w input), sp, skt, tcw, lsm,
  sin/cos local time, sin/cos day of year, insolation; outputs increments of t and q at 1000-200 hPa
  For step B also 2t and 2d: the station check (2026-11-06) shows ERA5's 2 m T diurnal cycle is too weak
  (warm nights -0.5..-0.8 K, cold days +0.6..+0.9 K in the tropics / SH; A2 copies it), so step B adds a
  2 m T / q station operator (lapse-rate corrected, as forecast_skill.py) to the theta-gradient. MLP 2 x 128, ~25k params.
- Radiance decoders are anchored to ERA5 (trained on ERA5 states), so they read a move away from ERA5
  as error. The theta-gradient therefore uses only obs with physical operators: raobs, ps, surface
  stations. Radiances still inform each window's x0 as in golden. AMSU-A is never assimilated: it is
  the independent check.

## Steps and gates (pre-registered)

0. Wiring: g_theta in the rollout (forecast_model / backends/aifs), config key `exp.model_correction`.
   Gate: g = 0 gives the golden 0104 cost to 1e-6; gradient check vs finite differences.
A. Offline, against ERA5: 5-day rollouts from ERA5 ICs, 2016-2021 every 4 days, loss = area-weighted
   t, q misfit to ERA5 at every 6-h step (checkpointed rollout, as fit_window_nudges). ~3 GPU-h.
   Gate on 2022 (never trained on): drift at 120 h (t300, q925 land, tcw) cut by >= 50%; 5-day
   forecast skill vs ERA5 and vs raobs/stations not worse than 1%.
   Prediction: drift cut 50-80%; forecast skill change within +-2% (drift is small vs rms).
R1. 2022 PITA with g_A, 40 windows: strong and weak (golden config), plus forecasts and AMSU-A.
   Prediction: strong day-5 O-A gap to golden closes by 10-25% (from +18% to +14-16% raob t);
   weak changes < 2%; AMSU-A 120-240 h improves 1-3% for both.
B. Co-learning against obs: training windows in 2015 (obs archive, IASI/ATMS available; 2022 kept
   clean), ~64 windows, 8 per MI300X node (one window per GPU, theta gradient all-reduced).
   Joint Adam on {x0_w} and theta, start theta = theta_A, penalty lambda ||theta - theta_A||^2,
   strong-constraint 120 h windows, 200 its per batch, 3-5 passes over the windows. Validation:
   held-out 2015 windows each pass (stop when validation O-A stops improving).
   Gate on 2022: strong + g_B beats strong + g_A on day 4-5 O-A and AMSU-A (significant), IC vs ERA5
   at 6 h not worse, and ERA5-IC forecasts with g_B no worse than with g_A vs raobs/stations.
   Prediction: closes a further 10-30% of the strong-weak gap; the rest is chaotic error growth.
   Guard: if training windows improve > 2x more than 2022 windows, theta is memorising -> shrink it.
W. If B passes: weak + g_B with Q halved (fewer, smaller jumps): should match golden at the start
   and day 1-3 and beat it in forecasts from the window end.

## Cost

Step 0 + A: ~1-2 days of code, ~3 GPU-h training, ~10 GPU-h evaluation.
Step B: ~2-3 days of code (multi-window joint optimisation with torch.distributed), ~13 node-h training.

## Results (2026-11-06)
- Step 0 gate passed (g = 0 reproduces golden 0104 costs exactly; job 25676811).
- Step A FAILED its gate twice (g_A: loss over 46 normalised t,q outputs, q dominated, t300 drift got worse; g_A2:
  per-variable equal weights, drift cut 17-72% but t300 RMSE +2.8%). R1 not run. Details in plan_20261105.md.
- Drift is ~1% of 5-day MSE, so it cannot explain the strong-weak gap. Revisit B with an objective on state-dependent error.

## Longer term: radiances through a physical operator (user, 2026-11-06)
We have a PyTorch CRTM. The learned decoders' biases (vs CRTM) are how the ERA5 biases were found. Goal down the road:
use all-surface, all-sky radiances to learn model physics. Staging:
1. B now: theta-gradient from raobs, ps, stations (decoders stay on x0 only).
2. B+: add CRTM clear-sky radiances to the theta-gradient, starting with ocean temperature-sounding channels (AMSU-A/ATMS
   5-9 type: small CRTM bias). Per-channel constant (VarBC-like) offsets only, so the bias model cannot swallow
   state-dependent model error; raobs (and GNSS-RO if added) anchor it. Then the AMSU-A check needs a replacement
   independent check (held-out ATMS satellite, or held-out channels).
3. Surface-sensitive channels over land/ice: need emissivity (we have the z control) and skt; the station 2 m diurnal
   result says skt/2t diurnal cycle is exactly what they would constrain.
4. All-sky: needs hydrometeors (A2 has none; only q, tcw). Either g / a diagnostic head outputs cloud water/ice, or
   the operator diagnoses them from q, t. This is where moist-physics learning becomes possible.
CRTM: `~/work/atms/external/crtm-torch` (radianced repo, NOAA-PSL/radianced; float64 coefficient store in
`external/crtm-coeff/float64`): forward, Jacobian and VJP in PyTorch; Surface.ocean / Surface.land; Cloud class with
CloudCoeff. SpcCoeff present for amsua_n19 only, atms_npp, atms_n20, iasi_metop-a/b (2015 AMSU-A on MetOp/NOAA-15/18
needs more coefficients). radianced docs: plan_better_jacobians.md, jacobian_notes.md (decoder TL vs CRTM).

## Revised direction after B0 (2026-11-06, discussed with the user)
(3) "Model right almost everywhere, wrong where ERA5 is weak" (diurnal cycle, deep convection, surface): step B in
WEAK constraint (jumps keep the unpredictable error), g sparse and regime-gated (L1/size penalty; local time, land,
instability, tcw inputs; ERA5-IC forecast guard), objective = regime-conditional bias vs obs (stations by local time,
raob q 550-800 hPa, later CRTM land surface channels, all-sky for convection). Gate on 2015: land-station 2t bias by local
time down >= 30%, raob q 550-800 hPa bias down, window O-A not worse. B0's MSE skill is the wrong metric for this.
Adjoint hypothesis (user): A2 never learned a useful TL/adjoint (off-manifold Jacobian unconstrained). Evidence: 400-it
strong lowers cost but worsens x0 vs ERA5; ill-conditioned nudge fit. Decisive test: synthetic raob/ps at real 2022
locations, strong 4D-Var on 10 windows, (a) identical twin (truth = A2 from ERA5) vs (b) ERA5 truth; optional (c) another
backend as truth. Twin error large -> gradient path is the bottleneck -> adjoint-aware co-learning of A2 (perturbed-IC
training, Jacobian regularisation, or learning-to-assimilate unrolls with LoRA). Prediction: twin removes ~half of the
x0 gap.
Extended jump check (tendency, neighbours, jump variance): jump_predict2.py, jobs 25698714 -> 25698715.
