# Adding GPS radio-occultation (GPSRO) bending-angle observations

Read this when working on: the GPSRO observation stream, the 1D
Abel-transform bending-angle forward operator, the `exp.observations.gpsro`
config schema, or anything else GPSRO-specific.

## Status: implemented and offline-validated (2026-10-20); no GPU smoke test yet

## 2026-10-20: GSI comparison -- three GSI-ported QC pieces added; thinning removed, then restored

A real GPU run (`ps+raobs+amv+gpsro`, aifs2-1.0deg-v1.0, 2015-01) showed
z500 forecast skill still clearly worse than `ps+raobs+amv` alone even
after the AMV box-thinning/gross-QC fix (see integrate_amv_satwnd.md) --
with AMV's own distortion removed, GPSRO looked like it had a real problem
of its own. Comparing this port against GSI's actual GPSRO code
(NOAA-EMC/GSI, commit `55faa5e7af4f0604bd1ed38b385fd762863ca294`,
`read_gps.f90`/`setupbend.f90`) found:

- **GSI does not thin GPSRO levels at all** -- confirmed directly (zero
  `ithin`/`thin` occurrences in `read_gps.f90`). This repo's greedy 1 km
  vertical thinning (`impact_height_spacing_m`) was a JAX-reference design
  choice, not something GSI does. **Tried removing it -- then restored it
  the same day** once a real run showed why: see "Thinning restored"
  below.
- **GSI reaches a fixed 55/45 km impact-height ceiling by extending its
  own discrete grid** above the model top (`nsig_ext`, a power-law
  refractivity continuation), rather than rejecting any tangent point
  above the model's native top outright. This repo's own closed-form
  operator needed NO math change to do the equivalent -- confirmed
  directly that `exponential_top_bending_angle` already extrapolates
  correctly for an impact height above `refractional_height[..., -1]`
  (a synthetic check: bending angle stays finite and decreases smoothly
  from `top+0km` to `top+50km`). Only the validity mask
  (`GPSRO_COLUMN_TANGENT_UNSUPPORTED`) was actually rejecting these --
  **added `top_extrapolation_km` (default 10 km)** to widen it. This
  directly addresses the real run's own diagnostics (checked in a prior
  turn): essentially 0% of levels between 32-40 km impact height were
  used, ~17% of the entire archive, purely from this rejection.
- **GSI's background QC for GPSRO is the same clamped-ratio form used for
  conventional/SATWND obs** (`ratio=|O-B|/clip(obserr,ermin,ermax)>gross`,
  `setupbend.f90` lines ~796-812) -- this repo's `background_check_stddev`
  was a plain, unclamped `stddev * error` multiplier. **Added
  `error_min_rad`/`error_max_rad`** (optional, default no clamping) to
  adopt GSI's form when wanted.
- **GSI has an explicit super-refraction (SR) detector** near the surface
  (`setupbend.f90`'s `crit_grad=157.0` N-units/km critical gradient,
  `alt<=5km` screen) that this repo had no equivalent for. **Added a
  simplified port** (`GPSRO_COLUMN_SUPER_REFRACTION`,
  `critical_gradient_n_per_km`/`sr_detection_fraction`/
  `sr_screen_height_km`, defaulting to GSI's own values) -- reject-only,
  no ray relocation above a detected SR layer (GSI's fuller behavior);
  see "Simplifications" below.

GSI's `ddnj<0` jacobian check and its full altitude/latitude/season
statistical cutoff were deliberately NOT ported -- the former is specific
to GSI's own numerical-quadrature forward operator (this repo's
closed-form design handles the analogous condition differently, via
`increasing_layer_bending_angle`, not by rejecting), and the latter's
numeric coefficients are tuned to GSI's own background/operator error
characteristics, not verified for this system -- see "GSI comparison"
discussion in conversation history for the full reasoning.

**Offline-validated** (no GPU; see "Validation" below for the specific
checks): top-extrapolation confirmed to raise the used fraction from 30.5%
to 51.5% on a synthetic uniform background column; SR detector confirmed
to fire on a synthetic spiked-moisture column and stay silent on a clean
one; all three new validity kwargs confirmed backward-compatible (defaults
reproduce prior behavior when left off).

### Thinning restored (same day)

Removing `impact_height_spacing_m` was tried first, matching GSI's own
practice literally. A real GPU run of the SAME config
(`ps+raobs+amv+gpsro`) with the untthinned reader showed why that doesn't
transfer: GSI doesn't need to thin GPSRO because its own forecast models
carry 60-150+ vertical levels, so a dense, unthinned profile still has
real degrees of freedom to fit against. This repo's model backends
(AIFS-single-2.0: 14 levels, top 10 hPa) don't -- a profile's simulated
bending angle is a smooth function of only ~14 column values regardless of
how many observed levels are fit against it, so every additional kept
level beyond that is mostly redundant, cross-correlated signal, not new
information, and this repo's `ecmwf_seed_v1` error model has no vertical
error-correlation term to down-weight that redundancy (unlike a real R
matrix would).

Confirmed directly by comparing two real runs' own log files (same
config, `logs/lw4dvar_aifs2_1p0deg_v1p0_ps_raobs_amv_gpsro_24122922.out`
[thinned] vs. `..._24193904.out` [unthinned]):

- Window-total level count: 212,640 (thinned, ~30.4 levels/profile) vs.
  1,113,755 (unthinned, ~159.2 levels/profile) -- a 5.2x increase, for the
  SAME 6,997 profiles.
- `gpsro`'s own `Jo` contribution per verification slot (epoch 90, same
  cycle, same slots): ~16,800-17,600 (thinned) vs. ~94,700-122,600
  (unthinned) -- a ~6-7x increase, consistent with the
  (levels-per-profile / 14) over-weighting argument above. In the
  unthinned run, `gpsro`'s `Jo` became the single largest observation
  term in nearly every slot, 2-4x bigger than raobs and 2-3x bigger than
  the already-fixed amv term -- the same symptom (one stream's sheer
  term-count dominating a simple summed `Jo = sum (innov/err)^2`) as the
  original AMV bug, just via vertical rather than horizontal
  over-sampling.

Restored `impact_height_spacing_m` (default 1000 m, unchanged from the
original design) -- confirmed back to mean 30.4/max 32 levels per profile
on the real archive, matching the original pre-removal numbers almost
exactly. The other three GSI-ported pieces (clamped gross check, SR
screen, model-top extension) were kept, since none of them multiply the
per-profile level count the way removing thinning did -- model-top
extension only reclaims levels within the already-configured
`impact_height_range_km` window, it doesn't add density within a profile.

A real GPU rerun with thinning restored + the three new QC pieces is the
natural next step, to confirm this combination is actually a net
improvement over `ps+raobs+amv` alone.

## Reference precedent

Ported from the `long-window-4dvar` (JAX/NeuralGCM) reference repo
(`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/greg-long-window-4dvar/long-window-4dvar`,
`sc_gpsro_goes16` branch), which already assimilates GPSRO bending angle
from the same kind of Hive-partitioned candidate-bending-angle parquet
archive this repo's `gpsro_parquet` directory holds. Two files there did
the real work:

- `gpsro_parquet.py` -- canonical-profile selection (filtering to
  `is_canonical_neutral_bending_angle`), read-time QC, and GREEDY vertical
  thinning (keep one level per `impact_height_spacing_m`, walking
  bottom-to-top).
- `gpsro_operator.py` -- a differentiable, closed-form 1D Abel-transform
  bending-angle simulator (exponential-refractivity layers, ROM SAF ROPP
  12 forward-model formulation), plus a frozen model-column validity
  predicate.

This port covers both, trimmed of the reference's JAX-retracing-avoidance
machinery (fixed capacities had to be pre-declared and manifest-cached
there; PyTorch has no such requirement -- same simplification already made
for raobs/amv) and its contract/provenance-validation ceremony (schema/
counts/clean-git-revision checks -- a converter-side testing concern,
already covered by that converter's own test suite).

## Reader (`gpsro_parquet.py`)

`load_gpsro_profiles` returns one ragged list of profile dicts per
verification-time slot (same per-slot-list convention as
`load_psobs_parquet`/`load_raobs_parquet`/`amv_parquet.load_amv_parquet`)
-- `get_gpsro` (in `long_window_4dvar_utils.py`) does the model-dependent
fixed-shape padding and horizontal-interpolation stencil, same division of
labor as raobs/amv.

Canonical selection + QC + thinning (`_canonical_profiles_from_file`) is a
close port of the reference's own algorithm (kept, despite GSI not
thinning GPSRO itself -- see "GSI comparison" / "Thinning restored"
above, the 14-level-model reason this repo still needs it):
1. Filter to `is_canonical_neutral_bending_angle = True` (the zero-
   frequency, ionosphere-corrected neutral product, confirmed also present
   in this repo's own archive schema).
2. Per-level QC bits: non-finite/non-positive/too-large bending angle,
   invalid geometry (lat/lon/impact-parameter/curvature/undulation out of
   physical range), impact height outside the configured window, duplicate
   impact parameters, optional source QFRO-flag and producer-confidence
   checks.
3. Greedy vertical thinning: sort by impact parameter (ascending height),
   walk bottom-to-top, keep a level only if its impact height is at least
   `impact_height_spacing_m` above the last KEPT level.
4. Assign each surviving profile to its nearest verification time within
   `time_tolerance_hours`.

**Offline-validated against the real archive** (2015-01-01 00Z,
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/prepbufr_to_parquet/data/gpsro_parquet`):
1,427,474 canonical candidate rows read -> 302 profiles / 9,167 levels
retained after QC/thinning, mean 30.4/max 32 levels per profile (one level
roughly every km over the configured 6-40 km window, as expected --
matches the original pre-removal numbers almost exactly). Impact heights
land exactly inside the configured 6-40 km window; bending angles range
~3e-5 to 2e-2 rad (textbook GPSRO magnitudes).

**Error model**: only `ecmwf_seed_v1` (the reference's "transparent
fractional-plus-floor seed" -- 20% of the bending angle near the surface,
tapering to 1% by 10 km, floored at 3e-6 rad) is implemented. The
reference's second option, `height_dependent_table` (a reviewed,
Desroziers-diagnosed impact-height-vs-error table), is NOT ported --
there is no such table for this repo's backends yet, and raobs's own
`q_error_g_kg` precedent shows that kind of table should come from a real
diagnosed run, not be guessed up front.

## Forward operator (`gpsro_operator.py`)

A line-by-line PyTorch port of the reference's JAX `gpsro_operator.py`:
`neutral_refractivity` (Smith-Weintraub formula), `geopotential_to_geometric_height_m`
(WGS-84 latitude-dependent gravity/radius), `refractional_height_m`
(`nr - radius_base`, centered to avoid subtracting two ~6.4e6 m values),
and the Abel integral itself -- `exponential_layer_bending_angle` (closed-
form solution for an exponentially-decreasing-refractivity layer, via the
scaled complementary error function) and `increasing_layer_bending_angle`
(a separate closed-form branch for layers where refractivity increases
with height, where the exponential-decay solution is ill-posed), summed
over layers plus an analytic extrapolation above the model top
(`exponential_top_bending_angle`).

**Two deliberate simplifications relative to the JAX port, both because
PyTorch already has what the reference had to build by hand**:
- The reference implements its own `erfcx` (scaled complementary error
  function) in two branches, specifically because `jax.scipy.special` has
  no native primitive for it. `torch.special.erfcx` exists natively (a
  properly numerically stable implementation) -- every call site uses it
  directly instead of porting that workaround.
- The reference flattens every leading batch dimension and uses
  `jax.vmap` over individual columns/layers, because that is how JAX
  vectorizes a scalar-signature function. PyTorch broadcasts natively, so
  this port keeps the layer axis as an ordinary trailing tensor dimension
  and reduces with `.sum(dim=-1)` -- no flatten/vmap/reshape machinery
  anywhere.

**Level-order fix required for this repo**: the JAX reference's NeuralGCM
convention decodes model columns top-to-bottom (model top first, surface
last) and flips to bottom-to-top before the Abel integral. This repo's
own backends decode the OPPOSITE way -- confirmed directly in
`backends/aifs/aifs_model.py`'s `_levels_by_base` construction:
`sort(key=..., reverse=True)  # high pressure (surface) first`, i.e.
bottom-to-top already. The port's functions therefore take bottom-to-top
input directly (matching `decoded[base]`'s native order) with no internal
flip, documented explicitly in `gpsro_operator.py`'s docstrings -- getting
this backwards would silently integrate the ray from the wrong end.

**NaN-gradient-contamination fix, ported directly and confirmed necessary**:
`simulate_bending_angles` substitutes a deliberately valid, state-
independent standard atmosphere/ray into EVERY masked-out slot BEFORE
evaluating the Abel integral, not just after (masking only the result is
too late -- a rejected ray's genuinely invalid geometry, e.g. reversing
refractional height near the model top, can produce a finite forward
value but a non-finite local gradient, and reverse-mode autodiff can still
encounter `0 * NaN` and contaminate the whole loss's gradient). Verified
directly in this port (see "Validation" below) that this actually matters:
without the substitution, gradients through masked observations go NaN.

### Validation

1. **Exact cross-check against the reference's live JAX implementation.**
   Built one realistic synthetic ISA-like atmospheric column (hydrostatic
   geopotential from a simple lapse-rate temperature/humidity profile, 14
   levels matching AIFS's set) and ran it through BOTH this port (as-is,
   bottom-to-top) and the reference's own `gpsro_operator.py` (same
   column, reversed to its expected top-to-bottom order) in the `jcm`
   conda env, which has JAX installed, for 7 impact heights spanning
   2-30 km. Bending angles agreed to **~2e-7 relative difference**
   (float32-precision-level noise) across every impact height; validity
   flags matched exactly, including the one deliberately out-of-range
   case (impact height below the surface's own refractional height).
2. **Gradient finiteness through masked/invalid observations.** With a
   batch of 5 impact heights including two clearly unsupported ones
   (60 km, above any reasonable model top; -500 m, nonsensical),
   `torch.isfinite` on every input gradient after `.backward()` through
   `simulate_bending_angles` was confirmed True -- the safe-substitution
   design works as intended.
3. **Full `get_gpsro` -> forward-operator pipeline, real archive data +
   a fake model/grid (no AIFS)**: correct tensor shapes
   `(n_obs, profile_capacity, level_capacity)`, no NaN propagation with
   unrealistic random thermodynamic fields (correctly rejects nearly
   everything instead of producing garbage), and -- with a physically
   realistic uniform ISA-like background column -- ~31% of levels passed
   every QC gate with sensible O-B statistics (background and observed
   bending angle in the same 1e-4 to 1e-2 rad range, mean |O-B| ~2.7e-4
   rad). The partial (not near-100%) accept rate here is expected: the
   test background was ONE global column, not a real spatially-varying
   analysis, so many real profiles' own geography/impact-height combinations
   fall outside what that single column's refractional-height range
   supports.

## Simplifications relative to the reference (deliberate, see each section)

- **Background/sanity check**: a plain fixed-stddev threshold
  (`background_check_stddev`, default 4.0) evaluated at the CURRENT state
  on every loss call -- the same convention as raobs's `q_bg_check` in
  this repo. The reference instead freezes a quantile-scaled-by-lead-time
  threshold against the BACKGROUND once (`_psobs_background_check_thresholds`,
  shared with its own psobs gross check) -- a real feature, but
  substantially more machinery than this first cut needed. GSI's own form
  (a clamped `ratio=|O-B|/clip(obserr,ermin,ermax)>gross`) IS available
  here now via `error_min_rad`/`error_max_rad` (both null by default, i.e.
  unclamped) -- see "GSI comparison" above.
- **No observation-side profile-shape QC.** The reference's
  `gpsro_profile_quality_control` (log-bending-angle jaggedness/
  fragmentation rejection, independent of any model) exists but defaults
  `enabled: False` in the reference's own config too -- not ported here
  either.
- **Model-column validity and the "maximum simulated bending angle" sanity
  bound are real, ported checks** (`bending_angle_columns_are_valid`,
  `maximum_simulated_bending_angle_rad`) -- these are NOT simplified away.
  As of 2026-10-20 this also includes a model-top extension
  (`top_extrapolation_km`) and a simplified super-refraction screen
  (`critical_gradient_n_per_km`/`sr_detection_fraction`/
  `sr_screen_height_km`), both ported from GSI rather than the JAX
  reference -- see "GSI comparison" above.
- **Super-refraction screen is reject-only, no ray relocation.** GSI's own
  SR handling, when it detects a near-critical layer ABOVE the ray's
  tangent, relocates the tangent's search window above that layer instead
  of rejecting (`setupbend.f90`'s `hob < top_layer_SR+1` branch) --  this
  port just rejects any ray at/below a detected SR layer. Simpler, and
  conservative (discards real information a relocated search could have
  kept), but not yet the fuller GSI behavior.
- **GSI's `ddnj<0` jacobian check and its altitude/latitude/season
  statistical cutoff are NOT ported.** The former checks a property of
  GSI's own numerical-quadrature integration points specifically (this
  repo's closed-form design handles the analogous increasing-gradient
  condition differently, via `increasing_layer_bending_angle`, rather than
  rejecting); the latter's numeric coefficients were empirically tuned to
  GSI's own background/operator error characteristics, not verified for
  this system.

## Plan: extend GPSRO to the ACE2 backend (2026-10-22, steps 0-4 DONE; step 5 -- the real GPU run -- not yet done)

GPSRO itself (reader, forward operator, QC -- everything above) needs no
redesign for this; the GSI (`55faa5e7af4f0604bd1ed38b385fd762863ca294`) and
`greg-long-window-4dvar` comparisons that shaped the operator are already
done. This is a backend-wiring task, same shape as raobs/amv/aircraft's
own ACE2 extensions (see integrate_prepbufr_obs.md/integrate_amv_satwnd.md/
integrate_aircraft_obs.md) -- `GPSRO_SUPPORTED_BACKENDS = ('aifs',
'aurora')` (`long_window_4dvar_utils.py`) is the only thing currently
stopping it -- EXCEPT two real physics questions below that are
structurally different from those three streams' own ACE2 extensions, not
just "repeat the same checklist."

**Checked directly against `backends/ace2/ace2_model.py` before writing this
plan** (so the easy parts of the raobs/amv/aircraft playbook are already
confirmed, not just assumed to carry over):
- **Level sets trivially match, no padding needed.** `decode_state`
  derives `'z'`/`'t'`/`'q'`/`'u'`/`'v'` all from the SAME `self._plevs`
  tuple (`DEFAULT_PLEVS = (1000, 925, 850, 700, 600, 500, 400, 300, 250,
  200, 150, 100, 50)`, 13 levels) -- unlike AIFS, where `'q'` omits the top
  two `'t'`/`'z'` levels and `get_gpsro` needs the `q_present_idx` padding
  logic for that. For ACE2, `t_levels == z_levels == q_levels` exactly, so
  that padding path is a no-op (still exercised, just trivially).
- **Level order matches gpsro_operator.py's required convention.**
  `_levels_by_base[b] = [(int(p), None) for p in self._plevs]` with
  `self._plevs` in `DEFAULT_PLEVS` order (1000 -> 50, i.e. surface-first,
  confirmed by `decode_state`'s own docstring: "surface-first") -- this
  matches the bottom-to-top order `gpsro_operator.py` requires (see "Level-
  order fix required for this repo" above). Getting this backwards would
  silently integrate from the wrong end, so this was checked directly in
  source, not assumed.

**Two things that are NOT just a repeat of the raobs/amv/aircraft
checklist:**

1. **GPSRO backprops through ACE2's derived `'z'`/geopotential for the
   FIRST time -- an unexercised gradient path, not covered by the existing
   fix.** The 2026-10-21 NaN-gradient fix (see
   integrate_prepbufr_obs.md's "Known issue found and FIXED" entry) patched
   `_to_pressure_levels`'s `t`/`u`/`v`/`q` linear-interpolation block (an
   `hi>lo`-guarded division that's 0/0 when a target level sits beyond the
   native layer-midpoint range). Reading the `'z'` derivation block right
   above it in the same function (`ace2_model.py` lines ~377-387): it uses
   `idx.clamp(max=nlay-1)` as a direct gather index plus an analytic
   `torch.exp(...)` extrapolation below ground -- structurally different
   from the interval-width-dependent weight that caused the t/u/v/q bug, so
   it likely does NOT share that exact failure mode. "Likely" is not good
   enough here, though: raobs and amv only ever read `'t'`/`'u'`/`'v'`/
   `'q'` for ACE2, never `'z'` -- GPSRO would be the first observation type
   to backprop through it. **Needs its own CPU-only gradient-finiteness
   probe** (`torch.isfinite` on `decode_state(only={'z'})`'s input gradient
   after a dummy `.backward()`, same technique already used to catch the
   t/u/v bug) before trusting a real run, not an assumption that the prior
   fix covers it.
2. **ACE2's `'q'` being `specific_total_water` (not pure specific humidity)
   graduates from an optional caveat to a load-bearing assumption.** For
   raobs this is a documented OPEN issue with a working config-level
   escape hatch: `observations.raobs.variables` just excludes
   `specific_humidity` for ACE2 (see integrate_prepbufr_obs.md's "Known
   issue... OPEN"), so raobs assimilates t/u/v fine regardless. GPSRO has
   no analogous escape hatch -- `gpsro_operator.py`'s Smith-Weintraub
   refractivity formula ALWAYS uses `q` as water-vapor partial pressure;
   there is no "assimilate without q" mode for a bending-angle operator.
   Extending GPSRO to ACE2 therefore inherits this open question in a form
   that can't be deferred by a config toggle -- it directly biases every
   simulated bending angle by however much condensate mass fraction
   pollutes ACE2's `'q'` relative to true vapor. Needs an explicit decision
   (quantify the vapor/condensate difference against ERA5 first, same check
   already recommended for raobs's q but never done, versus proceeding with
   it as a documented, accepted limitation) before trusting GPSRO O-B
   statistics on ACE2, not a silent inheritance of raobs's "not yet
   checked" status.

**A third thing worth checking before copying AIFS's defaults, not a
blocker**: ACE2's model top (`DEFAULT_PLEVS`'s 50 hPa) is much lower than
AIFS-single-2.0's (10 hPa). The existing `top_extrapolation_km` default
(10 km) was specifically sized to reclaim AIFS's own ~31-40 km band (see
"GSI comparison" above) -- a standard-atmosphere ballpark puts 50 hPa
around ~20 km, roughly DOUBLE the extrapolation distance needed to reach
the same 40 km `impact_height_range_km` ceiling. Before picking ACE2-
specific values, check the ACTUAL geometric height of ACE2's decoded 50 hPa
level directly (cheap: one `decode_state(only={'z'})` call,
`geopotential_to_geometric_height_m`, no GPU) rather than trusting a
textbook table -- then either widen `top_extrapolation_km` for ACE2
configs or narrow `gpsro_impact_height_range_km`'s upper bound, whichever
the real number supports.

**Plan, in dependency order**:

0. Quantify the two physics questions above offline (no code change, no
   GPU): (a) ACE2 `'q'` vs ERA5 true specific humidity, a few real
   dates/columns -- same check integrate_prepbufr_obs.md's "ACE2 'q'"
   section already flagged as the way to resolve this, just actually
   doing it now that GPSRO makes it load-bearing; (b) ACE2's real decoded
   geometric height at 50 hPa, to size `top_extrapolation_km`/
   `impact_height_range_km` for ACE2 instead of copying AIFS's numbers.
1. Add `'ace2'` to `GPSRO_SUPPORTED_BACKENDS` (`long_window_4dvar_utils.py`)
   -- mechanical, mirrors the raobs/amv/aircraft precedent.
2. CPU-only gradient-finiteness probe through `decode_state(only={'z'})`
   specifically (item 1 above), then through the full
   `gpsro_operator.simulate_bending_angles` chain fed ACE2-shaped inputs --
   confirm no NaN gradients before any real run.
3. Offline `get_gpsro` -> forward-operator pipeline probe against the real
   gpsro_parquet archive, reusing the same fake-model-object technique
   already used for raobs/aircraft's own ACE2 offline validation (ACE2's
   real F90 Gauss-Legendre 180x360 grid, real `DEFAULT_PLEVS`, the real
   `RectilinearBilinearInterpolator` as `grid_interp`) -- check shapes,
   finiteness, and a sane (not necessarily high, per the existing
   validation's own caveat about a single fake background column)
   accepted fraction.
4. New config(s): extend the `config_test_ace2_ps_raobs*.yml` family with
   `gpsro.enabled: True` and the ACE2-tuned `impact_height_range_km`/
   `top_extrapolation_km` from step 0b -- update the "also not in
   GPSRO_SUPPORTED_BACKENDS for ace2 yet" comments added to
   `config_test_ace2*.yml` on 2026-10-22, now stale once step 1 lands.
5. Real GPU smoke test -- the actual `ps+raobs+aircraft+amv+gpsro`
   ablation run, only after steps 0-4 pass. Same sequencing every other
   obs type here has followed: offline-validated first, GPU run later.

### Results (2026-10-22): steps 0-4 done, all CPU-only/offline, no GPU used

**Step 0a -- ACE2 `'q'` vs real ERA5 specific humidity, quantified, not
assumed.** Fetched true ERA5 specific humidity (ARCO-ERA5 `full_37`
pressure-level zarr, `gs://gcp-public-data-arco-era5`, anonymous read, same
source `ace2_ic.py` already uses) at ACE2's own 13 `DEFAULT_PLEVS`, same
date/grid as a real cached ACE2 IC (2014-12-30T00, conservative-regridded to
the F90 180x360 grid with `ace2_ic._regrid`), and compared directly against
`ACE2Model.decode_state`'s real decoded `'q'` for that same state (area-
weighted mean/rms diff, same metric `compare_ic_ace2.py` uses). Result: ACE2
is ~0.1-0.6 g/kg DRIER than ERA5 at 1000/925 hPa, then consistently ~0.06-
0.26 g/kg WETTER than ERA5 from 850-250 hPa (rms difference 15-70% of
ERA5's own spatial std in that band) -- the wetter-in-the-free-troposphere
sign is the physically expected direction for a condensate-inclusive field
being compared to pure vapor (clouds add mass there), but this is one
column/one date, not a climatology. **Also confirmed structurally
unresolvable from ACE2's own output**: `ACE2Model._channel_names` has NO
separate cloud/liquid/ice-water channel at all (checked directly, grepped
every channel name) -- `specific_total_water` is ACE2's only moisture
prognostic, so vapor and condensate can never be separated from ACE2's own
fields; the only way to even estimate the difference is the external ERA5
comparison just done, not further engineering on ACE2's side. **Decision**:
proceed with GPSRO on ACE2 as a documented, accepted limitation (this
repo's established "simplest correct thing first" pattern) rather than
block on a fix that doesn't exist -- but the "expected to be small outside
cloudy columns" language in integrate_prepbufr_obs.md's open raobs-`'q'`
issue is NOT fully supported by this check (15-70% of spatial std in the
850-250 hPa band is not obviously small) and should be read as a real,
confirmed limitation for both raobs's and gpsro's O-B statistics on ACE2,
not a speculative caveat.

**Step 0b -- ACE2's real model top, measured, not estimated from a
textbook.** Decoded a real cached ACE2 IC (2014-12-30T00) on CPU and ran
`gpsro_operator.geopotential_to_geometric_height_m` on `decode_state`'s own
`'z'` at every one of the 13 `DEFAULT_PLEVS`. The 50 hPa (top) level sits at
mean 20.44 km globally (min 18.99 / max 21.65 km) -- matching the
standard-atmosphere ballpark in the plan above, now a measured number. Full
per-level table (global min/mean/max, km): 1000 hPa -0.80/0.08/0.99, 925
-0.11/0.71/1.68, 850 0.63/1.40/2.41, 700 2.27/2.93/4.06, 600 3.42/4.13/5.33,
500 4.66/5.48/6.78, 400 6.16/7.12/8.53, 300 8.01/9.09/10.63, 250
9.18/10.33/11.96, 200 10.59/11.76/13.41, 150 12.40/13.61/15.27, 100
14.87/16.13/17.64, 50 18.99/20.44/21.65.

**Step 0c -- gradient-finiteness probe through `'z'`, the one genuinely new
risk, confirmed clean.** `decode_state(only={'z'})`'s backward pass on the
same real IC: gradient fully finite (0 non-finite entries), confirming the
structural read above (the `'z'` block's `idx.clamp`+analytic-`exp` form
doesn't share the `t`/`u`/`v`/`q` block's `hi>lo`-guarded-division bug).
Then ran the FULL `gpsro_operator.simulate_bending_angles` chain fed
ACE2-decoded `t`/`q`/`z` at 8 synthetic impact heights (2-50 km) and
`bending_angle_columns_are_valid` at `top_extrapolation_m=10000`: all 8
bending angles finite, backward pass fully finite, and the validity
predicate accepted 2-25 km but correctly rejected 35/50 km -- exactly
confirming step 0b's arithmetic (top 20.44 km + 10 km extrapolation =~
30.4 km ceiling) in actual code behavior, not just by hand.

**Decision on `impact_height_range_km`/`top_extrapolation_km` for ACE2**:
keep `top_extrapolation_km: 10.0` (the already-GPU-validated AIFS value,
not an untested ~20 km stretch) and narrow `impact_height_range_km` to
`[6., 30.]` (vs. AIFS's `[6., 40.]`) to match ACE2's own, lower, measured
top -- confirmed directly in the probe above, not guessed.

**Step 1**: `'ace2'` added to `GPSRO_SUPPORTED_BACKENDS`
(`long_window_4dvar_utils.py`), with a comment summarizing what was checked
(above) rather than just the raobs/amv precedent.

**Step 3 -- full offline pipeline probe, real archive + real ACE2 grid/IC,
no GPU**: `get_gpsro` against the real `gpsro_parquet` archive
(2015-01-01T00, `impact_height_range_km=[6.,30.]`) with ACE2's real
`RectilinearBilinearInterpolator` grid: 1,427,474 canonical rows -> 301
profiles / 6,516 levels retained (fewer than the AIFS doc's 302/9,167 at
`[6.,40.]`, as expected from the narrower window), shapes
`(1, 301, 23, 4)` for the horizontal stencil. Forward operator evaluated
against a REAL decoded ACE2 column (the 2014-12-30T00 IC, used here only to
exercise the math/shapes end-to-end -- NOT the same date as the obs, so
this is not an O-B accuracy claim): `model_equivalent` 100% finite within
the mask, 76.2% of levels passed QC+validity+gross-check, innovation
mean/std (-3.0e-5/3.4e-4 rad) in the expected GPSRO magnitude range. No
crash, no NaN propagation, no shape mismatch anywhere in the chain.

**Step 4**: `config_test_ace2_ps_raobs_aircraft_amv_gpsro.yml` added
(`ps+raobs+aircraft+amv+gpsro`, all five streams on) with `gpsro.enabled:
True` and the tuned `impact_height_range_km: [6., 30.]`/
`top_extrapolation_km: 10.0`; round-tripped through `load_config()` and
confirmed `gpsro_enabled: True` resolves with no error now that step 1
lifted the guard. Superseded an earlier same-day version of this file that
tested amv WITHOUT aircraft -- see "Observation-ablation series,
regenerated" below for why the whole `config_test_ace2*.yml` family was
restructured into an aircraft-before-amv ladder instead. The other
`config_test_ace2*.yml` rungs' stale "also not in GPSRO_SUPPORTED_BACKENDS"
comments (added earlier the same day) were corrected to point here instead.

### Observation-ablation series, regenerated (2026-10-22, same day)

The user asked for the whole `config_test_ace2*.yml` family (and matching
`run_ace2*.sh` slurm jobs) to be regenerated as a clean incremental ladder,
one stream added at a time, in the order **psobs -> raobs -> aircraft ->
amv -> gpsro** (aircraft BEFORE amv, unlike the order these streams were
originally built in). Five rungs now exist: `config_test_ace2_ps.yml`,
`_ps_raobs.yml`, `_ps_raobs_aircraft.yml`, `_ps_raobs_aircraft_amv.yml`,
`_ps_raobs_aircraft_amv_gpsro.yml` (plus `config_test_ace2.yml`, the
original tracked baseline, left untouched). Each config's `observations:`
block lists every stream explicitly (psobs/raobs/aircraft/amv/gpsro, in
that order) with `enabled: True`/`False` marking the rung, matching the
`config_test_aifs2-1.0deg-v1.0_ps_raobs_amv_gpsro.yml` precedent's style.

**Two pre-existing files were deleted to make room for this**:
`config_test_ace2_ps_raobs_amv.yml` and its slurm script
`run_ace2_ps_raobs_amv.sh` -- NOT a trivial removal: that config was the
actual basis for completed, documented GPU runs (job 24270091, the run
that first surfaced the ACE2 NaN-gradient-through-derived-fields bug, see
integrate_multiple_backends.md; job 24273858 and the sensitivity-sweep
jobs 24295238-40, see integrate_amv_satwnd.md's "Vertical thinning tied to
model levels"). Flagged this explicitly to the user before deleting (since
it meant those docs' job-ID citations would reference a config no longer
in the working tree); the user confirmed deletion anyway. Those docs'
own historical descriptions of what happened in those runs are unaffected
-- only the YAML source file (and its job script) are gone, superseded by
`config_test_ace2_ps_raobs_aircraft_amv.yml`, which tests a superset
(same ps+raobs+amv, plus aircraft) of what the deleted file tested. The
also-just-created `config_test_ace2_ps_raobs_amv_gpsro.yml` (this
session's own first draft of the gpsro extension, no run history) was
deleted too, superseded by `_ps_raobs_aircraft_amv_gpsro.yml`.

**Still not done**: step 5, the real GPU cycling run -- see "Not yet done"
below.

## Not yet done

- **No real AIFS/GPU smoke test of the restored design.**
  `config_test_aifs2-1.0deg-v1.0_ps_raobs_amv_gpsro.yml` (ps + raobs + amv
  + gpsro, same 200-cycle/12h/5-day-window setup as the other validated
  configs) is written and round-tripped through `load_config()`, but
  hasn't been run against the real model with the final 2026-10-20 state
  (thinning restored at 1000 m + top extrapolation + SR screen + clamped
  gross check together) -- see "Thinning restored" above for what's next.
- **No reviewed/Desroziers-style error profile** (see "Error model"
  above) -- starts with the reference's `ecmwf_seed_v1` placeholder.
- **Whether 1000 m is still the right thinning spacing for this repo's
  14-level models is itself an open question**, not just whether to thin
  at all. AIFS-single-2.0's own level spacing through the 6-40 km window
  is more like 2-10 km, coarser than 1000 m almost everywhere in that
  range -- 1000 m was the original design's (untuned) default, carried
  over as a known-working starting point, not verified to be the best
  spacing for THIS model's actual vertical resolution.
- **No real ACE2/GPU smoke test either.**
  `config_test_ace2_ps_raobs_aircraft_amv_gpsro.yml` (all five streams on)
  is written and offline-validated end-to-end (see "Plan: extend GPSRO to
  the ACE2 backend" above -- steps 0-4 all done, CPU-only) but has never
  been run against the real GPU model. Also inherits two ACE2-specific,
  confirmed (not speculative) limitations going into that run: ACE2's `'q'`
  (specific_total_water) is measurably wetter than true ERA5 vapor through
  850-250 hPa (15-70% of ERA5's own spatial std, one column/date, no
  climatology) with no way to separate vapor from condensate in ACE2's own
  output; and the impact-height window was narrowed to `[6.,30.]` km
  (vs. AIFS's `[6.,40.]`) because ACE2's measured model top (~20.4 km) is
  much lower than AIFS's, discarding real information between 30-40 km
  that AIFS's own config keeps.
