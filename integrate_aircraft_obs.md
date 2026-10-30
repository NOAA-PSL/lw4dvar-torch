# Adding aircraft (AIRCAR/AIRCFT) observations

Read this when working on: the aircraft temperature/wind observation
stream, why it is split out from `raobs` as its own `exp.observations.aircraft`
block, why it thins temperature and wind as two independent sub-streams, the
`aircraft_id` metadata field (kept for diagnostics, but NOT used for bias
correction -- see "Static aircraft temperature bias correction" below for
why per-tail correction turned out not to be buildable here), or the static
pressure x region temperature bias correction itself.

## Status: observation stream implemented (2026-10-22), offline-validated
only -- no real GPU cycling run yet. A static (not per-tail) temperature
bias correction was added 2026-10-23 -- see "Static aircraft temperature
bias correction" below. Built while restructuring the observation config
schema to mirror the `greg-long-window-4dvar` reference repo (splitting
aircraft out of `raobs`, and making every obs-type's yaml keys explicit) as
groundwork for adding bias correction per observation type -- see
integrate_amv_satwnd.md's "AMV cycling-bias investigation" for the
motivating problem (AMV's uncorrected, dominant loss weight injecting a
growing cycling bias; the reference repo's own `aircar_superobs.md` flags
"adaptive aircraft-specific bias correction" as a natural next step once
aircraft identity is tracked).

## Why aircraft is a new stream, not literally "split out of" raobs

The reference repo (`greg-long-window-4dvar/long-window-4dvar`) treats AIRCAR
as a distinct `observations:` stream from ADPUPA radiosondes
(`observation_sources.py`, `aircar_superobs.md`). In THIS repo, `raobs` only
ever read `ADPUPA` (`DEFAULT_RAOBS_CLASSES = ('ADPUPA',)` in
`raobs_parquet.py`) -- aircraft reports were never read at all before this.
So "splitting aircraft out of raobs" here means adding a new sibling stream
with its own config block, not separating already-assimilated data.

## Why aircraft is AMV-shaped, not raobs-shaped

Checked directly against the real archive
(`prepbufr_to_parquet/data/prepbufr_parquet/2015-01-01/00/{reports,levels}/
{AIRCAR,AIRCFT}`):

- **Row counts are AMV-scale**: 290,382 AIRCAR + 28,362 AIRCFT level rows in
  one 00Z cycle -- raobs's per-row Python loop (`raobs_parquet.py`'s
  `_profile_records`) doesn't scale to this.
- **Pressures are continuous**, not radiosonde mandatory levels -- raobs's
  hard `pressure_match_tolerance_hpa=0.05` snap-to-model-level reader would
  reject nearly every aircraft report.

So `aircraft_parquet.py` is ported from `amv_parquet.py`'s vectorized
joined-read (`np.searchsorted` report-id join, vectorized nearest-time
assignment), and `get_aircraft` reuses amv's native-pressure point stencil
(`_amv_horizontal_vertical_stencil`) and GSI-style box thinning
(`_amv_thin_slot`, generalized -- see "Implementation" below) rather than
raobs's mandatory-level matching.

## Why temperature and wind are two independent sub-streams, not one

The critical archive fact that shapes this whole design, checked directly:
AIRCAR/AIRCFT report temperature and wind as **mutually exclusive level
rows** -- `report_type` 133 (a mass-report BUFR message) carries
`temperature_c` with `u_wind_ms`/`v_wind_ms` both null; `report_type` 233 (a
wind-report message) carries `u_wind_ms`/`v_wind_ms` with `temperature_c`
null, for the SAME physical observation. Confirmed with a direct column
scan: `has_t=145330, has_uv=145052, has_both=0` (AIRCAR); `has_both=0`
(AIRCFT too, with 152 rows carrying neither). AIRCAR's `report_type` column
is a clean 145,330/145,052 split between exactly 133 and 233.

GSI's own `convinfo` treats these as separate thinning types for exactly
this reason (different `ithin`/`rmesh`/`pmesh` entries per `itype`, each
maintaining its own independent thinning grid). Thinning them together here
-- picking ONE row per shared horizontal x vertical x report_type box by
GSI's `crit1` time-closeness score -- would silently keep only a temperature
OR a wind row per box (whichever happens to score better), discarding the
other variable entirely for that box. Running two independent thinning
passes (one over temperature-bearing rows, one over wind-bearing rows) avoids
this and matches GSI's actual per-type treatment.

Consequence: `aircraft_traj` is `{'t': {...}, 'uv': {...}, <shared keys>}`,
not one flat dict of padded fields the way `amv_traj`/`raobs_traj` are. Each
sub-stream has its own capacity, mask, kept-observation fields, and node
stencil. `rmesh_km`/`pmesh_hpa`/`vertical_bin_mode`/
`background_check_stddev` apply identically to both (one thinning policy);
error floors/clamps are per-sub-stream (`error_floor_t_k` vs
`error_floor_ms`, since K and m/s are different units).

## Implementation

- **Reader** (`aircraft_parquet.py`, repo root): `load_aircraft_parquet`,
  ported from `amv_parquet.py`. Returns `{'t': {...}, 'uv': {...}}`, each a
  flat, per-slot-bucketed, un-thinned point set (same shape convention as
  `load_amv_parquet`). Temperature is sensible temperature (ported from
  `raobs_parquet.py`'s `common.sensible_temperature` -- the archive's
  `temperature_c` is virtual temperature wherever moisture was reported).
  `aircraft_id` (a string ACID identifier, e.g. `'UZVDPMJA'`) is carried
  through as metadata on every row -- NOT used for filtering, grouping, or
  deduplication in this pass.
- **Thinning** (`_amv_thin_slot`, `long_window_4dvar_utils.py`):
  generalized to accept a `components: dict[name -> array]` payload instead
  of hardcoded `u`/`v` params, plus an optional `extra_fields: dict[name ->
  array]` passthrough (arbitrary dtype, not used in grouping/scoring --
  carries `aircraft_id`) and an optional `satellite_id`/
  `separate_satellite_ids` (AMV-specific; `None` for aircraft, which has no
  comparable per-row identity to group by). `get_aircraft` calls this
  function TWICE per obs slot -- once for the temperature sub-stream
  (`components={'temperature': t}`), once for the wind sub-stream
  (`components={'u_component_of_wind': u, 'v_component_of_wind': v}`) --
  each pre-filtered at read time to rows where its own variable(s) are
  finite, so every kept observation in either sub-stream has a fully valid
  payload (no partial-NaN rows to special-case downstream).
  `_amv_horizontal_vertical_stencil` needed NO changes -- it only depends
  on lon/lat/pressure/levels, never on which variables are attached.
  **Regression-checked**: re-ran the offline validation this refactor could
  have silently broken (see "Offline validation" below) -- the generalized
  function reproduces the exact previously-documented AMV kept count
  (28,308 at the default 120 km/50 hPa mesh, 2015-01-01 00Z) byte-for-byte,
  confirming amv's own behavior is unchanged.
- **Forward operator** (`_compute_aircraft_observation_diagnostics_at_time`,
  `long_window_4dvar_utils.py`): a near-duplicate of
  `_compute_amv_observation_diagnostics_at_time`, evaluating the 't'
  sub-stream's stencil against `decoded['t']` and the 'uv' sub-stream's
  against `decoded['u']`/`decoded['v']`. Deliberately duplicated rather than
  generalized into one shared function -- it's short (under 50 lines), and
  this keeps amv's own validated function completely untouched.
- **Loss integration**: a `aircraft_traj=None`-gated branch (`_aircraft_term`,
  parallel to `_amv_term`) inside `compute_loss_4dvar`'s existing per-step
  loop -- same obs-slot grid as psobs/raobs/amv.
- **Diagnostics**: `save_aircraft_diagnostics` (parallel to
  `save_amv_diagnostics`), one netCDF file
  (`<date>_aircraft_diagnostics_*.nc`) with TWO independent slot dimensions
  (`t_slot`, `uv_slot`) instead of amv's single `superob_slot`, each
  carrying its own kept observation's lat/lon/pressure/report_type/
  `aircraft_id`/component_count plus the usual O-B/O-A/used fields.
- **Config**: `aircraft.*` keys documented in `config.yml.template`, all
  optional, mirroring amv's defaults (1h time tolerance, quality_mark_max 3,
  use_source_error True, rmesh_km 120.0, pmesh_hpa 50.0, pmesh vertical
  mode, background_check_stddev 4.0) plus aircraft-specific error defaults
  (fallback_error_std_t_k 1.5, fallback_error_std_ms 3.5, error_floor_t_k
  0.5, error_floor_ms 2.0 -- picked as conservative, round, physically
  reasonable starting points, NOT Desroziers-diagnosed; see "Not yet done").
- **Backend guard**: `aircraft_enabled` is rejected by `load_config` for any
  backend other than `aifs`/`aurora`/`ace2` (`AIRCRAFT_SUPPORTED_BACKENDS`),
  the same conservative restriction raobs/amv use.

## Offline validation (no model/GPU)

Ran against the real archive at
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/prepbufr_to_parquet/data/prepbufr_parquet`,
2015-01-01 00Z, under the `lwace2` conda env (has pyarrow), using a
`RectilinearBilinearInterpolator` (`backends/ace2/ace2_grid.py`, the same
class ACE2 uses for real) on a regular 360x180 lon/lat grid and ACE2's
`DEFAULT_PLEVS` as a fake model:

1. **AMV regression** (the generalized `_amv_thin_slot`): `get_amv` against
   the real archive reproduces the EXACT previously-documented kept count
   (56,821 accepted raw vectors -> 28,308 kept at the default 120 km/50 hPa
   mesh) -- confirms the box-thinning/selection logic is unchanged by the
   `components`/`extra_fields` generalization. (For `vertical_bin_mode:
   pmesh`, the kept count structurally cannot depend on the grid/model
   levels at all -- only `rmesh_km`/`pmesh_hpa`/grouping flags and the raw
   archive's own lat/lon/pressure/time/error feed the box key -- so this
   check is grid-independent and reproducible with any valid fake grid.)
2. **Aircraft reader + full `get_aircraft` pipeline**: 318,744 candidate
   level rows -> 48,204 temperature / 49,903 wind accepted after QC ->
   6,621 temperature / 6,688 wind kept after thinning. Physically sane
   throughout: temperature 204-303 K, |u| up to ~104 m/s, |v| up to ~55 m/s
   (jet-level cruise winds), lat [-67, 85], lon [0, 360), pressure
   [178, 1000] hPa (aircraft stay below the tropopause), `aircraft_id`
   non-empty strings on every kept observation.
3. **Forward operator correctness**: with the background gross check
   disabled, `used` exactly equals the per-sub-stream `mask` for every
   point (confirms the gather/weighting/masking wiring, independent of any
   QC decision); with it enabled (default `background_check_stddev: 4.0`)
   against a random/unrealistic fake background field, the fraction
   flagged `used` drops sharply (9-60% depending on variable) -- exactly
   the expected behavior of a working gross check rejecting large,
   nonphysical O-B innovations, not a bug.
4. **Config parsing**: `load_config` on an `aircraft: {enabled: True}`
   block (no other keys) resolves `aircraft_path` to `obspath` and leaves
   every other active config (`config_test_ace2_ps_raobs_amv.yml`,
   `config_test_aifs2-1.0deg-v1.0_ps_raobs_amv_gpsro.yml`, etc.) parsing to
   IDENTICAL `exp[...]` values as before this change -- a pure addition,
   not a behavior change to any existing stream.

## Not yet done

- **No real GPU cycling run.** Offline-validated only, same sequencing amv
  and raobs followed before their own first real runs. A natural next step,
  once scheduled: an ACE2 `ps+raobs+amv+aircraft` ablation run, parallel to
  the amv/gpsro real-run confirmations already done.
- **No reviewed/Desroziers-style error profile.** `fallback_error_std_t_k`/
  `fallback_error_std_ms`/`error_floor_t_k`/`error_floor_ms` are round,
  conservative starting points -- this starts with `use_source_error: True`
  (each report's own `temperature_error_c`/`wind_error_ms`) plus a flat
  floor, the same "simplest correct thing first" sequencing raobs and amv
  both followed before their own error-table refinements.
- **`aircraft_id` is metadata only.** It is threaded through the reader,
  thinning, and diagnostics, but used for nothing yet -- not thinning
  grouping, not a bias-correction term, not even an aircraft-level QC
  check. This is deliberate groundwork: the reference repo's own
  `aircar_superobs.md` describes a much larger phase-aware (flight/profile
  identity, ascent/descent/cruise classification, cruise thinning)
  redesign, explicitly flagged there as pre-production/not-yet-accepted
  policy on its own feature branch -- out of scope here. The near-term use
  this unblocks is simpler: a per-aircraft or per-region/level running-mean
  bias correction (see integrate_amv_satwnd.md's "AMV cycling-bias
  investigation" for the same idea applied to AMV) without needing full
  flight tracking first.
- **8-native-layer-aware thinning: decided 2026-10-22, not sensitivity-
  tested.** AMV's `vertical_bin_mode: nearest_level` option is directly
  reusable here (the generalized `_amv_thin_slot` takes it as a plain
  argument). Every `config_test_ace2_*.yml` now sets
  `aircraft.vertical_bin_mode: nearest_level`, same as `amv` -- applied as
  the better-justified default (ACE2's 8 native layers, plus aircraft's
  own bimodal/cruise-heavy pressure distribution being if anything a
  tighter density-vs-resolution mismatch than amv's), not because a
  sensitivity sweep confirmed it matters. integrate_amv_satwnd.md's
  "Vertical thinning tied to model levels" sweep found NO z500 skill
  difference between `pmesh` and `nearest_level` for amv on ACE2; the
  equivalent sweep has NOT been run for aircraft, so this remains an
  applied-by-default choice, not a validated one.

## GSI comparison: aircraft thinning/QC (2026-10-22, reviewed -- no action taken)

Researched how GSI (NOAA-EMC/GSI, pinned commit
`55faa5e7af4f0604bd1ed38b385fd762863ca294`, via the local `jswhit/GSI`
clone at `/scratch4/BMC/gsienkf/Jeffrey.Whitaker/GSI-jswhit`) reads/thins/QCs
AIRCAR/AIRCFT -- `read_prepbufr.F90`, `setupt.f90`/`setupw.f90`,
`qcmod.f90`'s `errormod_aircraft`, and the dedicated `aircraftinfo.f90`/
`aircraftobsqc.f90` modules -- to compare against this repo's own aircraft
implementation. **Recorded here as a reference for later work -- no code
changed in this pass.**

### GSI aircraft (AIRCAR/AIRCFT)

- **Thinning**: the same generic `convinfo`-driven `ithin`/`rmesh`/`pmesh`
  mechanism as every other conventional type, with independent entries for
  mass (T) vs. wind `kx` -- nothing aircraft-specific in the thinning code
  itself. This matches what's already in this doc's "Why temperature and
  wind are two independent sub-streams" section.
- **Vertical/horizontal hard gates**: aircraft winds (`itype` 230-239) are
  rejected outright above `presw < 126 hPa` -- a flat pressure ceiling, the
  same style of gate as the VADWND 226 hPa cutoff just above it in
  `setupw.f90`.
- **`errormod_aircraft`** (`qcmod.f90`) replaces the generic `errormod` for
  aircraft obs-error inflation: it widens the assigned error based on
  vertical spacing to the nearest usable neighboring level in the same
  sounding (same representativeness idea as the generic form used for
  raobs), but -- unique to aircraft, since a report is from a moving
  platform rather than a fixed station -- it also checks the **horizontal**
  distance between the current level and the neighboring level it borrows
  from (`hdist_aircraft`, default 60 km); if the neighbor is farther than
  that, it's treated as absent and the inflation falls back to a default
  half-grid-spacing error.
- **Per-tail-number temperature bias correction** -- the one genuinely
  aircraft-specific feature, and the biggest structural difference from
  every other obs type covered so far (`aircraftinfo.f90`, `setupt.f90`,
  mass types 130/131/133-135 only): a persistent table keyed by tail
  number (`taillist`/`idx_tail`), holding regression coefficients
  (`predt`) carried cycle-to-cycle like a VarBC file (`aircftbias_in`/
  `aircftbias_out`). Two predictor schemes: `aircraft_t_bc` regresses
  against instantaneous vertical velocity (`pred = [1, w, w^2]`, clamped at
  |w|<=30); `aircraft_t_bc_pof` instead buckets by discrete phase-of-flight
  code (ascending/descending/cruise) into three one-hot predictors. New or
  uninitialized tail numbers get their assigned error inflated (x1.2, or
  x1.5 more if |w|>=30) rather than rejected, until the online regression
  (Desroziers-style accumulated stats, `ostats_t`/`rstats_t`/`varA_t`) has
  enough data; stale tails can be retired after a month unseen
  (`cleanup_tail`).
- **Reject list** (`aircraftobsqc.f90`): a separate, much simpler mechanism
  -- a flat file (`current_bad_aircraft`) of blacklisted tail numbers, one
  column per variable (T/W/RH), checked before the gross check; any match
  forces `usage=450` (effectively excluded). Independent of the bias
  correction above.

### Comparison to this repo + recommendations

This repo's `get_aircraft` is already GSI-shaped on thinning (reuses AMV's
GSI-style box thinning, `rmesh_km`/`pmesh_hpa`, independently per
sub-stream) -- that part needs no change. The gaps are in per-platform
bias handling and a couple of GSI's hard gates. Checked what the real
archive can actually support before recommending anything novel:

- **GSI's own aircraft bias-correction predictors are not available in this
  archive -- checked directly, not assumed.** The normalized PREPBUFR
  parquet schema for AIRCAR/AIRCFT levels *does* carry a column matching
  GSI's `aircraft_t_bc` predictor, `instantaneous_altitude_rate_ms` (GSI's
  IALR), plus per-level `drift_longitude`/`drift_latitude`/
  `drift_time_hours` columns that would support per-level true position
  instead of the single report-level lat/lon this reader currently joins
  onto every level. All of these are **100% null** in the real archive
  (2015-01-01 00Z, checked for both AIRCAR -- 290,382 rows -- and AIRCFT --
  28,362 rows). GSI's phase-of-flight predictor (POAF) isn't even in the
  schema at all. This is the same "blocked on data, not code" situation as
  raobs's balloon drift gap (see `integrate_prepbufr_obs.md`'s GSI
  comparison) -- neither of GSI's two bias-correction predictor schemes,
  nor a drift-corrected per-level position, can be ported as-is.

**Recommendations, not yet implemented, highest-value first**:

1. **If per-tail bias correction is wanted, it has to be a simpler
   constant-per-tail running mean, not GSI's regression** -- `aircraft_id`
   is already threaded through the reader/thinning/diagnostics
   specifically for this (see "Not yet done" above). Since the vertical-
   velocity and phase-of-flight predictors GSI regresses against aren't
   populated in this archive, the only implementable version is a
   predictor-free, Desroziers-style running mean per tail (closest in
   spirit to GSI's `aircraft_t_bc_ext` "externally supplied" mode, but
   estimated online instead of from an external table) -- the same
   per-platform running-mean idea already floated for AMV's cycling bias
   in `integrate_amv_satwnd.md`. This is a genuine gap worth closing, but
   it is orthogonal to this repo's coarse vertical resolution: a tail's
   calibration offset doesn't depend on how many model levels exist.
2. **Do not add GSI's hard aircraft-wind pressure ceiling
   (`presw < 126 hPa` -> zero weight)** -- this repo's existing
   `background_check_stddev` gross check already rejects badly-performing
   high-altitude obs dynamically, and GSI's specific 126 hPa number is tied
   to its own analysis vertical grid, not validated for any backend here.
   Same "soft inflation over hard gate" preference already adopted for
   psobs's `zthresh`/orography check.
3. **Do not adopt `errormod_aircraft`'s horizontal-distance-gated
   vertical-representativeness inflation** -- that machinery exists
   because GSI treats each reported level independently within a 1D
   sounding-style vertical interpolation and has to guard against
   borrowing a spatially-displaced neighbor's error. This repo's forward
   operator instead evaluates one post-thinning point per kept
   observation, and GSI-style box thinning (`rmesh_km`/`pmesh_hpa`) already
   collapses nearby reports into one representative value per box --
   duplicating GSI's neighbor-search inflation on top of that would double
   up on a representativeness problem thinning already addresses
   structurally, not add real information.
4. **Check kept-obs-per-model-level at cruise altitude specifically, before
   trusting the AMV mesh-size sensitivity result unchanged.** Aircraft
   pressure is heavily bimodal, not spread like raobs's mandatory levels --
   checked directly in the real archive (AIRCAR, 2015-01-01 00Z): 33% of
   rows fall in 150-250 hPa (cruise) and 39% in 500-1000 hPa
   (ascent/descent/low-level), vs. only 28% in between. Cruise-level
   reports cluster right on top of the one or two coarse pressure levels
   AIFS/ACE2 actually carry near 200-250 hPa, which is a tighter mismatch
   between ob density and vertical resolution than AMV has (whose own
   `pmesh` vs. `nearest_level` sensitivity sweep found no z500 difference,
   `integrate_amv_satwnd.md`). Don't assume that result transfers --
   it hasn't been checked for aircraft's own vertical distribution.
5. **Reject list**: low cost, low priority. GSI's flat tail-number
   blacklist could be added later if diagnostics turn up specific
   consistently-bad tails, the same way psobs/raobs already use source
   quality marks -- no evidence of a problem tail in this dataset yet, so
   not worth building ahead of that evidence.

### Decision (2026-10-22): none of the above implemented

User's call: "No change is needed at this moment." Same posture as
`integrate_prepbufr_obs.md`'s psobs/raobs GSI comparison -- this is a
recorded reference for later work, not abandoned. Revisit if a real
cycling run surfaces a specific aircraft-related problem (e.g. a bad tail
number, or a skill regression traceable to cruise-level thinning), or if
the archive is ever regenerated with populated altitude-rate/drift columns
(which would reopen recommendation 1's data blocker).

## Static aircraft temperature bias correction (implemented 2026-10-23)

The predicted regression above showed up for real: the ACE2
`ps+raobs+aircraft` observation-ablation run (`test_ace2_6h_5d_50it_ps_raobs_aircraft`,
see `integrate_multiple_backends.md`) regresses z500/T850 skill vs.
`ps+raobs` alone, in both NH and SH. Pulled directly from the saved
`*_aircraft_diagnostics_*.nc` files (`t_omb`, grouped by `aircraft_id`):

- **NH** has a real, large per-tail bias -- dozens of high-volume tails
  (thousands of obs each) sit at -1.0 to -1.8 K, individually far beyond
  noise (SE ~0.02 K at that sample size), though the *pooled* NH mean is
  small (~-0.1 to -0.3 K) because tails partly cancel in aggregate.
- **SH** has a smaller but statistically unambiguous *consistent* warm
  bias: pooled mean +0.321 K over 31,716 obs, SE 0.0067 K (~48-sigma from
  zero), 72.5% of 6h-binned means positive, and all 5 of SH's n>=10 tails
  individually positive.
- **Wind** (u/v) shows no such pattern -- per-tail mean (u,v) O-B is a
  symmetric cloud centered on zero in every region, i.e. scatter, not bias
  -- matching GSI's own choice to never bias-correct aircraft wind.

**Why this isn't per-tail, despite recommendation 1 above**: the user
explicitly ruled out estimating a per-tail running mean from this repo's
own ACE2 cycling (that cycling has its own known issues, and using it to
estimate the correction risks correcting for ACE2's background error
rather than the aircraft sensor's own bias). The alternative -- borrowing
per-tail coefficients from a real external GSI run -- was checked directly
and is a dead end: tail-ID string matching against a real 2022 operational
`abias_air` (8708 tails) AND a real 2015-01-01 00Z operational `cnvstat`
(parsed directly, legacy Fortran-binary `diag_conv` format, since ncdiag
didn't exist yet in 2015) both have **zero overlap** with this archive's
own aircraft `aircraft_id`s -- confirmed for the full month, case/whitespace-
normalized, and even for the exact same cycle date/hour (so it isn't a
fleet-turnover-over-years effect; every PREPBUFR extraction appears to
independently anonymize tail numbers). GSI's own real per-tail predictors
(`w`, ascent/descent rate) are also unusable here regardless, since this
archive's `instantaneous_altitude_rate_ms` is 100% null (established
above). So per-tail correction isn't a "not yet" here -- it's not buildable
from any data this repo has access to.

**What's implemented instead**: a static (pressure bin x NH/Tropics/SH
region) lookup table of mean O-B (observation minus background/first-guess,
GSI's "OmF" -- NOT O-A, which is contaminated by that cycle's own partial
fit to the observation), built from a real, independent GSI operational
archive and applied as a fixed subtraction to the raw aircraft temperature
observation in `get_aircraft()`, before thinning -- completely decoupled
from ACE2's own cycling.

- `build_aircraft_temperature_bias_table.py` builds the table (one-off,
  not run as part of normal pipeline execution) from 76 real
  `diag_conv_t_ges.*_ensmean.nc4` GSI cycles, 2022-01-01 to 2022-01-19
  (~7.3M aircraft obs, itype 130/131/133/134/135), writing
  `aircraft_temperature_bias_table.json`. Cross-validated against the one
  real 2015-01-01 00Z cycle: the two agree well at cruise levels (150-250
  hPa, including the SH sign/magnitude) where most aircraft obs actually
  sit, less well in the mid-troposphere, where the single 2015 cycle is too
  small a sample (141K obs vs. 7.3M) to trust on its own. **The user is
  downloading more real 2015 `gdas1.t*z.cnvstat` files to rebuild this
  table from 2015-native data** -- rerun `build_aircraft_temperature_bias_table.py`
  once enough are available (see the script's own docstring for the path
  to point it at), or merge the two sources, rather than trusting the 2022
  table as final.
- Confirmed via the Zhu et al. (2015, *Mon. Wea. Rev.* 143(9):3774-3803,
  "Variational Correction of Aircraft Temperature Bias in the NCEP's GSI
  Analysis System" -- the paper that introduced `aircraft_t_bc` into GSI)
  that this is the physically right place to look: the bias comes from
  pitot-tube adiabatic-heating correction error, "largest around cruise
  level, where the Mach number is highest" -- matches the table's own
  150-300 hPa peak. The paper's own numbers (per-tail constant terms of
  -0.15 to -0.45 K in one 5-tail example, OmA/OmF std-dev reductions of
  13-17% at 300 hPa from bias correction) are consistent in scale with what
  this repo's own diagnostics found, and its min-data-count-per-cycle
  threshold (3) informed this design even though this implementation
  doesn't do per-cycle online updates.
- Config: `exp.observations.aircraft.bias_correction_enabled` (default
  `False`) and `.bias_correction_table` (default: the repo's own
  `aircraft_temperature_bias_table.json`) -- see `config.yml.template`.
  `_aircraft_temperature_bias_correction`/`_load_aircraft_temperature_bias_table`
  in `long_window_4dvar_utils.py` do the lookup; 0 correction wherever the
  table has no data for a (bin, region) combination (an unknown correction
  is left alone, not guessed at).
- Verified offline: the correction function against hand-picked
  (pressure, lat) pairs matches the table's own printed values exactly,
  and `load_config()` round-trips correctly with the flag on (confirmed
  the default-off config used by every existing/currently-running ACE2
  experiment is completely unaffected).

### Sensitivity tests t1/t2 (real GPU runs started 2026-10-23/24)

Two ACE2 configs, both derived from `config_test_ace2_ps_raobs_aircraft.yml`
-- see `sensitiveTest.record` for the one-line description of each, and
`run_ace2_ps_raobs_aircraft_t1.sh`/`_t2.sh` for the launch scripts:

- **t1** (`config_test_ace2_ps_raobs_aircraft_t1.yml`): aircraft temperature
  OFF, wind only -- `aircraft.report_types` restricted to the wind
  sub-stream's codes (`[230, 231, 233, 234, 235]`), which are disjoint from
  the temperature codes (checked directly, zero overlap) so this drops every
  temperature row at the source with no code change. Isolates how much of
  the regression is attributable to aircraft temperature specifically.
- **t2** (`config_test_ace2_ps_raobs_aircraft_t2.yml`): same baseline, with
  `bias_correction_enabled: True` -- the first real GPU exercise of the
  static bias-correction table above.

Both ran an initial real GPU cycling segment (t1 to 2015-01-14T06, 54
cycles; t2 to 2015-01-13T12, 51 cycles) and were added to
`z500err_ts_ace2_cycle_compare_parquet.py` /
`t850err_ts_ace2_cycle_compare_parquet.py`'s `experiments` dict. Early read
(partial window, not yet comparable to the ~101-cycle full baseline): in
the SH T850/z500 panels -- the region the regression showed up most
clearly -- **both t1 and t2 track noticeably below the full
`ps+raobs+aircraft` baseline, closer to `ps+raobs` alone**, over their
shared overlap period. Consistent with temperature being the main driver
of the SH regression and with the bias correction helping, not hurting --
but t1 and t2 track close enough to each other in this short window that
it isn't yet possible to tell whether removing aircraft temperature
entirely (t1) or correcting it (t2) performs better; that needs the longer
runs.

Both configs were then switched to `restart: True` (with `sdate` set to
each run's last completed cycle, and `ic_suffix` corrected from the
baseline's stale `'120h_50it'` to the actually-produced `'120h_100it'` --
the baseline config's `ic_suffix` had never been exercised for a real
restart before, so this mismatch wasn't caught until now) and resubmitted
to continue cycling further. Rerun the two comparison scripts once they've
caught up to the full baseline's cycle count for a real comparison.

**Not yet done**: the eventual full-month comparison (to confirm which, if
either, of t1/t2 actually closes the z500/T850 `ps+raobs+aircraft`
regression -- same before/after methodology already planned for AMV's own
bias-correction fix) is still open; current results are an early, partial-
window read only. Also open: whether splitting the bias-correction table
further by report type (130/131/133/134/135, i.e. AIREP vs. AMDAR vs.
MDCRS) sharpens the signal beyond pressure+region alone -- not checked;
and rebuilding the table from 2015-native `cnvstat` data once more cycles
are downloaded, per the note above.

## GSI-table correction vs. ACE2's own O-B (2026-10-26, reviewed -- explains why t2/t3 are flat)

With t1/t2/t3 all cycling to ~100 cycles
(`plots/z500err_ts_ace2_cycle_compare_lead6h_parquet.png`), the NH/SH
panels showed t2 (biascorr-2022) and t3 (biascorr-2015) tracking almost on
top of the uncorrected `ps+raobs+aircraft` baseline (green), while t1
(windonly) tracked close to `ps+raobs` alone -- i.e. the static bias table
correction wasn't visibly closing the regression, even though removing
aircraft temperature entirely did.

Checked directly whether the table's correction actually points the right
direction for ACE2's own background, by pooling `t_omb` (QC-passed,
`t_used==1`) from the bias-correction-OFF baseline run
(`test_ace2_6h_5d_50it_ps_raobs_aircraft`, 101 cycles, lead=0h slot only
per file to avoid the long-window diagnostics' cross-cycle overlap -- see
"Why this reads raw state files..." in the next section for the same
overlap issue), binned the same way as
`build_aircraft_temperature_bias_table.py` (pressure bin x NH/Tropics/SH)
and compared against `aircraft_temperature_bias_table.json`'s own values:

| pressure bin | region | ACE2 own O-B (K) | GSI table (K) | |
|---|---|---|---|---|
| 200-250 hPa | NH | -0.22 | +0.17 | sign flip |
| 200-250 hPa | SH | -0.13 | +0.55 | sign flip |
| 850-1000 hPa | NH | -0.13 | +0.39 | sign flip |
| 850-1000 hPa | SH | -0.20 | +0.08 | sign flip |
| 150-200 hPa | SH | +0.01 | +0.75 | same sign, table ~75x too big |
| 150-200 hPa | NH | +0.87 | +0.34 | same sign, table ~2.5x too small |

200-250 hPa is the single most heavily populated cruise bin (152k obs in
this sample). The GSI table is subtracting in the OPPOSITE direction from
ACE2's own actual background error there, in both NH and SH -- helping in
some bins, hurting in about as many others, netting out to ~no visible
z500/T850 skill change. Pooled over all pressures, ACE2's own NH/SH mean
O-B is tiny (-0.01 K NH, -0.03 K SH over the lead=0h sample) -- nowhere
near uniform enough for one region-wide number to matter; the real
structure is bin-by-bin and sign-alternating, and that structure doesn't
match GSI's 2022 operational table.

**Conclusion**: the flat t2/t3 result isn't evidence the correction
approach is wrong in principle -- it's that the correction was estimated
from the wrong model's background error (GSI's, not ACE2's). See "Where to
source the bias correction from" below for why the naive fix (re-estimate
from ACE2's own O-B) has its own, different problem.

## raobs-verified T/u/v profile across the observation-ablation series (2026-10-26/27)

`plots/raobs_tuv_profile_ace2_cycle_compare.py` -- bias/RMSE vertical
profiles of temperature and wind, verified against the real raobs (ADPUPA)
archive (NOT aircraft or AMV), background vs. analysis, across
`ps` / `ps+raobs` / `ps+raobs+aircraft` /
`ps+raobs+aircraft_t1(windonly)` / `ps+raobs+aircraft+amv` /
`ps+raobs+aircraft+amv+gpsro` (the same experiments the z500/T850 compare
scripts use, plus t1 from the sensitivity-test series above). Built to
check, independent of the z500/T850-vs-ERA5 metric, whether the aircraft
temperature regression is visible as a direct bias/RMSE shift in the
model's fit to an upper-air obs type the aircraft-carrying runs never
verify against directly.

**Why it reads raw state files instead of each run's own saved
`*_raobs_diagnostics_*.nc`**: `ps` alone never assimilates raobs, so it has
no such file at all -- offline evaluation is the only way to put it on
equal footing with the other runs. Also, `*_raobs_diagnostics_*.nc`'s
`*_oma` fields are ALL-NaN at `lead_time_hours==0` specifically (every
other lead is fine) -- not a save-routine bug: `compute_optimal`'s own
docstring (`long_window_4dvar_utils.py`) explains that the latent control
increment is only ever injected as a one-time forcing at the model's FIRST
forward step, so the earliest a corrected physical state exists is
`window_start + dt_verif` (6h here). There is structurally no analysis at
the literal window-start time, for any experiment.

**A real time-alignment bug found and fixed while building this**: the
first version compared this cycle's `*_control_inputs_*.nc` (background,
valid at `window_start`) against this cycle's `*_optimal_inputs_*.nc`
(analysis) assuming the same valid time -- but `*_optimal_inputs_*.nc`'s
own `valid_date` attribute is exactly `window_start + 6h`, per the
structural reason above. This produced a nonsense result (analysis RMSE
roughly double background RMSE for wind, i.e. assimilation apparently
making the fit worse). Fixed by comparing both at `window_start + 6h`:
analysis from this cycle's `optimal_inputs.nc` (already valid there), and
background from the NEXT cycle's `control_inputs.nc` -- built completely
independently (every config here uses `restart: False`, so each cycle's
background is its own fresh ERA5-seeded `nhrs_back`-hour forecast; ACE2 is
deterministic) but landing on the same real valid time. Confirmed directly
that this next-cycle `control_inputs.nc` agrees with this cycle's own
`control_forecast.nc` lead=6h step to ~0.1 K / ~0.1 m/s (floating-point-
level noise against a ~14 K / ~9 m/s field std, not a real discrepancy),
and that `control_forecast.nc`'s trimmed 5-level `save_levs` subset is why
the next-cycle-IC route is used instead (the full 13 standard levels are
only saved in the `*_inputs_*.nc` files, not the full trajectory file).
Both background and analysis are evaluated at the SAME raobs verification
slot (`raobs_traj`'s slot index 1, not 0) via the real
`get_raobs`/`_compute_raobs_observation_diagnostics_at_time`
(`long_window_4dvar_utils.py`), through a lightweight `_FakeModel`
stand-in (no GPU/model object needed -- same "fake model" pattern as the
offline validation above). Cross-checked against `ps_raobs`'s own internal
diagnostics file at one cycle: n_used matched exactly (6474/6650/6650 for
t/u/v), confirming the offline QC reproduces the real pipeline's accepted-
observation set exactly.

Spin-up (first 10 cycles) excluded; the six runs finished different cycle
counts (gpsro shortest), so the script intersects all available dates and
keeps cycle 11+ from the overlap -- 79 common cycles, 2015-01-03T12 to
2015-01-23T00, same for every experiment in every panel.

**Result**: `ps+raobs` (raobs the only upper-air obs type) has the
tightest fit to raobs in essentially every panel -- T, u, v, background
and analysis alike. Every other experiment, INCLUDING t1 (windonly, no
aircraft temperature at all), sits slightly worse, clustered together with
the full aircraft/amv/gpsro runs. This is the opposite of "more obs ->
lower RMSE" naive intuition, but not evidence the added obs types are
harmful: verifying against raobs specifically favors the raobs-only
baseline, since that's the only experiment where raobs is the SOLE
upper-air constraint in the cost function -- every other experiment makes
the shared latent increment satisfy raobs AND aircraft (AND amv/gpsro)
simultaneously, and a joint fit to multiple obs streams generically fits
any single one of them slightly worse than a fit to that stream alone.
The effect size is small relative to the `ps`-only (no raobs at all) gap.
Notable new piece: aircraft WIND alone (t1) also costs a small amount of
raobs wind fit, even though wind-only didn't show up as a z500 regression
in the earlier comparison -- so part of what's visible here is a general
"any added obs stream costs a little raobs-specific fit" effect, not
purely the aircraft-temperature-bias story.

**Not yet done**: the unbiased way to actually test whether aircraft/amv/
gpsro improve analysis quality is to verify against something none of the
experiments assimilate (ERA5, or a held-out raobs subset never fed to any
of them) rather than against raobs itself -- not built yet.

## Where to source the bias correction from (2026-10-26/27, discussed -- not yet implemented)

Working through what to try next for the aircraft temperature correction,
given the GSI-table cross-check above:

- **Any O-B-based table conflates sensor bias with the reference model's
  own background bias.** O-B = sensor_bias + (truth - background), not
  sensor_bias alone. This applies to GSI's table (GSI's own forecast model
  has *some* bias too, just small/well-tuned after years of its own VarBC
  applied across every obs type) AND to the natural "fix" of re-estimating
  the table from ACE2's own O-B instead -- which has the SAME conflation
  problem, just with ACE2's bias baked in instead of GSI's. This isn't
  only the earlier-flagged circularity concern (learning from a run that
  assimilated the very obs being corrected, which an ACE2-native table
  built from the raobs-only background would avoid) -- it's a more
  fundamental issue that affects ANY forecast-model background as the O-B
  reference, circular or not. Confirmed ACE2 does carry a real, nontrivial
  background bias of its own, directly in this session's own diagnostics:
  the raobs profile above shows `ace2-ps`'s background with a -1 to
  -1.5 K cold bias below 700 hPa, and even the raobs-assimilating runs
  aren't flat-zero bias at every level.
- **Raobs-collocation (`aircraft_T - raobs_T`, obs vs. obs, no model
  involved at all)** was raised as the cleanest fix in principle -- no
  forecast background anywhere in the estimate. Risk flagged before
  building it: collocation is likely sparse exactly where the real bias
  lives. The physical bias (pitot-tube adiabatic heating, Zhu et al. 2015)
  peaks at cruise level (150-250 hPa), but that's where aircraft-raobs
  collocation is hardest -- aircraft are mid-flight between airports at
  cruise, while a raobs balloon only passes through that altitude during a
  slow ascent from one fixed ground station. Collocation should be much
  better in the lower/mid-troposphere (aircraft climb/descent near
  airports, some of which coincide with raobs sites), which is exactly
  where the bias is smallest. SH/ocean raobs sparsity (only 5 stations
  with n>=10 in the existing bias table, see above) compounds this
  further. Not built -- a cheap feasibility check (count collocated pairs
  per pressure bin x region before committing to a full table) was
  proposed but not run.
- **ERA5-referenced bias (`aircraft_T - ERA5` at matched lat/lon/pressure/
  time)**: agreed as the most promising direction. Avoids both models'
  background bias entirely -- ERA5 is not a forecast background, and while
  its own DA does ingest aircraft obs, that influence is diluted across
  millions of multi-platform observations, making it a much cleaner truth
  proxy than any single model's raw background. This repo already has the
  ERA5-reading infrastructure (`backends/ace2/ace2_ic.py`'s `load_verif()`,
  used for z500/T850 truth) reusable for this, so no new tooling needed.

**Decision (2026-10-27): parked, not yet implemented.** User's call: "let
hold on doing that yet." Recorded here as the agreed next direction --
revisit by building the ERA5-referenced pressure x region bias table
(reusing `load_verif()`) the next time this is picked up.
