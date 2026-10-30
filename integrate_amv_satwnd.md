# Adding satellite-derived wind (AMV) observations from the PREPBUFR SATWND class

Read this when working on: the AMV (atmospheric motion vector) observation
stream, its box-thinning design, the `exp.observations.amv` config schema,
or the decision to read the PREPBUFR archive's `SATWND` class instead of
(for now) the separate raw-BUFR GOES-16 `amv_parquet` archive.

## Status: implemented; improves z500 skill over the first several cycling days, then degrades via a growing, uncorrected O-B bias under cycling (diagnosed 2026-10-21, not yet fixed) -- see "AMV cycling-bias investigation" below. Also now running on `model_backend: ace2` (confirmed stable improvement over ps+raobs through ~01-12 cycling); a vertical-thinning sensitivity sweep (pmesh50/120/200 vs. nearest_level) found NO effect on the NH/Tropics-vs-SH regional imbalance, ruling out vertical thinning as its driver -- see "Vertical thinning tied to model levels" below.

## 2026-10-20: GSI-style box thinning replaces composite-stencil superobbing

The first real GPU run (4 cycling experiments, `ps` / `ps+raobs` /
`ps+raobs+amv` / `ps+raobs+amv+gpsro`, aifs2-1.0deg-v1.0, 2015-01) showed
z500 forecast error getting WORSE, not better, once amv (and, to a lesser
extent, gpsro) was added -- the opposite of the expected monotonic
improvement. Diagnosis against that run's own diagnostics netCDFs plus a
comparison with GSI's actual operational SATWND code
(NOAA-EMC/GSI, commit `55faa5e7af4f0604bd1ed38b385fd762863ca294`,
`src/gsi/read_satwnd.f90`/`setupw.f90`) found two compounding problems with
the original composite-stencil-superobbing design (both now fixed):

1. **No gross-error/background QC at all for AMV** -- unlike psobs
   (`bg_check`), raobs-q (`q_bg_check`), and gpsro
   (`gpsro_background_check_stddev`), `_compute_amv_observation_diagnostics_at_time`
   had no `|O-B|`-vs-error rejection gate. The real run's own diagnostics
   showed AMV superobs with `|O-B|` up to 74 m/s (an ~18-sigma outlier
   against the ~4 m/s floor error) being fit at full quadratic weight --
   alone, the >20 m/s tail contributed more to that window's `Jo` than the
   entire psobs term. Fixed: a GSI-style clamped-ratio gross check (see
   "Implementation" below), matching `setupw.f90`'s
   `ratio=(O-B)/clip(obserr,ermin,ermax) > gross` rejection.
2. **Superobbing collapsed to near-raw density on this repo's irregular
   grid.** The reference repo's design (`satwnd_superobs.md`) groups raw
   vectors sharing the SAME fixed interpolation stencil -- on its REGULAR
   lat/lon grid that's unambiguously one native grid cell. Ported to this
   repo's k-NN `GridInterpolator`, "same stencil" became "same *ordered*
   k=4-nearest-neighbor tuple," which on an irregular grid is a much finer
   partition than one native grid cell -- empirically ~5-30 km (median
   27.6 km) vs. AIFS's ~112 km native spacing, confirmed by measuring
   nearest-neighbor spacing between superob locations in the real run's
   diagnostics. Compression was only ~1.13x (consistent with what was
   flagged, at the time, as "not a bug" below -- it IS the actual problem
   here), so ~2.9M AMV superobs/window entered the loss, >5x the combined
   ps+raobs count, at near-independent weight.

**GSI's own SATWND treatment turns out not to superob (average) at all.**
Checked directly in `read_satwnd.f90`: it uses the generic conventional-
data thinning utility (`convthin.f90`'s `make3grids`/`map3grids_m`) with a
physical horizontal x vertical mesh (`rmesh` km x `pmesh` mb, from
`convinfo`, INDEPENDENT of the model's own grid) and keeps exactly ONE
observation per mesh cell -- the one with the lowest "badness" score
`crit1 = |time_offset_hours|/6 + 0.5` (further adjusted for some types using
the AMV producer's own quality-indicator/expected-error BUFR fields, which
this repo's normalized parquet archive does not carry -- checked its schema
directly, only the generic `wind_quality`/`wind_error_ms` PREPBUFR columns
are present). GSI's own downstream background gross check
(`setupw.f90`, shared by every conventional type) is the
`ratio=(O-B)/clip(obserr,ermin,ermax) > gross` form adopted above.

This repo now mirrors that design: `_amv_thin_slot` (was
`_amv_superob_slot`) replaces inverse-variance averaging with GSI-style box
thinning (`amv_rmesh_km`/`amv_pmesh_hpa`, default 120 km / 50 hPa --
120 km matches the GSI User's Guide's own worked conventional-thinning
example; 50 hPa is a reasonable starting point, not a confirmed
operational SATWND value), tie-broken by time proximity then assigned
error (standing in for GSI's QI/EE, which isn't available here). See
"Implementation" and "Offline validation" below for what changed and how
it was checked.

### Real-run confirmation (2026-10-20)

A real GPU run (`aifs2-1.0deg-v1.0-pretraining-ps-raobs-amv-boxthin-bkgcheck`,
`config_test_aifs2-1.0deg-v1.0_ps_raobs_amv.yml`, new exp_name per the
"give every concurrent run its own exp_name" rule) confirmed the fix on
real data, not just synthetic tests: file timestamps verified it ran AFTER
the code fix (`2015-01-01T00_amv_diagnostics` dated 2026-10-20 19:44, vs.
the code edit at 18:20; the old, pre-fix run's own diagnostics are dated
2026-10-19 23:44, so the two are a genuine before/after pair). Reader-side
counts matched the offline estimate almost exactly (00Z slot: 28,062 u /
28,250 v kept, vs. the offline test's 28,308).

Comparing z500 error at lead=24h over the first 11 cycles
(2015-01-01 00Z - 2015-01-06 00Z, via
`plots/z500err_ts_aifs2-1.0deg-v1.0_cycle_compare_parquet_daterange.py`):
the fixed amv run is now the best or tied-best performer in NH/SH/Global,
and -- the key result, since this is what wasn't happening before --
**it now beats `ps+raobs` alone** in SH and Global, not just the old
(pre-fix) amv run. GPSRO on top of it (`ps+raobs+amv+gpsro`) is still the
worst performer over this window, now clearly isolated as a separate
problem rather than masked by amv's own bug -- see integrate_gpsro.md.

**Caveat: only an 11-cycle (5-day) window so far.** The ORIGINAL
full-month plot that motivated this whole investigation showed the old,
buggy amv run's divergence from `ps+raobs` growing slowly, not appearing
immediately -- it didn't clearly separate until ~day 10+. So this is a
genuinely encouraging early signal, not yet full confirmation the fix
holds over a long cycling period. **Rerunning the daterange plot over the
fuller 2015-01-01 -> 2015-02-01 period (or longer) is the natural next
step** to confirm red stays flat instead of drifting the way the old
green line eventually did.

## Why SATWND, not the raw GOES-16 `amv_parquet` archive

The original ask was to assimilate AMVs from
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/prepbufr_to_parquet/data/amv_parquet`
-- a separate converter (`convert_amv_bufr_to_parquet.py`, sibling repo)
that decodes the raw NOAA NNJA `amv/satwnd` BUFR archive, selecting GOES-16
(`SAID=270`) only, per its own `goes16_amv_implementation_plan.md`.

That archive is real but **unusable for this repo's 2015 test period**:
GOES-16 did not exist until Nov 2016, and the local `amv_parquet` archive
only covers 2015-01 through 2015-07. Checking every `.counts.json` in that
archive (800 cycle files) confirms `emitted_rows: 0` everywhere -- not a
bug, just a satellite-era mismatch with the only period actually fetched.

Separately, the SAME normalized PREPBUFR parquet archive that
`psobs_parquet.py`/`raobs_parquet.py` already read (produced by
`convert_prepbufr_to_parquet.py`, a different, more general converter) was
found to carry a **`SATWND` Table-A class**: 107,728 reports in a single
2015-01-01 00Z cycle, `report_type` values 242/243/245/246/250/252/253/
257/258/259 (the standard NCEP legacy AMV network -- GOES/JMA/EUMETSAT/
MODIS cloud-drift and water-vapor winds), already merged, de-duplicated,
and quality-controlled into the operational PREPBUFR feed, in the exact
same `reports`/`levels` join shape as every other class (single-level,
like `ADPSFC`/psobs -- confirmed in the sibling repo's own notes).

So the practical, immediately-usable AMV source for this repo's existing
2015 configs is `SATWND`, not the raw GOES-16 archive -- no sibling-repo
changes, no BUFR descriptor reverse-engineering, real data for the exact
period `config_test_aifs2-1.0deg-v1.0*.yml` already use.

## Reference precedent: `greg-long-window-4dvar`'s `sc_gpsro_goes16` branch

`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/greg-long-window-4dvar/long-window-4dvar`
(the JAX/NeuralGCM sibling solver, `sc_gpsro_goes16` branch) already
assimilates AMVs from BOTH sources, and its own written design decision
(`goes16_amv_implementation_plan.md`) settles the "should the two archives
be merged into one" question used to motivate this implementation:

> Keep quality thresholds, thinning, background checks, superob settings,
> and assigned observation errors in experiment configuration rather than
> in the upstream converter.

It unifies the two sources only at the assimilation layer: one observation
type (`observations.upper_air`), a `source: prepbufr | amv_parquet` switch
(`observation_sources.py`), automatic satellite-id exclusion from the
PREPBUFR stream when a concurrent raw GOES-16 stream covers the same
satellite (to avoid double-counting once GOES-16 eventually appears in
PREPBUFR too), and mandatory superobbing for both sources
(`satwnd_superobs.md`).

**This port implements only the `source: prepbufr`/SATWND half.** The raw
GOES-16 `amv_parquet` half is deliberately deferred -- not just because of
the 2015 data gap above, but because the config/dispatch seam
(`exp['amv_enabled']`, `get_amv`) is already shaped the way the reference
shapes it, so adding a second source later is a new loader module plus a
`source`/`exclude_satellite_ids` branch in `get_amv`, not a redesign. The
`exclude_satellite_ids` config key is already wired through (see
config.yml.template) even though nothing populates it automatically yet --
that automatic cross-stream exclusion is exactly what to add if/when a
GOES-16 stream is added.

## Implementation

- **Reader** (`amv_parquet.py`, repo root): `load_amv_parquet`, ported from
  the reference's `prepbufr_parquet.py:load_prepbufr_upper_air_points`,
  restricted to `SATWND`. Returns flat, per-slot-bucketed, UN-superobbed
  points (lat/lon/pressure/u/v/error_std/report_type/satellite_id) --
  model-grid interpolation and superobbing are deliberately left to the
  caller (`get_amv`), since they depend on the configured model's grid and
  pressure levels. Uses a vectorized `np.searchsorted` report_id join and a
  vectorized nearest-verification-time assignment (not a Python per-row
  loop like psobs/raobs use) -- SATWND's row count (1e5+ per cycle) makes
  that necessary; ported from the reference's own vectorized approach.
  Shares `prepbufr_parquet_common.py` with psobs/raobs.
- **Thinning** (`_amv_horizontal_vertical_stencil`, `_amv_thin_slot`,
  `long_window_4dvar_utils.py`): GSI-style box thinning (see the redesign
  section above), restricted to the SATWND-relevant subset (no AIRCAR
  flight/phase/profile-sampling machinery, which doesn't apply here). Raw
  SATWND wind vectors are grouped by a horizontal `rmesh_km` x vertical
  `pmesh_hpa` mesh cell (independent of the model's native grid -- a
  reduced near-equal-area lat/lon mesh, lat bands of height ~`rmesh_km`
  with longitude bins per band narrowed by `cos(lat)`), plus, by default,
  the same report type/satellite id. Exactly ONE member per group is kept
  -- the one with the lowest GSI-style `crit1 = time_offset_hours/6 + 0.5`
  score, tie-broken by assigned error (standing in for GSI's own
  QI/expected-error tie-break, unavailable in this repo's archive). The
  kept observation's own k=4 horizontal k-NN (`grid_interp.weights`, same
  mechanism psobs/raobs use) x log-pressure-bracket 2*k=8-node stencil is
  stored directly -- no averaging, so no in-group error-correlation
  combination is needed either (dropped entirely, along with
  `component_weighting`/`representativeness_error_ms`).
- **Forward operator** (`_compute_amv_observation_diagnostics_at_time`,
  `long_window_4dvar_utils.py`): a direct gather against the stored 2*k=8-
  node stencil (`decoded[base][node_level_idx, node_point_idx]`,
  weighted-summed) -- no `grid_interp.interp` call needed, since the
  stored weights already fold in both the horizontal and vertical blend.
  Also applies the GSI-style background gross check here (ported from
  `setupw.f90`): rejects an observation whose `|O-B|` exceeds
  `amv_background_check_stddev * clip(error, amv_error_min_ms,
  amv_error_max_ms)`, evaluated at the CURRENT state every loss call (same
  convention as raobs's `q_bg_check`/gpsro's `background_check_stddev`,
  simpler than GSI's own frozen, once-before-optimization check);
  `error_min_ms`/`error_max_ms` default to no clamping.
- **Loss integration**: a `amv_traj=None`-gated branch
  (`_amv_term`, parallel to `_raobs_term`) inside `compute_loss_4dvar`'s
  existing per-step loop -- NOT a second rollout, same obs-slot grid as
  psobs/raobs (`get_amv` reuses `psobs_traj['step_lo']`/`['alpha']`).
  `_resolve_loss_interp_specs` folds 'u'/'v' into the same
  time-interpolated decode when `amv_enabled`.
- **Diagnostics**: `save_amv_diagnostics` (parallel to
  `save_raobs_diagnostics`), one netCDF file
  (`<date>_amv_diagnostics_*.nc`) with per-variable O-B/O-A/used fields on
  `(time, superob_slot)` -- the `superob_*` dimension/variable naming is
  kept from the earlier design for diagnostics continuity, even though
  each slot is now one kept (not averaged) observation. Unlike raobs,
  there is no shared level-coordinate dimension (each kept observation has
  its OWN pressure, not a mandatory-level match), so
  pressure/lat/lon/report_type/satellite_id/component_count (now "raw
  vectors found in this thinning box, 1 kept") are recorded as per-(time,
  superob_slot) fields instead.
- **Config**: `amv.*` keys documented in `config.yml.template` -- all
  optional, defaulting to (1h time tolerance, quality_mark_max 3,
  use_source_error True, fallback_error_std_ms 7.0, error_floor_ms 4.0,
  separate_report_types/separate_satellite_ids True, rmesh_km 120.0,
  pmesh_hpa 50.0, background_check_stddev 4.0).
- **Backend guard**: `amv_enabled` is rejected by `load_config` for any
  backend other than `aifs`/`aurora` (`AMV_SUPPORTED_BACKENDS`), the same
  conservative restriction raobs uses -- 'u'/'v' decoding confirmed for
  those two, not checked for FCN3/ACE2/SFNO.

## Offline validation (no model/GPU)

All three checks below used the real archive at
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/prepbufr_to_parquet/data/prepbufr_parquet`,
2015-01-01 00/06/12Z, under the `aifs2` conda env (has `pyarrow`).

1. **Reader, in isolation**: 56,821/81,690/82,897 accepted wind vectors per
   slot (00/06/12Z) out of ~107K/cycle candidate rows. Physically sane
   throughout: lat [-89,89], lon [0,360), pressure 100-1000 hPa (satellite
   winds are cloud/WV-height-bounded, never reach the model's 10 hPa top),
   |u|/|v| up to ~90 m/s (jet-level cloud-drift winds), error_std 3.8-8.0
   m/s, report types exactly the known legacy AMV set. Still accurate --
   unaffected by the 2026-10-20 thinning redesign.
2. ~~Superob linearity test~~ (superseded, composite-stencil design
   removed 2026-10-20) -- see items 4-5 below for the thinning design's
   own validation.
3. **Full `get_amv` pipeline, real 2015-01-01 00Z data + a fake model/grid
   (no AIFS)** (re-run 2026-10-20 against the thinning redesign): a
   `FakeGridInterpolator` (regular 360x180 lon/lat grid, density close to
   AIFS-single-2.0-1.0deg-v1.0's O96 reduced grid) exercised `get_amv`
   end-to-end against the real archive -- 56,821 accepted raw vectors ->
   28,308 kept after thinning at the default 120 km/50 hPa mesh (~2.0x
   compression, vs. the superseded design's ~1.13x on the same slot),
   finite error_std everywhere, floor respected, `background_check_stddev`
   correctly threaded into the trajectory.
4. **Thinning-selection correctness, synthetic data**: a 3-member cluster
   (same box, equal time offsets) plus a 1 singleton (different box)
   confirmed exactly one kept observation per box, selecting the member
   with the lowest `crit1` (here, after a time-offset tie, the lowest
   assigned error) -- not an arbitrary or first-seen member.
5. **Background gross check, synthetic data**: a synthetic 74 m/s
   innovation (18.5x the 4 m/s error) against two well-fit (0 m/s)
   innovations in the same slot was correctly marked `used=False` for the
   outlier only, confirming the ported `ratio=|O-B|/error >
   background_check_stddev` rejection fires as intended -- this directly
   closes the gap identified from the real run (see the redesign section
   above).

## AMV cycling-bias investigation (2026-10-21)

A full 2015-01-01 -- 2015-02-06 cycling comparison (aifs2-1.0deg-v1.0,
`ps+raobs+amv(gsi-thin)` vs `ps+raobs`, z500 at lead=24h, 4 hemispheric
panels) showed the fixed amv run beating `ps+raobs` for the first several
cycles in every region, then falling behind and staying behind (NH before
~01-09, TR before ~01-05, SH before ~01-13) -- early benefit, then
persistent degradation, not a flat offset either direction.

**No per-observation-type loss weighting exists.** `compute_loss_4dvar`
just sums every stream's `Jo` unweighted (`long_window_4dvar_utils.py`'s
`J = sum(Jol); J = J + sum(Jamv); J = J + sum(Jgpsro); J = J + sum(Jraobs)`,
around line 2648) -- there is no `amv_weight`/`raobs_weight` config knob, so
each stream's influence on the gradient scales directly with its raw
observation count. Pulled from a real GSI-thinned run's own log
(`logs/lw4dvar_aifs2_1p0deg_v1p0_ps_raobs_amv_gpsro_24195052.out`): at
off-synoptic slots (no raobs, half of every 6-hourly window) the `J`
breakdown was ps 7.5%, amv 68.2%, gpsro 22.3% -- amv alone, even after
120 km/50 hPa GSI box-thinning, keeps ~130,000 u/v obs/slot, ~12x ps's count
and 5-10x raobs's synoptic-slot count.

**That dominant weight is what injects a persistent, partly-growing
systematic bias into the cycled background.** There is no bias correction
(VarBC) anywhere in the amv pipeline -- `_AMV_CONFIG_KEYS` only carries
`use_source_error`/`fallback_error_std_ms`/`error_floor_ms`, no per-type/
per-level additive bias term the way GSI runs VarBC on SATWND. Pulled real
numbers from the GSI-fixed run's own diagnostics
(`output/aifs2-1.0deg-v1.0-pretraining-ps-raobs-amv-boxthin-bkgcheck/
*_amv_diagnostics_*.nc`, mean `O-B`, i.e. real bias, not RMS):

| region/component | day 1 | ~day 5 | ~day 9-13 | ~day 16-18 | later |
|---|---|---|---|---|---|
| TR v-wind | +0.17 m/s | +0.26 (01-05) | +0.35-0.40 (01-09/13) | **+0.55 peak (01-16/18)** | settles ~+0.35-0.45 |
| SH v-wind | -0.50 (spinup) | -0.12 to -0.20 | **jumps to -0.32 at 01-12/13** | stays ~-0.2 to -0.3 | ~-0.2 |
| NH u-wind | -0.94 | -0.63 | -0.71 to -0.75 | -0.87 to -0.91 | oscillates -0.6 to -0.9 |

This tracks almost exactly with the onset dates above: TR's bias grows
essentially monotonically from day 1 (already past +0.25 by 01-05, where TR
amv stops beating `ps+raobs`); SH's `O-B` RMSE jumps from ~3.4-3.5 to >4.0
right at 01-12/13, matching the SH threshold almost to the day; NH's bias is
large from day 1 but its `O-B` RMSE keeps climbing through cycling (6.0 at
day 1 -> 4.3 by day 3 -> back up to ~5.0-5.2 by day 27+), a background
fitting amv progressively worse over the window. Critically, `ps+raobs`
alone shows no such trend over the same period (flat z500 error) --
ruling out generic AIFS-1.0deg-v1.0 climate drift as the cause, since that
would show up there too independent of amv.

**Mechanism**: a systematic (non-zero-mean) innovation, assimilated at
~70-90% of the total loss weight, doesn't average out -- it pulls the
analysis the same direction every 12h cycle. Since each cycle's analysis
becomes the next cycle's background, that pull compounds: early cycles
absorb real information (the observed early skill gain), but with nothing
removing the bias itself, the `O-B` misfit against amv actually GROWS over
cycling instead of converging to a flat residual -- a feedback loop, not a
one-time adjustment.

**Recommended fixes, not yet implemented**: (1) add AMV bias correction --
even a simple per-cycle/region/level running-mean `O-B` subtracted before
computing the loss innovation (a minimal VarBC); (2) add the missing
per-observation-type loss weight so amv's influence can be tuned
independently of its raw volume. Rerun the same `boxthin-bkgcheck` config
with (1) on and check whether `O-B` RMSE stops growing over cycling, before
touching anything else (gpsro error model, ACE2, ...).

**ACE2 as a drift-free control, 2026-10-21**: to separate "amv's own
cycling-bias problem" from "AIFS-1.0deg-v1.0's own climate drift" more
cleanly than the flat-`ps+raobs`-baseline argument above, `amv_enabled` was
extended to `model_backend: ace2` (`AMV_SUPPORTED_BACKENDS`, same change as
raobs -- see integrate_prepbufr_obs.md's "raobs admitted for
`model_backend: ace2`" section for what changed, why it was a pure
guard-list change, and the offline-validation evidence; `get_amv` itself has
no backend-conditional code). ACE2 is the one backend already shown to
cycle stably with ps obs alone over a long run (147 cycles, no detectable
drift, `integrate_multiple_backends.md`) -- if the same growing,
region-specific O-B bias pattern reappears there, that isolates the cause
to amv's own uncorrected assimilation; if it doesn't, that points to an
AIFS-specific interaction instead. New config:
`config_test_ace2_ps_raobs_amv.yml` / `run_ace2_ps_raobs_amv.sh` (jobs
24270091/24273858, plus the 24295238-40 sweep below -- **this specific
config/script pair was deleted 2026-10-22** when the `config_test_ace2*.yml`
family was regenerated into an incremental psobs->raobs->aircraft->amv->
gpsro ladder, see integrate_gpsro.md's "Observation-ablation series,
regenerated"; `config_test_ace2_ps_raobs_aircraft_amv.yml` tests a superset
of what this run tested). **Run 2026-10-21** (in a copy of this repo at
`.../add-amv-gpsro/lw4dvar-torch-test/`, same code, see below) -- 11 cycles
so far; see "Vertical thinning tied to model levels" below for what it
showed.

## Vertical thinning tied to model levels (2026-10-21)

A real ACE2 `ps`/`ps+raobs`/`ps+raobs+amv` cycling comparison (11 cycles,
2015-01-01 -- 01-11, lead=24h z500) confirmed amv still improves z500 skill
on this drift-free backend too -- but the benefit is dramatically uneven by
region: a large gap in SH (amv ~9-10 m vs. `ps+raobs` ~13-14.5 m), a modest
gap in NH (~8-9 m vs. ~9-10.5 m), and essentially no gap in the Tropics
(the two lines overlap throughout). Since amv is assimilated identically
everywhere, this regional unevenness pointed at amv's vertical thinning
rather than its horizontal coverage or QC.

**Diagnosis**: the default thinning (`_amv_thin_slot`, GSI's own
`pmesh_hpa=50` fixed-width vertical mesh) gives `floor(1000/50) -
floor(50/50) + 1 ~= 20` possible vertical bins across the
50-1000 hPa range AMV reads -- independent of the host model's vertical
resolution, because GSI's own `convinfo` `pmesh` value is tuned to
satellite-pixel correlation length, not to the model's resolving power
(GSI always runs 64-150+ level host models, so this never needed to be
model-aware). ACE2 has only **8 native hybrid-sigma layers** -- the 13
pressure levels amv reads are all *derived* from those same 8 values via
log-pressure interpolation (`ace2_model.py`'s `_to_pressure_levels`), so
there are really only ~8 independent vertical degrees of freedom the
analysis can express, not 20. At `pmesh_hpa=50`, amv can inject up to
~2.5x more vertically-independent-looking corrections than the model can
actually resolve -- the same class of over-sampling that forced GPSRO's
vertical thinning to be tightened beyond what GSI itself does on this
repo's low-level-count backends (`integrate_gpsro.md` "Thinning
restored"), just milder here since amv thinning was never off entirely.
Compounding this: `pmesh_hpa`'s fixed-width bins aren't anchored to the
model's actual levels at all, so the single kept observation per bin
(selected purely by GSI's time-closeness `crit1`, not by pressure) can
land anywhere within a bin's span -- potentially 50+ hPa from the nearest
real model level, even though the forward operator still interpolates
correctly at that observation's own true pressure (not a wrong-level bug,
but a representativeness one: the retained vertical sample isn't
necessarily where the model's structure actually needs it).

**Fixed**: `_amv_thin_slot`/`get_amv` gained an
`amv_vertical_bin_mode` option (`long_window_4dvar_utils.py`), `'pmesh'`
(default, GSI's scheme, UNCHANGED behavior -- existing AIFS configs are
unaffected) or `'nearest_level'` (new): bins by the observation's own
nearest model pressure level (`level_lo_idx`/`level_hi_idx`/
`vertical_frac`, already computed by `_amv_horizontal_vertical_stencil`
for the forward-operator stencil -- no new computation needed) instead of
a fixed-width mesh. This both caps the vertical bin count at the model's
own level count (13 for ACE2's derived levels -- not quite the full
reduction to 8 true native-layer degrees of freedom, which would need
backend-specific plumbing into ACE2's own 8-layer index rather than its
13 generically-exposed pressure levels, not done here) and guarantees
every kept observation is closer to its assigned level than to any other,
by construction (the Voronoi cell of that level in log-pressure) --
stronger than a combined time+distance score could promise, and avoids
inventing a new free parameter to weight time-freshness against pressure-
proximity. Selection *within* a bin is still GSI's own `crit1`
unchanged -- only where bin boundaries fall changes.
Config: `config_test_ace2_ps_raobs_amv.yml` now sets
`observations.amv.vertical_bin_mode: nearest_level`.
Offline-validated against the real archive (`'pmesh'` vs `'nearest_level'`,
same slot): both modes run cleanly, kept counts are comparable (35287 vs.
33176 for one slot) with `'nearest_level'` independently confirmed to
assign every kept observation to one of the 13 model levels as its true
nearest.

**Why 13 (derived levels), not 8 (true native layers)**: `get_amv`'s
forward operator (`decoded['u']`/`decoded['v']`, what the loss actually
compares against observations) is evaluated on ACE2's 13 *derived*
pressure levels (`_to_pressure_levels`'s hydrostatic/log-p interpolation
from the 8 native layers) regardless of which thinning scheme is used --
`'nearest_level'` only changes which observations survive thinning, not
how the survivors are evaluated. And the 13 derived levels are NOT 13
independent unknowns: ACE2's `control_variables` (what the 4D-Var
increment can actually move) are the native per-layer channel names only
(`air_temperature`/`specific_total_water`/`eastward_wind`/
`northward_wind` at k=0..7 -- `ace2_model.py`'s `resolve_columns`
explicitly rejects the derived `'t'`/`'u'`/`'v'`/`'z'` names as control
variables, "no packed columns -> KeyError"; they're valid `loss_variables`
only). So two AMV observations assigned to adjacent derived levels are
still constraining overlapping combinations of the same 8 real unknowns,
not two independent ones -- `'nearest_level'` (13 bins) is a real
improvement over `'pmesh'` (~20 bins at the default `pmesh_hpa=50`), but
not the full reduction to the model's true 8 degrees of freedom. Getting
all the way there would mean binning by nearest NATIVE layer index instead
of nearest derived pressure level -- needs new backend-specific plumbing
(ACE2's native layers are hybrid-sigma, so each layer's actual pressure
varies by location/time via `ak + bk*ps`, unlike the fixed 13-level ERA5
grid `model.pressure_levels()` generically exposes) -- not done, deferred
pending the sweep below.

**Sensitivity sweep, submitted 2026-10-21** (in
`.../add-amv-gpsro/lw4dvar-torch-test/`, a copy of this repo with the same
`_amv_thin_slot`/`vertical_bin_mode` code): three variants of
`config_test_ace2_ps_raobs_amv.yml`/`run_ace2_ps_raobs_amv.sh`, each with
its own `exp_name`/`rpath`/job name/log path, to separate two effects --
"does getting the vertical bin *count* roughly right help" vs. "does
getting bin *placement* (anchored to real levels) help on top of that":

| test | config | vertical_bin_mode | pmesh_hpa | job ID |
|---|---|---|---|---|
| 1 | `config_test_ace2_ps_raobs_amv_pmesh120.yml` | `pmesh` | 120 (~8 bins, matches true DOF count, arbitrary placement) | 24295238 |
| 2 | `config_test_ace2_ps_raobs_amv_pmesh200.yml` | `pmesh` | 200 (~5 bins, more aggressive, arbitrary placement) | 24295239 |
| 3 | `config_test_ace2_ps_raobs_amv_nearestlevel.yml` | `nearest_level` | unused (13 bins, anchored placement) | 24295240 |

`pmesh_hpa=120`'s own derivation ((1000-50)/8 ~= 120) is a bin-*count*
heuristic only -- it does not and cannot make `'pmesh'` mode's bins land
near ACE2's actual (unevenly-spaced, hybrid-sigma) native layers, so it
was deliberately left as-is rather than "corrected" once this distinction
was clarified; only a level-anchored scheme (`'nearest_level'`, or the
not-yet-built native-layer version above) can fix placement. Expected
ranking going in: `'nearest_level'` > `pmesh120` > `pmesh200`, since
`'nearest_level'` is the only one of the three addressing both bin count
AND placement. **Result (2026-10-22, evaluated): expectation wrong -- no
detectable difference between any of the four variants.** Diagnostic:
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/plots/
z500err_ts_ace2_cycle_compare_lead24h_parquet.png`, all four
`ps+raobs+amv(...)` lines (`pmesh50` default/baseline run
`config_test_ace2_ps_raobs_amv.yml`, job 24273858, plus the three sweep
configs/jobs above) overlapping almost exactly in every region/background
and analysis, clearly separated from (better than) `ps+raobs` alone but
not separated from EACH OTHER, over 2015-01-01 through ~01-12
(all four jobs ran to their time limit around that date) -- well past the
original 11-cycle sweep window and past the day-9/13 point where the
NH/TR/SH skill split in "Vertical thinning tied to model levels" above was
first characterized, so this isn't a too-short-to-tell null result.

This null result is NOT because the thinning change was a no-op: kept
`u`/`v` obs counts at the very first slot (t+0h, same raw archive, same
candidate pool across all four runs) were 28068/28254 (`pmesh50`) >
26772/26948 (`nearest_level`) > 24144/24321 (`pmesh120`) >
22923/23084 (`pmesh200`) -- an ~18% swing in kept volume between the two
extremes, each run's own log (`epoch 1, oind 0 (t+0h): amv J ..., obs used
{...}`) confirms the counts stayed fixed (not re-thinned) across epochs
within a run, and `nearest_level` genuinely changes bin *placement* (one
bin anchored per model level, not a fixed-width unanchored mesh) not just
count. So the mechanism worked as designed; it's the z500 skill that
didn't move.

**Conclusion: vertical-thinning bin count/placement is ruled out as the
(or a major) driver of the SH-vs-NH/Tropics regional AMV skill gap**
found in "Vertical thinning tied to model levels" above -- a ~2x volume
change and a qualitatively different placement scheme left lead=24h z500
error statistically indistinguishable in all four hemispheric panels, for
every region including SH (where the gap was largest). This strengthens
rather than displaces the leading suspect already named in "AMV
cycling-bias investigation" above (no VarBC / no per-observation-type loss
weight) as the next thing to try -- further tuning AMV's vertical
thinning is not expected to help and is deprioritized.

## Not yet done

- **Only an 11-cycle (5-day) real GPU run exists so far** (see "Real-run
  confirmation" above) -- encouraging, but not a full-length confirmation.
  **Rerunning over the fuller 2015-01-01 -> 2015-02-01 period (or longer)
  is the main open item** -- the original bug's divergence from
  `ps+raobs` only became visible after ~day 10 in the old (buggy) run, so
  a short window can't yet rule out a similar slow drift in the fixed
  version.
- **No reviewed/Desroziers-style error profile.** Raobs's `q_error_g_kg`
  table (superseding the initial fraction-of-ob error after a real O-B
  diagnosis) has no AMV analogue yet -- this starts with `use_source_error:
  True` (each report's own `wind_error_ms`) plus a flat floor, the same
  "simplest correct thing first" sequencing raobs followed before its own
  error-table refinement. GSI's own per-type/per-level `etabl_uv` table
  (`read_satwnd.f90`) would be the natural next step, but needs a real
  diagnosed climatology this repo doesn't have yet.
- **GSI's AMV-specific quality-indicator/expected-error (QI/EE) fields are
  not available.** Checked directly: this repo's normalized PREPBUFR
  parquet schema (shared across all classes) only carries the generic
  `wind_quality`/`wind_error_ms` PREPBUFR columns, not SATWND's own
  `qifn`/`qify`/`ee`/`pct1` BUFR sequences (GSI's PRIMARY SATWND QC gate,
  `read_satwnd.f90`'s `if (qifn < 85) qm=15`). Porting that would mean
  extracting those fields in the upstream `convert_prepbufr_to_parquet.py`
  converter -- a separate project outside this repo.
- **The raw GOES-16 `amv_parquet` stream is not implemented** -- see "Why
  SATWND" above. The `source`/`exclude_satellite_ids` seam is shaped for
  it, but no second reader, no automatic cross-stream satellite-id
  exclusion, and no GOES-16-era archive refresh exist yet.
- **No bias correction (VarBC) for AMV** -- see "AMV cycling-bias
  investigation" above. This is now the leading suspect for why
  `ps+raobs+amv` degrades over a month of cycling despite each individual
  analysis looking like a good fit; needs implementing and testing before
  anything else in this list.
- **No per-observation-type loss weight** -- see "AMV cycling-bias
  investigation" above; amv's influence on the gradient is currently an
  uncontrolled function of its raw (post-thinning) observation count, not a
  deliberate choice.
- **The ACE2 `ps+raobs+amv` run (11 cycles so far) hasn't run long enough
  to confirm or rule out the growing-bias cycling pattern found on
  aifs2-1.0deg-v1.0** -- see "ACE2 as a drift-free control" above; the
  original aifs2-1.0deg-v1.0 bug's divergence only became visible after
  ~day 10, so an 11-cycle (5.5 day at 12h cycling, or ~2.75 day at this
  run's 6h cycling) ACE2 run can't yet rule out a similar slow drift here.
  Rerunning longer is the natural next step once the vertical-thinning
  sensitivity sweep below has a result.
- ~~Vertical-thinning sensitivity sweep (pmesh120/pmesh200/nearest_level)~~
  (evaluated 2026-10-22) -- see "Vertical thinning tied to model levels"
  above: no detectable z500 difference between `pmesh50`/`pmesh120`/
  `pmesh200`/`nearest_level` despite a real ~18% swing in kept-obs volume
  and a genuine placement change, over a ~12-day run in every region.
  Vertical-thinning tuning is ruled out as the driver of the SH-vs-
  NH/Tropics gap; deprioritized in favor of VarBC/per-type loss weighting.
- **8-native-layer AMV thinning not implemented, and now deprioritized** --
  see "Vertical thinning tied to model levels" above; `'nearest_level'`
  bins to the 13 *derived* pressure levels, not the model's true 8 degrees
  of freedom, since that needs new plumbing for ACE2's per-point
  hybrid-sigma layer pressures. The sensitivity sweep now shows vertical
  bin count/placement doesn't move z500 skill at all, so this is not
  expected to help either and isn't worth building next.
