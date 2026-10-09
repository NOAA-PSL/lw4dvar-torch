# Adding radiosonde (raob) observations from Parquet, and a parquet-based psobs reader

Read this when working on: the PREPBUFR-parquet observation pipeline --
surface pressure (psobs) now read from parquet instead of the legacy `.txt`
files, radiosonde (raobs) temperature/wind/humidity profiles as a new obs
type, the `exp.observations` config schema, or the `pyarrow`-per-conda-env
gap this introduces.

## Status: ported from a sibling repo's drafted implementation (2026-10-07)

This was designed and implemented/validated in a separate working copy,
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch` (branch
`feature/add_sondes`, base commit `447af9e` -- before this repo's SFNO
backend, z500 diagnostics, and `cycle: False`/`n_init > 1` fix), and
recorded there in that repo's own `INTEGRATE_SONDES.md`. This document is
that write-up, ported and adapted into this repo (base commit `9d01fcf`)
together with the actual code. See "What changed in this port" below for
what's different from the original draft.

**User's explicit choice for this port**: do the full migration as
drafted -- `get_psobs` now reads the parquet archive unconditionally (no
`.txt`/parquet switch), so every `config_test_*.yml` in this repo needed
migrating to the new `exp.observations.psobs.path` schema, not just the
configs the original draft touched. See "What changed in this port".

## Goal and sequencing (user's plan)

1. Read surface-pressure obs from the normalized PREPBUFR parquet archive at
   `/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/longWin4DVar/prepbufr_to_parquet/data/prepbufr_parquet/<YYYY-MM-DD>/<HH>/`
   instead of the `.txt` files at
   `/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs`, and confirm the parquet
   path reproduces the existing `.txt`-based ps-obs assimilation (same O-B/
   O-A statistics, same analysis) before building anything new on top of it.
2. Once that comparison looks good, add radiosonde temperature, wind, and
   humidity observations from the same parquet archive.

Step 1 was validated in the original draft repo (see "Verification"
below); step 2 (raobs) was implemented there too, with a full-pipeline
smoke test that completed cleanly. Both are now ported here as working
code, not a plan.

## Part 1 -- How the reference repo does this

`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/greg-long-window-4dvar/long-window-4dvar`
is a sibling long-window 4D-Var solver (JAX/NeuralGCM backend, not this
repo's PyTorch backends) that already assimilates both surface pressure and
radiosonde profiles from parquet, from two parquet archive generations:

- **NNJA parquet** (`observation_sources.py`): daily, wide-schema tables
  (`psobs_<date>.parquet`, `raobs_<date>.parquet`) with one column per
  mandatory pressure level per variable (e.g. `TMDB_PRLC50000` = temperature
  at 500 hPa). Older/simpler format.
- **Normalized PREPBUFR parquet** (`prepbufr_parquet.py`, `prepbufr_parquet.md`):
  six-hourly, cycle-partitioned (`<YYYY-MM-DD>/<HH>/`), normalized into two
  tables per PREPBUFR class -- `reports/<CLASS>/part-00000.parquet` (one row
  per report: id, time, station id, lat/lon, elevation, report type) and
  `levels/<CLASS>/part-00000.parquet` (one row per report x level: pressure,
  category, temperature/height/wind values + per-variable PREPBUFR quality
  marks), joined on `report_id`. This is the same schema the
  `prepbufr_to_parquet` converter produces -- the `2015-01-01/00` sample's
  `reports/ADPSFC`, `levels/ADPSFC`, `reports/ADPUPA`, `levels/ADPUPA`
  partitions carry every column `prepbufr_parquet.py` reads (`report_id`,
  `observation_time`, `station_id`, `latitude`, `longitude`, `elevation_m`,
  `report_type`; `level_index`, `category`, `pressure_mb`, `temperature_c`,
  `height_m`, `u_wind_ms`, `v_wind_ms`, `*_quality`, `pressure_error_mb`),
  plus extra columns (humidity, background, drift, raw-wind fields) the
  reader simply ignores. Cycle directories (`2015-01-01/00`) also match
  `prepbufr_parquet.py`'s `_cycle_directories` layout exactly.

### Observation streams and config

`observation_sources.py:normalize_observation_streams` turns a config block

```yaml
observations:
  psobs:
    source: prepbufr   # or nnja
    path: .../prepbufr_parquet/
  raobs:
    source: prepbufr
    path: .../prepbufr_parquet/
    variables: [temperature, u_component_of_wind, v_component_of_wind]
```

into a canonical list of "streams", each with its own source, path,
capacity, manifest, variables, errors, and QC settings -- multiple streams
of the same observation type are allowed, each independently named.

### Radiosonde ("raobs") forward operator: mandatory-level matching, no vertical interpolation

The key simplifying design decision: `prepbufr_parquet.py:_profile_records`
only keeps a level row when its reported pressure matches one of the
model's own pressure levels to within 0.01 mb. That means vertical
interpolation is never needed in the loss -- the `raobs` branch just does:

1. Horizontal interpolation of the decoded 3-D field to the station lat/lon
   (a precomputed 4-point stencil over the model's *regular* lat/lon grid).
2. A model-levels-to-selected-levels gather, where `level_idx` is a fixed
   index map computed once per window from `np.isclose` against the model's
   pressure-level axis.
3. `(obs - model_equivalent) / error_std`, masked, summed into the loss --
   exactly the same shape as the existing ps-obs term.

Units are normalized at read time (`temperature_c + 273.15`, `height_m *
STANDARD_GRAVITY` -> geopotential), and PREPBUFR category-4
winds-by-height levels are excluded from the geopotential channel (their
`height_m` is the wind's vertical coordinate, not an independent height
observation).

### QC

`bufr_radiosonde.py` defines a shared bitmask (`QC_PHYSICAL_HEIGHT`,
`QC_NON_MONOTONIC_HEIGHT`, `QC_BELOW_STATION`, `QC_BACKGROUND_GROSS`,
`QC_PREPBUFR_QUALITY`) applied at read time for raobs: unphysical height,
non-monotonic height with pressure, height below station elevation (with a
100 m tolerance), and the source PREPBUFR quality mark itself, with a
configurable max accepted mark (default 3). Surface-pressure QC instead
removes rejected rows outright before the existing orography/background
check, rather than flagging them.

### Fixed-shape capacities, not dynamic obs counts

Because JAX recompiles whenever an array shape changes, every stream
resolves a capacity (max station/report count) once, auto-scanned and
cached in a JSON manifest, so a rerun with unchanged inputs reuses the scan
instead of rescanning parquet. This machinery exists *purely* for JAX
retracing avoidance -- not needed in this repo (see Part 2).

### Diagnostics

Per-stream O-B/O-A diagnostics are preserved and written per observation
type, including availability/QC counts.

## Part 2 -- How this repo (`lw4dvar-torch`) implements it

This repo's PyTorch backends are architecturally simpler for this than the
JAX reference in two ways worth keeping: (1) **no fixed-shape/retracing
requirement** -- PyTorch doesn't need padded, capacity-bounded obs arrays
the way JAX does, so the manifest/capacity-scanning machinery above was
skipped entirely; (2) the existing `GridInterpolator`/`grid_interp`
(`backends/aifs/aifs_grid.py`) already does general k-NN horizontal
interpolation over each backend's *native* (possibly reduced/unstructured)
grid, which is strictly more general than the reference repo's
regular-lat/lon 4-point bilinear stencil -- it needs no changes to serve
radiosonde profiles too.

### Step 1 -- Parquet surface-pressure reader, validated against the `.txt` path

Decided against a `txt`/`parquet` config switch: `get_psobs` in this repo
reads parquet unconditionally, with no dead code path. Implementation:

- `psobs_parquet.py` (repo root): `load_psobs_parquet`, ported from the
  reference's `prepbufr_parquet.py:load_prepbufr_psobs`/`_surface_records`
  (join `reports`/`levels` on `report_id` per 6-hourly cycle directory, keep
  `level_index == 0`, PREPBUFR-quality-mark QC, ADPSFC/SFCSHP class
  de-duplication by quality-then-report-type preference) -- with one
  deliberate behavioral change from the reference: an added
  single-nearest-report-per-station-per-slot collapse, so one slot gets at
  most one observation per station, matching what the legacy per-hour
  `.txt` files actually contained (the reference keeps every report inside
  `time_tolerance_hours`, which would otherwise inflate counts ~2-6x at a
  6h-spaced, 3h-tolerance slot). Only this module imports `pyarrow`
  (lazily), so nothing else in the solver needs it installed -- but see
  "What changed in this port" below: `get_psobs` *does* now import this
  module unconditionally, so every run needs `pyarrow` in its env.
- `get_psobs` (`long_window_4dvar_utils.py`) calls `load_psobs_parquet`
  directly in place of the old `psobs1_<YYYYMMDDHH>.txt` read; everything
  downstream (padding to `nobs_max`, `oberrstart`/`oberrdeltaperday`,
  `step_lo`/`alpha` time-bracketing, k-NN `grid_interp` precompute) is
  unchanged.
- New config keys (`config.yml.template`, under `exp.observations.psobs`):
  `time_tolerance_hours` (default `dt_obs/2`), `prepbufr_classes` (default
  `ADPSFC`, `SFCSHP`), `report_types` (default `180, 181, 187`),
  `quality_mark_max` (default 3), `use_source_error`, `error_std_hpa`
  (default 1.0 hPa, matching the legacy files' constant assigned error).
  `path` points at the parquet archive root, not a `.txt` directory.
- **Legacy report-type network, confirmed empirically**: counting obtype
  codes across a full day (2015-01-01) of the real `.txt` files gives
  exactly `{181: 4208, 187: 3584, 180: 1317, 120: 3}` -- i.e. ADPSFC
  (fixed-land 181, mesonet 187) and SFCSHP (ship/buoy 180), with 3 stray
  ADPUPA (120) rows treated as noise. `report_types` defaults to
  `(180, 181, 187)` to match. The archive also carries 183/281/282/284/287
  for the same two classes, which widen the network if included --
  excluded by default, overridable.
- **Offline validation against the real `.txt` files** (2015-01-01, slots
  00/06/12, `time_tolerance_hours=3`, no model/GPU involved): 10,645 parquet
  obs vs 9,112 `.txt` obs at the 00Z slot. Spatial nearest-neighbor matching
  (0.02 deg) finds 8,797 stations common to both (96.5% of the `.txt` set).
  For matched stations, the reported surface pressure agrees almost
  exactly: median absolute difference ~1e-5 hPa (floating-point noise), 93%
  agree within 0.2 hPa, 99% within 1.0 hPa. The ~1,850 parquet-only
  stations are new, not spurious duplicates or bugs -- the normalized
  archive is a network *expansion* over whatever wide-query product the
  legacy `.txt` files came from, not a like-for-like repeat.
- **Full assimilation comparison** (in the original draft repo): a
  12h/5-epoch AIFS-single-2.0 smoke test, run twice (parquet reader vs the
  untouched `.txt` reader in a frozen comparison checkout) with otherwise
  identical config -- see "Verification" below for the result summary.
- **Archive coverage**: as of this port (2026-10-07), the
  `prepbufr_to_parquet` archive covers `2015-01-01` through `2015-03-12`
  (confirmed by listing its date directories directly) -- cycling configs
  whose `sdate` + total span exceeds that range will hit a missing-cycle
  `FileNotFoundError`. See "What changed in this port" for which of this
  repo's configs were checked against this.

### Step 2 -- Radiosonde profiles (temperature, wind, humidity)

- **Mandatory-level matching confirmed viable, no vertical interpolation
  needed**: AIFS-single-2.0's own pressure levels (its README) --
  T/U/V/Z: 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925,
  1000 hPa; Q: same minus 10/50. 13 of these 14 coincide with standard WMO
  mandatory levels (only 600 hPa doesn't; conversely the archive's data has
  essentially nothing at 70/30/20 hPa, which AIFS doesn't carry either) --
  so **the model's own pressure-level axis is used directly as the match
  target** (`model.pressure_levels(base)`), not an external WMO list. This
  also means the obs-side level axis IS the model's level axis, in the same
  order -- the forward operator needs no level-index gather at all, just
  horizontal interpolation.
- **Humidity included, config-toggleable per variable**: `raobs.variables`
  is a plain list (`temperature`, `u_component_of_wind`, `v_component_of_wind`,
  `specific_humidity`) -- include or omit any one independently; default
  (when `raobs.enabled` but `variables` unset) is T+u+v, NOT humidity,
  since humidity has no reference-repo precedent. Separately, **omitting
  the whole `raobs` block (or `enabled: False`) still assimilates surface
  pressure only** -- every function down the call chain (`compute_optimal`,
  `compute_loss_4dvar`, `compute_ps_observation_hx`,
  `save_trajectory_diagnostics`) takes `raobs_traj=None` by default and
  skips the raobs term/diagnostics completely when it's None.
- **Backend guard**: `raobs.enabled` is rejected by `load_config` for any
  backend other than `aifs`/`aurora` (`RAOBS_SUPPORTED_BACKENDS`) -- the
  `'t'/'u'/'v'/'q'` packed-state base names are confirmed shared by AIFS and
  Aurora but have not been checked for FCN3/ACE2/SFNO, so (matching this
  repo's existing `ps_operator` guard pattern) it fails loudly rather than
  silently claim untested support.
- **Reader** (`raobs_parquet.py`): ported from `prepbufr_parquet.py`'s
  `_profile_records`/`load_prepbufr_raob_observations` (ADPUPA class only),
  restructured around a per-(slot, station) record dict rather than the
  reference's fixed-capacity JAX arrays (no retracing concern in PyTorch).
  Adds specific-humidity support (`specific_humidity_mg_kg` -> kg/kg,
  `/1.e6`) and a unit fix for temperature (`temperature_c` -> K, `+273.15`)
  -- confirmed by checking actual converted ranges against physical bounds
  (186-307 K; 2.3e-5-0.0196 kg/kg). Low-level parquet/cycle-directory
  helpers are factored out into a shared `prepbufr_parquet_common.py`.
- **Forward operator** (`_compute_raobs_observation_diagnostics_at_time`,
  `long_window_4dvar_utils.py`): per variable, `grid_interp.interp(idx, wts,
  decoded[base])` (the existing k-NN horizontal interpolator, unchanged --
  general enough for AIFS's reduced grid already) directly against
  `raobs_traj['values_<base>']`, masked by `mask_<base>` -- no vertical
  step, confirming the mandatory-level-matching design above.
- **QC** (`raobs_parquet.py`): physical height bounds, non-monotonic
  height-vs-pressure, below-station-elevation, source PREPBUFR quality mark
  -- reimplemented locally (no JAX/NeuralGCM dependency to import it from).
  Humidity gets one additional, reference-less QC: masked above a
  configurable pressure floor (default 300 hPa) -- confirmed empirically
  necessary: the real archive has essentially zero humidity reports above
  300 hPa already (checked directly: 0 accepted obs at every q-level above
  300 hPa for 2015-01-01's three cycles).
- **Error specification**: temperature/wind use a fixed absolute
  `error_std` (K, m/s, default `{t: 1.0, u: 2.0, v: 2.0}`). Humidity uses a
  FRACTION of the observed value instead (`q_error_fraction`, default 0.2),
  floored by `q_error_floor_kg_kg` (default 1e-5) so the effective error
  never collapses near-zero wherever q itself is tiny -- a flat absolute
  error makes no physical sense across q's full range, and has no
  reference-repo precedent.
- **Loss integration**: a `raobs_traj=None`-gated branch inside
  `compute_loss_4dvar`'s existing per-step loop (`_raobs_term`, parallel to
  `_obs_term`) -- NOT a second rollout. This works because `get_raobs`
  deliberately reuses psobs's exact obs-slot grid (same `step_lo`/`alpha`,
  same `dt_obs`-spaced verification times) rather than building an
  independent one: real ADPUPA launches are synoptic (00/06/12/18Z),
  matching whatever `dt_obs` a window is normally configured with anyway --
  confirmed by the real per-slot station counts (662/124/621 for
  00/06/12Z), which show the expected 00&12Z-heavy global launch pattern.
  `_resolve_loss_interp_specs` was extended to fold the enabled variables'
  model base names into the same `decode_state(only=...)`/time-interpolation
  specs the ps term already computes, so there's no duplicate decode either.
- **Diagnostics**: `save_raobs_diagnostics` (parallel to the existing ps
  diagnostics block in `save_trajectory_diagnostics`), one netCDF file
  (`<date>_raobs_diagnostics_*.nc`) with per-variable O-B/O-A/used/QC
  fields on `(time, level_<base>, station_slot)` -- `station_slot` is a
  plain index with no cross-time identity, the same convention psobs's
  `observation_slot` already uses.
- **Config**: `raobs.*` keys documented in `config.yml.template` (path,
  variables, classes, time tolerance, QC thresholds, pressure-match
  tolerance, error specification, humidity pressure floor, station
  capacity) -- all optional, all defaulting to behavior equivalent to "off"
  or to the choices justified above.

### Verification

- **Offline reader check** (`raobs_parquet.py`, no model/GPU): for
  2015-01-01 00/06/12Z (3h tolerance), reads 662/124/621 stations with
  per-level accepted counts that make physical sense throughout --
  standard mandatory levels well-populated, the one non-mandatory AIFS
  level (600 hPa) sparsely populated, and q correctly near-empty above
  300 hPa everywhere. Converted units in physically sane ranges (T
  186.85-306.65 K, q 2.3e-5-0.0196 kg/kg). **Re-run against this port's
  copy of the reader modules on 2026-10-07 -- identical counts
  (10645/10626/10271 psobs per slot; 662/124/621 raobs stations per slot),
  confirming the port changed nothing in the readers themselves.**
- **Full pipeline smoke test** (in the original draft repo): a 12h/5-epoch
  AIFS-single-2.0 run with `raobs.enabled: True` (T+u+v+q) completed
  cleanly (job 22812130) -- no Traceback/non-finite-loss errors, `raobs J`
  printed per epoch/slot alongside the existing ps `J` line, sensible
  per-variable obs-used counts, loss decreasing over 5 epochs.
- **Step 1 full assimilation comparison** (parquet vs `.txt`, 12h/5-epoch
  AIFS-single-2.0, same IC/model/window): obs counts per slot (00/06/12Z)
  parquet 10645/10626/10271 vs txt 9112/9691/9969 (parquet ~13-17% denser,
  as expected). Background z500 error effectively identical between the
  two runs (same IC/model, independent of which psobs source is read --
  this is a same-IC/same-model consistency check, and it passes). O-B RMS
  parquet 1.140/1.164/1.164 hPa vs txt 1.098/1.162/1.199 hPa (within ~5%);
  O-A RMS parquet 0.823/0.857 vs txt 0.841/0.983 (comparable, both show the
  expected O-B -> O-A reduction). Analyzed z500 error: parquet improved in
  all four regions (global/NH/tropics/SH) vs `before`; txt's NH *regressed*
  (3.88 -> 7.00) -- flagged as a real difference worth noting (not an
  obvious parquet-reader bug, since O-B/O-A RMS were comparable), but weak
  evidence from a single 5-epoch/one-case smoke test. A follow-up
  latent-increment-norm and physical-space-trajectory-diff check found both
  runs converged to matching-magnitude corrections (within ~1-2%),
  corroborating that nothing diverged or behaved inconsistently between the
  two obs sources -- see the original draft's `INTEGRATE_SONDES.md` for the
  full numeric write-up if this needs revisiting.

## What changed in this port (2026-10-07)

Ported from `/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch`
(branch `feature/add_sondes`, uncommitted working tree) onto this repo's
base (commit `9d01fcf`, which already has SFNO, the area-weight/`jc_ps_weight`
balance penalty, and the `cycle: False`/`n_init > 1` fix the draft's base
commit `447af9e` didn't). The code diffs (`long_window_4dvar_utils.py`,
`long_window_4dvar.py`, `config.yml.template`, `run_aifs.sh`) applied
cleanly via a three-way merge against each file's common ancestor -- none
of this repo's later work touched the same regions, so no manual conflict
resolution was needed. `psobs_parquet.py`, `raobs_parquet.py`,
`prepbufr_parquet_common.py` were copied unmodified and re-verified
end-to-end against the real archive from this repo's location (see
"Verification" above).

What needed repo-specific work beyond a straight copy:

- **Every `config_test_*.yml` in this repo migrated to the new
  `exp.observations.psobs`/`exp.observations.raobs` schema** -- the draft
  only touched the five configs that existed at its base commit; this repo
  additionally has `config_test_sfno.yml`, `config_test_sfno_ps.yml`, and
  the `aifs2-1.0deg-v1.0` pair (see `integrate_aifs2-1deg-v1.0_backend.md`).
  All nine were migrated: `obspath`/`nobs_max` under `exp:` ->
  `exp.observations.psobs.path`/`.nobs_max`, pointed at the real archive
  root (`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/longWin4DVar/
  prepbufr_to_parquet/data/prepbufr_parquet`), plus an explicit
  `observations.raobs.enabled: False` for clarity. Every migrated config
  was round-tripped through `load_config()` (under the `lwaifs2` env, which
  has `pyarrow`) and confirmed to load with the expected resolved
  `obspath`/`raobs_enabled` values.
- **`use_source_error: True` added for every config whose `oberrstart` was
  `-1` (or unset, which defaults to 0)** -- `config_test_aifs.yml`,
  `config_test_aifs1.yml`, `config_test_fcn3.yml`, `config_test_ace2.yml`,
  `config_test_aifs2-1.0deg-v1.0.yml`/`_ic.yml`. Those configs' old
  behavior read each report's own error from the `.txt` file's column 7;
  without this flag the parquet reader would silently fall back to a flat
  `error_std_hpa` (default 1.0 hPa) instead, a real behavioral change this
  port did NOT want to introduce silently. Configs with a positive fixed
  `oberrstart` (`config_test_aurora.yml`, `config_test_sfno.yml`,
  `config_test_sfno_ps.yml`) are unaffected either way, since a positive
  `oberrstart` always overrides the file/archive value in both the old and
  new code paths.
- **Found and fixed a pre-existing, unrelated bug while migrating
  `config_test_ace2.yml`**: it still had `cycle: False` with `n_init: 240`,
  which the already-merged "Require `cycle: True` when `n_init > 1`" fix
  (see `integrate_multiple_backends.md`) rejects at load time -- this
  config was never updated after that fix landed (both predate this port).
  Fixed by flipping it to `cycle: True`, independent of and unrelated to
  the raobs/parquet work itself.
- **Archive-coverage check for every migrated config's date range** against
  the archive's current `2015-01-01`-`2015-03-12` coverage: `config_test_
  ace2.yml` (`sdate` 2015-01-01, 240 cycles x 6h + a 5-day window =~
  2015-03-07) and `config_test_aifs2-1.0deg-v1.0.yml` (100 cycles x 12h + a
  5-day window =~ 2015-02-25) both fit within the archive's current
  coverage; the rest are single- or few-cycle smoke tests well inside it.
  None are known to exceed it as of this writing, but the archive's
  coverage can change -- re-check if a `FileNotFoundError` on a missing
  cycle directory ever appears.
- **An operational blocker this port surfaced, and its fix (2026-10-07)**:
  `get_psobs` now unconditionally needs `pyarrow`. Confirmed directly:
  `pyarrow` was present in the personal `lwaifs2` env (already used by
  `run_aifs.sh`/`run_aifs2-1.0deg-v1.0.sh`) but **absent from every shared
  env** (`aifs2`, `aifs1`, `fcstnet3`, `aurora`, `ace2`, all under
  `/scratch4/BMC/gsienkf/whitaker/conda/envs/`), none of which are
  group-writable by this account (so a normal `pip install` into them would
  fail with "Permission denied"). **Fixed** with a single
  `pip install --user pyarrow` (run once, via `aifs2`'s `python -m pip`):
  all five shared envs run the identical Python 3.12.14, and `pip`'s
  per-user site-packages directory (`~/.local/lib/python3.12/
  site-packages/`, confirmed `site.ENABLE_USER_SITE=True` in all five) is
  keyed by Python version, not by which env is active -- Python always
  adds it to `sys.path` alongside whichever env's own site-packages is
  active, so one install made `pyarrow==25.0.1` importable from all five
  envs' interpreters simultaneously (verified directly in each, and via a
  real `load_psobs_parquet` read under the previously-blocked `aifs2` env
  itself). Each affected `<name>-requirements.txt`
  (`aifs2`/`aifs1`/`fcn3`/`aurora`/`ace2`) now lists `pyarrow==25.0.1` to
  match. **This fix is scoped to this account**: it works because of who
  ran the install, not because the shared envs themselves changed -- a
  different user running these same envs still needs their own `--user`
  install (or `pyarrow` added to the shared env for real) until its owner
  does that; `run_fcn3.sh`/`run_aurora.sh`/`run_ace2.sh`/`run_sfno.sh`
  needed no env-activation change (unlike `run_aifs.sh`'s earlier
  `lwaifs2` switch) since the shared envs themselves now resolve `pyarrow`
  for this account.
  - **Found along the way, NOT fixed, unrelated to this port**: freshly
    freezing each of the five shared envs to update their
    `-requirements.txt` turned up one pre-existing discrepancy in every
    single one of them -- `requests==2.34.2` is listed in the tracked file
    but `import requests` fails (`ModuleNotFoundError`) in the live env,
    and there's no trace of it anywhere in `site-packages` (confirmed via
    `find`). The env directory's own mtime (2026-08-26) predates this port
    by weeks, so this isn't something caused by the `pyarrow` install --
    it's a stale tracked-file entry from whenever each env was last
    frozen, before `requests` was removed (deliberately or not) by
    whoever/whatever last touched these shared, not-group-writable envs.
    Left as-is (the `requests==2.34.2` line was NOT deleted from any
    `-requirements.txt`) since removing it wasn't part of this task and
    might be relevant to whoever manages these envs -- flagging it here
    rather than silently erasing the record.
- Not changed: the ad hoc smoke-test configs/launchers the original draft
  used for its own validation (`config_test_aifs2_psobs_parquet_12h.yml`,
  `config_test_aifs2_raobs_12h.yml`, `run_aifs2_psobs_parquet_smoke.sh`,
  `run_aifs2_raobs_smoke.sh`) were untracked ad hoc files there (per this
  repo's own convention -- see CLAUDE.md) and were not ported; the
  validation results they produced are summarized under "Verification"
  above instead.
