# Parquet observations: surface pressure (ps) and radiosondes (raobs)

Covers two observation types read from the same normalized PREPBUFR parquet
archive -- surface pressure and radiosonde (temperature/wind/humidity)
profiles -- but the config schema and the `exp.observations` loading
mechanism (see "Config schema" below) are the same ones a future
parquet-backed observation type (aircraft, AMV, GPSRO, ...) would extend,
which is why this doc is named for the general case rather than just
`psobs`/`raobs`.

Read this when working on: the ps-obs parquet reader (`src/psobs_parquet.py`),
the radiosonde (raobs) parquet reader (`src/raobs_parquet.py`), their shared
`src/prepbufr_parquet_common.py`, `get_psobs`/`get_raobs`, the
`exp.observations.psobs.*`/`exp.observations.raobs.*` config schema (or
adding a new observation type to `exp.observations`), the mini-ACE2 CI
fixture's synthetic obs, or the `pyarrow` dependency this introduces.

## Status: parquet-only, ported from a sibling repo's draft (2026-11-10)

`get_psobs` used to read the legacy per-hour-binned `.txt` files
(`psobs1_<YYYYMMDDHH>.txt`, one file per `dt_obs` slot) from
`/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs`. It now reads surface pressure
unconditionally from a normalized, cycle-partitioned PREPBUFR parquet
archive -- no `.txt` path left, by design (see "Why parquet-only" below).

The reader (`src/psobs_parquet.py`, `src/prepbufr_parquet_common.py`) is ported
near-verbatim from a sibling working copy
(`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/
add-raobs-parquet/lw4dvar-torch`, branch `add-raobs-parquet`, which also adds
radiosonde (raobs) observations -- at the time this section was written, not
yet ported here; this repo has since added raobs too, see "Adding
radiosonde (raobs) observations" below). That branch itself ports from
`greg-long-window-4dvar`'s JAX/NeuralGCM reference implementation
(`prepbufr_parquet.py`); see that repo's own `integrate_prepbufr_obs.md` for
the full archive-layout writeup.

## What changed

- **New files** (`src/`, only these import `pyarrow`, lazily):
  `src/prepbufr_parquet_common.py` (cycle-directory resolution, parquet-partition
  reads, station-key/QC helpers) and `src/psobs_parquet.py`
  (`load_psobs_parquet`: join `reports`/`levels` on `report_id` per 6-hourly
  cycle directory, keep `level_index == 0`, PREPBUFR-quality-mark QC,
  ADPSFC/SFCSHP class de-duplication by quality-then-report-type preference,
  plus a single-nearest-report-per-station-per-slot collapse so one slot
  gets at most one observation per station -- matching what the legacy
  per-hour `.txt` files actually contained; the archive otherwise keeps
  every report within `time_tolerance_hours`, which would inflate counts
  2-6x at a 6h-spaced, 3h-tolerance slot).
- **`get_psobs`** (`src/long_window_4dvar_utils.py`): calls `load_psobs_parquet`
  directly in place of the old `psobs1_<YYYYMMDDHH>.txt` read; everything
  downstream (padding to `nobs_max`, `oberrstart`/`oberrdeltaperday`,
  `step_lo`/`alpha` time-bracketing, k-NN `grid_interp` precompute) is
  unchanged. `nobs_max`, if present in the config, is still an upper bound
  -- `get_psobs` now raises `ValueError` instead of silently growing past it
  (the old reader scanned every file first to compute the true max; the
  parquet reader already has every slot's count in hand from one pass).
- **Config schema**: observation settings live under a new
  `exp.observations` block, one sub-block per observation type --
  `exp.observations.psobs` for now. `load_config` resolves it into the same
  flat internal keys `get_psobs` already read (`obspath`, `nobs_max`, etc.)
  via `_resolve_observations_config`; see "Config schema" below for the
  mechanism and how a future observation type extends it.
  `observations.psobs.path` is required and means the parquet archive ROOT
  (`<path>/<YYYY-MM-DD>/<HH>/{reports,levels}/<CLASS>/part-00000.parquet`),
  not a `.txt` directory. All other `observations.psobs` keys are optional
  (documented in `configs/config.yml.template`): `time_tolerance_hours` (default
  `0.5`) and `quality_mark_max` (default `4`) match the legacy `psobs1_*.txt`
  archive's actual generator (`/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs/
  read_ps_prepbufr1.py`: `halfwindow = 1800` seconds, `psqm <= 4`) -- found
  2026-11-11 while setting up a txt-vs-parquet cycling comparison; until then
  both defaults (`dt_obs/2`, `3`) were never actually derived from this
  archive, just carried over from the ported reference repo's own
  (unrelated) defaults. `prepbufr_classes` (default `ADPSFC`, `SFCSHP`),
  `report_types` (default `180, 181, 187` -- confirmed empirically to
  match the legacy `.txt` network: counting obtype codes across a full day
  of those files gives exactly `{181: 4208, 187: 3584, 180: 1317, 120: 3}`,
  i.e. ADPSFC fixed-land/mesonet and SFCSHP ship/buoy, with 3 stray ADPUPA
  rows as noise; the archive also carries 183/281/282/284/287 for the same
  two classes, excluded by default since that's a network *expansion*, not
  a like-for-like repeat), `error_std_hpa`
  (default 1.0 hPa, matching the legacy files' constant assigned error),
  `use_source_error` (default `False`), `nobs_max` (optional upper bound).
- **`oberrstart` semantics changed for `oberrstart <= 0`** (including the
  conventional `-1`, and the default `0` when unset): it used to mean "read
  each report's own error from the `.txt` file's column 7"; now it means
  "use `observations.psobs.error_std_hpa` (flat, default 1.0 hPa) unless
  `observations.psobs.use_source_error: True`, in which case use the
  report's own `pressure_error_mb` from the parquet archive instead."
  **Every tracked config whose `oberrstart` was `<= 0` or unset got an
  explicit `observations.psobs.use_source_error: True` added**, to preserve
  its old behavior rather than silently switching it to the flat default:
  `configs/config_test_aifs.yml`, `configs/config_test_aifs1.yml`,
  `configs/config_test_aifs2-1.0deg-v1.0.yml`, `configs/config_test_aifs3.yml`,
  `configs/config_test_ace2.yml`, `configs/config_test_ace2_polaris.yml`,
  `configs/config_test_fcn3.yml`. Configs with a positive fixed `oberrstart`
  (`configs/config_test_aurora.yml`, `configs/config_test_sfno.yml`,
  `configs/config_test_sfno_ps.yml`) are unaffected either way, since a positive
  `oberrstart` always overrides the file/archive value in both the old and
  new code paths.
- **Archive location and coverage**: every migrated `configs/config_test_*.yml`'s
  `observations.psobs.path` now points at
  `/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/
  prepbufr_to_parquet/data/prepbufr_parquet` (the account's own copy;
  confirmed group-readable at every level, including the per-cycle-hour
  subdirectories -- see "Known blocker" below for why this matters), which
  as of this writing covers `2015-01-01` through `2015-07-18`. Reading a
  known cycle (`2015-01-01` 00/06/12Z) from this copy reproduces the exact
  per-slot obs counts validated in the sibling repo's own writeup (10645,
  10626, 10271), confirming the copy is intact. `configs/config_test_ace2_polaris.yml`
  is a documented **TODO, not done**: Polaris has no access to any
  `scratch4` path at all (`docs/running_at_alcf.md`), so it needs its own archive
  copy under Eagle (`/lus/eagle/projects/moonshot-reanalysis/jwhitaker/`,
  replacing the old `.txt` psobs dir there) before it can actually run;
  until then its `observations.psobs.path` is a placeholder pointing at a path that doesn't
  exist yet.
- **Deliberately not using Sergey.Frolov's shared archive copy**
  (`/scratch4/BMC/gsienkf/Sergey.Frolov/obs/prepbufr_to_parquet/data/
  prepbufr_parquet/`): its per-cycle-hour directories deny this account read
  access even though the account is in the `gsienkf` group and the
  top-level directories list fine. Rather than chase that permissions gap,
  this account maintains and extends its own copy (above) directly --
  re-download/convert further dates into it as needed, the same way the
  current `2015-01-01`-`2015-07-18` coverage was produced.
- **`pyarrow` dependency**: added to every backend's `envs/*-requirements.txt`
  that didn't already have it (`aifs1`, `aifs2`, `fcn3`, `aurora`, `ace2`
  -- `aifs2-1.0deg-v1.0` and `aifs3` already had it via an unrelated
  dependency chain), and to `envs/requirements-ci.txt` (so `envs/requirements-ci-ace2.txt`,
  which includes it, picks it up too). Each production env needs
  `pyarrow==25.0.1` actually installed before this will run --
  `pip install` into the shared, not-group-writable conda envs under
  `/scratch4/BMC/gsienkf/whitaker/conda/envs/` needs that account's
  cooperation (or a per-account `pip install --user pyarrow`, which only
  helps the installing account -- see the sibling repo's own
  `integrate_prepbufr_obs.md` "What changed in this port" for how that was
  done there).

## Obs-count gap vs. the legacy `.txt` reader (found/fixed 2026-11-12)

While setting up a txt-vs-parquet cycling comparison (four jobs: aifs/ace2 x
`main`-branch-txt-reader/this-repo's-parquet-reader, all on the
`-longcyc-obsint-1hour` configs), the parquet reader's per-cycle total
assimilated ps obs count ran consistently **3-7% lower** than the legacy
`.txt` reader's, with the gap concentrated in ocean-heavy regions (Tropics,
SH) more than NH.

**Root cause**: PREPBUFR writes two generic placeholder station ids that are
NOT unique per physical platform -- `MASKSTID` (restricted/proprietary
mesonet feeds whose true identity is masked) and `SHIP` (vessels with no
assigned call sign). `prepbufr_parquet_common.py`'s `station_key()` treated
any non-empty `station_id` as a real identity, so `load_psobs_parquet`'s
per-station/per-slot nearest-report dedup (`psobs_parquet.py`, see its own
docstring) collapsed hundreds of genuinely distinct SFCSHP reports sharing
one of these placeholders down to effectively one per slot. Confirmed
against the real archive: one 2015-01-01 00Z cycle alone has 2556 of 19653
raw SFCSHP reports sharing the literal id `'MASKSTID'`, at clearly different
lat/lon each. The legacy generator
(`/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs/read_ps_prepbufr1.py`) does no
station-level dedup at all -- it prints every BUFR subset that passes its
own QC/type/time filters, so this loss has no equivalent on that side.

**Fix**: `station_key()` now routes `{'MASKSTID', 'SHIP'}` to the same
lat/lon-derived fallback key already used for a blank `station_id`, instead
of `'sid:'+station_id`. Checked for a better per-platform identity field
first (`restriction_flag`/`restriction_expiration_hours` are entirely
unpopulated in this archive; `report_sequence_number`/`process_number` are
batch metadata, only 12-120 distinct values across thousands of reports) --
lat/lon is the best signal actually available.

**Validation**: before/after comparison of `load_psobs_parquet`'s count
against the legacy `.txt` file's line count, across 28 cycle-hours (00/06/
12/18Z x 7 sample days spanning 2015-01-01 through 2015-01-29, the dev jobs'
actual cycling range): mean gap **-3.30% (std 0.20%) before -> -0.56% (std
0.06%) after**, stable across the whole range, not a single-day fluke. The
small residual is attributed to a minor time-tolerance boundary difference
(legacy's half-open `[center-0.5h, center+0.5h)` vs. this reader's closed
`abs_delta <= tolerance_seconds`) and a small number of coincidental lat/lon
collisions among distinct `MASKSTID`/`SHIP` platforms (this archive reports
ship/buoy positions to only ~0.01 deg, ~1 km, resolution) -- not something
further QC-parameter tuning can close.

**Checked and ruled out** as contributors (zero affected rows in the real
archive for the dates checked, so no code change made): the legacy
generator's extra observation-error sanity check (drops a report if its
`POE`/`pressure_error_mb` is NaN or >= 9.99 hPa; this reader never applies
that, falling back to the flat default instead when missing) and its
single-level-only requirement (`obs.shape[1] == 1`, vs. this reader's
`level_index == 0`). Noted here since they're still real, latent differences
from the legacy logic that could matter on a different date range or once
the archive itself changes.

**Testing**: `test_generic_placeholder_station_id_not_deduped` added to
`tests/unit/test_psobs_parquet.py` (two `MASKSTID` reports at different
locations must both survive the per-slot dedup).

**Re-run**: the four original `-longcyc-obsint-1hour` jobs predate this fix
and were not resubmitted (their saved diagnostics still reflect the old
dedup). A fresh pair, `-t1` suffixed so they don't overwrite that output --
`configs/config_test_ace2-longcyc-obsint-1hour-t1.yml` /
`configs/config_test_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-t1.yml`,
`slurm/run_ace2-longcyc-obsint-1hour-t1.sh` /
`slurm/run_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-t1.sh` (otherwise identical
configs, 5h walltime instead of the original 1-day) -- was submitted
2026-11-12 to confirm the fix end to end through the real solver path (jobs
26262652 ace2, 26262653 aifs).

## Off-cycle-hour slots and cycle-boundary duplication (found/fixed 2026-11-16)

**Bug** (PR #26 review): `prepbufr_parquet_common.py`'s `cycle_directories`
picked a cycle directory by checking whether the directory's *own nominal
hour* fell inside `[verification_time - tolerance, verification_time +
tolerance]`, not whether the directory could actually *contain* a report in
that range. For any verification time that isn't itself a 00/06/12/18Z
cycle hour (e.g. `dt_obs: 1`/`2`/`3` instead of `6`), this returned **zero
cycle directories -- and silently zero observations** -- even when the
correct 00Z partition had a report timestamped e.g. 01Z sitting right
there on disk. None of this repo's tracked configs hit it (all use
`dt_obs: 6`, so every slot already lands exactly on a cycle hour -- the
`-longcyc-obsint-1hour-*` configs' name describes `time_tolerance_hours`,
the width of the per-slot acceptance window, not `dt_obs`), so no past run's
results are affected.

**Fix**: `cycle_directories` now rounds each tolerance-window endpoint to
the *nearest* 6-hourly cycle (`_nearest_cycle`), reflecting that a cycle
directory's reports span its own nominal hour +/- 3h, not just the hour
itself. Exactly on the 3h midpoint between two cycles, the tie breaks
toward the verification time's own cycle, so a query centered exactly on a
cycle hour doesn't spuriously demand a neighboring directory that may not
exist. `tests/unit/test_psobs_parquet.py::test_off_cycle_slot_finds_report_in_earlier_cycle_partition`
covers the off-cycle-hour case directly.

**Cycle partitions genuinely overlap at their shared boundary -- but only
for some PREPBUFR classes.** Checked against this account's real archive
copy (`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/
prepbufr_to_parquet/`) and its generator (`convert_prepbufr_to_parquet.py`:
`observation_time = cycle_time + timedelta(hours=dhr)`, `dhr` taken straight
from the raw PREPBUFR `DHR` field, itself cut to `[-3, +3]` by NCEP's
operational dump upstream of this repo's archive entirely):

- `AIRCAR`/`AIRCFT` (aircraft, continuous reporting, not read by this
  repo): genuinely duplicated at the boundary -- e.g. 2015-01-01, reports
  timestamped exactly `03:00` appear **identically** (same station +
  lat/lon) in both the `00Z` partition (`397` `AIRCAR` + `92` `AIRCFT`
  rows, its own `+3h` edge) and the `06Z` partition (same `397`+`92`, its
  own `-3h` edge) -- a closed `[cycle-3h, cycle+3h]` window on both sides.
- **`ADPSFC`/`SFCSHP` (what `psobs_parquet.py` actually reads): no
  duplication.** Checked across all 201 days available locally, 57.5M
  surface-level rows: the observed offset range is `[-3.0h, +3.0h)` --
  closed at `-3h`, but the exact `+3.0h` offset never occurs in either
  partition (verified at three independent boundaries: 2015-01-01's 00Z/06Z
  and 12Z/18Z seams). A surface report exactly at a boundary instant (e.g.
  `03:00`) lives **only** in the later cycle's file (`06Z`, as its `-3h`
  edge) -- `00Z`'s own file has zero rows at its nominal `+3h` edge. So
  surface pressure is already a clean, non-overlapping `[cycle-3h,
  cycle+3h)` partition of time; nothing needed changing in the generator
  for this obs type.
- `ADPUPA` (radiosondes, not read by this repo; see "raobs design this
  repo doesn't need" above): concentrated tightly around each launch's own
  nominal hour (2015-01-01: 863 of 1252 `00Z` reports at exactly `+0.0h`
  offset; observed range only `-2.5h` to `+2.0h` across all four cycles) --
  nowhere near the `+-3h` boundary, so no overlap either way. Worth
  re-checking on whatever archive the raobs port (`add-raobs-parquet`
  branch) ends up using, rather than assuming this holds everywhere.

**Residual edge case, not exercised by any current config**: because
`ADPSFC`/`SFCSHP` partitions are clean (not duplicated), the fix's
midpoint tie-break has one narrow blind spot -- a query with
`time_tolerance_hours` exactly `3.0` centered exactly on a cycle hour
(`center=00:00`, window `[21:00, 03:00]` closed) resolves toward its own
cycle (`00Z`) and never opens the neighbor (`06Z`), so a surface report
timestamped exactly at the far edge (`03:00`, which lives only in `06Z`)
would be missed rather than double-counted. No tracked config uses
anything but the `0.5` default, so this is purely a documented limitation,
not a live bug.

## Config schema: `exp.observations`, extensible to future obs types

Observation settings are grouped under `exp.observations.<type>` on disk
(`psobs` and `raobs` so far), each with its own `path` -- rather than flat,
per-type-prefixed keys directly on `exp` -- specifically so a future
observation type (aircraft, AMV, GPSRO, ...) can be added without renaming
anything already here. This mirrors a convention already worked out and in
active use on a sibling branch,
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/
add-amv-gpsro/lw4dvar-torch` (branch `add-amv-gpsro`), which has `aircraft`
and `gpsro` blocks in addition to `raobs`/`psobs` -- see that branch's
`config_test_ace2_ps_raobs_aircraft_t3.yml` for a worked example with five
types enabled at once, and its own `integrate_prepbufr_obs.md` for the
per-type key tables and design notes on aircraft/amv/gpsro specifically
(thinning, bias correction, etc. -- none of which apply to psobs/raobs and
aren't duplicated here).

`src/long_window_4dvar_utils.py`'s `_PSOBS_CONFIG_KEYS`/`_RAOBS_CONFIG_KEYS`
(`{nested_key: flat_exp_key}` tables) and `_resolve_observations_config`
(called from `load_config`, right after `path_output` is resolved) do the
actual translation: `exp.observations.psobs.path` becomes `exp['obspath']`,
`exp.observations.raobs.variables` becomes `exp['raobs_variables']`, and so
on, raising `ValueError` on any key `_PSOBS_CONFIG_KEYS`/`_RAOBS_CONFIG_KEYS`
doesn't recognize, on a missing `psobs.path`, or on an unsupported/empty
`raobs.variables`. **Everything downstream of `load_config` --
`get_psobs`/`get_raobs` included -- still reads the flat keys it always
has**; this translation is the *only* place the nested schema exists.
Adding a new type means adding its own `_<TYPE>_CONFIG_KEYS` table and a
corresponding block in `_resolve_observations_config` (almost certainly with
its own `enabled` flag, since unlike psobs a new type would normally be
optional) -- copy the `raobs` block's pattern rather than re-deriving it,
and extend `_resolve_observations_config`'s
`unknown = set(observations) - {'psobs', 'raobs'}` check to admit the new
top-level key.

## Adding radiosonde (raobs) observations (2026-11-16)

Ported from the mature, already-validated "ps + raobs from parquet" feature
on the sibling branch above (`add-amv-gpsro/lw4dvar-torch`, ancestor commit
`8fce498`, branch `add-raobs-parquet` -- that branch's own
`integrate_prepbufr_obs.md` has the original design write-up, the
sensible-temperature and q-error/background-check fixes, and the forecast-
skill validation numbers), re-implemented against this repo's current
`exp.observations` schema and backend set rather than patched in
mechanically.

- **New files**: `src/raobs_parquet.py` (`load_raobs_parquet`: ADPUPA
  reports/levels joined per 6-hourly cycle directory, mandatory-level
  matching against the MODEL's own pressure levels -- see below --
  physical-height/non-monotonic-height/below-station/PREPBUFR-quality QC,
  an optional humidity pressure floor). `src/prepbufr_parquet_common.py`
  gained one addition, `sensible_temperature` (reads the archive's `events`
  partition): the archive's `temperature_c` is virtual temperature wherever
  humidity was reported (PREPBUFR's VIRTMP step), so `raobs_parquet.py`
  takes `t` from the first non-VIRTMP event instead, falling back to the
  input value when no such event exists. Everything else in `common.py`
  (cycle-directory resolution, the `MASKSTID`/`SHIP` station-id dedup fix,
  `station_key`) is shared unchanged with `psobs_parquet.py` and needed no
  raobs-specific change.
- **No vertical interpolation**: each requested variable (`temperature`,
  `u_component_of_wind`, `v_component_of_wind`, `specific_humidity`, mapped
  to packed-state base names `t`/`u`/`v`/`q` by `RAOBS_VARIABLE_MODEL_BASE`
  in `long_window_4dvar_utils.py`) is matched against
  `model.pressure_levels(base)` AT READ TIME (within
  `pressure_match_tolerance_hpa`, default 0.05 hPa) -- so the forward
  operator (`_compute_raobs_observation_diagnostics_at_time`) is horizontal
  interpolation only, the same k-NN/bilinear `grid_interp` every backend
  already builds for psobs, just evaluated at raobs's own station
  locations (raobs stations are not psobs stations, so the horizontal
  interpolation weights can't be shared -- only the *time*-slot grid is:
  `get_raobs` reuses `get_psobs`'s exact `step_lo`/`alpha` obs-slot grid
  rather than building an independent one, since real ADPUPA launches are
  synoptic and match whatever `dt_obs` a window is normally configured
  with anyway).
- **Backend support**: `RAOBS_SUPPORTED_BACKENDS = ('aifs', 'aurora',
  'ace2', 'ngcm', 'neuralgcm', 'graphcast')`. `aifs`/`aurora`/`ace2` were
  validated with real runs on the sibling branch; `ngcm`/`neuralgcm`/
  `graphcast` (added on this repo after that branch's base, so the sibling
  branch's own guard never saw them) were confirmed to expose the same
  `t`/`u`/`v`/`q` decode_state/`pressure_levels` base-name convention
  (`ngcm_model.py`'s `DERIVED`, `neuralgcm_model.py`'s `DECODED`,
  `graphcast_model.py`'s `SHORT` tables) and already get a generic grid
  interpolator from `get_grid_interpolator`, so no backend-specific work
  was needed to extend the guard to them. `fcn3`/`sfno` stay excluded --
  never checked, same conservative-guard style as the existing
  `ps_operator`/`reset_skt_ocean` guards elsewhere in `load_config`.
- **Loss/diagnostics integration**: a `raobs_traj=None`-gated branch
  parallel to the existing ps-obs one at every level -- `compute_loss_4dvar`
  gained a `_raobs_term` closure accumulated into the total loss alongside
  `_obs_term`'s; `compute_ps_observation_hx` gained a parallel
  `_record_raobs` collector (no extra `decode_state` call -- it already
  decodes every field unconditionally); `save_trajectory_diagnostics` gained
  a `save_raobs_diagnostics` call writing a separate
  `<date>_raobs_diagnostics_*.nc` (dims `time, level_<base>, station_slot`,
  O-B/O-A/used/QC per variable, mirroring the ps diagnostics file's
  conventions). Omitting `observations.raobs` (or `enabled: False`) leaves
  every one of these `None`/skipped, so existing ps-only configs are
  unaffected.
- **Error specification**: `t`/`u`/`v` use a fixed absolute `error_std`
  (K, m/s). `q` defaults to a fixed per-level table (`RAOBS_Q_ERROR_G_KG`,
  Desroziers estimates from the sibling branch's own ps+raobs run) plus a
  4-sigma background check (`RAOBS_Q_BG_CHECK`), rather than a flat
  fraction-of-observed-value error -- the fraction scheme weights dry obs
  most and pulls the analysis dry (see the sibling branch's own writeup for
  the numbers that motivated this).
- **ACE2-specific caveat, confirmed real (2026-11-17)**: exclude
  `specific_humidity` from `observations.raobs.variables` for
  `model_backend: ace2`. ACE2's packed-state humidity prognostic is
  `specific_total_water` (water vapor + cloud condensate), not a pure
  water-vapor specific humidity the way AIFS/Aurora's native `'q'` is --
  `ACE2Model.decode_state` derives the generic `'q'`/`specific_humidity`
  alias straight from it with no separation of the condensate term, and
  ACE2 has no separate cloud/liquid/ice-water channel to subtract it with.
  The add-amv-gpsro sibling branch's `integrate_prepbufr_obs.md` quantified
  this directly against real ERA5 (one column/date, F90 180x360): ACE2 is
  0.1-0.6 g/kg DRIER than ERA5 at 1000/925 hPa, then 0.06-0.26 g/kg WETTER
  from 850-250 hPa -- 15-70% of ERA5's own spatial std in that band, i.e. a
  confirmed, non-negligible bias, not a "usually small" one. Workaround
  (not a fix, since it's structurally unresolvable): `aifs`/`aurora`
  configs are unaffected and keep `specific_humidity` by default; only
  `ace2` configs need the explicit restriction.
- **`time_tolerance_hours` is deliberately NOT the same value as
  `observations.psobs`'s** (confirmed 2026-11-17): psobs defaults to 0.5h,
  matching the legacy `.txt` generator's 30-minute window for essentially
  instantaneous surface obs. Radiosondes are different -- a launch is
  scheduled relative to the synoptic hour but the ascent itself spans
  multiple hours, and only the two main synoptic hours (00/12Z) are
  launched densely; checked directly against this archive (2015-01-01):
  a 0.5h tolerance keeps 93-96% of 00/12Z stations but only 50-60% of the
  already much sparser 06/18Z ones (74 vs 124, 36 vs 72 stations). The
  default, `dt_obs / 2` (3h for a 6h cycle), is the right choice for raobs
  specifically -- don't lower it to match psobs's 0.5h.
- **Testing**: `tests/unit/test_raobs_parquet.py` (hand-crafted archives --
  mandatory-level tolerance, each QC bit, the sensible-temperature
  event-stack lookup and its fallback, the humidity pressure floor,
  multiple variables/slots, station-capacity overflow), extensions to
  `tests/unit/test_config.py` (unknown-key/backend-guard/empty-variable-list
  rejection, the resolved-flat-keys check, acceptance for every backend in
  `RAOBS_SUPPORTED_BACKENDS`), and a synthetic ADPUPA archive
  (`tests/make_fixtures.py`'s `_write_raobs_parquet`, at ACE2's own 13
  pressure levels) added to the mini-ACE2 fixture and exercised by
  `tests/ace2/test_ace2_mini.py` alongside psobs.

## First ps + raobs long-cycling experiments (2026-11-17)

Two one-month (300 x 6h cycles, `sdate: 2015-01-01T00`) long-cycling smoke
experiments, one per backend validated with real raobs runs so far:
`configs/config_test_ace2-longcyc-obsint-1hour-t1.yml` /
`slurm/run_ace2-longcyc-obsint-1hour-t1.sh` and
`configs/config_test_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-t1.yml` /
`slurm/run_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-t1.sh`. Both are gitignored
(`configs/config*longcyc*.yml`, `slurm/*longcyc*.sh` -- see `CLAUDE.md`'s
"left untracked by convention"), same as every prior `-t1`/`-t2`/`-t3`
config in this family.

Each is the sibling `add-parquet-ps-dev` repo's own ps-only long-cycling
comparison (`config_test_ace2-longcyc-obsint-1hour-t3.yml` /
`config_test_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-t3.yml`) plus an
`observations.raobs` block, using the add-amv-gpsro sibling's established
settings -- not the code-level 4-variable default -- with two deliberate
departures from a naive copy, found while reviewing the first draft:

- **ACE2's `raobs.variables` excludes `specific_humidity`**; AIFS2's keeps
  all four. See the "ACE2-specific caveat" bullet above -- ACE2's `'q'` is
  `specific_total_water`, a confirmed non-negligible bias against real
  radiosonde specific humidity, not a backend AIFS/Aurora share.
- **Both use `time_tolerance_hours: 3.0` (`dt_obs/2`), not psobs's `0.5`**,
  despite the instinct to match the two observation types' settings for
  consistency. See the `time_tolerance_hours` bullet above -- raobs and
  psobs have genuinely different temporal reporting characteristics
  (radiosonde ascents vs. near-instantaneous surface obs), not an
  oversight to reconcile.
- `-t1` (not `-t3`) because this repo has no prior `-t1`/`-t2` attempt at
  this experiment to collide with -- the sibling's `-t3` numbering reflects
  its own ps-only `-t1`/`-t2` history, not a shared sequence across repos.

**Readiness check before submission** found a real blocker, since fixed:
`backends/ace2/ACE2-ERA5` (the git submodule `path_model`/`model_name`
point at) was completely uninitialized in this checkout (`git submodule
status` showed the `-` "not initialized" prefix) -- the ACE2 job would have
failed immediately opening a nonexistent `ace2_era5_ckpt.tar`. The
documented fix (`README.md`'s "Getting the code":
`GIT_LFS_SKIP_SMUDGE=1 git submodule update --init` +
`backends/ace2/ace2_prefetch_checkpoint.py`) needs a login node; the
checkpoint and `forcing_2014.nc`/`forcing_2015.nc` (the only years this
run's `nhrs_back: 48`-from-`sdate`-forward span touches) were instead
copied in directly from the sibling `add-parquet-ps-dev` checkout, which
already had them. Verified byte-identical to that checkout's copies by
SHA256 (all three files match exactly) plus a zip-integrity check on the
checkpoint itself (112 entries, no corrupt members) -- not just a
file-size/existence check. The AIFS2-1.0deg-v1.0 job needed no equivalent
fix (its checkpoint is a plain file, not a submodule, and was already
present). Not fixed: `git submodule status` still shows
`backends/ace2/ACE2-ERA5` as uninitialized, since the files were copied in
directly rather than via `git submodule update --init`/`lfs pull` --
harmless for running the job (nothing in the code checks submodule state),
but a future `git submodule update` in this checkout may try to re-sync
this directory against the pinned commit and complain about or disturb the
now-present, untracked-from-git's-perspective content.

## ACE2 Tropics z500 degradation with raobs: diagnosed, not yet fixed (2026-11-17)

Comparing the two long-cycling experiments above against their ps-only
counterparts (the sibling `add-parquet-ps-dev` repo's own
`-longcyc-obsint-1hour-t3` runs) via region-split z500 and raobs-fit
comparison scripts (`plots/z500err_ts_txt_vs_parquet_cycle_compare.py`,
`plots/obs_assimilated_ts_txt_vs_parquet_cycle_compare*.py`,
`plots/raobs_tuv_profile_ps_raobs_cycle_compare*.py` -- one directory up
from this repo, `add-parquet-raobs-dev/plots/`, left untracked by
convention like every other analysis script; not part of this repo's own
`plots/` output):

- **AIFS-1deg**: z500 RMS error improves with ps+raobs vs. ps-only across
  NH/Tropics/SH, as expected from adding more observations.
- **ACE2**: improves in NH/SH but **gets WORSE in the Tropics specifically**
  -- the opposite of what adding raobs should do.

**Root cause, traced via the raobs fit itself (O-B/O-A bias+RMSE profiles,
by region)**: ACE2's RMSE vs. raobs is jagged and spikes sharply above
200 hPa for t/u/v (e.g. t: 0.84 K at 250 hPa -> 3.06 K at 150 hPa), while
AIFS's grows smoothly toward the stratosphere at the same obs
locations/times (t: 0.58 K -> 0.92 K over the same levels) -- confirming
this is an ACE2 representativity problem, not a property of the obs
themselves. Pulled the released ACE2-ERA5 checkpoint's own sigma
coordinates directly (`sigma_coordinates['ak']`/`['bk']` in the
checkpoint's `stepper` state): its 8 native hybrid layers have midpoints
at only ~26/98/199/330/503/689/847/964 hPa, so the derived t/u/v above
~200 hPa (`ace2_model.py`'s `_to_pressure_levels`) are coarse
linear-in-ln(p) interpolations between just 2-3 widely-spaced native
points -- genuinely poor vertical resolution there, not a bug. Meanwhile
`observations.raobs.error_std` (`t: 1.0` K, `u`/`v: 2.0` m/s) is a single
fixed value, the same regardless of level or backend, so at these levels
the DA badly over-trusts obs the model can't actually resolve.

**Why Tropics specifically**: splitting the same RMSE-by-level comparison
by region (NH/Tropics/SH) shows the **temperature** spike above 200 hPa is
global -- NH and SH jump almost as much as Tropics (SH slightly worse at
150 hPa). But the **wind** spike is concentrated in the Tropics: at
150 hPa, Tropics u-RMSE runs 60-80% higher than NH or SH (8.35 m/s vs.
4.44/5.23 m/s for that cycle's window) -- consistent with the Tropical
Easterly Jet (~150 hPa) and QBO (~50 hPa) being sharp, genuinely-tropical
wind features with no extratropical analogue at levels ACE2's coarse grid
can resolve, compounded by the Tropics' smaller baseline z500 variance
(this repo's own z500 comparison plot: Tropics spans ~3-10 m vs. NH/SH's
~10-25 m) making the same absolute noise injection show up as a much
larger relative degradation.

**Status**: diagnosed with data (level/region-resolved RMSE profiles,
the checkpoint's own sigma coordinates), not yet fixed. Candidate fix,
not yet tried: cap ACE2's `raobs.variables` assimilation to levels
<=200 hPa, or use a level/backend-dependent `error_std` that reflects
ACE2's actual (much coarser) skill above it -- analogous to the existing
`specific_humidity` exclusion above, just for t/u/v's upper-level
representativity instead of a whole-variable physical mismatch.

## PR #29 (`add-parquet-raobs-dev` -> `main`): merge conflict and review fixes (2026-11-17)

Before opening the PR, merged `main` into this branch
(`git merge main`, commit `0e7275b`) to pick up `main`'s `jc_z_weight`
feature (PR #27) first. Hit one real conflict:
`compute_loss_4dvar`'s signature -- this branch had independently added a
trailing `raobs_traj=None`, `main` a trailing `jcz_ref=None`. Resolved by
appending `raobs_traj` AFTER `jcz_ref` (`..., jc_ref=None, jcz_ref=None,
raobs_traj=None)`), not between `jc_ref` and `jcz_ref`: `tests/math/
problem.py`'s `Problem.cost()` calls this function **positionally**,
ending `..., self.jc_ref, self.jcz_ref` -- inserting `raobs_traj` any
earlier would have silently routed `self.jcz_ref` into the wrong
parameter with no error. Verified against the full suite (179 passed) and
the real end-to-end `make check-ace2` run before pushing.

[PR #29](https://github.com/NOAA-PSL/lw4dvar-torch/pull/29) then got two
rounds of GitHub Copilot's automated review, both fully addressed:

- **Round 1** (commit `a6ad7b9`, 5 issues): `backends/hybrid_levels.py` had
  the SAME NaN-control-gradient bug as `ace2_model.py`'s
  `_to_pressure_levels` (see "Found and fixed a real ACE2 backend bug" in
  the raobs section above) -- a second, independent copy of the identical
  derivation, newly reachable once `ngcm`/`neuralgcm`/`graphcast` were
  added to `RAOBS_SUPPORTED_BACKENDS`; same safe-denominator fix applied.
  Plus: `raobs_error_std` replaced the whole `{t, u, v}` default dict
  instead of merging over it (a partial override like `{t: 1.5}` would
  `KeyError('u')` on the first wind ob); a short/one-step window left a
  raobs base's diagnostics as `{}` instead of a correctly shaped empty
  array, `KeyError` in `_pad_leading`; the raobs loss term's per-base
  `used.sum().item()` (a CUDA sync) ran every obs slot of every epoch
  regardless of whether anything was printed; and the duplicate-observation
  height-metadata staleness bug below.
- **Round 2** (commit `a8efc6a`): enabling raobs with `specific_humidity`
  against an ACE2 checkpoint whose humidity is `specific_total_water` is
  now a hard `ValueError` (new `ACE2Model.humidity_family` accessor),
  instead of relying on every config author remembering the exclusion
  above by convention.
- **Round 3** (commit `985a5a2`, the review's "previously missed" findings
  on re-scan): duplicate `raobs.variables` entries are now rejected at
  config-load time (silently corrupted saved NetCDF row counts via
  `_pad_leading` being called twice for the same base); `raobs.error_std`/
  `q_error_*` settings are now validated (positive, finite, valid keys) at
  config-load time instead of producing a non-finite loss or a silently
  ignored typo; padded `station_slot` lat/lon/elev in the saved
  diagnostics are now masked with NaN instead of written as a literal
  station at (0N, 0E); and `raobs_parquet.py`'s duplicate-observation
  selection now ranks candidates by actual QC eligibility (own quality
  mark AND, if a height is attached, the physical-height/below-station
  checks) instead of quality mark alone, with heights that fail their own
  quality mark excluded from attachment entirely (previously used for QC
  regardless).

**Real-world impact, checked against the actual archive, not just
`make check`**: ran the pre-fix and post-fix `raobs_parquet.py` read logic
side by side against the real PREPBUFR archive both long-cycling runs
above actually use. The round-1 stale-height-metadata fix never fires in
that data (0 of 8081 AIFS station-records, 0 of 7869 ACE2 station-records
differ). The round-3 QC-ranking fix DOES fire: a handful to ~11 extra
temperature observations recovered per 5-day window out of 65-76k accepted
(e.g. AIFS 72438 -> 72441 t obs; ACE2 70109 -> 70112 first cycle, 65205 ->
65216 last cycle), u/v unaffected in every window checked -- small, and
strictly an improvement (recovers previously wrongly-rejected obs, never
drops a good one).

**Reproducibility implication**: because of that last fix, re-running
either `-ps_raobs-t1` config with the current code would NOT reproduce the
already-saved output bit-for-bit (the first round-3 item, after every
earlier fix round proved exactly reproducible against the real archive) --
expected to be within noise for any practical verification metric, but
worth knowing before treating a fresh rerun and the already-saved cycles
as identical.

**ps-only reproducibility** (checked separately, for the sibling
`add-parquet-ps-dev` repo's own `-t3` long-cycling runs): every change from
this PR sits behind `raobs_enabled`/`raobs_traj is not None` guards
(`get_raobs` returns `None` immediately when disabled; `raobs_parquet.py`
is only imported lazily inside it), except the `jc_ps_weight`
"reuse-the-loss-decode" refactor pulled in from `main` -- and that one is
forward-value-identical by construction (`ACE2Model.decode_state`'s `'sp'`
is a pure column slice of the packed state, unaffected by what else is
requested alongside it in the same decode's `only=` set). `psobs_parquet.py`
and `forecast_model.py` are byte-identical between the two repos, and
`prepbufr_parquet_common.py`'s diff is purely additive (new raobs-only
functions, zero changes to what `psobs_parquet.py` itself calls). A
ps-only rerun from this branch's current code should reproduce the sibling
repo's saved `-t3` results bit-for-bit.

## Why parquet-only (no `.txt`/parquet config switch)

The sibling repo's draft made this same unconditional-replacement call, but
could afford to because it predates any CI test depending on the `.txt`
reader. This repo's `main` has since grown `make check`/`make check-ace2`
(see `tests/README.md`), and `tests/ace2/test_ace2_mini.py` drives the real
solver end to end against a committed `.txt` obs fixture
(`tests/data/mini_ace2/psobs/psobs1_*.txt`) -- which an unconditional switch
would simply break.

Decided to keep the port's "no dead code, one reader" design anyway, and
rebuild the CI fixture itself as a synthetic parquet archive instead of
threading a `psobs_source: 'txt'|'parquet'` switch through `get_psobs` (the
user's explicit choice -- a config-switchable reader was considered and
rejected in favor of matching this repo's existing single-reader design
once the fixture itself no longer needs to be `.txt`).

### The CI fixture: synthetic, not a real-obs subsample

`tests/make_fixtures.py`'s `_write_psobs_parquet` generates a tiny
normalized-PREPBUFR-parquet archive (ADPSFC/SFCSHP partitions per cycle,
only the columns `src/psobs_parquet.py` actually reads) entirely in-process --
globally-scattered synthetic lat/lon, pressure ~1013 hPa + N(0, 4) hPa
noise, quality marks 0-3, per-ob error ~N(0.5, 0.15) hPa, fixed seed. This
replaces the old fixture's approach (subsample 2000 real reports per time
from the live `.txt` obs directory, Ursa-only) with something that rebuilds
anywhere `pip install pyarrow` reaches, with no dependency on any obs
archive, real or synthetic-upstream.

**This did require retuning `tests/ace2/test_ace2_mini.py`'s cost-drop
threshold** (`MAX_FINAL_TO_FIRST`): the old fixture's real obs, pulled from
an actual synoptic analysis, gave the tiny model's Jo term a much stronger,
more structured gradient signal than first-draft synthetic noise (~1013 +/-
12 hPa, uncorrelated across stations) did -- the first draft only reached a
33% cost reduction in 50 epochs against the existing 40%-reduction
threshold. Tightening the synthetic noise (12 -> 4 hPa) and per-ob error
(~1.0 -> ~0.5 hPa, i.e. a stronger pull per observation) recovered a 55%
reduction, comparable to the old fixture's observed 59%. This is a
legitimate recalibration against a deliberately-changed fixture, not a
loosened assertion against unchanged code -- the threshold itself
(`MAX_FINAL_TO_FIRST = 0.6`) did not move, only the documented "observed"
comment did.

## Testing

- `tests/unit/test_psobs_parquet.py`: `load_psobs_parquet` against tiny,
  hand-crafted parquet archives (not the mini-ACE2 fixture), one behavior
  per test -- acceptance within/outside time tolerance, report-type and
  quality-mark filtering, the non-surface-level skip, longitude wrapping,
  the single-nearest-report-per-station dedup and its quality-mark
  tie-break, flat-vs-source error, multiple independent slots, and the
  missing-cycle-directory/invalid-error-std error paths.
- `tests/unit/test_config.py`: `_resolve_observations_config`'s validation
  (unknown `observations`/`observations.psobs`/`observations.raobs` keys,
  missing `observations.psobs.path`, the raobs backend guard and
  unsupported/empty `variables`) and that a representative set of nested
  keys resolve to the expected flat `exp` keys
  (`test_observations_psobs_resolved_to_flat_keys`,
  `test_observations_raobs_resolved_to_flat_keys`), plus acceptance for
  every backend in `RAOBS_SUPPORTED_BACKENDS`.
- `tests/unit/test_raobs_parquet.py`: `load_raobs_parquet` against tiny,
  hand-crafted parquet archives (including an `events` partition for the
  sensible-temperature cases), one behavior per test -- mandatory-level
  pressure-match tolerance, each QC bit (physical height, non-monotonic
  height, below-station, PREPBUFR quality mark), the sensible-temperature
  event-stack lookup and its fallback to the raw TOB, the humidity
  pressure floor, multiple variables/slots, unsupported-variable and
  station-capacity-overflow errors.
- `tests/ace2/test_ace2_mini.py` (`make check-ace2`): now reads the
  synthetic parquet fixture via the real `get_psobs` path, with
  `observations.psobs.use_source_error: True` added to its config to
  exercise that branch too (matching the production configs that needed
  it, above); also enables `observations.raobs` (temperature/u/v) against
  `tests/make_fixtures.py`'s `_write_raobs_parquet`/
  `_ace2_background_profiles` fixture (the real mini-ACE2 checkpoint's own
  background + noise, not independent climatology -- see that function's
  docstring and `MAX_FINAL_TO_FIRST`'s updated "observed" comment). Finding
  and fixing this test's crash on ACE2 (a NaN-gradient bug in
  `ace2_model.py`'s derived pressure-level 't'/'u'/'v'/'q', never exercised
  before raobs) is its own write-up in `docs/integrate_multiple_backends.md`.
- All pass locally with a from-scratch `.venv-ci` (`rdhpcs-python/3.12` +
  `envs/requirements-ci.txt`/`envs/requirements-ci-ace2.txt`) as of this writing.
