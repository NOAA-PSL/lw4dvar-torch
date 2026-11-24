# Parquet observations: detailed history (psobs/raobs)

The full, round-by-round history behind `docs/integrate_multiple_parquetobs.md`
-- every Copilot/maintainer review response, every GSI-matching mechanism
tried and later superseded by a JEDI-consistent one (`use_source_error`,
the ermin/ermax gross-check clamp, the x1.2 error inflation -- all removed,
see "Simplifying raobs to JEDI-only" below), the investigations that
confirmed or disproved a reviewer's concern, and the exact tests/fixtures
touched at each step.

**Read the short doc first** (`docs/integrate_multiple_parquetobs.md`) for
what's actually implemented today. Come here only when you need the "why" --
why a mechanism looks the way it does, why an alternative (often GSI's own
approach) was tried and abandoned, or the exact reasoning behind a specific
review response. Sections are in chronological order; a later section can
supersede an earlier one's design without the earlier one being edited to
say so -- if something here contradicts the short doc, the short doc (and
the actual code) wins.

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

## raobs QC/thinning vs. GSI's actual implementation (2026-11-18)

Compared `raobs_parquet.py`/`long_window_4dvar_utils.py`'s raobs QC against
GSI's own ADPUPA (kx 120/220/221) handling, read directly from a
`global-workflow` checkout's `sorc/gsi.fd/{read_prepbufr.F90,qcmod.f90}`
and `fix/gsi/global_convinfo.txt` (not from memory/documentation of GSI --
the actual source this repo should be consistent with where it claims to
be).

**Consistent, confirmed directly against the source:**
- **No thinning**: `global_convinfo.txt`'s `ithin`/`ptime` columns are 0
  for kx 120/220/221 -- GSI thins none of ADPUPA (thinning there is for
  dense types like aircraft/profiler/mesonet). This repo does no raobs
  thinning either.
- **`time_tolerance_hours: 3.0`**: GSI's `twindow` column for 120/220/221
  is exactly 3.0h, matching the value this repo already settled on
  independently (see the `time_tolerance_hours` bullet above, derived from
  station counts against the real archive, not from GSI).
- **`quality_mark_max: 3`**: GSI's `lim_qm=4` (`read_prepbufr.F90`) means a
  level is used only if its quality mark is `< 4` -- identical to this
  repo's `value <= 3` convention.

**Differences, now understood rather than accidental:**
- **No vertical interpolation** (mandatory-level matching against the
  model's own pressure levels) vs. GSI's continuous log-p interpolation to
  each ob's exact reported pressure (`errormod`, every significant level
  too, not just mandatory ones) -- a deliberate simplification tied to the
  ML backends only exposing fixed pressure levels, not a bug.
- **Virtual/sensible temperature**: this repo converts the OBSERVATION to
  sensible T (event-stack walk to the first non-VIRTMP event,
  `sensible_temperature`); GSI's global/GDAS path instead keeps the ob as
  virtual T and converts the BACKGROUND to virtual space for the
  comparison (our approach instead matches GSI's own regional/RTMA
  branch). Same correction, opposite side of O-B -- appropriate here since
  the ML backends' state is sensible T with no virtual-T operator.
- **Height-bound/below-station/non-monotonicity QC is this repo's own
  addition**, not a port of anything GSI's reader does -- GSI trusts the
  quality marks already assigned upstream by NCEP's prepdata/complex-QC
  programs instead of re-deriving height checks itself. Loose bounds, so
  low risk of over-rejecting; kept as a defensive extra.
- **q error/cutoff**: GSI uses a smooth pressure-dependent error table
  (`setupq.f90`'s `ptablq`); this repo's fixed per-level table
  (`RAOBS_Q_ERROR_G_KG`) plays the same role, plus an optional hard
  `humidity_min_pressure_hpa` floor GSI has no equivalent of.
- **`VIRTMP`'s program code is hardcoded to `8`** here vs. GSI reading it
  from the BUFR table at runtime (`ufbqcd(lunin,'VIRTMP',ivtcd)`). Not
  changed -- low risk, low value until an archive built with a different
  BUFR table version is actually hit.

**Fixed as a result**: GSI applies an O-B background/gross-error check to
every conventional variable (`errormod` + convinfo's `gross`/`ermax`/
`ermin`, plus flat error inflation for quality marks 3/7 and above 100/50
hPa) -- t/u/v raobs here had NO such check at all, only the read-time
PREPBUFR-quality-mark/height QC (`q` already had one, `q_bg_check`/
`RAOBS_Q_BG_CHECK`, 4-sigma). In a strong-constraint 4D-Var minimizing a
smooth cost, an outlier that still passes its PREPBUFR quality mark has
more leverage than in GSI's iterative variational QC -- plausibly relevant
to the still-open "ACE2 Tropics z500 degradation with raobs" investigation
above. Added `tuv_bg_check`/`RAOBS_TUV_BG_CHECK` (default 4.0, one shared
multiplier for t/u/v, mirroring how they already share one `error_std`
dict) to `_compute_raobs_observation_diagnostics_at_time`, following the
existing `q_bg_check` pattern exactly (`used = used & (|O-B| <= bg_check *
error)`, `.detach()`ed, re-evaluated every epoch at the current state).
Config keys/validation/template entry added alongside; not yet re-run
against the longcyc ACE2/AIFS2 experiments to see whether it actually
changes the z500 finding -- that's the natural next step.

## raobs `use_source_error`: the archive's own GSI error, not hand-curated (2026-11-18)

Following on from the GSI comparison above: rather than hand-curate GSI's
`prepobs_errtable.global` values into Python constants (considered, judged
not worth the parsing risk/maintenance for a legacy fixed-width Fortran
format), checked whether the real archive already carries the per-
observation error GSI itself would use -- it does. The `levels` table for
every PREPBUFR class (confirmed via `/scratch4/BMC/gsienkf/Bo.Huang/
expCodes/ML/lw4dvar-torch-dev/prepbufr_to_parquet/convert_prepbufr_to_parquet.py`)
carries `temperature_error_c`/`wind_error_ms`/`humidity_error_tenths_percent`
(the PREPBUFR TOE/WOE/QOE mnemonics), parallel to `pressure_error_mb`
which `psobs_parquet.py`'s `use_source_error` already reads. Sampling a
real 2015-01-07 00Z ADPUPA partition: `temperature_error_c` ranges
0.8-1.5 K, `wind_error_ms` 1.4-3.2 m/s, `humidity_error_tenths_percent`
flat 2.0 -- matching GSI's own `prepobs_errtable.global` type-120/220
values exactly (see the GSI-comparison section above), because these
fields are stamped onto each observation by NCEP's own operational
prepdata processing using that same table.

**Added `observations.raobs.use_source_error`** (`raobs_parquet.py`,
`get_raobs`/`_compute_raobs_observation_diagnostics_at_time` in
`long_window_4dvar_utils.py`), mirroring `psobs_parquet.py`'s existing
flag:
- **t/u/v**: `_VARIABLE_ERROR_COLUMNS` maps each base to its archive
  column (`u`/`v` share `wind_error_ms` -- GSI's wind error is one vector
  quantity, not separate per component) -- a direct per-level value, NaN
  (missing/non-positive) falling back to the configured flat `error_std`.
  `raobs_traj['error_<base>']` becomes a per-(level, station) tensor
  instead of a flat float when source error is in use; `_compute_raobs_
  observation_diagnostics_at_time`'s else branch now accepts either.
- **q**: `humidity_error_tenths_percent` is RH-based (confirmed as
  `percent/10` from the converter's own field metadata: a stored `2.0`
  means 20% RH, not 2%), not kg/kg -- converted via saturation specific
  humidity (`_saturation_specific_humidity`: the same Magnus-Tetens +
  eps formula as `backends/ace2/ace2_ic.py`'s `_saturation_vapor_pressure`/
  `_specific_humidity_from_dewpoint`, duplicated rather than imported
  since `src/` must stay usable without any backend's dependencies) at
  the archive's own `background_temperature_c`/`background_pressure_mb`.
  `raobs_traj['source_error_q']` overrides the existing table/fraction
  `effective_error` per element wherever finite, rather than being pre-
  filled like t/u/v -- q already has a non-trivial fallback hierarchy
  (`error_q_level` table, or fraction-of-observed-value) that isn't a
  single flat value to pre-fill with.

**Known limitation, flagged rather than fixed (decided 2026-11-18)**: the
qsat conversion for q uses the archive's own STATIC `background_*`
fields -- the historical GFS/GDAS background interpolated to the
observation at archive-creation time -- not this run's own model state.
GSI itself recomputes qsat fresh from its OWN live background every
analysis (`setupq.f90`'s `genqsat(ges_qsat, ges_tsen, ges_prsl, ...)`,
confirmed directly in the GSI source, not from any stored observation
field). Doing the same here would mean moving the RH-to-kg/kg conversion
out of `raobs_parquet.py`'s one-time read and into the per-epoch loss
hot path (`_compute_raobs_observation_diagnostics_at_time`), needing a
torch qsat implementation and careful `.detach()`-ing so the error
denominator can't be gamed by the minimizer -- real complexity weighed
against the static approach's small inaccuracy (a frozen, historical
background standing in for the live one). Decision: ship the static
version now; revisit the live-background version only if it turns out to
matter in practice.

**Testing**: `tests/unit/test_raobs_parquet.py` -- `use_source_error`
disabled omits the `source_error_<v>` keys entirely; reads real
`temperature_error_c`/`wind_error_ms` values and falls back to NaN when
missing or non-positive; wind error shared between u and v; q's
qsat conversion checked against `_saturation_specific_humidity` called
directly (not a hand re-derivation of the physics -- that formula is
already validated against `backends/ace2/ace2_ic.py`'s matching
implementation) and against a missing-background NaN case.
`tests/unit/test_solver_helpers.py` -- `_compute_raobs_observation_
diagnostics_at_time` reads a per-station tensor `error_<base>` correctly
(not broadcasting a single scalar) for t/u/v, and for q, overrides the
fraction-based error per element while falling back to it wherever the
source value is NaN. `tests/unit/test_config.py` -- the new
`use_source_error` key resolves to `raobs_use_source_error`.

**`use_source_error` now defaults to `True`** (`get_raobs`,
`configs/config.yml.template`), rather than requiring it to be opted
into -- the real archive value is the better default now that it exists.
`tests/ace2/mini_ace2.py` explicitly sets `False` for its raobs block:
the synthetic fixture (`tests/make_fixtures.py`'s `_write_raobs_parquet`)
has none of `temperature_error_c`/`wind_error_ms`/
`humidity_error_tenths_percent`/`background_*` -- `read_partition` would
raise on the missing columns otherwise. Golden fixture regenerated to
match (the config echo in `/config/exp/observations/raobs` now includes
the key).

## Correction: the archive's `use_source_error` value is NOT GSI's actual DA error (2026-11-18)

The section above claimed the archive's embedded TOE/WOE/QOE value
"matches GSI's own `prepobs_errtable.global` ... exactly" and is "the
per-observation error GSI itself would use." That was wrong -- or more
precisely, incomplete: the numbers happen to land in the right range, but
tracing `read_prepbufr.F90` line-by-line (not just comparing magnitudes)
shows GSI does not actually use the embedded value in its Jo weighting.

In `read_prepbufr.F90`:
- Line 1226: `call ufbint(lunin,obserr,8,255,levs,oestr)` with
  `oestr='POE QOE TOE NUL WOE NUL PWE'` -- reads the embedded PREPBUFR
  error into `obserr`. This is the exact value our converter copies into
  `temperature_error_c`/`wind_error_ms`/`humidity_error_tenths_percent`.
- Line 1231: `if(oberrflg .and. kx<=300) then` -- immediately after,
  whenever the error-table fix file loaded successfully (`oberrflg`,
  true in every real operational run -- `prepobs_errtable.global` is
  always linked in) and the type is conventional (`kx<=300`, true for
  every raob `kx`: 120 for T/Q, 220/221 for u/v), `obserr` is
  **unconditionally overwritten**, both in the `njqc` branch (separate
  `etabl_t`/`etabl_q`/`etabl_uv`/`etabl_ps` tables, lines 1244-1496) and
  the non-`njqc` branch (one combined `etabl`, lines 1497-1529:
  `obserr(3,k)=(one-del)*etabl(itypex,k1,2)+del*etabl(itypex,k2,2)` for T,
  column 3 for Q, column 4 for UV, column 1 for PS, column 6 for PW) --
  interpolated fresh from `prepobs_errtable.global` by type and pressure,
  with nothing of the embedded BUFR value surviving.

So: the embedded value is read, then discarded, in the actual operational
GDAS. `prepobs_errtable.global` is a GSI/GDAS fix file -- confirmed
independently via `exglobal_atmos_analysis.sh`'s
`OBERROR=${FIXgfs}/gsi/prepobs_errtable.global` / `${NLN} "${OBERROR}"
errtable`, read every cycle by `converr.f90`'s `converr_read` -- and *it*,
not the archive's embedded field, is what GSI's Jo term actually weights
by. The archive's TOE/WOE/QOE is a real PREPBUFR quantity (confirmed from
the converter's own field metadata) stamped on by an earlier processing
stage, but it's upstream of, and superseded by, GSI's own ingest.

**Consequence: `raobs_use_source_error` now defaults back to `False`**
(`get_raobs` in `long_window_4dvar_utils.py`, `configs/config.yml.template`).
The configured `error_std` table (settable to GSI's own
`prepobs_errtable.global` values, as in the config template's example) is
closer to what GSI actually assimilates with than the archive's embedded
value is. `use_source_error: True` remains available as an opt-in for
experimenting with the archive's own per-level value, with the docstrings
in `raobs_parquet.py` and the config template corrected to say so plainly
-- it is a real, physically sensible per-observation error, just not the
one GSI's own DA run would have used for that same observation.

`tests/ace2/mini_ace2.py`'s explicit `'use_source_error': False` for its
raobs block is now redundant with the default but left in place (with an
updated comment) for clarity, since the synthetic fixture still has none
of the real error columns regardless of the default's value.

## GSI's ermin/ermax gross-check clamp for raobs (2026-11-18)

Following directly from `use_source_error`: now that the error itself is
GSI's real value, GSI's own gross check -- a **clamped** ratio,
`|O-B| / clip(error, ermin, ermax) > gross` (`global_convinfo.txt`, kx
120/220/221: `gross=8.0` for t/uv/q, `ermin`/`ermax` per variable: t
1.3/5.6 K, uv 1.4/6.1 m/s, q 10/100 %RH) -- is directly comparable, not
just a borrowed threshold against an arbitrary error the way
`tuv_bg_check`'s original `4.0` was. Bumped `RAOBS_TUV_BG_CHECK`/
`RAOBS_Q_BG_CHECK` from `4.0` to GSI's actual `gross=8.0`
(`long_window_4dvar_utils.py`) and added the `ermin`/`ermax` clamp.

**Where the clamp applies -- confirmed, not assumed, by reading GSI's
source**: initially implemented the clamp by baking it into the stored
source error itself (in `raobs_parquet.py`, before it ever reaches the
solver) -- simpler, but wrong. Real archive values at 850 hPa (T: 0.8 K)
sit *below* GSI's own `ermin` (1.3 K), and clamping them up would have
flattened exactly the per-level structure that `use_source_error` exists
to capture. Checked GSI's `setupq.f90` directly: the `ermin`/`ermax`-
clamped `obserror` computed there (`max(cermin*0.01, min(cermax*0.01,
data(ier,i)))`) is never fed into the actual Jo weight -- that comes from
the unclamped raw table value via `errorx=(data(ier,i)+dprpx)*qsges`. A
*different*, later-computed `obserrlm` (clamped, but derived from the
already-finalized weight) reads as a diagnostic-only quantity. So GSI's
clamp bounds the gross-check *decision*, not the weight.

**Fix**: moved the clamp out of `raobs_parquet.py` entirely --
`source_error_<v>`/`source_error_q` stay the real, unclamped archive
value (the actual Jo weight, via `effective_error`). The clamp now lives
in `_compute_raobs_observation_diagnostics_at_time`, applied only to a
local `qc_error` used for the gross-check comparison:
- **t/u/v**: `RAOBS_TUV_ERMIN`/`ERMAX` are flat per-variable constants
  (no archive dependency) -- `qc_error = clamp(effective_error, ermin,
  ermax)`, only when `error_<base>` is a tensor (`use_source_error`); a
  flat `error_std` is never clamped.
- **q**: GSI's bounds are RH-percent, not kg/kg, so they vary with qsat
  the same way the source error itself does. `raobs_parquet.py` now also
  returns `source_qsat_q` (the qsat used for the RH->kg/kg conversion,
  kept separately, NaN wherever `source_error_q` is too) so the gross
  check can derive `ermin_kgkg = 0.10*qsat`, `ermax_kgkg = 1.00*qsat` at
  the comparison point, without touching `source_error_q` itself. Where
  no source qsat exists (table/fraction error in use), `qc_error` is just
  `effective_error`, unclamped -- there's nothing GSI-sourced to bound
  against.
- Only applied when `use_source_error` is in effect, by construction --
  never to this repo's own flat/table defaults (e.g. the mini-ACE2
  fixture's deliberately inflated `error_std`, which must not be silently
  overridden).

**Testing**: `tests/unit/test_raobs_parquet.py` -- `source_error_t`
confirmed to stay unclamped (0.8 K, below GSI's own ermin, passes
through unmodified) and `source_qsat_q` exposed/NaN-propagated alongside
`source_error_q`. `tests/unit/test_solver_helpers.py` -- for both t/u/v
and q, `effective_error` (the Jo weight) stays unclamped while the gross-
check decision flips because the comparison's `qc_error` is clamped; a
flat `error_std` is confirmed untouched by the clamp.

**Not yet done**: re-validate `RAOBS_Q_BG_CHECK`'s bump from `4.0` to
`8.0` against the real 10-cycle AIFS experiment that originally
validated `4.0` (for the older table/fraction q-error path, not
`use_source_error`) -- changing it is a deliberate "mimic GSI" choice,
not yet re-confirmed to help or hurt skill in that same experiment.

## GSI's x1.2 error inflation for raobs (2026-11-18)

The remaining piece from the GSI comparison: `read_prepbufr.F90` also
inflates the error it writes out (`cdata_all(1,iout)=toe`, etc.) by x1.2,
multiplicatively, on two independent triggers:
- **Quality mark 3 or 7** (`inflate_error = qm==3 .or. qm==7`): applies
  to t, u/v, AND q alike (`toe=toe*r1_2`, `woe=woe*r1_2`,
  `qoe=qoe*r1_2` -- confirmed q gets this too by reading `read_prepbufr.F90`
  directly; an earlier answer in this conversation wrongly said q doesn't).
- **Above a variable-specific pressure**: t only, above 100 hPa
  (`if(ppb < r100)toe=toe*r1_2`); u/v only, above 50 hPa
  (`if(obsdat(1,k) < r50)woe=woe*r1_2`). q has no pressure-based trigger.

Unlike the ermin/ermax clamp (gross-check only, see above), this
inflation modifies the value GSI actually uses for weighting -- `toe`/
`woe`/`qoe` in `cdata_all` is what `setupt`/`setupw`/`setupq.f90` read
back as `data(ier,i)`, both for the Jo weight (`errorx=(data(ier,i)+
dprpx)*qsges`-style) and for the later ermin/ermax-clamped gross check.
So `_gsi_inflate_error` (`raobs_parquet.py`) is applied directly to
`source_error_<v>`/`source_error_q` at read time, inflating the value
itself -- not a separate gross-check-only quantity the way the clamp is.
It composes correctly with the existing clamp with no changes needed on
that side: the clamp already operates on `effective_error`, which now
already includes this inflation, exactly mirroring GSI's own order of
operations (inflate at read time, clamp only for the gross check later).

Only applied when `use_source_error` is in effect (same reasoning as the
clamp) -- this repo's own flat/table defaults are never touched.

**Testing**: `tests/unit/test_raobs_parquet.py` -- each trigger checked
independently (qm 3, qm 7 stacked with the pressure trigger, t's 100 hPa
vs. wind's 50 hPa threshold not being interchangeable, q's qm-only
inflation with no pressure trigger even at a very low pressure), plus a
good-quality-mark case confirming no spurious inflation.

## PR #29 review comment: UFO's sonde QC config, three follow-ups (2026-11-18)

A review comment on PR #29 (from `jswhit`, generated with Claude Code)
compared this work against JEDI UFO's GFS sonde QC config
(`sonde_gfs_qc.yaml`, reproduces GSI's sonde QC), an independent,
actively-maintained reference. Its T/wind error tables matched
`prepobs_errtable.global`'s type-120/220 numbers in this account's GSI
checkout exactly -- useful independent corroboration of the GSI
comparison work above. Its `ObsErrorFactorConventional` point (GSI's
inflation for several obs falling in one model layer) is correctly
reasoned to be a no-op here: mandatory-level-only matching means at most
one observation per model level, so that trigger can't fire. Three
concrete follow-ups acted on:

1. **`error_std` as scalar-or-table**: each of `error_std`'s t/u/v
   entries may now be a `{hPa: value}` table (same ln(p) interpolation as
   `q_error_g_kg`, via a new shared helper `_interp_level_table`) instead
   of only a flat scalar -- e.g. the UFO/GSI table itself, closing the one
   gap flagged earlier this session (the flat `error_std` default was
   never GSI-derived). `get_raobs` broadcasts a table across every obs
   slot/station once (`error_<base>` becomes a tensor); when combined
   with `use_source_error`, the table is the per-element fallback instead
   of a flat value. **Important design point**: GSI's ermin/ermax clamp
   must NOT apply to a configured table (even the identical UFO/GSI
   numbers) -- only to the genuine archive value. Since `error_<base>`
   being a tensor no longer implies "from the archive" (it can now also
   mean "a configured table"), added an explicit `is_source_error_<base>`
   flag to gate the clamp correctly; `torch.is_tensor(...)` alone is no
   longer the right test.
2. **Per-variable `bg_check` with an absolute limit**: `observations.
   raobs.bg_check` (optional) gives UFO/GSI's actual per-variable gross-
   check form -- `{t: {threshold, absolute}, u: ..., v: ..., q: ...}` --
   rejecting if EITHER the ratio OR the absolute limit fires. Resolved by
   a new `_raobs_gross_check_threshold(raobs_traj, base, shared_key)`:
   the per-variable override if configured, else the existing shared
   `tuv_bg_check`/`q_bg_check` scalar with no absolute limit --
   fully additive/backward-compatible, default behavior unchanged.
   Deliberately did NOT change `tuv_bg_check`/`q_bg_check`'s own default
   (8.0, GSI's convinfo `gross`) to UFO's numbers (7.0/6.0/8.0) -- same
   "different GSI vintages, different tuned numbers" pattern hit
   throughout this session (e.g. the pasted table that matched neither
   `prepobs_errtable.global` nor any `_nqcf` variant in this checkout);
   the STRUCTURE is the valuable, low-risk part, the exact numbers are a
   config choice left to whoever runs it.
3. **Physical bounds QC**: a new `QC_PHYSICAL_BOUNDS` flag
   (`raobs_parquet.py`), same numbers as UFO's config -- T 195-327 K, q
   1e-8-0.035 kg/kg, |u|/|v| <= 130 m/s. Applied at read time (independent
   of the background/model state, like the existing height-based checks),
   not configurable (fixed constants, like the ermin/ermax clamp).
   **Caught a real issue while testing this**: several existing test
   fixtures relied on `level()`'s default `temperature=300.` -- that
   column is Celsius (`temperature_c`), so it converts to 573 K, always
   unphysical, just never checked before. Only one existing test actually
   asserted on `mask_t` being True with that default; fixed by giving it
   a realistic Celsius value instead of relaxing the new bound.
   NOT implemented: UFO's wind-direction-vs-background check (needs the
   live background at the ob's location, so it belongs in the forward
   operator, not here -- lower priority, deferred).

The wind-direction check aside, `bg_check`'s absolute limit, and
`error_std`'s table form are all gated/additive so none of them touch
existing default behavior except the physical-bounds QC, which is
unconditional -- verified against the real ACE2 fixture before trusting
it doesn't reject anything there.

## Balloon drift time: `use_drift_time` (2026-11-18)

A separate PR #29 review thread (https://github.com/NOAA-PSL/lw4dvar-torch/pull/29#issuecomment-6994064088)
raised balloon drift time as a potentially serious issue: using the
report's single launch time for every level means upper-level
observations get compared against the wrong background by hours. One
suggestion was to modify the parquet converter to derive this from
PREPBUFR's `HRDR` mnemonic. Checked first: the parquet archive already
carries `drift_time_hours` per level, confirmed directly against real
data (one 218-level sounding, report 460, 2015-01-07 00Z): −0.35h at the
surface (1009 hPa), crossing zero around 438 hPa, growing to **+1.62h by
5 hPa** -- exactly the magnitude the concern was about, and already
available with no converter change needed.

**Added `observations.raobs.use_drift_time`** (default **False** --
unlike `use_source_error`, this changes WHICH background a level is
compared against, not just an error magnitude, so introduced the same
cautious way `use_source_error` itself was before its own default got
flipped): `_profile_records` (`raobs_parquet.py`) already computed the
slot-assignment time independently per LEVEL ROW, it just happened to
read the same report-level time for every level of a report (nothing
else varied it) -- so using `report_time + timedelta(hours=
drift_time_hours[level])` instead, falling back to the report's own time
wherever `drift_time_hours` is missing/non-finite, needed no
restructuring: one profile's levels splitting across multiple obs slots
is already handled by the existing `(slot_index, station_key)` keying,
since each slot already tolerates "this station has only some levels
here." Nothing downstream (`get_raobs`'s per-slot tensor building, the
forward operator) needed to change at all.

**Testing**: `tests/unit/test_raobs_parquet.py` -- a two-level profile
(850/500 hPa) with drift pushing the 500 hPa level into the next 6h
slot: confirmed it moves there with `use_drift_time: True` and stays
with the 850 hPa level in the original slot with the default `False`;
missing `drift_time_hours` falls back to the report's own time. (Hit an
unrelated test-construction bug along the way: both levels defaulted to
the same `height`, tripping the pre-existing non-monotonic-height QC
once they shared one slot -- fixed by giving them realistic, increasing
heights, not a drift-time issue.)

**Not yet done**: this only corrects the LOCATION-IN-TIME each level is
matched against (which existing, discrete obs slot). It doesn't touch
`alpha`'s own per-slot (not per-observation) background time-
interpolation -- that's the separate, larger "per-observation 4D
interpolation" architecture question discussed earlier (GSI's
`tintrp31` interpolates to each observation's own exact time from
several in-window background snapshots; this repo interpolates once per
slot, shared by every observation assigned to it). `use_drift_time`
narrows the mismatch between a level's actual time and its assigned
slot's nominal time; it doesn't eliminate it the way true per-
observation time-interpolation would. Also not yet done: using
`drift_longitude`/`drift_latitude` (present in the same archive,
confirmed earlier) for the forward operator's horizontal interpolation
location -- a related but separate, slightly larger change (needs
per-level interpolation weights, not per-station).

## Session recap (2026-11-19): `use_source_error`'s real relationship to GSI, finer raobs time interpolation, four new longcyc trials

Picking up directly from the "Correction" section above. This session's
own earlier correction (flip `use_source_error` to `False` because GSI's
`read_prepbufr.F90` overwrites the embedded value rather than reading
it) was right about MECHANISM but incomplete about CONSEQUENCE -- a
follow-up investigation, prompted by a user push-back for more rigor,
settled the actual question: **is the archive's embedded value usable
as a stand-in for `prepobs_errtable.global` in our own DA?** Yes,
empirically confirmed, not just inferred.

**What was verified, concretely**: reproduced GSI's own linear-in-
pressure `etabl` interpolation (`read_prepbufr.F90`'s non-`njqc` branch,
lines 1497-1529 -- confirmed `njqc=.false.` is GSI's actual default and
is never overridden anywhere in this account's global-workflow
checkout's `parm/`, so that branch is the one operationally active) from
the real `prepobs_errtable.global` fix file, for kx=120 (T/Q) and
kx=220/221 (UV) -- exactly the types this repo's raobs reader uses --
then compared it level-by-level against the real 2015-01-07 00Z ADPUPA
archive partition's `temperature_error_c`/`wind_error_ms`/
`humidity_error_tenths_percent` (~64,000 levels):

| type | match after rounding the GSI prediction to 1 decimal |
|---|---|
| T (kx=120) | 29157/29595 exact; all 438 remaining are exact-half rounding-tie artifacts (e.g. predicted 0.85 rounds to the archive's 0.9 under round-half-away-from-zero, 0.8 under Python's banker's rounding) |
| UV (kx=220) | 33881/33955, same rounding-tie story |
| UV (kx=221) | 832/832 -- 100% |
| Q (kx=120) | 26375/26375 -- 100% (trivial: this table's Q column is flat 2.0, no interpolation variance) |

Every discrepancy is explained by the archive's 1-decimal BUFR storage
precision, not a different source or formula. Also checked: for
quality-mark-3 observations, the archive's stored value is the RAW,
pre-inflation table value (ratio ~0.99-1.0 against the raw prediction,
not ~1.2) -- confirming GSI's own x1.2 quality-mark inflation
(`docs` section above) is applied by GSI itself at read time, never
baked into the archived mnemonic, so this repo's own `_gsi_inflate_error`
is correctly additive on top of `use_source_error`, not double-counting
anything.

Independent corroboration: PR #29 comment
[#issuecomment-6994962917](https://github.com/NOAA-PSL/lw4dvar-torch/pull/29#issuecomment-6994962917),
`jswhit` directly: *"The operational DA uses the prepobs_errtable.global
... The values encoded in the prepbufr are not used."* -- correct about
mechanism (GSI recomputes, doesn't reuse the BUFR field), and not in
tension with the verification above: the recomputed value and the
archived value are numerically the same (to rounding), because both
derive from the same table and formula, just at different times (archive
creation vs. analysis time).

**Net conclusion, stated precisely**: the archive's TOE/WOE/QOE values
ARE encoded in the PREPBUFR (literal mnemonics, not invented by this
repo's converter) AND are NOT read by GSI's own DA (overwritten by a
fresh `etabl` recompute) AND are numerically consistent with
`prepobs_errtable.global` to BUFR storage precision -- so
`use_source_error: True` is a faithful, verified stand-in for "use
`prepobs_errtable.global` in our own DA," for raobs t/u/v/q (kx
120/220/221) specifically. **Decided: leave the code default at `False`**
(unchanged from the "Correction" section above -- opt-in, not required)
but the three new `-t2` configs below turn it on explicitly for raobs,
now on this stronger evidence rather than the original, looser "same
range" comparison.

**Still open, not investigated this session**:
- `psobs` has no background/gross check at all (unlike raobs, which now
  has `tuv_bg_check`/`bg_check`). Identified while reviewing
  PR #29 comment 6993377418's UFO reference table, which gives
  `stationPressure`: threshold 3.6σ, absolute limit 990 Pa -- a
  reasonable default if/when this gets implemented, mirroring raobs'
  `_raobs_gross_check_threshold` pattern. Flagged, not built.
- `observations.psobs.error_std_hpa` (default 1.0) could be tightened to
  ~1.2 hPa, the middle of `prepobs_errtable.global`'s own PS-column
  range (1.3/1.1/1.2 mb for kx 180/181/187, this repo's own
  `psobs_report_types` default) -- confirmed directly from the fix file,
  not yet applied to any tracked config.
- The PR #29 thread has not been updated with this session's stronger
  verification (the "likely same lineage" draft reply from the prior
  session is now supersedable by the precise finding above) -- still
  pending the user's decision on how/whether to post it.

**Config-template-only change (no code/behavior change)**: per PR #29
comment
[#issuecomment-6993629238](https://github.com/NOAA-PSL/lw4dvar-torch/pull/29#issuecomment-6993629238),
confirmed `get_raobs` already reuses `psobs_traj['step_lo']`/`['alpha']`
(`long_window_4dvar_utils.py:1570-1571`) and
`raobs_time_tolerance_hours` already defaults to `dt_obs / 2.`
(`:1394`) -- so lowering `windows.*.dt_obs` (e.g. to `1`) already
sharpens the background each raob level is compared against to within
`dt_obs/2` of its own report time, with **no new code**, as long as
`observations.raobs.time_tolerance_hours` is set explicitly (otherwise
it silently collapses to `dt_obs/2`, e.g. 0.5h, matching psobs's much
tighter convention and dropping most raobs). Documented both sides of
this in `configs/config.yml.template`: a new paragraph under
`windows.*.dt_obs` describing the shared-slot mechanism and the S2-tide
caveat (dt_obs is shared with psobs), and a new paragraph under
`observations.raobs.time_tolerance_hours` warning it must be set
explicitly whenever `dt_obs` is lowered.

**Four new long-cycling trials** (gitignored by convention, like the
existing `-t1` pair -- `config*longcyc*.yml`/`slurm/*longcyc*.sh`
patterns in `.gitignore`), each `-t1`'s own config plus: `windows.'5'.
dt_obs: 1` (was `6`, applied to ALL FOUR, including the two ps-only
configs, so ps-only and ps+raobs share the identical time-interpolation
scheme -- isolating "does adding raobs help" as the only difference
between a ps-only/ps+raobs pair, at the cost of also exposing ps itself
to the S2-tide risk in all four) and `observations.raobs.
use_source_error: True` in the two ps+raobs configs (psobs's was already
`True` in `-t1`):

| # | config | slurm script | job ID |
|---|---|---|---|
| ps-only, AIFS 1.0deg | `config_test_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-ps_only-t2.yml` | `run_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-ps_only-t2.sh` | 26868150 |
| ps+raobs, AIFS 1.0deg | `config_test_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-ps_raobs-t2.yml` | `run_aifs2-1.0deg-v1.0-longcyc-obsint-1hour-ps_raobs-t2.sh` | 26868151 |
| ps-only, ACE2 | `config_test_ace2-longcyc-obsint-1hour-ps_only-t2.yml` | `run_ace2-longcyc-obsint-1hour-ps_only-t2.sh` | 26868148 |
| ps+raobs, ACE2 | `config_test_ace2-longcyc-obsint-1hour-ps_raobs-t2.yml` | `run_ace2-longcyc-obsint-1hour-ps_raobs-t2.sh` | 26868149 |

All four loaded cleanly through `load_config` and pass
`tests/unit/test_config.py`; checkpoints, IC caches and the `lwaifs2` env
were confirmed present before submitting; no output-directory collision
with any existing `exp_name`. Submitted 2026-11-19, 5h walltime each (1
month / 300 6h cycles requested, same as `-t1`).

**Pick up here tomorrow**:
1. Check `squeue`/job `.out`/`.err` for all four job IDs above. If any
   hit its 5h walltime before finishing 300 cycles (as both `-t1` runs
   did, needing their own `-1rst` configs), make the analogous `-1rst`
   config/script pair: `restart: True`, `sdate`/`rpath` advanced to the
   last cycle with a complete `optimal_inputs.pkl`, same `exp_name`/
   output dir, and resubmit.
2. Decide on the two "still open" items above (ps background check, ps
   `error_std_hpa` value) and whether/how to update the PR #29 thread
   with this session's stronger `use_source_error` verification.
3. Once all four `-t2` runs complete, compare against their `-t1`
   counterparts (dt_obs: 6, raobs `use_source_error` unset) and the
   ps-only vs ps+raobs pairs against each other -- region-split z500 and
   raobs-fit diagnostics, same approach as the ACE2 Tropics z500
   investigation section above.

## `psobs.time_tolerance_hours` now defaults to `dt_obs/2`, matching raobs (2026-11-19)

While explaining the `-t1`/`-t2` ps-obs-count jump above in a follow-up
session, worked out precisely how `time_tolerance_hours` and `dt_obs`
interact in `cycle_directories`/the per-report `argmin`-over-all-slots
assignment (`psobs_parquet.py`/`raobs_parquet.py`): each raw report is
claimed by its single nearest obs slot, then kept only if that slot is
within `time_tolerance_hours` -- never duplicated across slots, never
independently re-fetched per slot.

**Key property, confirmed both algebraically and by direct simulation**:
when `time_tolerance_hours == dt_obs/2` (exactly half the slot spacing),
consecutive slots' acceptance windows tile the whole window edge to edge
with no gaps and no overlap, for ANY `dt_obs` -- so this formula captures
100% of the archive's reports regardless of how finely `dt_obs` slices the
window, just redistributing them across however many slots result. A
200k-sample simulation confirmed this exactly: `dt_obs:1`/`tol:0.5h` and
`dt_obs:6`/`tol:3.0h` (both `== dt_obs/2`) each keep 100% of synthetic
reports uniformly scattered across a 120h window; only a tolerance
*smaller* than `dt_obs/2` loses anything (`dt_obs:6`/`tol:0.5h`, the
previous psobs default, kept only ~16.5% in the same simulation --
consistent with the real archive's measured ~17% psobs coverage at the
mainline `dt_obs: 6`).

**This corrects an earlier wrong inference in this same investigation**:
raobs's own `time_tolerance_hours` default (`dt_obs/2`) was assumed to
need pinning to a fixed `3.0` whenever `dt_obs` is lowered, reasoning (by
analogy to the real `93-96%`-vs-`50-60%` station-count gap measured
earlier -- see "time_tolerance_hours is deliberately NOT the same value
as observations.psobs's" above) that `dt_obs: 1`'s default collapsing to
`0.5h` would similarly drop most sparse-cycle stations. That gap was
real, but it was measured entirely at a FIXED `dt_obs: 6` comparing a
mismatched fixed `0.5h` (psobs's old fixed convention, far narrower than
the `6h` spacing's own `3h` half-width) against the matched `dt_obs/2`
formula -- not a property of lowering `dt_obs` itself. At `dt_obs: 1`,
`dt_obs/2 = 0.5h` IS the matched formula (half of the new, narrower `1h`
spacing), so it tiles perfectly and loses nothing, as the simulation
above confirms directly.

**Changed**: `get_psobs` (`long_window_4dvar_utils.py`) now defaults
`psobs_time_tolerance_hours` to `dt_obs / 2.` instead of a fixed `0.5`
(previously independent of `dt_obs` entirely, carried over from the
legacy `psobs1_*.txt` generator's own `halfwindow = 1800s`). raobs's own
default was already `dt_obs/2` and needed no code change. The mini-ACE2
CI fixture is unaffected: `tests/make_fixtures.py`'s `_write_psobs_parquet`
places every synthetic report's `observation_time` exactly on its cycle
boundary (zero jitter), so widening the tolerance window changes nothing
there (`make check-ace2` reconfirmed passing).

**Consequence for every tracked production config**: none of them set
`observations.psobs.time_tolerance_hours` explicitly, and all run at the
mainline `dt_obs: 6` -- so this changes their *effective* psobs tolerance
from `0.5h` to `3.0h`, i.e. from ~17% to 100% of each 6h cycle's surface
reports captured. This is a real, intentional behavior change to every
backend's ps-obs assimilation density, not just the longcyc raobs
experiments above -- set `observations.psobs.time_tolerance_hours: 0.5`
explicitly on any config that needs to keep reproducing the old
legacy-`.txt`-matching density. Not yet re-run against any tracked
production config to see how this affects skill; the longcyc `-t2`
comparison above already exercises the new default directly (it never
set `psobs.time_tolerance_hours` explicitly either).

**Config template** (`configs/config.yml.template`): both the
`observations.psobs.time_tolerance_hours` and
`observations.raobs.time_tolerance_hours` comments rewritten to state the
tiling property directly and drop the now-incorrect "must pin explicitly
when lowering dt_obs" guidance; the `windows.*.dt_obs` comment's reference
to pairing with a pinned raobs `3.0` was softened the same way.

## raobs vertical interpolation: `exact`/`nearest`/`log-linear`, following JEDI's `VertInterp` (2026-11-19)

Read JEDI's actual `ufo`/`oops` source (`gh api`, not documentation/memory)
to compare this repo's raobs forward operator against JEDI's own design,
prompted by the still-open "ACE2 Tropics z500 degradation with raobs"
investigation above and a general question of how closely this repo's
horizontal/vertical/time interpolation choices track JEDI's.

**Findings, horizontal**: `oops::UnstructuredInterpolator`
(`src/oops/generic/UnstructuredInterpolator.h`) does Delaunay-triangulation
+ 3-point barycentric weights, exact for linear fields on ANY unstructured
mesh, computed once per source geometry by the model-interface layer
(`oops::GetValues`), shared by every obs type/operator. This repo's own
`grid_interp` (`get_grid_interpolator`) already defaults every backend to
an exact-for-linear-fields method too (`ReducedGaussianBilinearInterpolator`/
`BilinearGridInterpolator`/`RectilinearBilinearInterpolator`, one per grid
topology, since 2026-10-08) -- already well-aligned; the only gap is
needing a new per-topology class for a future non-row-structured mesh,
where JEDI's one triangulation algorithm would just work. No action taken
here.

**Findings, time**: `oops::GetValues` has `doLinearTimeInterpolation_`
(linear vs. nearest-neighbor between background snapshots) -- the same
role as this repo's `step_lo`/`alpha` linear time-bracketing, already in
place. Already aligned, no action needed.

**Findings, vertical -- the real gap**: `ufo::VertInterp`
(`src/ufo/operators/vertinterp/ufo_vertinterp_mod.F90`,
`src/ufo/utils/vert_interp.F90`) interpolates UNCONDITIONALLY from the
model's own vertical GeoVaL profile to each observation's own exact
reported coordinate (`log-linear` is JEDI's actual default for pressure --
confirmed directly in the sonde QC config,
`test/testinput/instrumentTests/Sonde/sonde_gfs_qc.yaml`: `obs operator:
VertInterp, vertical coordinate: air_pressure, observation vertical
coordinate: pressure, interpolation method: log-linear`). It never rejects
for being far from a model level -- out-of-range clamps to the nearest
edge level's value (`vert_interp_weights`'s `wi=1, wf=1.0`/`wi=nlev-1,
wf=0.0`). All QC (physical bounds, background/gross-error check) happens
AFTER interpolation, as separate filters -- exactly the role this repo's
own `tuv_bg_check`/`q_bg_check`/`bg_check` already plays. This repo's
actual design (before this section) discarded any level not within
`pressure_match_tolerance_hpa` (`0.05` hPa) of a model level -- i.e. the
proximity check WAS the rejection mechanism, something JEDI's design
doesn't have at all.

**Decided**: add `observations.raobs.vertical_interpolation` (`'exact'`
default / `'nearest'` / `'log-linear'`), closely tracking JEDI's own
`nearest-neighbor`/`log-linear` `VertInterp` methods, rather than inventing
a third scheme -- at the user's explicit request to keep a coarse-model-
safe nearest-level option alongside real interpolation. `'exact'` keeps
today's behavior byte-for-bit; `'nearest'`/`'log-linear'` both drop the
`pressure_match_tolerance_hpa` discard (every level kept, snapped to its
nearest model level unconditionally -- fixing the real failure mode for a
coarse model: a level getting ZERO observations because nothing happened
to land within the tight tolerance). `'log-linear'` additionally corrects
the comparison for the kept level not sitting exactly on its assigned
model level, via JEDI's own boundary rule (clamp-extrapolate, never
reject).

**Implementation** (`'exact'` default preserves every tracked config/test
byte-for-bit -- purely additive):
- `raobs_parquet.py`'s `_profile_records`/`load_raobs_parquet`: new
  `vertical_interpolation` parameter. `'nearest'`/`'log-linear'` drop the
  tolerance discard and add the kept level's own real pressure
  (`record['pressure']`/`pressure_<v>` output, same shape as `values_<v>`)
  for the forward operator to use. The existing per-`(variable,
  level_index)` duplicate-selection priority gained a third tie-break --
  distance to the assigned level -- since without the tight tolerance,
  genuinely different raw levels can now compete for one bucket (moot for
  `'exact'`, where the tolerance already made it so).
- `long_window_4dvar_utils.py`: new `_raobs_vertical_interp_weights`
  helper -- a direct port of `vert_interp_weights`'s bracket-and-clamp
  logic (ln(p), adjacent model level, edge-clamp) -- called once per
  window in `get_raobs` (same "compute once, not per epoch" pattern as
  `interp_idx`/`interp_wts`/`step_lo`/`alpha`), producing
  `vert_neighbor_idx_<base>`/`vert_alpha_<base>` tensors.
  `_compute_raobs_observation_diagnostics_at_time` gains one new step for
  `'log-linear'` only: gather the neighbor level and blend by `alpha`
  before forming the innovation (`alpha=0` is a no-op, reproducing
  `'nearest'`/`'exact'` exactly). `save_raobs_diagnostics` additively
  writes `<base>_real_pressure` when not `'exact'`.
- `_RAOBS_CONFIG_KEYS`/`load_config` validate the three values;
  `configs/config.yml.template` documents all three with the JEDI
  correspondence.

**Testing**: `tests/unit/test_raobs_parquet.py` -- a level outside the old
`0.05 hPa` tolerance is discarded under `'exact'` but kept under
`'nearest'`/`'log-linear'`; the distance tie-break when two raw levels
compete for one bucket; `'exact'` never populates the new `pressure_<v>`
field (regression guard). `tests/unit/test_solver_helpers.py` --
`_raobs_vertical_interp_weights` directly: exact match gives `alpha=0`,
a level exactly between two model levels gives `alpha=0.5` with the
correct neighbor, out-of-range clamps to self with `alpha=0` rather than
rejecting; the full forward-operator blend reproduces hand-computed
values; every pre-existing hand-built `raobs_traj` fixture (none of which
set `vertical_interpolation`) is confirmed to still take the unmodified
path (`raobs_traj.get('vertical_interpolation', 'exact')`, not direct
indexing -- needed because those fixtures predate this key).
`tests/unit/test_config.py` -- accepts the three values, rejects anything
else, confirms the resolved flat key and its `'exact'` default.

**Not yet done**: the mini-ACE2 CI fixture (`tests/ace2/test_ace2_mini.py`)
was not extended to exercise `'nearest'`/`'log-linear'` end to end (not
run in this environment either -- `fme` isn't importable here); the error-
level tables (`error_std`/`q_error_g_kg`) still evaluate at the nominal
model-level pressure rather than the kept observation's own real pressure
for `'nearest'`/`'log-linear'` -- a small, flagged secondary accuracy
improvement, not bundled into this change; and no tracked config has been
switched to `'nearest'`/`'log-linear'` yet to see how it affects the
still-open ACE2 Tropics z500 investigation above.

## raobs balloon-drift horizontal position: `use_drift_position`, following JEDI's `ProfileHorizontalDrift` (2026-11-19)

Read JEDI's own source again (`ufo/variabletransforms/Cal_ProfileHorizontalDrift.{h,cc}`,
`ufo/variabletransforms/Formulas.cc`'s `horizontalDrift`) to see how it
handles balloon drift's HORIZONTAL component -- the "also not yet done"
item flagged in the "Balloon drift time" section above.

**How JEDI does it**: as an upstream variable transform (`ProfileHorizontalDrift`),
run BEFORE any obs operator -- it overwrites `MetaData/latitude`/
`longitude`/`dateTime` in place, per LEVEL (computed from the sonde's own
measured wind via a Laroche-Sarrazin ballistic integration, fixed 5.16 m/s
ascent speed). This works with zero changes to `VertInterp`/`GetValues`
because JEDI's ObsSpace already treats each level as its own independent
row with its own location -- `GetValues` already interpolates per row,
uniformly for every obs type; there is no "one location shared across many
levels" concept to generalize in the first place. Confirmed the archive
already carries `drift_longitude`/`drift_latitude` directly (absolute
degrees, same convention as `latitude`/`longitude` -- checked a real
218-level sounding: drifts from its launch position 100.27E/5.30N to
99.94E/5.45N by 6 hPa), so unlike JEDI's general case this repo doesn't
need to derive drift from wind at all -- just read it.

**Why this repo's situation differs**: `get_raobs` computes horizontal
interpolation weights ONCE PER STATION (`grid_interp.weights(lon_np[j],
lat_np[j], ...)`), shared across every level of that station's profile --
the opposite of JEDI's per-row design. Adopting the correction meant
introducing per-level position where none existed, not just reading one
more column.

**Added `observations.raobs.use_drift_position`** (default `False`,
independent of both `use_drift_time` and `vertical_interpolation` -- any
combination is valid):
- `raobs_parquet.py`'s `_profile_records`/`load_raobs_parquet`: each kept
  level additionally records its own drift-corrected `(lat, lon)`
  (`record['level_position']`/`lat_<v>`/`lon_<v>` output, same shape as
  `pressure_<v>`), falling back to the station's own launch position
  wherever `drift_longitude`/`drift_latitude` is missing/non-finite.
- `long_window_4dvar_utils.py`: when enabled, `get_raobs` computes
  per-`(slot, level)` horizontal weights PER BASE (`interp_idx_<base>`/
  `interp_wts_<base>`, shape `(n_obs, n_levels, capacity, k)` -- each base
  needed its own, since each base maps raw levels to its own `levels_hpa[base]`
  independently) instead of the single shared `interp_idx`/`interp_wts`.
  New `_raobs_gather_per_level` helper applies them in
  `_compute_raobs_observation_diagnostics_at_time` (advanced indexing with
  two same-shaped index tensors -- avoids materializing a `(capacity,
  n_points)` intermediate, which would be disastrous for AIFS's 542080-point
  grid). When disabled, the forward operator takes the exact same
  `grid_interp.interp` path as before -- zero behavior change.
  `save_raobs_diagnostics` additively writes `<base>_drift_lat`/
  `<base>_drift_lon`.

**Testing**: `tests/unit/test_raobs_parquet.py` -- each level gets its own
drift position when enabled; falls back to launch position when the
archive columns are missing/non-finite; the feature is a no-op (no
`lat_<v>`/`lon_<v>` keys at all) when disabled. `tests/unit/
test_solver_helpers.py` -- `_raobs_gather_per_level` reproduces `grid_interp.
interp`'s shared-weight result when weights happen to be constant across
levels; picks genuinely different grid points per level when they differ;
the full forward operator correctly applies distinct per-level weights
end to end. `tests/unit/test_config.py` -- new key defaults unset (`False`
default applied by `get_raobs`), accepted/resolved when set.

**Verified against the real archive** (not just synthetic fixtures):
`use_drift_position=False` produces no `lat_t`/`lon_t` keys at all
(regression-safe); `True` recovers per-level positions that genuinely
vary across a real sounding (one station's latitude ranged 70.94-71.17
across its kept levels). `make check`/`make check-ace2` both still pass
with the new code present (default off) -- the mini-ACE2 CI fixture
itself isn't extended to exercise `use_drift_position: True` end to end,
since (like `use_source_error`) its synthetic archive has no
`drift_longitude`/`drift_latitude` columns at all.

## raobs t/u/v/q observation error: GSI's real table, not a UFO test fixture or Desroziers estimates (2026-11-19)

Followed up on the `use_source_error` discussion above with a direct
question: how does JEDI actually define raobs observation error, and is
that consistent with what this repo does? Answer, worked out by reading
the real `JCSDA/ufo` source (not documentation/memory) and a real
NOAA-EMC operational config, not by trusting any one reference in
isolation:

**UFO itself is inconsistent about what an "RH error" table means.**
`sonde_gfs_qc.yaml` (the reference PR #29's review comment described)
assigns q's error via `ObsFunction/ObsErrorModelStepwiseLinear` +
`scale_factor_var: ObsValue/specificHumidity` -- confirmed directly in
that function's own source (`obserr[iv][jobs] = error *
(*obvalues)[iv][jobs]`): a pressure-dependent fraction of the **observed**
q, no qsat, no temperature, no saturation-vapor-pressure formula anywhere.
But the exact same table of numbers (`errors: [0.2, 0.4, 0.8]` at
`xvals: [25000, 20000, 10]` Pa) also appears, identically labeled `# RH
error`, in a *different* UFO function's own unit test --
`ObsFunction/ObsErrorModelHumidity` (`function_obserrorhumidity.yaml`) --
which DOES do the real conversion: interpolate the RH-error table to the
ob's pressure, then multiply by a genuine saturation specific humidity
computed from the **live model background** (`GeoVaLs/air_temperature`/
`air_pressure`, via a real Magnus-Tetens-family formula,
`SatVaporPres_fromTemp`). So the same numbers get used two different ways
in UFO depending on which function happens to consume them; the sonde
reference config's own wiring isn't the authoritative pattern to copy.

**A real operational config settles which mechanism GSI/JEDI actually
use for sondes**: `NOAA-EMC/RDASApp`'s
`parm/jcb-rdas/observations/atmosphere/adpupa_specificHumidity_120.yaml.j2`
(ADPUPA, kx 120 -- exactly this repo's raobs) supports both via a
`use_error_inflation` toggle, defaulting to the fraction-of-observed-q
path (`ObsErrorModelStepwiseLinear`) paired with several *additional*
GSI-`errormod`-replicating inflation filters -- `ObsErrorModelHumidity`
is the non-default, opt-out alternative.

**The deciding evidence**: a direct comment from a GSI-source maintainer
on PR #29
([#6992427303](https://github.com/NOAA-PSL/lw4dvar-torch/pull/29#issuecomment-6992427303)),
posting GSI's actual `prepobs_errtable.global` table (T, RH/10, ps, uv
columns, 1000-5 hPa) and stating plainly: *"specific humidity error
computed using rh error * qsat (where qsat is saturation specific
humidity computed from background)."* That's authoritative in a way
neither UFO reference config is on its own -- it's GSI's own stated
mechanism, not inferred from a test fixture's wiring.

**Decided**: adopt that GSI table as the single source for t/u/v/q,
replacing both the previous UFO-YAML-sourced `RAOBS_TUV_ERROR_STD`
(checked against this table and found to disagree at several levels --
e.g. 900 hPa T: UFO's test fixture said `0.9`, GSI's real table says
`1.0`; 100 hPa T: `0.8` vs `1.0` -- confirming the UFO test config is not
a reliable stand-in for the real fix file) and the Desroziers-based
`RAOBS_Q_ERROR_G_KG`/the `q_error_fraction`/`q_error_floor_kg_kg`
fraction-of-observed-q fallback (removed entirely, not just
deprioritized -- GSI's real mechanism is RH*qsat, not a fraction of the
observation). The qsat conversion reuses the archive's own static
`background_temperature_c`/`background_pressure_mb` fields -- the same
simplification (live GSI background vs. this repo's static archived one)
already made and documented for `use_source_error`'s own q path above,
not a new one.

**Implementation**:
- `long_window_4dvar_utils.py`: `RAOBS_TUV_ERROR_STD` updated to GSI's
  real T/uv columns (25 levels, 1000-5 hPa, transcribed and verified
  pair-by-pair against the posted table). New `RAOBS_Q_ERROR_RH_PERCENT`
  (the RH/10 column, x10 for percent) replaces `RAOBS_Q_ERROR_G_KG`
  entirely. Deliberately kept as a SEPARATE constant, not folded into
  `RAOBS_TUV_ERROR_STD['q']`: different units (RH percent vs. absolute
  K/m-s -- a `['q']` entry in the TUV dict would look usable the same way
  `['t']`/`['u']`/`['v']` are and silently wouldn't be) and a different
  consumption path (t/u/v's table is interpolated once per MODEL level in
  `get_raobs`; q's needs each observation's own real pressure and static
  background, so it's interpolated per RAW ARCHIVE ROW in
  `raobs_parquet.py` instead).
- `raobs_parquet.py`: `background_temperature_c`/`background_pressure_mb`
  are now read whenever `'q' in variables`, not just under
  `use_source_error` -- the DEFAULT q error needs qsat too now. New
  `_interp_rh_percent_table` (ln(p) interpolation, duplicated from
  `long_window_4dvar_utils.py`'s `_interp_level_table` rather than
  imported, matching this module's existing no-solver-dependency
  convention) interpolates the RH-percent table to each kept level's own
  real pressure; `default_error = (rh_percent/100.) * qsat` -- NOT
  inflated by GSI's x1.2 rule, which stays scoped to the genuine
  archive-sourced value only (same principle as `is_source_error_<base>`
  for t/u/v). New `default_error_q` output (same shape as `values_q`),
  always populated whenever q is requested, independent of
  `use_source_error`.
- `get_raobs`/`_compute_raobs_observation_diagnostics_at_time`: q's
  `effective_error` simplifies to `raobs_traj['default_error_q'][oind]`,
  overridden per-element wherever `source_error_q` (the real archive
  value, `use_source_error: True`) is finite -- the override logic itself
  is unchanged, only what it falls back to. `q_error_fraction`/
  `q_error_floor_kg_kg`/`q_error_g_kg` removed from `_RAOBS_CONFIG_KEYS`
  and their validation; new `q_error_rh_percent` (`{hPa: percent}` table,
  merged over `RAOBS_Q_ERROR_RH_PERCENT` the same way `error_std`
  overrides `RAOBS_TUV_ERROR_STD`) added.
- `configs/config.yml.template`: both tables rewritten with the new
  source and numbers; the q section rewritten to explain the RH*qsat
  mechanism and why it isn't a fraction-of-observed-value scheme.

**Testing**: `tests/unit/test_raobs_parquet.py` -- default q error
matches a hand-computed RH-percent-table-interpolated-times-qsat value;
absent entirely when no table is given; coexists correctly with
`source_error_q` when both `use_source_error` and `q_error_rh_percent`
are set (independent fields, the override happens downstream).
`tests/unit/test_solver_helpers.py` -- the three hand-built `raobs_traj`
fixtures that used `error_q_fraction`/`error_q_floor` rewritten to set
`default_error_q` directly (same asserted numeric behavior, new field
name). `tests/unit/test_config.py` -- new `q_error_rh_percent` table
accepted/resolved and its own invalid-value rejection, mirroring
`error_std`'s existing tests. `make check` and `make check-ace2` both
pass (the mini-ACE2 fixture doesn't enable `specific_humidity` for raobs
at all, so is unaffected either way).

**Known, deliberately unaddressed consequence**: the four `-t1`/`-t2`-family
longcyc configs that explicitly set `q_error_g_kg`/`q_error_fraction`/
`q_error_floor_kg_kg` (gitignored, not tracked -- `config_test_*-longcyc-
*.yml`) now fail `_RAOBS_CONFIG_KEYS`'s unknown-key validation. Checked
what they actually contain before leaving them alone (at the user's
explicit request, "ignore the longcyc yamls for now"): the four ACE2
configs never actually exercised these keys at all (ACE2's raobs
variables exclude `specific_humidity` per the existing total-water
caveat); the four AIFS2 configs' values exactly matched the OLD defaults
(not customized), so removing the three lines would be a real behavior
change for q on any future rerun (old saved outputs are unaffected). Not
fixed -- these 3 lines in each of the 8 files are what would need
removing whenever this is picked back up.

## Raob station pressure: confirmed assimilated in both GSI and JEDI, not implemented here (2026-11-19)

Follow-up question from the GSI table above: jswhit's posted table has a
`ps` column (flat `0.70` hPa at every level) -- is that usable for this
repo's existing `observations.psobs.error_std_hpa`? **No** -- that table
is explicitly GSI's radiosonde (kx 120/220/221) error table, and its `ps`
column is the error for **the raob's own station pressure**, reported
alongside its profile. This repo's `psobs` reads a completely different
PREPBUFR class/kx entirely (ADPSFC/SFCSHP, 180/181/187) -- unrelated
observations, not interchangeable sources. (Separately, and still open
from an earlier session: `psobs.error_std_hpa` could be tightened using
kx 180/181/187's OWN real `prepobs_errtable.global` PS values, `1.3/1.1/
1.2` mb respectively -- a different, already-identified number, not this
one.)

**Checked whether GSI/JEDI actually assimilate the raob's own station
pressure at all** (this repo currently doesn't -- `RAOBS_VARIABLE_MODEL_
BASE` only maps `temperature`/`u_component_of_wind`/`v_component_of_wind`/
`specific_humidity`, no station-pressure option exists). Confirmed yes,
via a real, CI-exercised NOAA-EMC RDASApp config:
`parm/jcb-rdas/observations/atmosphere/adpupa_stationPressure_120.yaml.j2`
(part of the same RRFS/JEDI pipeline the q config above came from --
shows up in that repo's own `rrfs-fv3jedi-*-analysis.ref`/`-conv-upperair.
ref` regression-test outputs, so genuinely exercised, not dead config).
Its filter-step comments cite real GSI routines by name (`setupps.f90`,
`read_prepbufr.f90`), confirming this mirrors actual GSI behavior:
- **Forward operator**: `SfcCorrected` (`correction scheme to use: GSL`)
  -- a genuine terrain-height correction using `GeoVaLs/geopotential_
  height`/`geopotential_height_at_surface` and the observed temperature,
  the same conceptual role this repo's own `preduce`/`get_surface_
  pressure` elevation reduction already plays for `psobs`.
- **Error table**: a 33-point `ObsErrorModelStepwiseLinear` in Pa --
  `68.115` Pa (~0.68 hPa) at the surface, growing smoothly to `577.03` Pa
  (~5.77 hPa) by 50 hPa, then literally `1E11` (effectively infinite,
  i.e. switched off) above that -- station pressure is a surface-only
  quantity, and GSI's real table encodes that by disabling it entirely
  at upper levels rather than using one flat value everywhere the way
  jswhit's quick-reference table's single `0.70` number would suggest.
- Further `ObsErrorFactorSfcPressure` terrain-mismatch inflation (the
  same function the real `psobs`-equivalent config uses) plus the usual
  quality-mark/low-pressure x1.2 inflation and gross-error/bounds checks.

**Not implemented here, and not scoped yet**: this would be a genuinely
new raobs capability (a new variable, a new forward-operator path
deciding on an equivalent terrain-correction operator, a new error
table sourced from this RDASApp config rather than jswhit's single flat
number), not a tweak to something existing -- flagged as a candidate
future addition if/when picked up, deliberately not scoped further this
session.

## PR #29 Copilot review (review #5997548256): 9 correctness fixes (2026-11-20)

GitHub Copilot's automated review on PR #29 found 9 new issues, spanning
the vertical-interpolation/drift-position combination, the gross-check
clamp's granularity, two config dict-merge crashes, and four
`raobs_parquet.py` read-time bugs. All nine fixed, each with a new
regression test (`tests/unit/test_solver_helpers.py`,
`tests/unit/test_raobs_parquet.py`, `tests/unit/test_config.py`).

- **`log-linear` + `use_drift_position` mixed horizontal locations**
  (High): with both set, the vertical blend's neighbor-level value was
  gathered straight out of `model_obspace` (`torch.gather(model_obspace,
  0, neighbor_idx)`) -- but under `use_drift_position`, each row of
  `model_obspace` was already horizontally interpolated at ITS OWN
  level's drift position, so gathering the neighbor's row reused the
  NEIGHBOR's position instead of the current level's. Fixed by giving
  `_raobs_gather_per_level` an optional `field_level_idx` parameter
  (which row of the decoded field to sample) independent of which row of
  `idx`/`wts` (whose horizontal weights to apply) -- the neighbor blend
  now samples the neighbor's pressure level at the CURRENT level's own
  position (`long_window_4dvar_utils.py`).
- **A missing effective error could reach the loss as non-finite**
  (High): `used` only screened a non-finite/non-positive `effective_error`
  through the optional background check -- `q_bg_check: null` (or
  `tuv_bg_check: null`) left it unscreened, so a read-time-QC-passing
  observation with a NaN error (e.g. a missing archived background
  leaving both `default_error_q` and `source_error_q` NaN) could reach
  `_raobs_term` and make the total loss non-finite.
  `_compute_raobs_observation_diagnostics_at_time` now ANDs `used` with
  `isfinite(effective_error) & (effective_error > 0)` unconditionally,
  after either branch.
- **The GSI ermin/ermax clamp was gated per-VARIABLE, not per-CELL**
  (Medium): `is_source_error_<base>` was a single bool for the whole
  t/u/v variable (`'source_error_'+base in data`), so a cell that fell
  back to `error_std` (the archive had no value there) got clamped
  anyway, as if it were a genuine archive value. Now a per-(obs, level,
  station) bool tensor, set only where that cell's own archive value was
  actually used; the clamp applies `torch.where(is_source_valid[oind],
  clamped, qc_error)` instead of an all-or-nothing Python `if`.
- **`observations.raobs.error_std: null` / `q_error_rh_percent` merge
  bug** (Medium x2): `{**RAOBS_TUV_ERROR_STD, **exp.get('raobs_error_std',
  {})}` crashed on an explicit YAML `null` (`exp.get` still returns the
  stored `None`, not the default, when the key is present) -- fixed with
  `exp.get('raobs_error_std') or {}`. Separately, `q_error_rh_percent`'s
  per-PRESSURE-KNOT table was merged the same way
  (`{**RAOBS_Q_ERROR_RH_PERCENT, **user_table}`), which is wrong for a
  knot table specifically: a partial override like `{1000: 20, 500: 10}`
  left the DEFAULT table's own knots (almost certainly including 850 hPa
  itself) sitting between the user's two entries, so ln(p)-interpolating
  at 850 hPa returned the default's own value there instead of
  interpolating between the user's 1000/500 hPa entries. Fixed to
  `exp.get('raobs_q_error_rh_percent') or RAOBS_Q_ERROR_RH_PERCENT` --
  the configured table replaces the default wholesale, never merges with
  it knot-by-knot (`get_raobs`).
- **Cycle-partition discovery didn't account for `use_drift_time`**
  (Medium): `_profile_records` called `common.cycle_directories` with
  the UN-drifted `time_tolerance_hours`, so a report sitting right at
  (or just outside) the undrifted window's edge -- e.g. a 20:30Z launch
  with a first slot at 00Z and a 3h tolerance, where `_nearest_cycle`'s
  own tie-break at the exact 3h midpoint favors the query's own cycle --
  could have its home cycle directory never opened at all, even though
  its drift-corrected time (21:30Z, +1h drift) belongs inside the
  window. Fixed by widening discovery by a new `_RAOBS_MAX_DRIFT_HOURS`
  (3.0) margin, tolerating missing padding directories at the archive's
  own date-range boundary (`common.cycle_directories` gained an
  `allow_missing` parameter for exactly this, default `False` so every
  other caller -- `psobs_parquet.py` included -- is unaffected) while
  the original, unwidened request still raises as before if genuinely
  missing; the per-level acceptance check still uses the real,
  un-widened tolerance against the corrected time.
- **Duplicate selection ignored physical-value bounds / the humidity
  floor** (Medium x2): `_profile_records`'s duplicate-selection priority
  bucketed on quality mark and height eligibility only, not the
  physical-value bounds (`_RAOBS_PHYSICAL_BOUNDS`) applied later as QC --
  so an impossible value with a merely better quality mark could win a
  duplicate slot over a value that would have passed every check,
  permanently losing it once the winning (impossible) value failed QC
  downstream. Now folded into the same priority tuple's first bucket.
  Separately, `humidity_min_pressure_hpa` was checked against the
  matched MODEL level's nominal pressure (in `load_raobs_parquet`, after
  selection), not each candidate's own REPORTED pressure -- wrong in
  'nearest'/'log-linear' modes, where a kept level's real pressure can
  differ from its assigned model level (e.g. 290 hPa snapped to 300 hPa)
  enough to matter at the floor. Moved into `_profile_records`, checked
  against each raw row's own pressure BEFORE it ever becomes a
  duplicate-selection candidate; the old nominal-level check in
  `load_raobs_parquet` removed as dead/wrong code.
- **`source_qsat_q` could be finite with no corresponding
  `source_error_q`** (Medium): the two were populated independently --
  `source_qsat_q` from the archive's background T/p alone, `source_error_q`
  additionally needing a valid QOE (`humidity_error_tenths_percent`). A
  report with valid background but missing/invalid QOE got a finite
  qsat paired with a NaN source error, which the forward operator's
  ermin/ermax clamp (keyed off `source_qsat_q`'s presence) assumes never
  happens. `load_raobs_parquet` now only records `source_qsat_q` where
  the corresponding `source_error_q` cell is already finite.

## Simplifying raobs to JEDI-only: removed `use_source_error`, GSI's ermin/ermax clamp, and GSI's x1.2 inflation (2026-11-23)

Follow-up to the GSI-vs-JEDI comparison work above (the "raobs QC/thinning
vs. GSI's actual implementation", "GSI's ermin/ermax gross-check clamp",
and "GSI's x1.2 error inflation" sections), prompted by the user's explicit
goal of keeping this repo's raobs implementation JEDI-consistent rather
than tracking two reference systems (GSI and JEDI/UFO) with genuinely
different, sometimes incompatible mechanisms. Scoped first (not changed
blind), confirming against the real `JCSDA/ufo` source (not memory) that
the three removed pieces have no JEDI counterpart:

- **`observations.raobs.use_source_error`** (the archive's own embedded
  PREPBUFR TOE/WOE/QOE value) -- removed entirely. GSI itself reads this
  value then unconditionally overwrites it from `prepobs_errtable.global`
  before assimilating (confirmed directly in `read_prepbufr.F90`, see the
  "Correction" section above); it has no JEDI/UFO equivalent either --
  `ufo`'s own error filters (`ObsErrorModelHumidity`/
  `ObsErrorModelStepwiseLinear`) always derive error from a table plus
  the background, never from an embedded observation-file value.
- **GSI's ermin/ermax gross-check clamp** (`RAOBS_TUV_ERMIN`/`ERMAX`,
  `RAOBS_Q_ERMIN_PERCENT`/`ERMAX_PERCENT`) -- removed. This was only ever
  reachable when `use_source_error` was in effect, so it's dead code once
  that's gone; it was also never found anywhere in `ufo`'s real gross-
  check filter (`BackgroundCheck`), which bounds the *innovation* with an
  absolute limit instead of clamping the *error value* the way GSI's
  `setupq.f90` does -- a genuinely different mechanism, not a stand-in.
- **GSI's x1.2 quality-mark/pressure error inflation**
  (`_gsi_inflate_error` in `raobs_parquet.py`) -- removed. Also only ever
  reachable under `use_source_error`; no equivalent trigger (quality-mark-
  3/7, or above a fixed pressure) was found in `ufo`'s own sonde QC
  reference config (`sonde_gfs_qc.yaml`)'s active filter list.

**Kept unchanged, confirmed JEDI-consistent**: `vertical_interpolation`
(`'log-linear'` is a direct port of `ufo::VertInterp`'s own bracket/clamp-
to-edge logic, confirmed against `vert_interp.F90` directly), the
`error_std`/`q_error_rh_percent` tables and the RH-fraction-times-qsat-
from-background q error mechanism (matches `ufo`'s own
`ObsErrorModelHumidity`, confirmed by reading `ObsErrorModelHumidity.cc`
directly), `bg_check`'s per-variable `{threshold, absolute}` form (UFO's
own `BackgroundCheck` shape), and `QC_PHYSICAL_BOUNDS` (UFO's own sonde
config numbers).

**Left as-is, deliberately not touched this pass**: no thinning,
`quality_mark_max` (both already confirmed JEDI-consistent too --
`sonde_gfs_qc.yaml` has no thinning filter, and a commented-out `PreQC`
filter there uses the identical `maxvalue: 3` threshold), the virtual-to-
sensible-temperature conversion (`sensible_temperature`, confirmed to have
no equivalent inside `ufo` itself -- any such conversion in a real JEDI
pipeline happens upstream, in `ioda-converters`, not checked further here),
and the height-bound/non-monotonic-height/below-station QC (this repo's
own addition; `ufo`'s own `ConventionalProfileProcessing` filter covers
a related but much larger scope -- hydrostatic/superadiabatic/sign-change
checks -- and isn't invoked by the GSI-replicating reference config this
repo has anchored its comparisons on, so porting it was judged out of
scope for this pass). `use_drift_time`/`use_drift_position` are also left
as-is -- JEDI's own `ProfileHorizontalDrift` derives drift from wind via a
formula, while this repo reads the archive's own precomputed drift
columns directly; a different implementation of the same correction, not
a GSI-only mechanism, so nothing to remove.

**Fallout**: `configs/config.yml.template`'s `use_source_error`/ermin-
ermax-clamp comments removed; `tests/unit/test_config.py`,
`tests/unit/test_raobs_parquet.py`, `tests/unit/test_solver_helpers.py`
had their `use_source_error`/clamp/inflation-specific tests removed
(coverage for the surviving `error_std`-table-as-tensor path kept via a
new, narrower test); `tests/ace2/mini_ace2.py`'s explicit
`'use_source_error': False` (already redundant with the default) dropped,
with `tests/data/golden/mini_ace2.json`'s config echo updated to match.
`make check` (285 passed) and `make check-ace2` both pass.

**Not reopened**: the four already-run long-cycling experiments
(`-ps_raobs-t1` through `-t4`) are unaffected -- all four explicitly set
`use_source_error: False` already (confirmed before removing the key), so
none of the removed code paths were ever exercised for them.

## `observations.raobs.physical_bounds`: made configurable (2026-11-23)

`_RAOBS_PHYSICAL_BOUNDS` (`raobs_parquet.py`) was a hardcoded Python dict
with no config path at all -- the only stated reason was "fixed constants,
like the ermin/ermax clamp" (the "PR #29 review comment" section above),
an analogy that no longer has a partner now that the ermin/ermax clamp
itself was removed above. Every other GSI/JEDI-sourced default in this
schema (`error_std`, `q_error_rh_percent`, `bg_check`) is configurable;
this one was the odd one out, with no principled reason it needed to stay
fixed.

**Added `observations.raobs.physical_bounds`**: an optional `{base: [min,
max]}` override, defaulting to today's values (same as UFO's sonde
`Bounds Check` filters). `load_raobs_parquet`/`_profile_records`
(`raobs_parquet.py`) gained a `physical_bounds=None` parameter (None keeps
`_RAOBS_PHYSICAL_BOUNDS`, used at both duplicate-selection time and the
final QC flag); `get_raobs` (`long_window_4dvar_utils.py`) merges a
configured override **per-variable-key** with the default
(`{**_RAOBS_PHYSICAL_BOUNDS, **(exp.get('raobs_physical_bounds') or {})}`)
-- unlike `q_error_rh_percent`'s pressure-knot table (where a partial
merge was the actual bug PR #29's review caught), each `physical_bounds`
entry is a complete, self-contained `[min, max]` pair, so overriding one
variable and leaving the others at their default is the correct,
intended behavior here, not a hazard to guard against.

Validated at config-load time like `bg_check`: keys must be a subset of
`('t', 'u', 'v', 'q')`, each value a 2-element `[min, max]` with
`min < max`, both finite. `configs/config.yml.template` documents the
default explicitly. Tests: `tests/unit/test_raobs_parquet.py` (a widened
override rescues a value the default would reject; a narrowed one rejects
one the default would accept; confirms `load_raobs_parquet` itself does
no per-variable merging -- that's `get_raobs`'s job), `tests/unit/
test_config.py` (the per-variable merge, via a real `get_raobs` call; the
three new validation-rejection cases). `make check` (292 passed) and
`make check-ace2` both pass.

## PR #29's third Copilot review: 15 of 16 findings fixed (2026-11-23)

Diagnosed and fixed in order of severity (reasoning and exact code checked
against upstream references before any fix -- see the PR discussion for
the full diagnosis):

- **Virtual-temperature-only levels assimilated as sensible** (High,
  `common.sensible_temperature`): a level whose TOB event stack is 100%
  VIRTMP (no plain non-VIRTMP event at all) silently kept the virtual
  temperature as if it were sensible. Now returns NaN for that case
  (excluded downstream like any other missing value); the input-value
  fallback is kept only for a level with no TOB events in the archive at
  all.
- **Balloon drift time double-counted DHR** (High, `raobs_parquet.py`):
  `drift_time_hours` (PREPBUFR's HRDR) is cycle-relative, the SAME
  referent as the report's own DHR -- confirmed against both GSI's
  `read_prepbufr.F90` (HRDR replaces, not adds to, the header time) and
  the RDASApp `bufr2ioda_adpupa_prepbufr.py` converter (`hrdr*3600 +
  cycleTimeSinceEpoch`). The code was anchoring to `report_time` (==
  `cycle_time + DHR`) instead of `cycle_time`, double-counting DHR.
  Fixed by parsing the cycle time back out of the archive's own
  `<root>/<YYYY-MM-DD>/<HH>` partition directory (`common.
  cycle_time_from_directory`, new) rather than adding an archive column.
- **Gross-check mask recomputed against the live optimization state**
  (High, deferred): matches an existing, pre-PR-29 pattern already in
  the ps-obs path (`_compute_ps_observation_diagnostics_at_time`'s own
  `gross_check_failed`) -- fixing it properly needs a reusable "decode
  the background once" mechanism for both obs types, planned as its own
  follow-up task rather than a narrow raobs-only patch that would
  diverge from psobs's behavior.
- **Both drift corrections now default to True in code** (`get_raobs`),
  not just in the committed template, per explicit request -- matches
  the PR description's own stated default.
- **Empty `error_std`/`q_error_rh_percent` pressure tables rejected at
  config-load time** instead of reaching `np.interp` with no sample
  points.
- **t/u/v error tables evaluated at each observation's own real reported
  pressure in `nearest`/`log-linear` modes** (`get_raobs`), not the
  nominal model-level pressure -- matches what `q`'s error already did.
- **raobs cost now included in the per-slot loss output**
  (`compute_loss_4dvar` returns a third value, `Jraobs_np`/
  `all_loss_raobs`, kept separate from the existing `all_loss` so
  psobs-only golden values are unaffected).
- **Non-positive reported pressure rejected** before matching/
  interpolation (was reaching `np.log` in `log-linear` mode).
- **Profile identity now includes the report's own launch time**
  (`raobs_parquet.py`'s `profile_key = (station_key, report_time)`), so
  two different launches from the same physical station landing in the
  same slot no longer merge into one profile.
- **Duplicate humidity ranking now also requires a usable computed
  error** (finite, positive `default_error_q`) before ranking by quality
  mark, so an unusable-background duplicate can't win over a usable one.
- **Station records with no retained observation at all are dropped**
  (e.g. a q-only sounding with every level below
  `humidity_min_pressure_hpa`) instead of consuming a station slot with
  an all-False mask.
- **Non-monotonic-height QC now checked once on the COMPLETE profile**
  (across every `use_drift_time` slot it was split into), not per-slot
  -- an inversion straddling two slots was previously invisible to
  either slot's own check.
- Added: a VIRTMP-only-stack test, a duplicate-q-by-usable-error test, a
  two-launches-same-station test, a non-monotonic-height-across-slots
  test, a non-positive-pressure test, a cycle-time-vs-report-time
  anchoring regression test (`tests/unit/test_raobs_parquet.py`); a
  multi-dimensional `_interp_level_table` test and an independent
  finite-difference gradient check on the raobs cost term
  (`tests/unit/test_solver_helpers.py`, since `tests/math/`'s toy-model
  gates never pass `raobs_traj` at all); an assertion that the mini-ACE2
  raobs diagnostics record nonzero `t_used`/`u_used`/`v_used`
  (`tests/ace2/test_ace2_mini.py` -- not run in this session, no fme/
  checkpoint/network available; syntax-checked only). `tests/ace2/
  mini_ace2.py`'s raobs config now explicitly sets `use_drift_time:
  False, use_drift_position: False`, since that fixture's reports have
  no `drift_time_hours`/`drift_longitude`/`drift_latitude` columns at
  all and would otherwise hit the new True-by-default (PR #29's review,
  this round).
- One existing test (`test_use_drift_time_widens_partition_discovery_
  for_a_boundary_report`) was built around the OLD, double-counting
  formula's arithmetic and needed its scenario reworked; one
  (`test_humidity_pressure_floor_checked_against_real_pressure_not_
  nominal_level`) now asserts zero stations landed instead of one
  all-False station, per the dropped-empty-record fix above.
- Not fixed (acceptable per Copilot's own stated alternative): the
  `psobs.time_tolerance_hours` default change from the earlier
  `dt_obs/2` entry above is already extensively documented, including
  the exact override (`time_tolerance_hours: 0.5`) for anyone who needs
  the old legacy-matching density -- no code change needed.
- `make check`: 300 passed (up from 292), plus 12 **pre-existing,
  unrelated** failures (`test_tracked_configs_load`) -- `_RAOBS_CONFIG_KEYS`
  dropped `q_error_fraction`/`q_error_g_kg`/`q_error_floor_kg_kg` in the
  JEDI-only-error/QC simplification above, but 12 tracked
  `config_test_*.yml` files still set them; confirmed present before any
  of this round's changes (`git stash` + rerun). Not fixed in this round
  (out of scope for the Copilot review); flagged to the user instead.
  `make check-ace2` not run (no fme/checkpoint/network in this session).

## Finding #1, fixed: gross-check QC frozen at the background (2026-11-23)

The one deferred finding from the round above: both `_compute_ps_
observation_diagnostics_at_time` and `_compute_raobs_observation_
diagnostics_at_time` recomputed their background/gross check from the
LIVE, currently-optimizing state at every epoch, not the background --
letting the optimizer lower Jo by pushing an observation out of QC
instead of actually fitting it (Copilot's own worked example: two unit-
error wind obs at 0 and 12, 8-sigma threshold -- state 6 costs 72 fitting
both; state 12 drops the first observation and costs 0, a LOWER cost
from a worse fit). Confirmed this is not raobs-specific: psobs's own
`gross_check_failed` had the identical pattern already, pre-dating PR #29
entirely. Fixed for both, unconditionally (no new config flag) -- this
is a correctness fix matching traditional 4DVar/GSI/JEDI practice, and
this module's own saved-diagnostics file already claimed
`'quality_control_reference': 'background'`, previously true only of
what got saved after the fact, not what the optimizer itself used.

**Mechanism**: `compute_ps_observation_hx` (`@torch.no_grad()`) already
existed for exactly this -- `save_trajectory_diagnostics` was already
calling it with `step_start=0` on the uncorrected background, purely for
post-hoc reporting. `compute_optimal` now makes that SAME call once,
before the AdamW loop, and stores the result (`psobs_traj['background_hx']`/
`raobs_traj['background_hx']`, plus the derived boolean `bg_used`:
psobs's qc_flag bit 8 resolved once; raobs's full per-base `used`).
Both diagnostics functions gained `frozen_qc=False`: when `True`, the
background-check comparison is replaced by a lookup into `bg_used`
instead of being recomputed from the live innovation -- availability/
interpolation/orography (raobs: the archive mask/isfinite(error)) stay
live either way, since orography is state-independent anyway and
interpolation must keep catching a live state going non-finite.
`compute_loss_4dvar`'s `_obs_term`/`_raobs_term` pass `frozen_qc=(psobs_
traj.get('bg_used') is not None)` (automatically True via `compute_optimal`,
automatically False -- today's behavior, unchanged -- for a caller like
`tests/math/`'s toy-model gates that builds its own `psobs_traj` without
that step). `compute_ps_observation_hx`'s own calls (used for BOTH the
background bootstrap above and `save_trajectory_diagnostics`'s analysis-
side snapshot) never set it -- both must always score fresh at whatever
state they're handed, not reuse the frozen decision; freezing the
analysis-side snapshot too would have defeated its entire point ("what
does the final analysis score right now"). `save_trajectory_diagnostics`
now reuses `psobs_traj['background_hx']`/`raobs_traj['background_hx']`
instead of recomputing that same no-grad pass a second time (falls back
to recomputing if called without `compute_optimal` having run first --
not the production call pattern, but keeps the function usable
standalone).

**Golden values moved** (expected -- QC active in all 5 toy configs,
with 4 planted gross-error obs specifically meant to exercise the
background check): regenerated via `python tests/make_fixtures.py
golden`. `tests/data/golden/mini_ace2.json` **was** also regenerated in
this session, via `python tests/make_fixtures.py golden --ace2` run
against this same, fully-fixed solver code, using the `ace2` conda env
(`/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2` -- has fme/pyarrow/
pytest already; confirmed) and the mini-ACE2 test checkpoint already
cached at `~/.cache/lw4dvar` (no network needed). Its loss trajectory
moved as expected: last-epoch cost 2721 -> 1255 (was 1493 pre-fix,
i.e. -54% now vs. -45% before) -- `make check-ace2` passed against it
(1 passed, 1 unrelated jax skip). `tests/ace2/test_ace2_mini.py:35`'s
comment is updated below to cite these new numbers instead of the
pre-fix ones it was never updated to reflect.

**Tests**: `tests/unit/test_solver_helpers.py` -- Copilot's own worked
example reproduced directly (`frozen_qc=False` reproduces the pathology
byte-for-byte: cost 72 at state 6, 0 at state 12, obs dropped;
`frozen_qc=True` fixes it: cost 144 at state 12, obs kept).
`tests/golden/test_golden_toy.py` -- a new test confirms
`save_trajectory_diagnostics`'s saved `surface_pressure_used` matches
`compute_optimal`'s own cached `background_hx['used']` exactly (not just
"doesn't crash" -- genuinely the same values), with `save_xr_trajectory`
stubbed out since `ToyModel` has no `_levels_by_base` (a real backend's
attribute the toy model never needed before this test). `make check`:
302 passed (up from 300), same 12 pre-existing unrelated failures.

## PR #29's fourth Copilot review: 5 of 8 findings fixed (2026-11-23)

Submitted right after the previous round's push; 13 of the previous
round's 16 findings marked resolved. Diagnosed and fixed:

- **Missing drift columns crashed the whole levels read** (High,
  `raobs_parquet.py`): flipping `use_drift_time`/`use_drift_position` to
  default True (previous round) meant any archive without
  `drift_time_hours`/`drift_longitude`/`drift_latitude` -- not missing a
  VALUE, missing the COLUMN entirely, e.g. `tests/data/mini_ace2/raobs`
  -- failed to load outright, instead of falling back as documented.
  Fixed by checking each partition's own schema (`common.
  available_columns`, new) before requesting the optional drift columns,
  independently per `(cycle, prepbufr_class)` -- a required column
  missing still raises as before. Confirmed end to end: temporarily
  removed `tests/ace2/mini_ace2.py`'s explicit `use_drift_time: False`/
  `use_drift_position: False` (added last round specifically to dodge
  this crash) and reran `make check-ace2` -- no crash, identical
  computed values, the only diff was the now-absent config echo. Kept
  the explicit config anyway (clearer documented intent for this
  fixture), just softened its comment: no longer *required* to avoid a
  crash, kept for clarity.
- **Mini-ACE2 golden fixture "predates frozen-background QC"** (High):
  the data was actually already correct -- verified the committed
  `mini_ace2.json`'s loss trajectory had genuinely moved (1493 -> 1255
  last-epoch cost) consistent with the frozen-QC fix actually running.
  The bug was in the *documentation*: this doc's earlier revision left
  the golden-regeneration status claim unresolved after the fact (see
  git history for the exact wording at the time), and `tests/ace2/
  test_ace2_mini.py:35`'s comment still cited the pre-fix observed
  numbers (-46%, not the new -54%). Both corrected below/above -- this
  doc now consistently states the fixture was regenerated with the
  final solver, with no stale claim left anywhere in it.
- **Drift latitude not range-checked before overwriting launch
  coordinates** (Medium, `raobs_parquet.py`): the launch latitude gets
  `-90 <= lat <= 90`; the drift-position override only checked
  `isfinite`. A finite-but-garbage drift latitude would have reached
  horizontal interpolation directly. Added the same range check.
- **Non-monotonic-height QC still hid a losing bucket-mate's height**
  (Medium, `raobs_parquet.py`, second round on the same mechanism): the
  previous round's profile-wide height tracking still collapsed to one
  winner per *(variable, matched-model-level)* bucket -- inherited from
  the pre-existing per-slot duplicate-selection design, just applied
  profile-wide. In `nearest`/`log-linear` modes, two genuinely different
  real levels from different `use_drift_time` slots that happen to snap
  to the same coarse model bucket could still hide an inversion: only
  the bucket's assimilation WINNER's height ever reached the check.
  Re-keyed the tracking by the row's own REAL reported pressure instead
  (`profile_source_priority`/`profile_source_heights`, replacing
  `profile_priority`/`profile_heights`) -- two rows at the SAME real
  pressure are still one genuine duplicate (same convention this
  module's own duplicate-selection tests already used); two different
  real pressures now both stay visible regardless of which one wins its
  model-level bucket for assimilation. The monotonicity check itself now
  sorts by real pressure (descending) instead of by matched model level,
  so it still orders correctly when two rows share one bucket. Hit the
  exact class of test-construction bug the first round's own docstring
  warned about: three existing duplicate-selection tests gave competing
  rows the SAME default height, which used to be invisible (only one
  winner's height ever reached the old bucket-keyed check) but is now
  correctly visible and triggered false monotonicity failures -- fixed
  by giving `test_relaxed_modes_distance_tiebreak_prefers_closer_
  duplicate` physically distinct heights; the other two already had one
  row's height excluded before reaching the check (ineligible/wrong
  bucket) so needed no change once the real-pressure key stopped
  conflating them. Still NOT a check of the literal complete archive
  sounding -- a level whose drift-corrected time falls outside every
  slot's tolerance window is never read into any record at all, same as
  every other QC here; the fix's own comment says so now instead of
  overclaiming completeness.
- **VIRTMP event-selection test only had one candidate after filtering**
  (Medium, test coverage, carried over from two rounds ago):
  `test_sensible_temperature_used_instead_of_virtual` couldn't actually
  tell "first non-VIRTMP event by stack order" apart from "the only one
  left" -- strengthened with three non-VIRTMP candidates at distinct
  `event_index` values, given in shuffled input-list order, confirming
  the selection is by stack order and not list order or value magnitude.
- **Independent radiosonde cost/gradient coverage, broadened** (Medium,
  test coverage, carried over from two rounds ago): the previous round's
  gradient check was an isolated unit test of the diagnostics function
  alone. Added a toy t/q radiosonde trajectory to `tests/math/
  problem.py`'s `Problem` (`raobs=True`, sharing psobs's own obs-slot
  grid, so it gets a between-step slot for free; one masked-at-read-time
  cell per slot; `q` uses a tensor `default_error_q`, `t` a flat scalar
  error -- both real code paths), wired through the REAL
  `compute_loss_4dvar`, with new `gradient_error`/`taylor_order`/
  checkpointing-consistency gate tests (`tests/math/
  test_cost_gradient.py`) for both the background-check-off and -on
  paths, plus a sanity check that the added cost is actually nonzero.
  Needed `_resolve_loss_interp_specs` to see `raobs_enabled`/
  `raobs_variables` in `Problem.exp` before `interp_specs` is resolved --
  already handled correctly in the real code (folds raobs's variables
  into the sub-6h time-interpolation set), `Problem` just hadn't been
  exercising that path before.
- Not fixed (no code change needed): "document the widened psobs
  tolerance" -- `configs/config.yml.template` and this doc already
  explain the `dt_obs/2` default, why, and the `time_tolerance_hours:
  0.5` override; the only thing not mentioning it is the PR's own
  GitHub description text, which isn't a code change.
- `make check`: 312 passed (up from 302), same 12 pre-existing unrelated
  failures. `make check-ace2`: passed, confirmed on the `ace2` conda env
  (has fme/pyarrow/pytest already) with the cached mini-ACE2 checkpoint.

## PR #29's fifth Copilot review: the math gates' frozen-QC gap (2026-11-24)

3 of the previous round's 6 open findings resolved; one genuinely new
one -- the previous round's own toy-raobs gate coverage never actually
exercised the production QC path it was meant to validate.

**"Test gates with frozen background QC masks"** (Medium, `tests/math/
problem.py`): `Problem._build_raobs_traj` set `tuv_bg_check`/
`q_bg_check` (the LIVE-recompute QC branch) but never populated
`raobs_traj['bg_used']` (the FROZEN-mask branch `compute_optimal` itself
always uses in production, per two rounds ago's fix). Since
`compute_loss_4dvar` only sets `frozen_qc=True` when `bg_used` is
present, every gradient/Taylor/checkpointing gate added last round was
silently running the live branch -- the one real runs never take.
Fixed by calling the SAME `compute_ps_observation_hx(step_start=0)`
`compute_optimal` calls, on the toy problem's own uncorrected background
state, and attaching the resulting `used` mask as `bg_used` before
returning the trajectory -- now `frozen_qc` resolves to `True`
automatically for every `raobs=True` gate test, exactly matching
production. Confirmed: `gradient_error`/`taylor_order` unchanged in
magnitude (expected -- at a random `x0` close to the background, the
frozen and live verdicts agree; the whole point is that they can
diverge once the state actually moves, which these gates don't do).

Carried over, assessed as already resolved or out of scope (no new
information since the previous round):
- "Regenerate the ACE2 golden fixture with the final solver" -- the
  data and the live documentation were already correct; the flag looks
  triggered by this doc's OWN historical narration quoting the old
  stale phrasing verbatim for context. Reworded above to stop citing
  that exact phrase, with no data or code change.
- "Check complete source profiles before slot and bucket filtering" --
  only the bucket-collapsing half was fixed two rounds ago; the time-
  cutoff-excludes-out-of-window-levels half remains a deliberate,
  documented, pre-existing architectural limitation (same before any of
  this work), not reopened this round.
- "Document the widened psobs tolerance" / "Preserve the psobs default
  window" -- unchanged assessment: already documented in code, only the
  PR's own description text doesn't mention it.
- "Add independent radiosonde cost and gradient coverage" -- was the
  parent of this round's actual fix; resolves together with it.

`make check`: 312 passed, same 12 pre-existing unrelated failures. No
golden fixtures affected (this is test-only plumbing, not solver code).

## Closing the remaining half of "check complete source profiles" (2026-11-24)

The other half of a finding Copilot kept re-raising across three
reviews: `_profile_records`'s per-level loop discarded a level entirely
-- before any height-order bookkeeping ran -- whenever its drift-
corrected time missed every verification slot's tolerance window. A
sounding where that excluded level was inverted relative to a kept,
in-window sibling therefore had its inversion go completely undetected:
the in-window level would pass height-order QC even though the complete
sounding it came from was bad. This can only happen with
`use_drift_time: True` (now the code default) -- without drift every
level of one report shares the same `report_time` and so the same
accept/reject verdict, making the gap moot.

**Explicitly requested** (reproducibility with the earlier `-ps_raobs-
t*` runs' exact obs counts is not a concern here): replaced the early
`continue` on the time-tolerance check with a `time_accepted` flag.
`records.setdefault(...)` (and everything that writes into that per-
slot `record`) is now gated on `time_accepted`; the profile-wide
`profile_source_priority`/`profile_source_heights` tracking (added two
rounds ago) runs unconditionally, regardless of `time_accepted` -- so
an out-of-window level's height still reaches the monotonicity check
and can still reject an in-window sibling's whole variable, even though
the out-of-window level itself is never assimilated. This only
*tightens* QC (an out-of-window level can newly cause an in-window
level to be rejected; it can never newly cause one to be accepted),
and the gap it closes requires a fairly specific combination
(`use_drift_time: True`, a level drifting outside every slot's
tolerance, and a real inversion against a kept sibling) that none of
the tracked configs have been run against yet.

**Test**: `tests/unit/test_raobs_parquet.py::
test_non_monotonic_height_visible_when_sibling_drifts_outside_every_
slots_window` -- one profile, one level in-window (850 hPa) and one
drifted 5h outside the only slot's 3h tolerance (500 hPa), inverted
relative to each other. Confirmed this test fails against the pre-fix
code (`git stash` the source change, rerun) and passes with the fix.

`make check`: 313 passed (up from 312), same 12 pre-existing unrelated
failures.

## PR #29 review comment #7137120727 (jswhit): psobs tolerance and frozen QC reverted to opt-in, with independent defaults (2026-11-24)

A maintainer review of two changes from this PR that altered **every
ps-only run**, not just raobs runs -- both confirmed valid against the
actual code before changing anything:

1. **`psobs.time_tolerance_hours` default (0.5h -> `dt_obs/2`, added in
   "`psobs.time_tolerance_hours` now defaults to `dt_obs/2`, matching
   raobs" above):** `get_psobs`/`psobs_parquet.py` is the only ps-obs
   path now (no legacy txt-reader branch), and zero of the 36 tracked
   configs set this key explicitly -- so every mainline ps-only config
   (ACE2/AIFS/AIFS3, all `dt_obs: 6`) silently moved from 0.5h to 3.0h
   tolerance, ~6x more assimilated obs, with `jc_ps_weight`/
   `weight_decay` still tuned at the old count.
2. **The background/gross check frozen for psobs too, with no opt-out
   (added in "PR #29's third Copilot review: 15 of 16 findings fixed" /
   "Finding #1, fixed: gross-check QC frozen at the background" above):**
   `compute_optimal` always set `psobs_traj['bg_used']`, so every
   ps-only run moved from live to frozen QC, with no config flag to
   turn it off. Valid for a 6h window (GSI/JEDI practice); for the new
   5-day window the background error late in the window is large, so
   freezing at the background risks rejecting late-window obs the
   analysis would have fit -- untested against a real long-window
   production config either way.

**Fixed, both made independently configurable with independent
defaults rather than a single shared toggle:**

- `time_tolerance_hours`: psobs's own default reverted to a flat `0.5`
  (`long_window_4dvar_utils.py`'s `get_psobs`), matching the legacy
  txt-reader generator and `main` -- raobs keeps its own `dt_obs/2`
  default unchanged (raobs has no pre-existing mainline value to
  revert to). Set `observations.psobs.time_tolerance_hours` explicitly
  to opt into the wider, dt_obs/2 tolerance instead.
- `observations.psobs.frozen_qc` / `observations.raobs.frozen_qc`
  (`_PSOBS_CONFIG_KEYS`/`_RAOBS_CONFIG_KEYS`): `compute_optimal` now
  only populates `psobs_traj['bg_used']`/`raobs_traj['bg_used']` when
  the respective flag is set, instead of unconditionally -- everything
  downstream (`compute_loss_4dvar`'s own `psobs_frozen_qc`/
  `raobs_frozen_qc` derivation, `_compute_ps_observation_diagnostics_
  at_time`/`_compute_raobs_observation_diagnostics_at_time`'s
  `frozen_qc` branches) already keyed off whether `'bg_used'` was
  *present*, not a passed-down bool, so gating the population was the
  only change needed.
  - Both default to **False** (live QC) -- not merely for mutual
    consistency, but because GSI applies the SAME live, every-outer-
    loop gross check to raobs as to ps (`errormod` in
    `read_prepbufr.F90`, the same per-kx `convinfo` gross value this
    module's own `RAOBS_TUV_BG_CHECK`/`RAOBS_Q_BG_CHECK` defaults
    already copy, 8.0). So there is no more a GSI precedent for
    freezing raobs than for freezing ps; defaulting only psobs to live
    QC and raobs to frozen would have been the one-sided choice.
    (An initial version of this fix defaulted `raobs_frozen_qc` to
    `True` specifically to keep the directly-tested fix for Copilot's
    adversarial gaming example -- `tests/unit/test_solver_helpers.py::
    test_raobs_frozen_qc_prevents_gaming_the_gross_check`, two unit-
    error wind obs at 0 and 12 against one shared state, 8-sigma gross
    check: live QC lets the optimizer "fit" by moving the state to 12
    and having the gross check disqualify the first observation
    entirely, cost 72 -> 0; frozen QC holds both obs accounted for,
    cost 72 -> 144, removing the incentive -- reasoning raobs had no
    established practice of its own to lean on. Revisited once the
    GSI precedent above was confirmed to cover raobs too, not just ps.)
    This leaves the adversarial example live-by-default for both obs
    types, accepted as a known tradeoff; one caveat applying to either
    type, not just raobs: GSI's own outer loop runs only ~2-3 times, so
    its "live" QC is re-derived that often, while this module's AdamW
    inner loop runs ~100 epochs -- a finer-grained, more aggressive
    version of GSI's practice, with correspondingly more opportunity
    for the adversarial example to matter in practice.
  - Both independently overridable to `True` either direction,
    regardless of the other's setting.

**Golden fixtures:** the 5 toy fixtures (`toy_default/obden_jcps/
control_stride2/ps_operator/jc_z.json`) build their obs synthetically in
`tests/math/problem.py` -- they never call `load_psobs_parquet`, so
`time_tolerance_hours` never touched them, and they never pass
`raobs_traj`, so only psobs's frozen-QC flag applied. Confirmed (`git
show <pre-frozen-QC commit>^:<path>` vs. the freshly regenerated file,
byte for byte) that reverting `psobs_frozen_qc` to its new default
reproduces exactly what these 5 fixtures held before frozen QC was ever
introduced -- no "new" values, an exact revert to previously-validated
ones (confirmed a second time against `origin/main` directly: all 5
are byte-identical to `main`'s own current fixtures). `mini_ace2.json`
(`tests/ace2/`) does exercise both psobs and raobs through the real
config path and sets neither flag explicitly, so it picks up the new
psobs defaults (live QC, a narrower 0.5h tolerance instead of the mini
fixture's `dt_obs: 6` -> 3.0h) and was regenerated once for that --
its own sanity checks (monotonic cost decrease, nonzero `t`/`u`/`v`
raobs obs used) still pass; only the exact numeric values moved, as
expected from an intentional default change. Flipping `raobs_frozen_qc`
to `False` afterward needed no further regeneration of this fixture --
the synthetic raobs obs it carries never land close enough to the
gross-check threshold for live vs. frozen QC to change which ones pass.

**New test**:
`tests/golden/test_golden_toy.py::
test_frozen_qc_is_config_driven_and_defaults_to_live_for_both`
-- builds a toy psobs+raobs problem, resets `raobs_traj['bg_used']` to
the unpopulated state `compute_optimal` actually receives in production
(`tests/math/problem.py`'s own harness pre-populates it unconditionally
for its *own* gradient-gate tests, which would otherwise mask this),
and confirms: the default leaves both psobs and raobs live, and each
flag independently overrides to frozen. Mutation-tested by temporarily
flipping the psobs default and confirming the test catches it.

`make check`: 318 passed (up from 313), same 12 pre-existing unrelated
failures. `make check-ace2`: passed, against the regenerated
`mini_ace2.json`.

## `raobs_frozen_qc` default reconsidered: False, matching psobs (2026-11-24)

The asymmetric defaults above (`psobs_frozen_qc: False`, `raobs_frozen_qc:
True`) were revisited once the GSI precedent cited for `psobs_frozen_qc`
was checked against raobs too: GSI applies the SAME live, every-outer-
loop gross check to raobs as to ps (`errormod` in `read_prepbufr.F90`,
the same per-kx `convinfo` gross value `RAOBS_TUV_BG_CHECK`/
`RAOBS_Q_BG_CHECK` already copy, 8.0) -- so there is no more a GSI
precedent for freezing raobs than for freezing ps. `raobs_frozen_qc` now
also defaults to **False**. This leaves Copilot's adversarial gaming
example (`test_raobs_frozen_qc_prevents_gaming_the_gross_check`) live-
by-default for both obs types, accepted as a known tradeoff rather than
closed off by default; one caveat applying to either type, not just
raobs: GSI's own outer loop runs only ~2-3 times, so its "live" QC is
re-derived that often, while this module's AdamW inner loop runs ~100
epochs -- a finer-grained, more aggressive version of GSI's practice.

No further golden-fixture regeneration was needed: `mini_ace2.json`
(already regenerated above for the psobs default) needed no second
regeneration -- its synthetic raobs obs never land close enough to the
gross-check threshold for live vs. frozen QC to change which ones pass.
The 5 toy fixtures don't exercise raobs at all, so are unaffected either
way. Renamed the new test to `test_frozen_qc_is_config_driven_and_
defaults_to_live_for_both` and updated its assertions/docstring to match.

## PR #29 review comment #7137120727: smaller points addressed (2026-11-24)

The remaining smaller points from the same review:

- **`sensible_temperature`'s event_index-ordering assumption**: checked,
  not just commented on. The real archive
  (`prepbufr_to_parquet/data/prepbufr_parquet/<date>/<HH>/README.md`)
  states its events table is "in NCEPLIBS order" with "event_index = 0
  is current" -- i.e. lower event_index is more recent, confirming the
  function's existing assumption against the archive's own documented
  convention rather than leaving it unverified. Docstring updated to
  cite this directly.
- **Raob Jo not area/obs-density weighted**: documented in `_raobs_term`
  rather than given a speculative weight -- unlike `obs_density_box_deg`
  (added against a specific documented ps-obs clustering problem), there
  is no evidence yet that raobs's lack of an equivalent actually causes
  an imbalance in practice (raobs soundings are far sparser and more
  uniform in per-level/variable count than psobs reports); revisit with
  a real weight only if a validation run shows it's needed.
- **GSI vs. UFO background-check defaults**: kept GSI's flat 8.0
  (`RAOBS_TUV_BG_CHECK`/`RAOBS_Q_BG_CHECK`) as the default rather than
  switching to UFO's sonde-specific values (T 7.0x+9.0K, wind 6.0x+19m/s)
  -- documented why: this module already treats UFO's reference config
  as secondary to a real GSI-sourced value wherever they disagree (the
  error-table comment already noted UFO's own table doesn't match GSI's
  at every level), so adopting UFO's gross-check numbers specifically
  while rejecting its error table would mix the two references
  inconsistently. UFO's form is available via the existing `bg_check`
  config override; not the default.
- **Level snapping by raw hPa, not ln(p)** (`nearest`/`log-linear`
  modes): documented as a known Minor approximation (per the review's
  own framing) rather than changed -- only matters in the upper
  atmosphere where levels are log-spaced but numerically close in hPa;
  `exact` mode's tight tolerance makes the two metrics indistinguishable
  regardless.
- **Drift-position weight-loop timing at `dt_obs: 1`**: actually timed,
  not just estimated. Benchmarked the real loop (`backends/aifs/
  aifs_grid.py`'s `GridInterpolator`, a real N320-scale, 542080-point
  k-d tree -- AIFS's worst case among the raobs-supported backends) at
  a representative 5-day-window obs-slot count: ~0.4s total (all four
  t/u/v/q variables) at the mainline `dt_obs: 6`, ~2.1s at `dt_obs: 1`.
  Both are a one-time per-window setup cost (not per-epoch, not
  per-outer-loop), well under 1% of the ~300-400s/cycle observed in the
  real `-t5` SLURM runs. No fix needed -- confirms the reviewer's own
  "fine at dt_obs:6; worth timing at dt_obs:1" framing, with the timing
  now actually done.
- **Doc/comment size**: inspected rather than blindly shortened --
  sampling several of the longest in-source "PR #29's review" comments
  found them to be dense but genuinely load-bearing technical rationale
  (non-obvious invariants this repo's own CLAUDE.md says comments
  SHOULD explain), not narrative bloat; trimming them would hurt
  maintainability, not help it. What WAS bloat, per CLAUDE.md's own
  stated policy against task-referencing comments that "rot as the
  codebase evolves": the bare `(PR #29's review)` attribution tag
  itself, carrying zero information once PR #29 closes. Stripped it
  (and its minor variants: "second round", "now three rounds on this
  mechanism", etc.) from all ~40 sites across `raobs_parquet.py`/
  `long_window_4dvar_utils.py`/`prepbufr_parquet_common.py` where it
  carried no unique information beyond "this was reviewed", keeping
  every site's substantive explanation intact. Left the handful of
  genuine citations alone (named-maintainer attributions like
  "jswhit's ... review comment", specific comment-ID references like
  `#6992427303`, and UFO reference-file citations) -- those carry real
  provenance, not task-tracking noise. The doc itself
  (`docs/integrate_multiple_parquetobs.md`) was left growing by design
  -- per this repo's own CLAUDE.md, it's the dedicated historical record
  future sessions read before touching this area, not comment bloat in
  the functional code.

`make check`: 318 passed, same 12 pre-existing unrelated failures (no
logic changed in this round -- comments and docstrings only).
`make check-ace2`: passed.

## `bg_check` is now the ONLY raobs background-check mechanism; its default is UFO's table, not GSI's flat 8.0 (2026-11-24)

Reconsidering the GSI-vs-UFO `bg_check` default decision above: the
reasoning there leaned on an analogy to the error-table decision (GSI's
real `prepobs_errtable.global` preferred over UFO's own table, since
GSI's was independently confirmed authoritative and UFO's was shown to
diverge from it). That analogy doesn't hold for the gross check --
`error_std` was a choice between two competing *estimates of the same
physical quantity* (GSI's won on evidence); `bg_check` is a choice
between two *QC designs* (flat multiplier vs. per-variable-plus-
absolute-limit), where there's no evidence GSI's flat form is more
correct, and jswhit -- the same person who confirmed GSI's error table
as authoritative -- explicitly suggested UFO's richer form as the
default. `452a9db` (the JEDI-only simplification commit) also already
noted `bg_check`'s *mechanism* was built UFO-shaped from the start
("already JEDI-consistent") -- only the scalar fallback default was
GSI's.

**Changed**: `RAOBS_TUV_BG_CHECK`/`RAOBS_Q_BG_CHECK` (GSI's flat 8.0,
no absolute limit, with `bg_check` as an optional per-variable override
on top) replaced entirely by a single `RAOBS_BG_CHECK` default dict --
UFO's real sonde QC config values (T: `{threshold: 7.0, absolute: 9.0}`,
wind: `{threshold: 6.0, absolute: 19.0}`, q: `{threshold: 8.0}`, no
absolute -- UFO's own config doesn't give q one either).
`observations.raobs.bg_check` is now the ONLY way to configure this --
no separate scalar knob. A variable not listed in a configured
`bg_check` keeps the FULL default dict for that variable (whole-variable
replacement, not a per-field merge -- same convention as `error_std`/
`physical_bounds`); a variable explicitly set to `null` disables the
check entirely for it (replacing the old `tuv_bg_check: null`/
`q_bg_check: null` convention). `_raobs_gross_check_threshold` dropped
its `shared_key` parameter and scalar fallback -- it now just reads the
resolved `bg_check_threshold_<base>`/`bg_check_absolute_<base>` fields
`get_raobs` already wrote.

One caveat, documented rather than treated as blocking: UFO's relative
thresholds were tuned against UFO's own error table, not GSI's
`prepobs_errtable` this repo actually uses (already confirmed to diverge
from UFO's at some levels) -- so the effective relative cutoff shifts
slightly wherever the two tables disagree. The absolute floor (9K,
19 m/s) is error-table-independent and still catches gross outliers
regardless.

**Tracked configs**: `tuv_bg_check`/`q_bg_check` had zero explicit uses
anywhere; `q_bg_check: 4.0` (the independently-validated value from
before the q-error-table switch to GSI's RH-based mechanism) was set in
all 14 tracked `-t1`..`-t5` experiment configs -- migrated to
`bg_check: {q: {threshold: 4.0}}` in each (t/u/v keep the new UFO
defaults in all of them; none previously overrode t/u/v).

**Tests**: `tests/unit/test_config.py` -- the old scalar-validation test
cases now assert `tuv_bg_check`/`q_bg_check` are rejected as unknown
keys (they're no longer resolvable keys at all) rather than testing a
validation path that no longer exists; added a `bg_check: {..., u:
None}` case to `test_observations_raobs_bg_check_accepted` to cover the
new null-disables-a-variable path explicitly. `tests/unit/
test_solver_helpers.py` -- `_raobs_traj_case`'s hand-built dicts and the
`_raobs_gross_check_threshold` direct-call tests updated to the 1-arg
signature and `bg_check_threshold_<base>`-only fields;
`test_raobs_gross_check_threshold_resolves_override_then_shared`
(tested scalar-vs-override precedence, now moot) replaced with
`test_raobs_gross_check_threshold_reads_resolved_fields`. `tests/math/
problem.py`'s `_build_raobs_traj` writes `bg_check_threshold_t`/
`bg_check_threshold_q` directly instead of the removed `tuv_bg_check`/
`q_bg_check` scalar keys -- no change to its own public `raobs_bg_check`
parameter, callers in `tests/math/test_cost_gradient.py` are unaffected.

**Golden fixtures**: none needed regeneration. The 5 toy fixtures never
exercise raobs at all. `mini_ace2.json` does, and sets no `bg_check`
override, so it now resolves UFO's defaults instead of GSI's flat
8.0/8.0 -- but `make check-ace2` passed against the ALREADY-regenerated
fixture (from the `frozen_qc` default changes above) with no further
changes needed, confirming none of that fixture's synthetic obs sit
close enough to either threshold's boundary for the default change to
flip any accept/reject decision.

`make check`: 318 passed, same 12 pre-existing unrelated failures.
`make check-ace2`: passed.

## `sensible_temperature`'s event_index assumption: deeper verification (2026-11-24)

The earlier confirmation cited only the archive's generated README.
Asked to check more directly -- the converter source and the parquet
files themselves -- rather than trust a doc file. Three independent
checks, all converging:

1. **The converter's own code**
   (`/scratch4/BMC/gsienkf/Sergey.Frolov/obs/prepbufr_to_parquet/
   convert_prepbufr_to_parquet.py`, the real source for this archive):
   `read_events`/`append_event_group` get `event_index` directly from
   `ncepbufr._bufrlib.ufbevn(...)`'s own array index -- NCEPLIBS'
   actual Fortran `UFBEVN` routine, the same one GSI's own
   `read_prepbufr.f90` uses to read PREPBUFR event stacks operationally.
   This isn't a convention the converter invented; it's BUFRLIB's native
   ordering, passed through unchanged. The converter also writes an
   explicit `is_current = (event_index == 0)` column and
   `event_order` dataset metadata (`event_index=0 is the current/
   top-of-stack event`) -- the README is generated FROM this same
   converter's `write_readme`, not a separately-maintained doc that
   could have drifted from the code.
2. **Cross-table empirical check against real data** (2015-01-01 00Z,
   ADPUPA, this account's own archive copy): the `levels` table stores
   each level's "current" value in its own column (`temperature_c`),
   written by a code path entirely separate from the event loop. Pulled
   both tables and checked directly: across 28,288 real levels with a
   TOB event stack, `temperature_c` matched `event_index == 0`'s value
   in ALL 28,288 cases -- a dataset-wide consistency check that assumes
   nothing from any documentation.
3. **A real example, checked for physical sense**: report_id 0,
   level_index 0 -- event_index 0: value 21.5 C, program_code 8
   (VIRTMP); event_index 1: value 19.6 C, program_code 1. Virtual
   temperature (Tv) must exceed sensible temperature (T) for any
   positive specific humidity -- 21.5 > 19.6 confirms event_index 0
   really is the VIRTMP-corrected value and event_index 1 the
   pre-VIRTMP sensible one, exactly matching what `sensible_temperature`
   assumes and selects.

Docstring in `prepbufr_parquet_common.py` updated to cite all three
instead of just the README. No code change -- the function's existing
behavior was already correct; this closes out the "please confirm"
ask with real evidence instead of a documentation citation one level
removed from the actual mechanism.

## `sensible_temperature`: a real bug found while digging deeper, and fixed (2026-11-24)

Pushing further on the same verification (checking whether VIRTMP is
*always* at event_index 0, not just confirming event_index 0 means
"current" in general) turned up a genuine, previously-undetected bug --
not the one Copilot originally guessed at (reversed archive ordering,
disproven above), but a real one found by testing the assumption harder
than the original review comment did.

**The bug**: `sensible_temperature` picked the lowest-`event_index` event
whose own `program_code != VIRTMP_PROGRAM_CODE` (8). That's not the same
as "the lowest event_index above VIRTMP's own position" -- and in 239 of
15,498 real VIRTMP-bearing `(report_id, level_index)` groups checked
(2015-01-01 00Z, ADPUPA), it isn't: a later event (`program_code 10`,
not VIRTMP) sits ABOVE VIRTMP in the stack (lower `event_index`),
carrying VIRTMP's own value forward unchanged -- confirmed exhaustively,
its value matches the VIRTMP event's value in all 239 cases. The old
filter picked that event up as "sensible" since it isn't itself tagged
VIRTMP, when it's still virtual temperature. `event_index` is confirmed
contiguous from 0 with no gaps (checked across 252,452 real groups, 5
PREPBUFR classes), so the true sensible value always sits at exactly
`virtmp_event_index + 1` -- confirmed empirically, not merely assumed.

**Fixed**: anchor on VIRTMP's own `event_index` (found per group, lowest
one if more than one VIRTMP event is ever found for the same group --
not observed in the real archive, confirmed exactly one per group there,
but handled defensively rather than assumed) and exclude everything at
or above that position (`event_index <= virtmp_event_index`), regardless
of its own `program_code`, instead of excluding only the literal VIRTMP-
tagged event. A group with no VIRTMP event at all excludes nothing
(sentinel position -1), unchanged from before.

**Test**: `tests/unit/test_raobs_parquet.py::
test_sensible_temperature_skips_event_above_virtmp_too`, built from the
real report_id=621/level_index=14 example (event_index 0: value 14.7,
program_code 10; event_index 1: VIRTMP, value 14.7; event_index 2: the
real sensible value, 14.6). Mutation-tested: reverted to the old
`program_code != VIRTMP_PROGRAM_CODE` filter and confirmed this new test
fails (returns 287.85 K, the virtual temperature, instead of 287.75 K),
then restored the fix. The pre-existing
`test_sensible_temperature_used_instead_of_virtual` had VIRTMP at the
HIGHEST event_index in its fixture (an arrangement that cannot occur in
the real archive, since VIRTMP is derived FROM the sensible value and so
must be more recent / lower-indexed than it) -- corrected to place
VIRTMP at the lowest index instead, preserving the test's original
intent (stack order, not input-list order or value magnitude, decides
the selection) with a now-realistic fixture.

`make check`: 319 passed (up from 318), same 12 pre-existing unrelated
failures. `make check-ace2`: passed, no fixture regeneration needed --
the mini-ACE2 synthetic fixture's VIRTMP arrangement doesn't trigger
this edge case.

Also fixed, separately: the `'nearest'` vertical-interpolation mode's
comment and `configs/config.yml.template`'s matching note both claimed
"raw hPa distance, not ln(p)" was a "Minor" deviation from JEDI's own
convention. Read directly from a real local clone of `JCSDA/ufo`
(`src/ufo/operators/vertinterp/ufo_vertinterp_mod.F90`,
`src/ufo/utils/vert_interp.F90`): JEDI's own `nearest-neighbor` method
also uses raw, un-logged distance -- `log()` is applied only on the
`log-linear` branch (`tmp`/`tmp2 = log(...)`), not generally. So this
repo's `'nearest'` mode isn't a deviation from JEDI at all; it's an
exact match. Comments corrected in both files to cite the real source
instead of a false divergence claim.

## PR #29 Copilot review (#6047104139): 6 of 7 findings fixed, all verified (2026-11-24)

All 6 "New" findings confirmed valid and fixed (the 7th, "Add
independent radiosonde cost and gradient coverage", was carried over
from 2026-11-17 and subsumed by the live/frozen-QC fix below). Every
fix mutation-tested where practical: temporarily reintroduce the bug,
confirm the new/changed test actually fails, then restore.

- **`'exact'`-mode-discarded levels still skipped profile-wide height
  tracking** (Medium, `raobs_parquet.py`): the same class of gap already
  fixed twice before for `time_accepted`, found again for a different
  gate. A level failing `pressure_match_tolerance_hpa` (the default
  mode) `continue`d before ever reaching the profile-wide
  `profile_source_priority`/`profile_source_heights` bookkeeping, so it
  could never reject an in-tolerance sibling for a height inversion.
  Fixed: replaced the early `continue` with an `exact_ok` flag that
  gates only the assimilation-bucket-update block
  (`if time_accepted and exact_ok:`), leaving the profile-wide tracking
  unconditional on it, same as `time_accepted`. New test
  `test_non_monotonic_height_visible_when_sibling_fails_exact_mode_tolerance`,
  built from Copilot's own example (850 hPa/1500 m kept, an unmatched
  830 hPa/1000 m sibling -- a real inversion since lower pressure must
  mean higher altitude). Mutation-confirmed.
- **Toy math-gates only ever exercised frozen QC for raobs** (Medium,
  `tests/math/problem.py`/`test_cost_gradient.py`): `_build_raobs_traj`
  unconditionally installed `bg_used`, so every raobs finite-
  difference/Taylor/checkpointing gate tested the opt-in path and zero
  gates tested the actual default (live QC, after this PR's own earlier
  default flip). Fixed: new `Problem.__init__` parameter
  `install_raobs_bg_used` (default `True`, preserving this module's own
  already-validated behavior for existing variants) controls whether
  `_build_raobs_traj` installs the cached mask at all. New VARIANTS
  entry `'raobs_bg_check_live'` (`install_raobs_bg_used=False`);
  `test_raobs_taylor_remainder_is_second_order`/
  `test_raobs_checkpointing_does_not_change_the_gradient` parametrized
  over both masks, the latter also given a real `bg_check=3.0` threshold
  it didn't have before (otherwise its own parametrization would have
  been a no-op, same reasoning as the plain `'raobs'` VARIANTS entry
  having no threshold at all). Named `install_raobs_bg_used`, not
  `raobs_frozen_qc`, specifically to avoid colliding with the real
  `exp['raobs_frozen_qc']` config key an existing test
  (`test_frozen_qc_is_config_driven_and_defaults_to_live_for_both`)
  needs to pass through `Problem`'s `**exp` catch-all -- the first
  attempt used the colliding name and broke that test (`Problem.__init__`
  consumed it as its own new parameter instead of letting it flow to
  `exp`); caught immediately by `make check`, renamed before verifying
  further. Updated the stale comments at `problem.py`'s former line 192
  and `test_cost_gradient.py`'s former line 34, which both claimed
  frozen QC was what `compute_optimal`/production "actually uses" --
  true before this PR's own frozen-QC default flip, false after.
- **`test_save_trajectory_diagnostics_reuses_background_pass` didn't
  actually prove reuse** (Medium, `tests/golden/test_golden_toy.py`):
  `ToyModel` is deterministic, so even a silent regression to
  recomputing the background pass instead of reusing the cache would
  give the identical saved mask, passing the test's old output-only
  comparison. Wrapped `compute_ps_observation_hx` with a call-counting
  spy; the real expected count turned out to be 2, not 1 (`compute_optimal`'s
  own background pass, plus `save_trajectory_diagnostics`'s own always-
  fresh ANALYSIS-state call, a second, genuinely different, non-reused
  call that's supposed to happen) -- confirmed by reading
  `save_trajectory_diagnostics`'s actual code, not assumed. Mutation-
  tested by forcing the background branch to recompute (expected 3,
  got 3, test correctly failed); restored.
- **`level()` test helper's default temperature was physically
  impossible** (Medium, `tests/unit/test_raobs_parquet.py`):
  `temperature=300.` written straight to `temperature_c` is 573.15 K,
  outside the 195-327 K physical-bounds check -- any test using the
  helper without an explicit override had its `'t'` value already
  rejected regardless of what QC mechanism the test actually intended
  to exercise. Changed the default to `15.` (288.15 K, physically
  valid); all 48 existing tests in the file still passed unchanged
  (confirming none of them were *actually* relying on the broken
  default, just at risk of it) -- spot-verified
  `test_below_station_elevation_qc_rejected` by mutation-testing its
  real QC gate directly (there are two below-station checks in this
  file -- one feeding duplicate-selection priority, one the final QC
  flag; the first mutation attempt disabled the wrong one and the test
  still passed, which itself confirmed the two are independent;
  disabling the real gate made it fail as expected). Removed two
  now-unnecessary `temperature=-10.` overrides (with stale comments
  explaining why they used to be needed) that predated this fix.
- **The `q_error_rh_percent` config template example was invalid
  YAML** (Low, `configs/config.yml.template`): written as comma-
  separated `key: value` pairs with no enclosing `{...}`, unlike the
  `error_std`/`bg_check` examples right above it. Confirmed with a
  direct `yaml.safe_load` round-trip: the old text raises
  `YAMLError: mapping values are not allowed here`; wrapped in `{...}`,
  it parses to the intended 25-entry dict.
- **Missing archive cycles (including interior gaps) silently
  skipped, undetectably** (Medium, `prepbufr_parquet_common.py`):
  `cycle_directories(..., allow_missing=True)` skipped any missing
  directory with no distinction between the archive's own date-range
  edge (expected) and a genuine interior gap (not). Added a `logger`
  parameter; a skipped directory now gets an INFO line (not WARNING --
  the expected case and the one worth flagging look identical from
  here, so a human still has to judge which one occurred) naming the
  path. Threaded through `_profile_records`/`load_raobs_parquet`,
  which already received a real `logger` from `get_raobs` in
  production -- previously just never passed further down.

`make check`: 325 passed (up from 319), same 12 pre-existing unrelated
failures. `make check-ace2`: passed, no fixture regeneration needed.
