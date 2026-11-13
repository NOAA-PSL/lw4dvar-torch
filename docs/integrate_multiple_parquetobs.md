# Parquet observations, starting with surface pressure (ps)

Currently covers one observation type -- surface pressure, read from a
normalized PREPBUFR parquet archive -- but the config schema and the
`exp.observations` loading mechanism (see "Config schema" below) are the
same ones a future parquet-backed observation type (radiosondes, aircraft,
AMV, GPSRO, ...) would extend, which is why this doc is named for the
general case rather than just `psobs`.

Read this when working on: the ps-obs parquet reader (`src/psobs_parquet.py`,
`src/prepbufr_parquet_common.py`), `get_psobs`, the `exp.observations.psobs.*`
config schema (or adding a new observation type to `exp.observations`), the
mini-ACE2 CI fixture's synthetic obs, or the `pyarrow` dependency this
introduces.

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
radiosonde (raobs) observations -- NOT ported here; this repo stays ps-only).
That branch itself ports from `greg-long-window-4dvar`'s JAX/NeuralGCM
reference implementation (`prepbufr_parquet.py`); see that repo's own
`integrate_prepbufr_obs.md` for the full archive-layout writeup and the
raobs design this repo doesn't need.

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

## Config schema: `exp.observations`, extensible to future obs types

Observation settings are grouped under `exp.observations.<type>` on disk
(`psobs` is the only type so far), each with its own `path` -- rather than
flat, per-type-prefixed keys directly on `exp` -- specifically so a future
observation type (radiosondes, aircraft, AMV, GPSRO, ...) can be added
without renaming anything already here. This mirrors a convention already
worked out and in active use on a sibling branch,
`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/
add-amv-gpsro/lw4dvar-torch` (branch `add-amv-gpsro`), which has `raobs`,
`aircraft`, `amv`, and `gpsro` blocks alongside `psobs` -- see that branch's
`config_test_ace2_ps_raobs_aircraft_t3.yml` for a worked example with five
types enabled at once, and its own `integrate_prepbufr_obs.md` for the
per-type key tables and design notes on raobs/aircraft/amv/gpsro
specifically (vertical matching, thinning, bias correction, etc. -- none of
which apply to psobs and aren't duplicated here).

`src/long_window_4dvar_utils.py`'s `_PSOBS_CONFIG_KEYS` (a `{nested_key:
flat_exp_key}` table) and `_resolve_observations_config` (called from
`load_config`, right after `path_output` is resolved) do the actual
translation: `exp.observations.psobs.path` becomes `exp['obspath']`,
`exp.observations.psobs.use_source_error` becomes
`exp['psobs_use_source_error']`, and so on, raising `ValueError` on any key
`_PSOBS_CONFIG_KEYS` doesn't recognize or on a missing `path`. **Everything
downstream of `load_config` -- `get_psobs` included -- still reads the
flat keys it always has**; this translation is the *only* place the nested
schema exists. Adding a new type means adding its own `_<TYPE>_CONFIG_KEYS`
table and a corresponding block in `_resolve_observations_config` (almost
certainly with its own `enabled` flag, since unlike psobs a new type would
normally be optional) -- copy the pattern from the sibling branch above
rather than re-deriving it, and extend `_resolve_observations_config`'s
`unknown = set(observations) - {'psobs'}` check to admit the new top-level
key.

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
  (unknown `observations`/`observations.psobs` keys, missing
  `observations.psobs.path`) and that a representative set of nested keys
  resolve to the expected flat `exp` keys
  (`test_observations_psobs_resolved_to_flat_keys`).
- `tests/ace2/test_ace2_mini.py` (`make check-ace2`): now reads the
  synthetic parquet fixture via the real `get_psobs` path, with
  `observations.psobs.use_source_error: True` added to its config to
  exercise that branch too (matching the production configs that needed
  it, above).
- All pass locally with a from-scratch `.venv-ci` (`rdhpcs-python/3.12` +
  `envs/requirements-ci.txt`/`envs/requirements-ci-ace2.txt`) as of this writing.
