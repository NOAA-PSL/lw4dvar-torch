# Parquet observations: surface pressure (ps) and radiosondes (raobs)

Covers two observation types read from the same normalized PREPBUFR parquet
archive -- surface pressure and radiosonde (temperature/wind/humidity)
profiles. The config schema and the `exp.observations` loading mechanism
(see "Config schema" below) are the same ones a future parquet-backed
observation type (aircraft, AMV, GPSRO, ...) would extend.

Read this when working on: the ps-obs parquet reader (`src/psobs_parquet.py`),
the radiosonde (raobs) parquet reader (`src/raobs_parquet.py`), their shared
`src/prepbufr_parquet_common.py`, `get_psobs`/`get_raobs`, the
`exp.observations.psobs.*`/`exp.observations.raobs.*` config schema (or
adding a new observation type to `exp.observations`), the mini-ACE2 CI
fixture's synthetic obs, or the `pyarrow` dependency this introduces.

This is a **current-state summary**, not a changelog -- it describes what's
implemented today, not how it got there. For the detailed round-by-round
history (every review response, every GSI-matching mechanism tried and
later superseded by a JEDI-consistent one, every investigation that
confirmed or disproved a specific concern), see
`docs/integrate_multiple_parquetobs_history.md`. If the two ever disagree,
this doc and the actual code win.

## Architecture

Both readers pull from the same normalized, cycle-partitioned PREPBUFR
parquet archive: `<path>/<YYYY-MM-DD>/<HH>/{reports,levels,events}/<CLASS>/
part-00000.parquet`, Hive-partitioned by PREPBUFR class (`ADPSFC`/`SFCSHP`
for psobs; `ADPUPA` for raobs). `reports` is one row per BUFR
subset/report; `levels` holds each level's current (top-of-stack) values;
`events` holds the complete PREPBUFR correction-event stack per
(report, level) -- used only for recovering sensible temperature where
VIRTMP was applied (see "QC check" below). Only `src/psobs_parquet.py`,
`src/raobs_parquet.py`, and `src/prepbufr_parquet_common.py` import
`pyarrow` (lazily), so nothing else in the solver needs it installed.

`src/prepbufr_parquet_common.py` holds what both readers share: cycle-
directory resolution, parquet-partition reads, station-key/quality-mark
helpers, and `sensible_temperature` (used by raobs only). `get_psobs`/
`get_raobs` (`src/long_window_4dvar_utils.py`) call the readers and build
each observation type's own trajectory dict (`psobs_traj`/`raobs_traj`)
for `compute_loss_4dvar`/`compute_optimal`.

Raobs is only validated for `model_backend` in `aifs`, `aurora`, `ace2`,
`ngcm`, `neuralgcm`, `graphcast` (`RAOBS_SUPPORTED_BACKENDS`) -- the
`'t'`/`'u'`/`'v'`/`'q'` base-name forward operator has not been checked
for `fcn3`/`sfno`.

## Config schema: `exp.observations`, extensible to future obs types

Observation settings are grouped under `exp.observations.<type>` (`psobs`
and `raobs` so far), each with its own `path`, rather than flat,
per-type-prefixed keys directly on `exp` -- so a future observation type
can be added without renaming anything already here.
`_PSOBS_CONFIG_KEYS`/`_RAOBS_CONFIG_KEYS` (`{nested_key: flat_exp_key}`
tables in `src/long_window_4dvar_utils.py`) and `_resolve_observations_config`
(called from `load_config`) do the translation -- `observations.psobs.path`
becomes `exp['obspath']`, `observations.raobs.variables` becomes
`exp['raobs_variables']`, and so on, raising `ValueError` on any
unrecognized key. **Everything downstream of `load_config` -- `get_psobs`/
`get_raobs` included -- reads only the flat keys**; this translation is
the *only* place the nested schema exists. Adding a new type means adding
its own `_<TYPE>_CONFIG_KEYS` table and a corresponding block in
`_resolve_observations_config` (copy the `raobs` block's pattern).

`observations.psobs.path` is required (the parquet archive root).
`observations.raobs.path` defaults to psobs's own path (same archive,
`ADPUPA` class) if omitted. An absent/omitted `raobs` block, or
`enabled: False`, assimilates surface pressure only.

All defaults below are current as of this writing -- check
`configs/config.yml.template` (the authoritative, always-current
reference) if this doc and the template ever disagree.

## Observation error

- **psobs**: a flat `error_std_hpa` (default `1.0` hPa) applied to every
  observation, or `use_source_error: True` to use each report's own
  archived `pressure_error_mb` value instead.
- **raobs t/u/v**: `error_std` -- GSI's real `prepobs_errtable.global`
  table by default (one `{hPa: value}` table per variable, ln(p)-
  interpolated between entries, held constant beyond the table's range;
  u and v share one table, GSI's wind error being one vector quantity).
  Overridable per variable with a flat scalar or a custom table.
- **raobs q**: `q_error_rh_percent` -- a `{hPa: percent}` RH-error table
  (GSI's real table by default), converted to kg/kg via saturation
  specific humidity (qsat) computed from the archive's own static
  `background_temperature_c`/`background_pressure_mb` -- the same
  RH-error-table-times-qsat-from-background mechanism as UFO's
  `ObsErrorModelHumidity` (not a flat per-level guess, not a fraction of
  the observed value).

## QC check

Three layers, applied in this order:

1. **Read-time QC** (`raobs_parquet.py`, always applied, not configurable
   away except via the specific knobs below): PREPBUFR `quality_mark_max`
   (psobs default `4`, raobs default `3`); for raobs, a physical height
   range (`height_min_m`/`height_max_m`), a below-station-elevation
   tolerance (`below_station_tolerance_m`), and a non-monotonic-height-
   with-pressure check computed once per COMPLETE source profile (every
   raw level the archive has for that report, regardless of which slot or
   model-level bucket it's assigned to) -- an inversion anywhere in the
   real profile rejects the whole variable for that profile. A per-
   variable physical sanity bound (`physical_bounds`, UFO's real sonde
   `Bounds Check` numbers by default: T 195-327 K, q 1e-8-0.035 kg/kg,
   |u|,|v| <= 130 m/s) is always applied, independent of the background/
   model state.
2. **Background (gross) check** (`bg_check` for raobs, evaluated at the
   current/background state, per variable): reject `|O-B| > threshold *
   qc_error`, OR (if given) `|O-B| > an absolute limit` regardless of
   `qc_error` -- UFO's real per-variable sonde QC config values by
   default (T: `{threshold: 7.0, absolute: 9.0}`, wind: `{threshold: 6.0,
   absolute: 19.0}`, q: `{threshold: 8.0}`, no absolute). This is the
   ONLY background-check mechanism for raobs -- no separate flat-scalar
   fallback. A variable not listed in a configured override keeps the
   full default dict for that variable; a variable set to `null`
   disables its check entirely. psobs has its own pre-existing,
   independent `bg_check`/`zthresh`/`zconst` orography-aware check
   (unrelated to raobs's mechanism, not covered by this doc).
3. **Frozen vs. live QC** (`frozen_qc`, independently configurable for
   `observations.psobs`/`observations.raobs`, both default **False** --
   live QC): if `True`, the background/gross check above is evaluated
   ONCE against the uncorrected background before optimization and held
   fixed for the whole minimization; if `False` (default), it's
   recomputed from the live, still-optimizing state every epoch, matching
   GSI's own practice of re-evaluating QC every outer loop. Live QC has a
   known, accepted tradeoff: the optimizer can lower Jo by pushing an
   observation's live O-A far enough to fail the gross check (dropping it
   from the loss) instead of fitting it -- see
   `tests/unit/test_solver_helpers.py::
   test_raobs_frozen_qc_prevents_gaming_the_gross_check` for a concrete
   adversarial example. Set `frozen_qc: True` to close that off for a
   given run.

**VIRTMP handling** (raobs temperature only): the archive's current TOB
is virtual temperature wherever moisture was reported. `sensible_temperature`
(`prepbufr_parquet_common.py`) recovers sensible temperature from the
PREPBUFR event stack by anchoring on VIRTMP's own `event_index` and
excluding every event at or above that position (not just the literal
VIRTMP-tagged one -- a later, value-preserving event can sit above VIRTMP
in the stack). A level whose only TOB event is VIRTMP has no recoverable
sensible value and is excluded like any other missing observation. GSI's
own Tv-operator path (assimilating virtual temperature directly where
it's available) is not implemented -- levels with no recoverable sensible
value are dropped instead of being assimilated as Tv.

## Balloon drift: time and position

- `use_drift_time` (default **True**): assign each LEVEL to its own
  nearest obs slot using the archive's own balloon-drift-corrected
  `drift_time_hours` (near 0 at the surface, growing smoothly with
  altitude for a deep sounding), instead of every level of a profile
  sharing the report's single launch time. Falls back to the launch time
  wherever `drift_time_hours` is missing for a level.
- `use_drift_position` (default **True**), independent of `use_drift_time`:
  compare each level against the model at that level's own balloon-
  drift-corrected horizontal position (`drift_longitude`/`drift_latitude`),
  instead of every level sharing the station's single launch position.
  Falls back to the launch position wherever missing.
- Both are read-time-graceful: an archive partition with neither drift
  column at all (predating when the archive started recording them)
  falls back to launch time/position for every level, rather than raising.

## Vertical interpolation

`vertical_interpolation` controls how a raobs level is matched to the
model's own pressure levels:

- `'exact'` (default): mandatory-level matching -- a level is kept only
  if within `pressure_match_tolerance_hpa` (default `0.05` hPa) of a
  model level; everything else discarded. No vertical interpolation.
- `'nearest'`: every level kept, snapped to its nearest model level by
  raw hPa distance regardless of how far (no discard) -- so a coarse
  model still gets an observation at every one of its levels. Still no
  vertical correction (the comparison pretends the ob sits exactly on
  that level). Confirmed consistent with JEDI's own `nearest-neighbor`
  `VertInterp` method (read directly from the real `ufo` source): JEDI's
  nearest-neighbor also uses raw, un-logged distance -- ln(p) is a
  `log-linear`-specific step in JEDI, not a general convention this mode
  falls short of.
- `'log-linear'`: same relaxed matching as `'nearest'`, PLUS a vertical
  correction -- the model is log-linearly interpolated between the
  assigned level and its adjacent neighbor to the observation's own real
  pressure, including JEDI's own boundary rule (clamp-extrapolate to the
  edge level, never reject for being out of the model's vertical range).
  A direct port of JEDI's `vert_interp_weights` bracket-and-clamp logic.

## Raobs Jo weighting

Raobs's cost term (`_raobs_term`) is a straight sum of `(innovation /
effective_error) ** 2` over every used (variable, level, station) cell --
**no area or obs-density weighting**, unlike psobs's optional
`obs_density_box_deg` (which rescales ps Jo so its weights sum to the
number of used ps obs, countering a documented ps-obs clustering
problem). This means the ps/raobs Jo balance depends on how many levels
and variables each sounding happens to contribute, not just on how well
either fits. Deliberately left this way rather than given a speculative
weight: there's no evidence yet that raobs's lack of an equivalent
mechanism causes a real imbalance in practice (raobs soundings are far
sparser and more uniform in per-level/variable count than psobs reports).
Revisit with a real weight only if a validation run shows it's needed.

## Other settings

- `variables` (raobs, default all four -- `temperature`,
  `u_component_of_wind`, `v_component_of_wind`, `specific_humidity`):
  which variables to assimilate. **`model_backend: ace2` should drop
  `specific_humidity`** -- ACE2's `'q'` is specific *total water*
  (vapor + cloud condensate), not pure water-vapor specific humidity, a
  confirmed non-negligible bias with no way to subtract out.
- `prepbufr_classes`: which PREPBUFR classes to read (psobs default
  `['ADPSFC', 'SFCSHP']`; raobs default `['ADPUPA']`).
  `report_types` (psobs only, default `[180, 181, 187]`): PREPBUFR
  report-type codes kept, `null` to disable the filter.
- `time_tolerance_hours`: max `|report time - slot time|` accepted.
  psobs defaults to a flat `0.5` h (matching the legacy `.txt` reader,
  independent of `dt_obs`); raobs defaults to `dt_obs/2` (tiles the
  window edge to edge with no gaps for any `dt_obs`).
- `humidity_min_pressure_hpa` (raobs, default `300.`): mask `'q'` at any
  matched level with pressure below this (radiosonde humidity sensors
  are unreliable in the upper troposphere); `null` disables the floor.
- `station_capacity` (raobs): fixed per-slot padding size; by default
  sized to the largest slot actually read, set explicitly only if a
  stable shape across configs/runs is needed.

## Testing

- `tests/unit/test_psobs_parquet.py` / `tests/unit/test_raobs_parquet.py`:
  each reader against tiny, hand-crafted parquet archives (not the
  mini-ACE2 fixture), one behavior per test.
- `tests/unit/test_config.py`: `_resolve_observations_config`'s
  validation and flat-key resolution.
- `tests/ace2/test_ace2_mini.py` (`make check-ace2`): the real solver
  end to end against a synthetic parquet archive (`tests/make_fixtures.py`),
  both psobs and raobs (temperature/u/v) enabled.
