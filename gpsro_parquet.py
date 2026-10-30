"""GPS radio-occultation (GPSRO) bending-angle observation reader for the
Hive-partitioned `gpsro_parquet` archive (`<root>/year=YYYY/month=MM/
day=DD/gdas.YYYYMMDD.tCCz.gpsro.parquet`, one flat candidate-table file per
cycle -- see integrate_gpsro.md and the archive's own
`contract/gpsro_parquet_schema.md`).

Ported from the `long-window-4dvar` (JAX/NeuralGCM) reference repo's
`gpsro_parquet.py` (`_canonical_profiles_from_file`/`select_gpsro_profiles`/
`gpsro_error_std_rad`), trimmed of its provenance/contract-validation
machinery (schema/counts/clean-git-revision checks -- a converter-side
testing concern already covered by that converter's own test suite, not a
runtime reader concern) and its JAX fixed-shape capacity-manifest/caching
machinery (not needed in PyTorch -- see raobs_parquet.py's module
docstring for the same point made about the sibling radiosonde reader).
The canonical-profile selection, read-time QC, vertical thinning, and
error-model logic are kept faithful to the reference.

**2026-10-20: greedy 1 km vertical thinning removed, then restored the
same day.** GSI's own `read_gps.f90`/`setupbend.f90` confirmed GSI does
not thin GPSRO levels at all -- but GSI's own forecast models carry
60-150+ vertical levels, so a dense, untinned profile still has real
degrees of freedom to fit against. This repo's model backends (AIFS-
single-2.0: 14 levels, top 10 hPa) do not -- a profile's simulated
bending angle is a smooth function of only ~14 column values regardless
of how many observed levels are fit against it, so removing thinning
inflated GPSRO's effective weight in the loss by roughly (levels-per-
profile / 14), confirmed directly in a real run: GPSRO's `Jo` grew ~6-7x
and became the single largest observation term, dwarfing raobs/amv in
most verification slots. Restored to the original `impact_height_spacing_m`
default (1000 m) -- see integrate_gpsro.md "GSI comparison" for the full
reasoning and the log evidence.

Only this module imports `pyarrow`, lazily, so nothing else in the solver
needs it installed.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def _pyarrow():
    try:
        import pyarrow.compute as compute
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError(
            'Reading observations from the GPSRO parquet archive requires '
            "pyarrow (pip install pyarrow into this run's conda env)."
        ) from exc
    return compute, parquet


# Read-time QC bits (recorded per level, before any model-dependent check --
# model-column validity and the background check are evaluated later, at
# loss-evaluation time, and use a plain boolean mask instead -- see
# get_gpsro/integrate_gpsro.md for why this repo keeps that part simpler
# than the reference's frozen, quantile-scaled scheme).
GPSRO_QC_INVALID_VALUE = np.int16(1)
GPSRO_QC_INVALID_GEOMETRY = np.int16(2)
GPSRO_QC_IMPACT_HEIGHT_RANGE = np.int16(4)
GPSRO_QC_DUPLICATE_IMPACT = np.int16(8)
GPSRO_QC_THINNING = np.int16(16)
GPSRO_QC_SOURCE_FLAG = np.int16(32)
GPSRO_QC_CONFIDENCE = np.int16(64)
GPSRO_QC_FLAG_MASKS = np.asarray([
    GPSRO_QC_INVALID_VALUE, GPSRO_QC_INVALID_GEOMETRY, GPSRO_QC_IMPACT_HEIGHT_RANGE,
    GPSRO_QC_DUPLICATE_IMPACT, GPSRO_QC_THINNING, GPSRO_QC_SOURCE_FLAG, GPSRO_QC_CONFIDENCE,
], dtype=np.int16)
GPSRO_QC_FLAG_MEANINGS = (
    'invalid_value invalid_geometry impact_height_range duplicate_impact '
    'thinning source_flag confidence'
)

GPSRO_CANONICAL_COLUMNS = (
    'profile_id', 'level_index', 'source_message_index', 'source_subset_index',
    'observation_time', 'satellite_id', 'transmitter_id', 'instrument_id',
    'generating_center_id', 'latitude_deg_north', 'longitude_deg_east',
    'impact_parameter_m', 'bending_angle_rad', 'bending_angle_error_rad',
    'mean_frequency_hz', 'vertical_significance', 'first_order_statistics',
    'percent_confidence', 'profile_percent_confidence', 'quality_flags_ro',
    'earth_local_radius_curvature_m', 'geoid_undulation_m',
    'is_canonical_neutral_bending_angle',
)


def _utc_datetime(value):
    if isinstance(value, np.datetime64):
        nanoseconds = value.astype('datetime64[ns]').astype(np.int64)
        return datetime.fromtimestamp(nanoseconds / 1.e9, tz=timezone.utc).replace(tzinfo=None)
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.strptime(str(value), '%Y-%m-%dT%H')
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def gpsro_files_for_times(directory, verification_times, tolerance_hours):
    """Return cycle files on Hive `year=/month=/day=` days relevant to the
    requested verification times (a one-day halo handles UTC date
    boundaries without parsing filenames)."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError('GPSRO directory not found: '+str(root))
    times = [_utc_datetime(value) for value in verification_times]
    if not times:
        raise ValueError('GPSRO verification_times must not be empty.')
    halo = timedelta(hours=float(tolerance_hours) + 24.)
    first_day = (min(times) - halo).date()
    last_day = (max(times) + halo).date()
    paths = []
    day = first_day
    while day <= last_day:
        parent = (
            root / ('year='+str(day.year).zfill(4))
            / ('month='+str(day.month).zfill(2)) / ('day='+str(day.day).zfill(2))
        )
        if parent.is_dir():
            paths.extend(sorted(parent.glob('*.parquet')))
        day += timedelta(days=1)
    if not paths:
        raise FileNotFoundError(
            'No GPSRO cycle files found for the requested UTC dates under '+str(root)
        )
    return sorted(set(path.resolve() for path in paths))


def _nullable_numpy(column, *, fill_value=np.nan, dtype=np.float64):
    values = column.to_numpy(zero_copy_only=False)
    if column.null_count:
        mask = np.asarray(column.is_null())
        values = np.asarray(values).copy()
        values[mask] = fill_value
    return np.asarray(values, dtype=dtype)


def _nullable_integer(column, *, dtype=np.int32, fill_value=-1):
    values = np.asarray(column.to_pylist(), dtype=object)
    return np.fromiter(
        (fill_value if value is None else int(value) for value in values),
        dtype=dtype, count=len(values),
    )


def _profile_boundaries(profile_ids):
    if not profile_ids:
        return np.asarray([0], dtype=np.int64)
    transitions = np.flatnonzero(
        np.asarray(profile_ids[1:], dtype=object) != np.asarray(profile_ids[:-1], dtype=object)
    ) + 1
    return np.concatenate((
        np.asarray([0], dtype=np.int64), transitions.astype(np.int64),
        np.asarray([len(profile_ids)], dtype=np.int64),
    ))


def _canonical_profiles_from_file(
    path, *, impact_height_range_m, impact_height_spacing_m, maximum_bending_angle_rad,
    quality_flags_disqualifying_mask, minimum_percent_confidence,
):
    """Read, QC, and vertically thin one cycle file's canonical profiles."""
    compute, parquet = _pyarrow()
    table = parquet.read_table(path, columns=list(GPSRO_CANONICAL_COLUMNS))
    table = table.filter(compute.equal(table['is_canonical_neutral_bending_angle'], True))
    if not table.num_rows:
        return [], {'canonical_rows': 0, 'retained_rows': 0}

    profile_ids = table['profile_id'].to_pylist()
    boundaries = _profile_boundaries(profile_ids)

    columns = {
        'latitude_deg_north': _nullable_numpy(table['latitude_deg_north']),
        'longitude_deg_east': _nullable_numpy(table['longitude_deg_east']),
        'impact_parameter_m': _nullable_numpy(table['impact_parameter_m']),
        'bending_angle_rad': _nullable_numpy(table['bending_angle_rad']),
        'bending_angle_error_rad': _nullable_numpy(table['bending_angle_error_rad']),
        'percent_confidence': _nullable_integer(table['percent_confidence'], dtype=np.int16),
        'quality_flags_ro': _nullable_integer(table['quality_flags_ro'], dtype=np.int64),
        'earth_local_radius_curvature_m': _nullable_numpy(table['earth_local_radius_curvature_m']),
        'geoid_undulation_m': _nullable_numpy(table['geoid_undulation_m']),
        'satellite_id': _nullable_integer(table['satellite_id']),
    }
    observation_times = table['observation_time'].to_numpy(zero_copy_only=False).astype('datetime64[ns]')

    profiles = []
    retained_rows = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        profile_id = profile_ids[start]
        indices = np.arange(start, stop)
        qc_flag = np.zeros(indices.size, dtype=np.int16)

        bending_angle = columns['bending_angle_rad'][indices]
        invalid_value = (
            ~np.isfinite(bending_angle) | (bending_angle <= 0.)
            | (bending_angle > maximum_bending_angle_rad)
        )
        qc_flag[invalid_value] |= GPSRO_QC_INVALID_VALUE

        latitude = columns['latitude_deg_north'][indices]
        longitude = columns['longitude_deg_east'][indices]
        impact_parameter = columns['impact_parameter_m'][indices]
        curvature = columns['earth_local_radius_curvature_m'][indices]
        undulation = columns['geoid_undulation_m'][indices]
        invalid_geometry = (
            ~np.isfinite(latitude) | (latitude < -90.) | (latitude > 90.)
            | ~np.isfinite(longitude) | ~np.isfinite(impact_parameter) | (impact_parameter <= 0.)
            | ~np.isfinite(curvature) | (curvature < 5.e6) | (curvature > 8.e6)
            | ~np.isfinite(undulation) | (np.abs(undulation) > 2000.)
        )
        qc_flag[invalid_geometry] |= GPSRO_QC_INVALID_GEOMETRY

        impact_height = impact_parameter - curvature - undulation
        outside_height_range = (
            ~np.isfinite(impact_height)
            | (impact_height < impact_height_range_m[0]) | (impact_height > impact_height_range_m[1])
        )
        qc_flag[outside_height_range] |= GPSRO_QC_IMPACT_HEIGHT_RANGE

        if quality_flags_disqualifying_mask:
            source_flag_raw = columns['quality_flags_ro'][indices]
            source_flag = (source_flag_raw >= 0) & (source_flag_raw & quality_flags_disqualifying_mask).astype(bool)
            qc_flag[source_flag] |= GPSRO_QC_SOURCE_FLAG

        if minimum_percent_confidence is not None:
            confidence = columns['percent_confidence'][indices]
            confidence_failure = (confidence >= 0) & (confidence < minimum_percent_confidence)
            qc_flag[confidence_failure] |= GPSRO_QC_CONFIDENCE

        order = np.argsort(impact_parameter, kind='stable')
        indices = indices[order]
        qc_flag = qc_flag[order]
        impact_height = impact_height[order]
        impact_parameter = impact_parameter[order]
        if indices.size > 1:
            repeated = impact_parameter[1:] == impact_parameter[:-1]
            duplicate = np.zeros(indices.size, dtype=bool)
            duplicate[:-1] |= repeated
            duplicate[1:] |= repeated
            qc_flag[duplicate] |= GPSRO_QC_DUPLICATE_IMPACT

        # Greedy vertical thinning, ported from the reference: walk the
        # profile bottom-to-top (already sorted by impact_parameter, i.e.
        # by height), keeping the lowest surviving level in each
        # `impact_height_spacing_m` window. Restored 2026-10-20 (removed
        # earlier the same day, then reinstated -- see module docstring):
        # this repo's 14-level model backends have far fewer real vertical
        # degrees of freedom than GSI's own 60-150+ level models, so
        # assimilating every native GPSRO level (vs. GSI, which doesn't
        # thin at all) inflates this stream's effective weight in the
        # loss by roughly (levels-per-profile / 14).
        if impact_height_spacing_m > 0.:
            previous_height = -np.inf
            for level in range(indices.size):
                if qc_flag[level] != 0:
                    continue
                if impact_height[level] < previous_height + impact_height_spacing_m:
                    qc_flag[level] |= GPSRO_QC_THINNING
                else:
                    previous_height = impact_height[level]

        keep = qc_flag == 0
        if not np.any(keep):
            continue
        kept_indices = indices[keep]
        retained_rows += int(keep.sum())
        profiles.append({
            'profile_id': profile_id,
            'observation_time': observation_times[start],
            'satellite_id': int(columns['satellite_id'][kept_indices[0]]),
            'latitude_deg_north': columns['latitude_deg_north'][kept_indices],
            'longitude_deg_east': columns['longitude_deg_east'][kept_indices],
            'impact_height_m': (
                columns['impact_parameter_m'][kept_indices]
                - columns['earth_local_radius_curvature_m'][kept_indices]
                - columns['geoid_undulation_m'][kept_indices]
            ),
            'bending_angle_rad': columns['bending_angle_rad'][kept_indices],
            'source_error_rad': columns['bending_angle_error_rad'][kept_indices],
            'earth_local_radius_curvature_m': columns['earth_local_radius_curvature_m'][kept_indices],
            'geoid_undulation_m': columns['geoid_undulation_m'][kept_indices],
        })
    return profiles, {'canonical_rows': table.num_rows, 'retained_rows': retained_rows}


def select_gpsro_profiles(directory, verification_times, *, time_tolerance_hours, **qc_kwargs):
    """Select, QC, thin, and assign canonical GPSRO profiles to their
    nearest verification time (ported from the reference's
    `select_gpsro_profiles`, without its JAX-retracing file-cache
    machinery)."""
    times = [_utc_datetime(value) for value in verification_times]
    paths = gpsro_files_for_times(directory, times, time_tolerance_hours)

    profiles = []
    canonical_rows = 0
    retained_rows = 0
    seen_profile_ids = set()
    for path in paths:
        selected, counts = _canonical_profiles_from_file(path, **qc_kwargs)
        canonical_rows += counts['canonical_rows']
        retained_rows += counts['retained_rows']
        for profile in selected:
            if profile['profile_id'] in seen_profile_ids:
                raise ValueError('Duplicate GPSRO profile ID across cycle files: '+profile['profile_id'])
            seen_profile_ids.add(profile['profile_id'])
            profiles.append(profile)

    verification_ns = np.asarray(times, dtype='datetime64[ns]').astype(np.int64)
    tolerance_ns = int(float(time_tolerance_hours) * 3600. * 1.e9)
    by_time = [[] for _ in times]
    unassigned_profiles = 0
    for profile in profiles:
        profile_ns = profile['observation_time'].astype('datetime64[ns]').astype(np.int64)
        distances = np.abs(verification_ns - profile_ns)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tolerance_ns:
            by_time[nearest].append(profile)
        else:
            unassigned_profiles += 1
    for selected in by_time:
        selected.sort(key=lambda profile: profile['profile_id'])

    return {
        'by_time': by_time,
        'counts': {
            'canonical_rows_read': canonical_rows,
            'source_retained_rows': retained_rows,
            'source_retained_profiles': len(profiles),
            'assigned_profiles': sum(len(value) for value in by_time),
            'assigned_levels': sum(len(p['bending_angle_rad']) for value in by_time for p in value),
            'unassigned_profiles': unassigned_profiles,
            'profiles_by_time': [len(value) for value in by_time],
        },
    }


def gpsro_seed_error_std_rad(
    bending_angle_rad, impact_height_m, *,
    fractional_at_0_km=.20, fractional_at_10_km=.01, absolute_floor_rad=3.e-6,
):
    """Transparent fractional-plus-floor seed error model (the reference's
    'ecmwf_seed_v1') -- a simple analytic placeholder, not a reviewed/
    Desroziers-diagnosed error profile (see integrate_gpsro.md)."""
    interpolation_fraction = np.clip(np.asarray(impact_height_m) / 10000., 0., 1.)
    fractional = (
        float(fractional_at_0_km)
        + interpolation_fraction * (float(fractional_at_10_km) - float(fractional_at_0_km))
    )
    return np.maximum(fractional * np.abs(bending_angle_rad), float(absolute_floor_rad))


def load_gpsro_profiles(
    directory, verification_times, *,
    time_tolerance_hours=1.5, impact_height_range_km=(6., 40.), impact_height_spacing_m=1000.,
    maximum_bending_angle_rad=.1, quality_flags_disqualifying_mask=0,
    minimum_percent_confidence=None, error_model=None, logger=None,
):
    """Select, QC, thin, and error-assign canonical GPSRO profiles into one
    ragged list of profile dicts per verification-time slot (same
    per-slot-list convention as `load_psobs_parquet`/`load_raobs_parquet`
    -- fixed-shape padding is the CALLER's job, since it depends on the
    configured model, same division of labor as `get_raobs`/`get_amv`).
    Each profile dict carries `latitude_deg_north`/`longitude_deg_east`/
    `impact_height_m`/`bending_angle_rad`/`error_std_rad`/
    `earth_local_radius_curvature_m`/`geoid_undulation_m` (1-D arrays, one
    per surviving level) and a scalar `satellite_id`.
    """
    impact_height_range_m = tuple(1000. * float(v) for v in impact_height_range_km)
    error_model = {} if error_model is None else dict(error_model)
    error_model_name = str(error_model.get('name', 'ecmwf_seed_v1')).lower()
    if error_model_name != 'ecmwf_seed_v1':
        raise ValueError(
            f"Unsupported gpsro error_model.name {error_model_name!r} -- only "
            "'ecmwf_seed_v1' is implemented so far (see integrate_gpsro.md)."
        )
    error_model_kwargs = {k: v for k, v in error_model.items() if k != 'name'}

    selected = select_gpsro_profiles(
        directory, verification_times, time_tolerance_hours=time_tolerance_hours,
        impact_height_range_m=impact_height_range_m, impact_height_spacing_m=impact_height_spacing_m,
        maximum_bending_angle_rad=maximum_bending_angle_rad,
        quality_flags_disqualifying_mask=quality_flags_disqualifying_mask,
        minimum_percent_confidence=minimum_percent_confidence,
    )
    for profiles in selected['by_time']:
        for profile in profiles:
            profile['error_std_rad'] = gpsro_seed_error_std_rad(
                profile['bending_angle_rad'], profile['impact_height_m'], **error_model_kwargs,
            )
    if logger is not None:
        counts = selected['counts']
        logger.info(
            'gpsro parquet: '+str(counts['canonical_rows_read'])+' canonical rows read, '
            +str(counts['assigned_profiles'])+' profiles / '+str(counts['assigned_levels'])
            +' levels retained after QC/thinning, per-slot profile counts '
            +str(counts['profiles_by_time'])
        )
    return selected
