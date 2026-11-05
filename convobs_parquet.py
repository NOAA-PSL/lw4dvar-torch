"""Upper-air conventional observation reader (temperature, humidity, wind at the
reported pressure) for the normalized PREPBUFR parquet archive, for forecast
verification profiles (`diagnostics/forecast_skill.py`).

Platform groups (PREPBUFR classes):
  raob      ADPUPA  every level except the surface level (category 0):
                    mandatory, significant, tropopause, max wind; balloon drift
                    positions where reported
  aircraft  AIRCAR, AIRCFT
  satwind   SATWND  (subtype: the satellite id, as GSI's convinfo keys them)
  profiler  PROFLR, VADWND
RASSDA (RASS virtual temperature) is left out.

Temperature is sensible temperature (`common.sensible_temperature`: the
current TOB is virtual where moisture was reported). Read-time QC: quality
mark <= `quality_mark_max`, finite values, bounds; obs errors from the
archive (NaN where it has none). No de-duplication -- every level is an
observation. Background QC (convinfo use flags, gross check, vertical range)
is the caller's.
"""

import numpy as np

import prepbufr_parquet_common as common


GROUPS = {
    'raob': ('ADPUPA',),
    'aircraft': ('AIRCAR', 'AIRCFT'),
    'satwind': ('SATWND',),
    'profiler': ('PROFLR', 'VADWND'),
}
GROUP_CODES = {name: i for i, name in enumerate(GROUPS)}
FIELDS = ('group', 'obtype', 'subtype', 'lon', 'lat', 'p', 't', 'q', 'u', 'v', 'err_t', 'err_q', 'err_uv')

_REPORT_COLUMNS = ['report_id', 'observation_time', 'latitude', 'longitude', 'report_type', 'report_subtype',
                   'satellite_id']
_LEVEL_COLUMNS = ['report_id', 'level_index', 'category', 'pressure_mb', 'temperature_c', 'specific_humidity_mg_kg',
                  'u_wind_ms', 'v_wind_ms', 'temperature_quality', 'humidity_quality', 'wind_quality',
                  'temperature_error_c', 'humidity_error_tenths_percent', 'wind_error_ms', 'drift_latitude',
                  'drift_longitude']


def _absent_class(cycle, prepbufr_class):
    """True when the cycle has no partition for the class because its PREPBUFR file has no such reports: the manifest
    says the conversion was complete and lists none (e.g. PROFLR in about half the 2022 cycles). Any other missing
    partition still raises in common.class_file."""
    if (cycle / 'reports' / str(prepbufr_class) / 'part-00000.parquet').is_file():
        return False
    try:
        import json
        m = json.load(open(cycle / 'metadata' / 'manifest.json'))
    except (OSError, ValueError):
        return False
    return bool(m.get('complete_source_conversion')) and prepbufr_class not in m.get('reports_by_class', {})


def load_convobs_parquet(directory, verification_times, time_tolerance_hours=1.5, groups=tuple(GROUPS),
                         quality_mark_max=3, logger=None):
    """Upper-air conventional obs per slot.

    Returns {field: [array per slot]} with FIELDS: 'group' (GROUP_CODES),
    'obtype'/'subtype' (PREPBUFR report type; subtype, or the satellite id for
    satwinds), 'lon' [0, 360), 'lat', 'p' (hPa), 't' (K, sensible), 'q' (kg/kg),
    'u'/'v' (m/s), 'err_t' (K), 'err_q' (relative humidity fraction), 'err_uv'
    (m/s); a missing or QC-failed variable is NaN.
    """
    verification = np.asarray([np.datetime64(common.to_datetime(v), 's') for v in verification_times])
    tolerance = float(time_tolerance_hours) * 3600.
    cycles = common.cycle_directories(directory, verification_times, time_tolerance_hours)
    out = {f: [[] for _ in verification_times] for f in FIELDS}
    for cycle in cycles:
        for name in groups:
            for prepbufr_class in GROUPS[name]:
                if _absent_class(cycle, prepbufr_class):
                    continue
                reports = common.read_partition(common.class_file(cycle, 'reports', prepbufr_class), _REPORT_COLUMNS)
                levels = common.read_partition(common.class_file(cycle, 'levels', prepbufr_class), _LEVEL_COLUMNS)
                rid = common.column(reports, 'report_id', np.int64)
                row_of = {int(r): i for i, r in enumerate(rid)}
                level_rid = common.column(levels, 'report_id', np.int64)
                rrow = np.array([row_of.get(int(r), -1) for r in level_rid], dtype=np.int64)
                col = lambda n: common.column(levels, n)  # noqa: E731
                rcol = lambda n, dtype=np.float64: common.column(reports, n, dtype)[rrow]  # noqa: E731

                delta = (common.timestamps(reports)[rrow][:, None] - verification[None, :]).astype(
                    'timedelta64[s]').astype(np.float64)
                slot = np.argmin(np.abs(delta), axis=1)
                dt = delta[np.arange(len(slot)), slot]
                p = col('pressure_mb')
                lat, lon = rcol('latitude'), np.mod(rcol('longitude'), 360.)
                dlat, dlon = col('drift_latitude'), col('drift_longitude')
                drift = np.isfinite(dlat) & np.isfinite(dlon)
                lat, lon = np.where(drift, dlat, lat), np.where(drift, np.mod(dlon, 360.), lon)
                ok = (rrow >= 0) & (np.abs(dt) <= tolerance) & np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90.)
                ok &= np.isfinite(p) & (p > 0.) & (p < 1100.) & (col('category') != 0)

                def passed(qcol):
                    qm = col(qcol)
                    return np.isfinite(qm) & (qm <= quality_mark_max)

                t = col('temperature_c')
                if np.isfinite(t).any():
                    t = common.sensible_temperature(cycle, prepbufr_class, level_rid,
                                                    common.column(levels, 'level_index', np.int64), t)
                t = np.where(passed('temperature_quality') & (t > -100.) & (t < 60.), t + 273.15, np.nan)
                q = col('specific_humidity_mg_kg') * 1e-6
                q = np.where(passed('humidity_quality') & (q > 0.) & (q < 0.05), q, np.nan)
                u, v = col('u_wind_ms'), col('v_wind_ms')
                wind = passed('wind_quality') & np.isfinite(u) & np.isfinite(v) & (np.hypot(u, v) < 150.)
                u, v = np.where(wind, u, np.nan), np.where(wind, v, np.nan)
                ok &= np.isfinite(t) | np.isfinite(q) | np.isfinite(u)
                sub = rcol('satellite_id' if name == 'satwind' else 'report_subtype')
                values = {
                    'group': np.full(len(p), GROUP_CODES[name]), 'obtype': rcol('report_type'),
                    'subtype': np.nan_to_num(sub, nan=0.), 'lon': lon, 'lat': lat, 'p': p, 't': t, 'q': q,
                    'u': u, 'v': v, 'err_t': col('temperature_error_c'),
                    'err_q': col('humidity_error_tenths_percent') / 10., 'err_uv': col('wind_error_ms'),
                }
                for s in np.unique(slot[ok]):
                    m = ok & (slot == s)
                    for f in FIELDS:
                        out[f][s].append(values[f][m])
    dtypes = {'group': np.int8, 'obtype': np.int32, 'subtype': np.int32}
    out = {f: [np.concatenate(v).astype(dtypes.get(f, np.float32)) if v else np.zeros(0, dtypes.get(f, np.float32))
               for v in vals] for f, vals in out.items()}
    if logger is not None:
        for name, code in GROUP_CODES.items():
            n = [int((x == code).sum()) for x in out['group']]
            logger.info(f'convobs parquet {name}: {sum(n)} obs, per slot {n}')
    return out
