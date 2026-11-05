"""Near-surface observation reader (2 m temperature and humidity, 10 m wind)
for the normalized PREPBUFR parquet archive -- surface stations and
scatterometer winds, for forecast verification (`diagnostics/forecast_skill.py`).

Networks (PREPBUFR class, mass report types, wind report types):
  land    ADPSFC  181, 183 (synoptic / METAR)  281, 284
  marine  SFCSHP  180, 183 (ship, buoy, C-MAN) 280, 282, 284
  scat    ASCATW  --                           290 (ASCAT)
US mesonets (ADPSFC 187/287) are left out by default: they would outnumber
every other land station in the NH. Mass and wind are separate reports in
PREPBUFR (1xx vs 2xx), so each comes back as its own row.

Temperature is sensible temperature: PREPBUFR's current TOB is virtual
temperature wherever humidity was reported (program code 8, VIRTMP), so the
value is taken from the event stack's first non-VIRTMP event.

Read-time QC: the current-event quality mark must be <= `quality_mark_max`
or 9 (good, monitored -- what GDAS gives land T/q/wind, which it does not
assimilate), finite values, physical bounds. One report per station per slot,
nearest in time, then lowest quality mark (scatterometer cells are kept as
they are). Background-dependent QC (orography mismatch, gross check) is the
caller's.
"""

import numpy as np

import prepbufr_parquet_common as common
from convobs_parquet import _absent_class as absent_class  # a class the cycle's PREPBUFR file lacks


NETWORKS = {
    'land': ('ADPSFC', (181, 183), (281, 284)),
    'marine': ('SFCSHP', (180, 183), (280, 282, 284)),
    'scat': ('ASCATW', (), (290,)),
}
NETWORK_CODES = {name: i for i, name in enumerate(NETWORKS)}
MONITORED_QM = 9
FIELDS = ('network', 'obtype', 'subtype', 'lon', 'lat', 'elev', 'dt_hours', 't', 'q', 'u', 'v', 'p')

_REPORT_COLUMNS = ['report_id', 'observation_time', 'station_id', 'latitude', 'longitude', 'elevation_m',
                   'report_type', 'report_subtype', 'satellite_id']
_LEVEL_COLUMNS = ['report_id', 'level_index', 'pressure_mb', 'temperature_c', 'specific_humidity_mg_kg',
                  'u_wind_ms', 'v_wind_ms', 'temperature_quality', 'humidity_quality', 'wind_quality']


def _qm_ok(qm, maximum):
    return np.isfinite(qm) & ((qm <= maximum) | (qm == MONITORED_QM))


def load_sfcobs_parquet(directory, verification_times, time_tolerance_hours=1., networks=tuple(NETWORKS),
                        quality_mark_max=3, logger=None):
    """Near-surface obs per slot.

    Returns {field: [array per slot]} with FIELDS: 'network' (NETWORK_CODES),
    'obtype'/'subtype' (PREPBUFR report type / subtype; satellite id for scat), 'lon' [0, 360),
    'lat', 'elev' (m), 'dt_hours' (obs minus slot time), 't' (K, sensible),
    'q' (kg/kg), 'u'/'v' (m/s), 'p' (station pressure, hPa); a variable that
    is missing or fails read-time QC is NaN.
    """
    verification = np.asarray([np.datetime64(common.to_datetime(v), 's') for v in verification_times])
    tolerance = float(time_tolerance_hours) * 3600.
    cycles = common.cycle_directories(directory, verification_times, time_tolerance_hours)
    best = {}
    rows = []
    for cycle in cycles:
        for name in networks:
            prepbufr_class, mass_types, wind_types = NETWORKS[name]
            if absent_class(cycle, prepbufr_class):
                continue
            reports = common.read_partition(common.class_file(cycle, 'reports', prepbufr_class), _REPORT_COLUMNS)
            levels = common.read_partition(common.class_file(cycle, 'levels', prepbufr_class), _LEVEL_COLUMNS)
            level_rid = common.column(levels, 'report_id', np.int64)
            surface = common.column(levels, 'level_index', np.int32) == 0
            rid = common.column(reports, 'report_id', np.int64)
            row_of = {int(r): i for i, r in enumerate(rid)}
            rrow = np.array([row_of[int(r)] for r in level_rid[surface]], dtype=np.int64)
            lrid = level_rid[surface]
            col = lambda n: common.column(levels, n)[surface]  # noqa: E731
            rtype = common.column(reports, 'report_type', np.int32)[rrow]
            # GSI's convinfo subtype for ASCAT (290) is the satellite id (4 MetOp-A, 3 MetOp-B)
            sub = common.column(reports, 'satellite_id' if name == 'scat' else 'report_subtype')[rrow]
            sub = np.nan_to_num(sub, nan=0.).astype(np.int32)
            lat = common.column(reports, 'latitude')[rrow]
            lon = np.mod(common.column(reports, 'longitude')[rrow], 360.)
            elev = common.column(reports, 'elevation_m')[rrow]
            obs_time = common.timestamps(reports)[rrow]
            sid = np.asarray(reports['station_id'].to_pylist(), dtype=object)[rrow]

            delta = (obs_time[:, None] - verification[None, :]).astype('timedelta64[s]').astype(np.float64)
            slot = np.argmin(np.abs(delta), axis=1)
            dt = delta[np.arange(len(slot)), slot]
            ok = (np.abs(dt) <= tolerance) & np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90.)
            is_mass, is_wind = np.isin(rtype, mass_types), np.isin(rtype, wind_types)
            ok &= is_mass | is_wind

            p = col('pressure_mb')
            t = col('temperature_c')
            if is_mass.any():
                t = common.sensible_temperature(cycle, prepbufr_class, lrid, np.zeros_like(lrid), t)
            t = np.where(is_mass & _qm_ok(col('temperature_quality'), quality_mark_max)
                         & (t > -90.) & (t < 60.), t + 273.15, np.nan)
            q = col('specific_humidity_mg_kg') * 1e-6
            q = np.where(is_mass & _qm_ok(col('humidity_quality'), quality_mark_max) & (q > 0.) & (q < 0.05), q, np.nan)
            wind_ok = is_wind & _qm_ok(col('wind_quality'), quality_mark_max)
            u, v = col('u_wind_ms'), col('v_wind_ms')
            wind_ok &= np.isfinite(u) & np.isfinite(v) & (np.hypot(u, v) < 75.)
            u, v = np.where(wind_ok, u, np.nan), np.where(wind_ok, v, np.nan)
            p = np.where((p > 400.) & (p < 1100.), p, np.nan)
            ok &= np.isfinite(t) | np.isfinite(q) | np.isfinite(u)
            qm = np.where(is_mass, col('temperature_quality'), col('wind_quality'))

            code = NETWORK_CODES[name]
            for i in np.flatnonzero(ok):
                record = (code, rtype[i], sub[i], lon[i], lat[i], elev[i], dt[i] / 3600., t[i], q[i], u[i], v[i], p[i])
                if name == 'scat':
                    rows.append((slot[i], record))
                    continue
                key = (slot[i], code, bool(is_mass[i]), sid[i], round(lat[i], 2), round(lon[i], 2))
                rank = (abs(dt[i]), qm[i] if np.isfinite(qm[i]) else np.inf)
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, record)
    rows += [(key[0], record) for key, (_, record) in best.items()]

    out = {f: [[] for _ in verification_times] for f in FIELDS}
    for s, record in rows:
        for f, value in zip(FIELDS, record):
            out[f][s].append(value)
    dtypes = {'network': np.int8, 'obtype': np.int32, 'subtype': np.int32}
    out = {f: [np.asarray(v, dtype=dtypes.get(f, np.float32)) for v in vals] for f, vals in out.items()}
    if logger is not None:
        for name, code in NETWORK_CODES.items():
            n = [int((x == code).sum()) for x in out['network']]
            logger.info(f'sfcobs parquet {name}: {sum(n)} obs, per slot {n}')
    return out
