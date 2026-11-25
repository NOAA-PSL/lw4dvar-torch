#!/usr/bin/env python
"""Fit / evaluate the static radiosonde temperature bias table used by raob_tbias.py (2026-11-24, user: "build raob
dataset for 2016 and use it to develop corrections to raob. then apply it on 2015").

O - ERA5 for every ADPUPA temperature at the mandatory levels, from the normalized PREPBUFR parquet archive:
  * sensible temperature (first non-VIRTMP event, as raobs_parquet.py), PREPBUFR quality mark <= 3,
    level height not more than 100 m below the station (as the solver's reader), |O - ERA5| < 8 K
  * ERA5 = 37-level O96 zarr at the cycle time (soundings are within +-3 h of it), nearest grid point
  * solar elevation at the station and report time -> night (< -5 deg) / twilight / day (> 15 deg)

fit:  aggregate by (station, level, class) and (latitude band, level, class); write the table:
        band value    = band mean where n >= 50
        station value = (sum + K * band) / (n + K), K = 30 soundings, for stations with n >= 10 in that cell
                        (shrinks sparse stations toward their band; a station with 300 soundings keeps ~90% of its mean)
eval: O - ERA5 statistics on another period, raw and corrected with a table: mean by region and level, and the rms
      over stations of the per-station mean (how much station-to-station bias is left).

Usage (anemoi-dev, from lwatms):
  python fit_raob_tbias.py fit  --start 2016-01-01 --end 2016-12-31 --out figures/raob_tbias/raob_tbias_2016.json
  python fit_raob_tbias.py eval --start 2015-01-01 --end 2015-05-31 --table figures/raob_tbias/raob_tbias_2016.json
"""
import argparse
import json
import os
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree

import prepbufr_parquet_common as common
from raob_tbias import RaobTBias, solar_class, solar_elevation

ARCHIVE = Path('/scratch4/BMC/gsienkf/Sergey.Frolov/obs/prepbufr_to_parquet/data/prepbufr_parquet')
ZARR = '/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr'
LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
BANDS = [-90, -60, -30, -10, 10, 30, 60, 90]
K_SHRINK, N_STATION_MIN, N_BAND_MIN = 30.0, 10, 50
QM_MAX, BELOW_STATION_M, MAX_DEP = 3, 100.0, 8.0
_Z = {}


def _era5():
    if not _Z:
        Z = zarr.open(ZARR, 'r'); V = list(Z.attrs['variables'])
        la, lo = np.deg2rad(np.asarray(Z['latitudes'][:])), np.deg2rad(np.asarray(Z['longitudes'][:]))
        _Z.update(Z=Z, dates=np.asarray(Z['dates'][:]).astype('datetime64[s]'), vidx=[V.index(f't_{p}') for p in LEVELS],
                  tree=cKDTree(np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]))
    return _Z


def one_cycle(cycle_dir):
    """rows (station_id, lat, level index, class, O - ERA5) for one cycle directory YYYY-MM-DD/HH"""
    E = _era5()
    t0 = np.datetime64(datetime.strptime(f'{cycle_dir.parent.name}T{cycle_dir.name}', '%Y-%m-%dT%H'), 's')
    k = np.where(E['dates'] == t0)[0]
    try:
        reports = common.read_partition(common.class_file(cycle_dir, 'reports', 'ADPUPA'),
                                        ['report_id', 'observation_time', 'station_id', 'latitude', 'longitude', 'elevation_m'])
        levels = common.read_partition(common.class_file(cycle_dir, 'levels', 'ADPUPA'),
                                       ['report_id', 'level_index', 'pressure_mb', 'height_m', 'temperature_c', 'temperature_quality'])
    except Exception as e:                                   # noqa: BLE001  (missing class file for a cycle)
        return [], f'{cycle_dir}: {e}'
    if not len(k):
        return [], f'{cycle_dir}: no ERA5'
    rid = common.column(reports, 'report_id', np.int64); rrow = {int(r): i for i, r in enumerate(rid)}
    rtime = common.timestamps(reports); sid = reports['station_id'].to_pylist()
    lat, lon, elev = (common.column(reports, c) for c in ('latitude', 'longitude', 'elevation_m'))
    lrid = common.column(levels, 'report_id', np.int64); p = common.column(levels, 'pressure_mb')
    h = common.column(levels, 'height_m'); qm = common.column(levels, 'temperature_quality')
    t = common.sensible_temperature(cycle_dir, 'ADPUPA', lrid, common.column(levels, 'level_index', np.int64),
                                    common.column(levels, 'temperature_c')) + 273.15
    lev = np.asarray(LEVELS, dtype=np.float64)
    li = np.argmin(np.abs(p[:, None] - lev[None, :]), axis=1)
    ok = np.isfinite(t) & np.isfinite(p) & (np.abs(p - lev[li]) <= 0.05) & np.array([common.quality_passes(q, QM_MAX) for q in qm])
    ri = np.array([rrow.get(int(r), -1) for r in lrid]); ok &= ri >= 0
    idx = np.where(ok)[0]
    if not len(idx):
        return [], None
    ri = ri[idx]; la_, lo_ = lat[ri], lon[ri]
    good = np.isfinite(la_) & np.isfinite(lo_) & ~np.isnat(rtime[ri])
    hh, ee = h[idx], elev[ri]
    good &= ~(np.isfinite(hh) & np.isfinite(ee) & (hh < ee - BELOW_STATION_M))
    idx, ri, la_, lo_ = idx[good], ri[good], la_[good], lo_[good]
    era = np.asarray(E['Z']['data'].oindex[int(k[0]), E['vidx'], 0, :])          # (n_level, n_grid)
    r = np.deg2rad(la_), np.deg2rad(lo_)
    _, g = E['tree'].query(np.c_[np.cos(r[0]) * np.cos(r[1]), np.cos(r[0]) * np.sin(r[1]), np.sin(r[0])])
    dep = t[idx] - era[li[idx], g]
    cls = solar_class(np.array([solar_elevation(a, b, rtime[j]) for a, b, j in zip(la_, lo_, ri)]))
    keep = np.abs(dep) < MAX_DEP
    return [(str(sid[j]).strip(), float(a), int(l_), int(c), float(d), str(rtime[j])) for j, a, l_, c, d, kk
            in zip(ri, la_, li[idx], cls, dep, keep) if kk], None


def collect(start, end, workers):
    days = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end - start).days + 1)]
    cycles = [ARCHIVE / d / c for d in days for c in ('00', '06', '12', '18') if (ARCHIVE / d / c).is_dir()]
    print(f'{len(cycles)} cycle directories {start:%Y-%m-%d} .. {end:%Y-%m-%d}', flush=True)
    rows, notes = [], []
    with Pool(workers) as pool:
        for i, (r, note) in enumerate(pool.imap(one_cycle, cycles, chunksize=4)):
            rows += r
            if note:
                notes.append(note)
            if i % 200 == 0:
                print(f'  {i}/{len(cycles)} cycles, {len(rows):,} values', flush=True)
    for n in notes[:10]:
        print('  note:', n)
    print(f'{len(rows):,} values, {len(notes)} cycle notes', flush=True)
    return rows


def band_of(lat):
    return int(np.clip(np.searchsorted(BANDS, lat, side='right') - 1, 0, len(BANDS) - 2))


def fit(rows, out, period):
    nb, nl = len(BANDS) - 1, len(LEVELS)
    B = np.zeros((nb, nl, 3, 2)); S = {}
    for s, la, li, c, d, _ in rows:
        B[band_of(la), li, c] += (1, d)
        a = S.setdefault(s, np.zeros((nl, 3, 2))); a[li, c] += (1, d)
    band = np.where(B[..., 0] >= N_BAND_MIN, B[..., 1] / np.maximum(B[..., 0], 1), np.nan)
    st_lat = {}
    for s, la, *_ in rows:
        st_lat.setdefault(s, la)
    station = {}
    for s, a in S.items():
        bvals = np.nan_to_num(band[band_of(st_lat[s])])
        v = np.where(a[..., 0] >= N_STATION_MIN, (a[..., 1] + K_SHRINK * bvals) / (a[..., 0] + K_SHRINK), np.nan)
        if np.isfinite(v).any():
            station[s] = np.round(v, 4).tolist()
    tab = {'meta': {'fit_period': period, 'written': datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'), 'n_values': len(rows),
                    'n_stations': len(station), 'k_shrink': K_SHRINK, 'n_station_min': N_STATION_MIN, 'n_band_min': N_BAND_MIN,
                    'reference': 'ERA5 37-level O96, nearest grid point, cycle time', 'qc': f'qm<={QM_MAX}, |O-E|<{MAX_DEP} K'},
           'levels': LEVELS, 'classes': ['night', 'twilight', 'day'], 'bands': BANDS,
           'station': station, 'band': np.round(band, 4).tolist()}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(tab, open(out, 'w'))
    print(f'wrote {out}: {len(station)} stations')
    print('band means (K), night / twilight / day, at 850 / 500 / 250 / 100 hPa:')
    for b in range(nb):
        print(f'  {BANDS[b]:+4d}..{BANDS[b + 1]:+3d}: ' + '   '.join(
            '/'.join(f'{band[b, LEVELS.index(p), c]:+.2f}' for c in range(3)) for p in (850, 500, 250, 100)))


def evaluate(rows, table):
    tb = RaobTBias(table)
    reg = {'NH': lambda la: la >= 20, 'TR': lambda la: abs(la) < 20, 'SH': lambda la: la <= -20}
    raw = np.array([d for *_, d, _ in rows]); lat = np.array([r[1] for r in rows]); li = np.array([r[2] for r in rows])
    # same lookup as RaobTBias.__call__, with the solar class the rows already carry
    corr = np.empty(len(rows))
    for i, (s, la, l_, c, d, _) in enumerate(rows):
        row = tb.station.get(s)
        v = row[l_, c] if row is not None else np.nan
        if not np.isfinite(v):
            v = tb.band[band_of(la), l_, c]
        corr[i] = v if np.isfinite(v) else 0.0
    adj = raw - corr
    print(f'\n{len(rows):,} values; table {tb.meta.get("fit_period")}; stations covered '
          f'{np.mean([r[0] in tb.station for r in rows]) * 100:.0f}% of values')
    print('mean O - ERA5 (K) raw -> corrected, and rms over stations of the per-station mean (>= 20 values):')
    print('  level   ' + '   '.join(f'{rg + " mean":>16s}' for rg in reg) + '   ' + '   '.join(f'{rg + " stn rms":>16s}' for rg in reg))
    sid = np.array([r[0] for r in rows])
    for p in (925, 850, 700, 500, 300, 250, 200, 100, 50):
        m = li == LEVELS.index(p); cells = []; cells2 = []
        for rg, fn in reg.items():
            mm = m & fn(lat)
            cells.append(f'{raw[mm].mean():+.3f} -> {adj[mm].mean():+.3f}')
            u, inv, cnt = np.unique(sid[mm], return_inverse=True, return_counts=True)
            mr = np.bincount(inv, raw[mm]) / cnt; ma = np.bincount(inv, adj[mm]) / cnt; ok = cnt >= 20
            cells2.append(f'{np.sqrt(np.mean(mr[ok] ** 2)):.3f} -> {np.sqrt(np.mean(ma[ok] ** 2)):.3f}')
        print(f'  {p:5d}   ' + '   '.join(f'{c:>16s}' for c in cells) + '   ' + '   '.join(f'{c:>16s}' for c in cells2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fit', 'eval'])
    ap.add_argument('--start', required=True); ap.add_argument('--end', required=True)
    ap.add_argument('--out'); ap.add_argument('--table'); ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    s, e = datetime.strptime(a.start, '%Y-%m-%d'), datetime.strptime(a.end, '%Y-%m-%d')
    rows = collect(s, e, a.workers)
    if a.mode == 'fit':
        fit(rows, a.out, f'{a.start}..{a.end}')
    else:
        evaluate(rows, a.table)
