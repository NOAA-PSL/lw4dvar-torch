"""ATMS O-B / O-A of a 4D-Var run vs the UFS GEFSv13 replay's GSI O-F (ges) and O-A (anl), same obs.

Our obs are y = Tb - BC(replay ges); each of our 6-h obs slots is one replay cycle (+-3 h).
Match on (lat, lon, y ch5) against the ges file; the anl file has the same obs in the same order.
Usage: python atms_vs_replay_oma.py RUN_DIR FIRST_WINDOW NWIN OUT_CSV
"""
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

RAW = '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/{}/{}/diag_atms_npp_{}.{}_control.nc4'
SCENES = {0: 'ocean_clear', 4: 'ocean_cloudy', 1: 'land/snow/ice', 2: 'land/snow/ice', 3: 'land/snow/ice'}
LEADBINS = [(0, 24), (48, 72), (96, 120)]


def key(lat, lon, y5):
    return (np.round(lat, 2).astype(np.float64) * 1e9 + np.round(np.mod(lon, 360.), 2) * 1e3
            + np.round(y5, 1))


_cache = {}


def replay(cyc):
    if cyc in _cache:
        return _cache[cyc]
    out = {}
    for kind, root in (('ges', 'raw'), ('anl', 'raw_anl')):
        ds = xr.open_dataset(RAW.format(root, cyc[:6], kind, cyc))
        n = ds.sizes['nobs'] // 22
        r = lambda v: ds[v].values.reshape(n, 22)  # noqa: E731
        assert (r('Channel_Index') == np.arange(1, 23)).all()
        out[kind] = dict(lat=r('Latitude')[:, 0], lon=r('Longitude')[:, 0], tb=r('Observation'),
                         bc=r('Obs_Minus_Forecast_unadjusted') - r('Obs_Minus_Forecast_adjusted'),
                         omf=r('Obs_Minus_Forecast_adjusted'), used=r('Inverse_Observation_Error') > 0)
    g, a = out['ges'], out['anl']
    assert np.array_equal(g['lat'], a['lat']) and np.array_equal(g['lon'], a['lon'])
    k = pd.Series(np.arange(len(g['lat'])), index=key(g['lat'], g['lon'], g['tb'][:, 4] - g['bc'][:, 4]))
    _cache[cyc] = (k[~k.index.duplicated()], g, a)
    return _cache[cyc]


def main():
    run, first, nwin, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    t0 = datetime.strptime(first, '%Y-%m-%dT%H')
    rows = []
    for w in range(nwin):
        ws = t0 + timedelta(hours=12 * w)
        ds = xr.open_dataset(f'{run}/{ws:%Y-%m-%dT%H}_atms_diagnostics_120h_100it.nc')
        chans = ds.channel.values
        for it, t in enumerate(pd.to_datetime(ds.time.values)):
            lead = int(ds.lead_time_hours.values[it])
            if lead == 0:
                continue  # slot 0 has no analysis value
            d = ds.isel(time=it)
            scr = d.screened.values.astype(bool)
            if not scr.any():
                continue
            k, g, a = replay(f'{t:%Y%m%d%H}')
            y5 = d.observation.sel(channel=5).values[scr]
            m = k.reindex(key(d.latitude.values[scr], d.longitude.values[scr], y5)).to_numpy()
            ok = np.isfinite(m)
            mi = m[ok].astype(int)
            sc = d.scene.values[scr][ok]
            for ic, c in enumerate(chans):
                rows.append(pd.DataFrame(dict(
                    window=w, lead=lead, channel=int(c), scene=sc,
                    used=d.used.values[scr, ic][ok].astype(bool),
                    omb=d.omb.values[scr, ic][ok], oma=d.oma.values[scr, ic][ok],
                    r_used=g['used'][mi, c - 1], r_omf=g['omf'][mi, c - 1], r_oma=a['omf'][mi, c - 1])))
        print(ws, flush=True)
    df = pd.concat(rows, ignore_index=True)
    df['scene'] = df['scene'].map(SCENES)
    df.to_parquet(out.replace('.csv', '.parquet'))
    rms = lambda x: np.sqrt(np.mean(np.square(x)))  # noqa: E731
    stats = []
    for (c, s), g in df.groupby(['channel', 'scene']):
        both = g[g.used & g.r_used]
        rec = dict(channel=c, scene=s, n_ours=int(g.used.sum()), n_gsi=int(g.r_used.sum()), n_both=len(both),
                   gsi_omf=rms(both.r_omf), gsi_oma=rms(both.r_oma))
        for l0, l1 in LEADBINS:
            b = both[(both.lead > l0) & (both.lead <= l1)]
            rec[f'oma_{l0}_{l1}'] = rms(b.oma)
        rec['omb_all'] = rms(both.omb)
        rec['oma_all'] = rms(both.oma)
        gsi_only = g[g.r_used]
        rec['gsi_oma_gsiused'] = rms(gsi_only.r_oma)
        rec['ours_oma_gsiused'] = rms(gsi_only.oma)
        stats.append(rec)
    st = pd.DataFrame(stats)
    st.to_csv(out, index=False, float_format='%.3f')
    with pd.option_context('display.width', 250, 'display.max_rows', 200):
        print(st.round(3).to_string(index=False))


main()
