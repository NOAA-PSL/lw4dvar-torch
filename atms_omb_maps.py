"""Maps of ATMS O-B and O-A from a 4D-Var run's diagnostics next to the replay's GSI O-F (bias
corrected, `omfbc`) for the same obs, over a range of hours into one window.

The run's obs are y = Tb - BC with the replay's BC (radianced build_lw4dvar_obs.py), so
O-B = y - H_decoder(x_b) and the replay's omfbc = y - H_CRTM(x_replay 6-h forecast). Obs are
matched on (lat, lon, y of channel 5). The replay diags have no analysis departure.

Usage: python atms_omb_maps.py RUN_DIR WINDOW_START LEAD0 LEAD1 OUT_PNG [channels ...]
  e.g. python atms_omb_maps.py output/lwatms_6mo_all_tsens 2015-01-01T12 24 48 figures/atms_omb/c2.png 5 7 18 22
"""

import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

PROCESSED = '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/processed/atms_{}.parquet'
SCENES = ['ocean_clear', 'land', 'snow', 'seaice', 'ocean_cloudy']


def key(lat, lon, y5):
    return (np.round(lat, 2).astype(np.float64) * 1e9 + np.round(np.mod(lon, 360.), 2) * 1e3
            + np.round(y5, 1))


def main():
    run, start, lead0, lead1, out = sys.argv[1:6]
    chans = [int(c) for c in sys.argv[6:]] or [5, 7, 18, 22]
    lead0, lead1 = float(lead0), float(lead1)
    t0 = datetime.strptime(start, '%Y-%m-%dT%H')
    ds = xr.open_dataset(f'{run}/{start}_atms_diagnostics_120h_100it.nc')
    lt = ds.lead_time_hours.values
    ds = ds.isel(time=(lt >= lead0) & (lt <= lead1))
    scr = ds.screened.values.astype(bool)
    lat, lon = ds.latitude.values[scr], ds.longitude.values[scr]
    scene = ds.scene.values[scr]
    obs_ch5 = ds.observation.sel(channel=5).values[scr]
    ours = {c: {k: ds[k].sel(channel=c).values[scr] for k in ('omb', 'oma', 'used')} for c in chans}

    # replay diags: cycles whose +-3 h window overlaps [t0 + lead0 - 3h, t0 + lead1 + 3h]
    v0, v1 = t0 + timedelta(hours=lead0 - 3), t0 + timedelta(hours=lead1 + 3)
    months = sorted({f'{v0:%Y%m}', f'{v1:%Y%m}'})
    cols = ['cycle', 'dt_hours', 'lat', 'lon', 'sat'] + [f'{p}_{c:02d}' for c in set(chans) | {5} for p in ('tb', 'omf', 'omfbc', 'qc')]
    rp = pd.concat([pd.read_parquet(PROCESSED.format(m), columns=cols) for m in months])
    t = pd.to_datetime(rp['cycle']) + pd.to_timedelta(rp['dt_hours'], unit='h')
    rp = rp[(t >= v0) & (t < v1) & (rp['sat'] == 'npp')]
    y5 = rp['tb_05'] - (rp['omf_05'] - rp['omfbc_05'])
    rk = pd.Series(np.arange(len(rp)), index=key(rp['lat'].to_numpy(), rp['lon'].to_numpy(), y5.to_numpy()))
    rk = rk[~rk.index.duplicated()]
    ok_ = key(lat, lon, obs_ch5)
    m = rk.reindex(ok_).to_numpy()
    matched = np.isfinite(m)
    mi = m[matched].astype(int)
    print(f'{run} {start} +{lead0:g}..{lead1:g} h: {scr.sum()} screened obs, {matched.sum()} matched to the replay '
          f'({len(rp)} replay S-NPP obs in the period)')

    fig, axes = plt.subplots(len(chans), 4, figsize=(22, 3.3 * len(chans)), squeeze=False)
    for i, c in enumerate(chans):
        o = ours[c]
        r_omf = rp[f'omfbc_{c:02d}'].to_numpy()
        r_qc = rp[f'qc_{c:02d}'].to_numpy()
        lim = np.nanpercentile(np.abs(o['omb'][o['used'] == 1]), 98) if (o['used'] == 1).any() else 2.
        panels = [
            ('run O-B (used)', lon, lat, np.where(o['used'] == 1, o['omb'], np.nan)),
            ('run O-A (used)', lon, lat, np.where(o['used'] == 1, o['oma'], np.nan)),
            ('replay O-F, same obs', lon[matched], lat[matched], np.where(o['used'][matched] == 1, r_omf[mi], np.nan)),
            ('replay O-F, all obs passing GSI QC', np.mod(rp['lon'].to_numpy(), 360.), rp['lat'].to_numpy(),
             np.where(r_qc == 0, r_omf, np.nan)),
        ]
        for j, (title, x, y, v) in enumerate(panels):
            ax = axes[i, j]
            good = np.isfinite(v)
            sc = ax.scatter(x[good], y[good], c=v[good], s=1.2, cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True)
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_title(f'ch {c} {title}\nn {good.sum()}, mean {np.nanmean(v):+.2f}, rms {np.sqrt(np.nanmean(v ** 2)):.2f} K',
                         fontsize=9)
            ax.tick_params(labelsize=7)
        fig.colorbar(sc, ax=axes[i, :].tolist(), shrink=0.9, pad=0.01, label='K')
        u = o['used'][matched] == 1
        print(f'ch {c}: matched used obs n {u.sum()}: run O-B rms {np.sqrt(np.nanmean(o["omb"][matched][u] ** 2)):.3f} '
              f'mean {np.nanmean(o["omb"][matched][u]):+.3f}; O-A rms {np.sqrt(np.nanmean(o["oma"][matched][u] ** 2)):.3f}; '
              f'replay O-F rms {np.sqrt(np.nanmean(r_omf[mi][u] ** 2)):.3f} mean {np.nanmean(r_omf[mi][u]):+.3f}; '
              f'corr(run O-B, replay O-F) {np.corrcoef(o["omb"][matched][u], r_omf[mi][u])[0, 1]:.2f}')
        for k, sc_name in enumerate(SCENES):
            s_ = u & (scene[matched] == k)
            if s_.sum() > 50:
                print(f'   {sc_name:12s} n {s_.sum():6d}: run O-B rms {np.sqrt(np.nanmean(o["omb"][matched][s_] ** 2)):.3f}, '
                      f'O-A {np.sqrt(np.nanmean(o["oma"][matched][s_] ** 2)):.3f}, replay O-F {np.sqrt(np.nanmean(r_omf[mi][s_] ** 2)):.3f}')
    fig.suptitle(f'ATMS departures, {run.rstrip("/").split("/")[-1]}, window {start}, +{lead0:g} to +{lead1:g} h '
                 f'(valid {t0 + timedelta(hours=lead0):%m-%d %HZ} to {t0 + timedelta(hours=lead1):%m-%d %HZ})', y=1.0)
    fig.savefig(out, dpi=100, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
