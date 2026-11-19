"""Where does ps + raobs hurt q850? Per-grid-point RMSE vs ERA5 over the
analysis trajectories of the first 10 cycles (every 12 h of each window),
mapped as the difference from ps only (Bo's runs), for ps + raobs, ps + raobs
with 10-degree obs-density weighting (Bo) and ps + ATMS (ours); dots = raob
stations that reported humidity in cycle 1. Writes
figures/compare_10cyc/q850_rmse_diff_map.png. Extra runs can be appended as
label=dir arguments.
"""

import glob
import sys
from datetime import datetime

import matplotlib
matplotlib.use('agg')
import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, 'diagnostics')
sys.path.insert(0, 'backends/aifs')
import aifs_ic  # noqa: E402
import z500err_window as zw  # noqa: E402

CACHE = '/scratch4/BMC/gsienkf/Sergey.Frolov/ic_cache/ic_cache_aifs2-1.0deg-v1.0/'
B = '/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/lw4dvar-torch-dev/add-raobs-parquet/lw4dvar-torch-test1/output/smoke_test_aifs2-1.0deg-v1.0-pretraining-'
REF = B + 'ps-parquet-new'
RUNS = [('ps + raobs (Bo)', B + 'ps-raobs-parquet-new'),
        ('ps + raobs, 10° density weighting (Bo)', B + 'ps-raobs-parquet-new-obsdenwt'),
        ('ps + ATMS ch 5-7 (ours)', 'output/lwatms_e1_ps_atms')]
RUNS += [tuple(a.rsplit('=', 1)) for a in sys.argv[1:]]
OUT = 'figures/compare_10cyc/q850_rmse_diff_map.png'
_era = {}


def msq(d):
    """mean over cycles 1-10 and every other trajectory time of (q850_a - ERA5)^2, (g/kg)^2"""
    acc, n = 0., 0
    for f in sorted(glob.glob(d + '/*_optimal_forecast_120h_100it.nc'))[:10]:
        ds = xr.open_dataset(f)
        start = zw._parse_nc_date(ds.attrs['trajectory_start_date'])
        q = ds.q.sel(level_q=850).values * 1e3
        for t in range(0, q.shape[0], 2):
            vd = zw.add_hours(start, 6 * t)
            if vd not in _era:
                _era[vd] = aifs_ic.read_single_date_fields(None, datetime.strptime(vd, '%Y-%m-%dT%H'), CACHE)['q_850'] * 1e3
            acc = acc + (q[t] - _era[vd]) ** 2
            n += 1
    return acc / n, ds.latitude.values, ds.longitude.values % 360


ref, lat, lon = msq(REF)
xyz = lambda la, lo: np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),  # noqa: E731
                               np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))], -1)
glat, glon = np.arange(-89.5, 90, 1.0), np.arange(0, 360, 1.0)
LO, LA = np.meshgrid(glon, glat)
_, nn = cKDTree(xyz(lat, lon)).query(xyz(LA.ravel(), LO.ravel()))

r = xr.open_dataset(sorted(glob.glob(B + 'ps-raobs-parquet-new/*_raobs_diagnostics_120h_100it.nc'))[0])
has_q = r.q_used.values.astype(bool).any(axis=1)
slat, slon = r.station_lat.values[has_q], r.station_lon.values[has_q]

fig, axes = plt.subplots(len(RUNS), 1, figsize=(10, 4.6 * len(RUNS)),
                         subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180)))
axes = np.atleast_1d(axes)
w = zw.region_masks(lat)
for ax, (label, d) in zip(axes, RUNS):
    m, _, _ = msq(d)
    diff = np.sqrt(m) - np.sqrt(ref)
    g = diff[nn].reshape(LA.shape)
    pm = ax.pcolormesh(glon, glat, g, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.8, vmax=0.8, shading='auto')
    ax.coastlines(lw=0.4)
    ax.scatter(slon, slat, s=3, c='k', transform=ccrs.PlateCarree(), lw=0)
    stats = '  '.join(f'{k} {zw.getrms(np.sqrt(m) - 0, v) - zw.getrms(np.sqrt(ref), v):+.3f}' for k, v in w.items() if k != 'Global')
    ax.set_title(f'{label} minus ps only: q850 RMSE (g/kg); red = worse\nregion-mean RMSE change: {stats}', fontsize=10)
    fig.colorbar(pm, ax=ax, shrink=0.75, label='g/kg')
    print(label, 'done', flush=True)
fig.suptitle('q850 RMSE vs ERA5, analysis trajectories of cycles 1-10; dots: raob stations with humidity', y=1.0)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print('wrote', OUT)
