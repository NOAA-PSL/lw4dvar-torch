"""Drift of the fake-cycled free runs (Stage 0 E2, eagle/long-window/PLAN_levels_norm_capacity.md).

For each run and cycle: the background at cycle start (lead 0 of <T0>_control_forecast_*.nc, i.e. the
free-running state) minus ERA5 from the model's own training zarr (aifs37: era5_37 for the levels,
v8 for the surface; Bo's model: v8 for everything). Accumulates
  - global area-weighted bias and rms per variable and level, per cycle (time series);
  - zonal-mean bias (latitude x level) and point bias maps, averaged over the last LAST cycles;
  - low-level bias binned by surface elevation (the pressure-level / below-ground hypothesis).
Writes OUT/drift_<run>.npz. Plot with drift_plot.py.
Usage (CPU, anemoi-dev env): python eval/layer_models/drift.py OUT RUN_DIR [RUN_DIR ...] [--last 60]
"""
import argparse, glob, os, sys
from datetime import datetime, timedelta
import numpy as np, xarray as xr, zarr

TRUTHS = {'era5_37': '/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/'
                     'era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr',
          'v8': '/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/data/training/'
                'era5-o96-1979-2023-6h-v8.zarr'}
GRAV = 9.80665
LEVS = [1000, 925, 850, 700, 500, 300, 250, 100, 50, 10]
SCALE = {'z': 1 / GRAV, 't': 1.0, 'u': 1.0, 'v': 1.0, 'q': 1e3}
SFC = {'sp': 0.01, 'msl': 0.01, '2t': 1.0, 'tcw': 1.0, 'skt': 1.0}
ZBINS = [-500, 250, 1000, 2000, 3500, 7000]   # surface elevation bins (m)
BANDS = {'SP': (-90, -60), 'SO': (-60, -40), 'SM': (-40, -20), 'TR': (-20, 20), 'NM': (20, 40), 'NE': (40, 60), 'NP': (60, 90.1)}
SNAP = ['t_850', 't_250', 't_50', 'tcw', 'sp', 'z_500', 'q_850']


class Zarr:
    def __init__(self, path):
        z = zarr.open(path, 'r')
        self.data = z['data']
        self.idx = {n: i for i, n in enumerate(z.attrs['variables'])}
        self.start = datetime.fromisoformat(str(z['dates'][0])[:19])
        self.lat = np.asarray(z['latitudes'][:])

    def get(self, date, names):
        i, r = divmod((date - self.start).total_seconds(), 6 * 3600)
        assert not r, date
        block = np.asarray(self.data[int(i), :, 0, :])
        return {n: block[self.idx[n]] for n in names if n in self.idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('runs', nargs='+')
    ap.add_argument('--last', type=int, default=60, help='cycles at the end averaged for maps/zonal means')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    v8 = Zarr(TRUTHS['v8']); e37 = Zarr(TRUTHS['era5_37'])
    lat = v8.lat
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()  # O96 is near equal-area; cos weight as Jeff's aifs_drift.py
    orog = v8.get(v8.start, ['z'])['z'] / GRAV
    for run in a.runs:
        name = os.path.basename(run.rstrip('/'))
        files = sorted(glob.glob(f'{run}/*_control_forecast_*.nc'))
        cands = [f'eval/layer_models/cfg/{name}.yml', f'a37tmp/freerun/config_{name}.yml'] + sorted(glob.glob(f'a37tmp/*/cfg/config_{name}.yml'))
        cfg = next((c for c in cands if os.path.exists(c)), None)
        if cfg is None:
            raise FileNotFoundError(f'no config for {name}: tried {cands}')
        is37 = 'aifs37: True' in open(cfg).read()
        lev_truth = e37 if is37 else v8
        ts, dates, acc, n_acc, snaps = {}, [], {}, 0, {}
        for k, fn in enumerate(files):
            with xr.open_dataset(fn) as ds:
                d0 = datetime.fromisoformat(ds.attrs['trajectory_start_date'][:19])
                fields = {}
                for v in 'ztuvq':
                    if v not in ds:
                        continue
                    lv = f'level_{v}'
                    for L in LEVS:
                        if L in ds[lv].values:
                            fields[f'{v}_{L}'] = ds[v].isel(time=0).sel({lv: L}).values.astype(np.float64)
                for s in SFC:
                    if s in ds:
                        fields[s] = ds[s].isel(time=0).values.astype(np.float64)
            tl = lev_truth.get(d0, [f for f in fields if '_' in f])
            tl.update(v8.get(d0, [f for f in fields if '_' not in f]))
            dates.append(d0)
            take = k >= len(files) - a.last
            for f, x in fields.items():
                if f not in tl:
                    continue
                sc = SCALE[f.split('_')[0]] if '_' in f else SFC[f]
                d = (x - tl[f]) * sc
                ts.setdefault(f + '|bias', []).append(float((w * d).sum()))
                ts.setdefault(f + '|rms', []).append(float(np.sqrt((w * d * d).sum())))
                for b, (lo, hi) in BANDS.items():   # latitude-band bias time series
                    m = (lat >= lo) & (lat < hi)
                    ts.setdefault(f'{f}|bias|{b}', []).append(float((w * d * m).sum() / (w * m).sum()))
                if k % 10 == 0 and f in SNAP:       # bias maps every 5 days, to locate onsets
                    snaps.setdefault(f, []).append(d.astype(np.float32))
                if take:
                    acc[f] = acc.get(f, 0) + d
            n_acc += take
            if k % 30 == 0:
                print(name, k, d0, 'sp bias %.2f hPa' % ts.get('sp|bias', [np.nan])[-1],
                      'z500 rms %.1f m' % ts.get('z_500|rms', [np.nan])[-1], flush=True)
        mean_map = {f: v / max(n_acc, 1) for f, v in acc.items()}
        # bias of the mean map, by elevation bin, for the low levels and sp
        elev = {}
        for f in ['t_1000', 't_925', 't_850', 'z_1000', 'z_850', 'z_500', 'sp', '2t']:
            if f in mean_map:
                elev[f] = [float((w * mean_map[f] * m).sum() / (w * m).sum())
                           for m in [(orog >= lo) & (orog < hi) for lo, hi in zip(ZBINS[:-1], ZBINS[1:])]]
        np.savez_compressed(f'{a.out}/drift_{name}.npz', dates=np.array([d.isoformat() for d in dates]),
                            lat=lat, orog=orog, zbins=np.array(ZBINS), n_last=n_acc,
                            ts={k: np.array(v) for k, v in ts.items()}, maps=mean_map, elev=elev,
                            snaps={k: np.stack(v) for k, v in snaps.items()})
        print('wrote', f'{a.out}/drift_{name}.npz', len(dates), 'cycles', flush=True)


if __name__ == '__main__':
    main()
