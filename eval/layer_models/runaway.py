"""Where and when the L3 free run runs away: per cycle, the free-running state (lead 0 of each control_forecast) minus
D3 (the layer truth L3 trained on), by layer, for the south polar cap (< -60), the Southern Ocean (-60..-45), the
tropics and the globe; plus global-mean sp (dry-mass fixer check) and its model-minus-truth.
Usage (anemoi-dev, from lwatms): python eval/layer_models/runaway.py OUT.npz RUN_DIR [STRIDE=2]"""
import glob, sys
from datetime import datetime
import numpy as np, xarray as xr, zarr
out, run = sys.argv[1], sys.argv[2]; stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
Z = zarr.open('/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_ml37/chunks/era5-o96-ml37-1979-01-01T00-2023-12-31T18.zarr', 'r')
V = list(Z.attrs['variables']); ZD = np.asarray(Z['dates'][:]).astype('datetime64[s]')
L = 37; fams = ['t', 'q', 'u', 'v']
names = [f'{f}_ml{k:02d}' for f in fams for k in range(L)] + ['sp', '2t', 'skt', 'msl']
files = sorted(glob.glob(f'{run}/*_control_forecast_*.nc'))[::stride]
lat = None; R = {}; dates = []; B = []; G = []
for f in files:
    t = np.datetime64(datetime.strptime(f.split('/')[-1][:13], '%Y-%m-%dT%H'))
    i = np.where(ZD == t)[0]
    if not len(i): continue
    ds = xr.open_dataset(f).isel(time=0)
    if lat is None:
        lat = ds.latitude.values; w = np.cos(np.deg2rad(lat))
        R = {'SP': lat < -60, 'SO': (lat >= -60) & (lat < -45), 'TR': np.abs(lat) < 20, 'NP': lat > 60, 'GL': np.ones_like(lat, bool)}
    tr = np.asarray(Z['data'][int(i[0]), [V.index(n) for n in names], 0, :], np.float64)
    m = np.stack([ds[n].values.astype(np.float64) for n in names])
    d = m - tr
    B.append(np.stack([[np.average(d[k][r], weights=w[r]) for k in range(len(names))] for r in R.values()]))
    G.append([np.average(m[-4], weights=w), np.average(tr[-4], weights=w)])
    dates.append(str(t)); print(dates[-1], f"sp glob model-truth {G[-1][0] - G[-1][1]:+.1f} Pa; SP t_ml36 {B[-1][0][names.index('t_ml36')]:+.2f} q_ml36 {B[-1][0][names.index('q_ml36')]*1e3:+.3f} g/kg", flush=True)
np.savez(out, dates=np.array(dates), B=np.array(B), G=np.array(G), names=np.array(names), regions=np.array(list(R)))
