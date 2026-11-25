"""ATMS decoder-bias table for assimilating the stratospheric channels (step (a), E-H1): the a1q70 decoder's mean
(y_o - BC) - H(x_ERA5) per channel, lat band and day, from the tracker's ref8_*.npz (era5_ref.py, all scenes, 2015),
15-day centred running mean of the 6-hourly band sums. Read by long_window_4dvar_utils.get_atmsobs
(`observations.atms.extra_bc`), which subtracts it from y_o for the listed channels.
Usage: python tools/decoder_tables/make_extra_bc.py OUT.json [CHANNELS=9,10,11,12]"""
import glob
import os, json, sys
import numpy as np
from datetime import datetime
out = sys.argv[1]
chs = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else '9,10,11,12').split(',')]
fs = sorted(glob.glob(os.environ.get('REF', 'tools/decoder_tables/ref8_*.npz')))
d0 = np.load(fs[0]); allch = list(d0['channels']); bands = d0['bands'][:5].tolist()
dates = np.concatenate([np.load(f)['dates'] for f in fs]); S = np.concatenate([np.load(f)['S'] for f in fs])
if S.ndim == 5:   # mw_ref.py layout (date, stat, band, satellite, ch): all satellites = last
    S = S[:, :, :, -1]
o = np.argsort(dates); dates, S = dates[o], S[o]
days = sorted({d[:8] for d in dates})
day = np.array([d[:8] for d in dates])
ic = [allch.index(c) for c in chs]
bc = []
for k, dd in enumerate(days):
    t = datetime.strptime(dd, '%Y%m%d')
    sel = np.array([abs((datetime.strptime(x, '%Y%m%d') - t).days) <= 7 for x in day])
    n = S[sel, 0, :5][..., ic].sum(0); s = S[sel, 1, :5][..., ic].sum(0)
    bc.append(np.where(n > 100, s / np.maximum(n, 1), 0.).round(4).tolist())   # (band, ch)
json.dump({'channels': chs, 'bands': bands, 'days': days, 'bc': bc,
           'source': 'tools/decoder_tables/ref8_*.npz, 15-day running mean, d = (y_o - BC) - H(x_ERA5)'}, open(out, 'w'))
b = np.array(bc)
for j, c in enumerate(chs):
    print(f'ch{c}', ' '.join(f'{lo}..{hi}: {b[0, i, j]:+.2f}->{b[len(days)//2, i, j]:+.2f}->{b[-1, i, j]:+.2f}' for i, (lo, hi) in enumerate(bands)))
print(out, len(days), 'days', days[0], days[-1])
