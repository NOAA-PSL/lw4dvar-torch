"""E1 summary (notes/integrate_atms_obs.md): per cycle, for each run directory --
global z500 error before/after (from the log), ps O-B/O-A RMS (used obs), and
ATMS O-B/O-A RMS per channel for assimilated (used), withheld (screened,
passing the background check) and monitored obs.

O-B of cycle k >= 2 scores the forecast from cycle k-1's analysis.

Usage: python e1_summary.py <exp_name>:<log> [<exp_name>:<log> ...]
"""

import glob
import re
import sys

import numpy as np
import xarray as xr


def rms(x):
    x = x[np.isfinite(x)]
    return np.sqrt((x ** 2).mean()) if x.size else np.nan


for arg in sys.argv[1:]:
    name, log = arg.split(':')
    z = {}
    for line in open(log):
        m = re.match(r'z500err (before|after) (\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)', line)
        if m:
            z.setdefault(m.group(2), {})[m.group(1)] = float(m.group(3))
    print(f'\n== {name}')
    hdr = 'cycle          z500 b->a     ps omb->oma'
    files = sorted(glob.glob(f'output/{name}/*_observation_diagnostics_*.nc'))
    rows = []
    for f in files:
        date = f.split('/')[-1][:13]
        ps = xr.open_dataset(f)
        used = ps.surface_pressure_used.values.astype(bool)
        row = f'{date}  '
        zz = z.get(sorted(z)[len(rows)]) if len(rows) < len(z) else None
        row += f"{zz['before']:5.2f}->{zz['after']:5.2f}  " if zz else ' ' * 14
        row += f'{rms(ps.surface_pressure_omb.values[used]):5.3f}->{rms(ps.surface_pressure_oma.values[used]):5.3f}'
        af = f.replace('_observation_diagnostics_', '_atms_diagnostics_')
        if glob.glob(af):
            a = xr.open_dataset(af)
            ch = a.channel.values
            assim = set(a.attrs['assimilated_channels'].tolist()) if np.size(a.attrs['assimilated_channels']) else set()
            wh = (a.withheld.values.astype(bool) & a.screened.values.astype(bool))[..., None]
            ok = np.abs(a.omb.values) <= 4 * a.sigma_o.values  # background check, as in the loss
            parts = []
            for i, c in enumerate(ch):
                u = a.used.values[..., i].astype(bool)
                w = wh[..., 0] & ok[..., i]
                parts.append(f'{c}:{rms(a.omb.values[..., i][u]):.2f}/{rms(a.oma.values[..., i][u]):.2f}'
                             f'|wh {rms(a.omb.values[..., i][w]):.2f}/{rms(a.oma.values[..., i][w]):.2f}'
                             + ('*' if c in assim else ''))
            row += '   atms O-B/O-A ' + ' '.join(parts)
        rows.append(row)
    print(hdr + ('   atms ch:used O-B/O-A|withheld O-B/O-A (* assimilated)' if any('atms' in r for r in rows) else ''))
    print('\n'.join(rows))
