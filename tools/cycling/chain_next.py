"""Unattended cycling chains (notes/integrate_aifs37_models.md, "Week-long chain plan").

Writes the config for the next chunk of a run and prints its path, or DONE / STOP.
The chunk resumes from the newest complete analysis in the run's rpath (restart: True, sdate = that date),
so a job that fails or times out only loses its partial cycle and the next job in the chain picks up.
STOP after two attempts in a row without progress (a deterministic failure).

Usage: python chain_next.py BASE_CFG LAST_DATE CHUNK [EXTRA_EXP_LINE ...]
  BASE_CFG   the run's chunk-1 config (restart False, sdate = first cycle)
  LAST_DATE  the last analysis date, e.g. 2015-06-11T12
  CHUNK      cycles per job at most
  EXTRA      lines added under `exp:`, e.g. "  aifs_attention: 'flex_sliding_window'" on MI300X
"""
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

base, last, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
extra = sys.argv[4:]
txt = open(base).read()


def val(key):
    return re.search(rf"^  {key}: *'?([^'\s#]+)'?", txt, re.M).group(1)


rpath, suffix, sdate, dt = val('rpath'), val('ic_suffix'), val('sdate'), int(val('dt_init'))
F = '%Y-%m-%dT%H'
t_last = datetime.strptime(last, F)
os.makedirs(rpath + 'chain', exist_ok=True)
done = sorted(os.path.basename(p)[:13] for p in glob.glob(f'{rpath}*_optimal_inputs_{suffix}.pkl')
              if os.path.exists(p.replace('_optimal_inputs_', '_optimal_forecast_').replace('.pkl', '.nc')))
done = [d for d in done if datetime.strptime(d, F) <= t_last]
L = done[-1] if done else None

state_f = rpath + 'chain/state.json'
state = json.load(open(state_f)) if os.path.exists(state_f) else {'last_seen': 'none', 'fails': -1}
fails = state['fails'] + 1 if state['last_seen'] == str(L) else 0
json.dump({'last_seen': str(L), 'fails': fails}, open(state_f, 'w'))
if L == last:
    print('DONE')
    sys.exit()
if fails >= 2:
    print('STOP')
    sys.exit()

if L is None:
    restart, start = False, sdate
    n = min(chunk, int((t_last - datetime.strptime(sdate, F)) / timedelta(hours=dt)) + 1)
else:
    restart, start = True, L
    n = min(chunk, int((t_last - datetime.strptime(L, F)) / timedelta(hours=dt)))
out = re.sub(r"^  restart: .*$", f"  restart: {restart}  # chain_next.py: from {L or 'ERA5'}", txt, count=1, flags=re.M)
out = re.sub(r"^  sdate: .*$", f"  sdate: '{start}'", out, count=1, flags=re.M)
out = re.sub(r"^  n_init: .*$", f"  n_init: {n}", out, count=1, flags=re.M)
if extra:
    out = out.replace('exp:\n', 'exp:\n' + ''.join(e + '\n' for e in extra), 1)
cfg = f"{rpath}chain/chunk_{start}_n{n}.yml"
open(cfg, 'w').write(out)
print(cfg)
