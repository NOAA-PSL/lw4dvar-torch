"""Keep a scene_bc table's correction only on the listed channels (beta := 0 elsewhere), 2026-11-23: H2b applies it to
ATMS 5-7 only (it removes the NA/NH +0.1 K ch5-7 residual; on ch18-22 it overshoots in cloudy bins).
Usage: python restrict_channels.py IN.json OUT.json 5,6,7"""
import json
import sys

t = json.load(open(sys.argv[1])); keep = [int(c) for c in sys.argv[3].split(',')]
for k, sc in enumerate(t['beta']):
    for j, ch in enumerate(t['channels']):
        if ch not in keep:
            sc[j] = [0.0] * len(sc[j])
t['applied_channels'] = keep; t['source'] += f'; restricted to channels {keep} (restrict_channels.py)'
json.dump(t, open(sys.argv[2], 'w')); print('wrote', sys.argv[2], 'channels', keep)
