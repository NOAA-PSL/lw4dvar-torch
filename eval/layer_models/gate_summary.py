"""Gate table for an L3 variant vs T1 / L3 (free-run drift G2 and forecast bias G3), from drift_fr_r2<arm>_*.npz and
figures/fcst_bias/<arm>.npz. Usage (anemoi-dev, from lwatms): python eval/layer_models/gate_summary.py <arm>"""
import glob, os, sys
import numpy as np
arm = sys.argv[1]
# reference rows (T1, L3) come from Sergey's tree when they are not in yours (REF_LWATMS, read-only)
REF = os.environ.get('REF_LWATMS', '/scratch4/BMC/gsienkf/Sergey.Frolov/lwatms')
def find(rel):
    return rel if os.path.exists(rel) else os.path.join(REF, rel)
print(f'L3 variant {arm}: free-run drift (global bias at day 30/60/90; z500 rms) -- G2: t850 |b| < 1 K, tcw < 2, t50 < 3 K, z500 rms < 120 m')
for r in ['r2t1_final', 'r2l3_final'] + sorted(os.path.basename(f)[9:-4] for f in glob.glob(f'output/freerun_diag/drift_fr_r2{arm}_*.npz')):
    f = find(f'output/freerun_diag/drift_fr_{r}.npz')
    if not os.path.exists(f):
        continue
    ts = np.load(f, allow_pickle=True)['ts'].item()
    g = lambda k: np.asarray(ts[k])[[59, 119, 179]] if k in ts and len(ts[k]) >= 180 else np.full(3, np.nan)  # noqa: E731
    row = {k: g(k) for k in ('t_850|bias', 'tcw|bias', 't_50|bias', 'z_500|rms', 't_850|bias|SP', 'tcw|bias|SP')}
    ok = all(abs(row['t_850|bias'][i]) < 1 and abs(row['tcw|bias'][i]) < 2 and abs(row['t_50|bias'][i]) < 3 and row['z_500|rms'][i] < 120 for i in range(3))
    print(f"{r:16s} t850 {np.round(row['t_850|bias'], 2)} tcw {np.round(row['tcw|bias'], 2)} t50 {np.round(row['t_50|bias'], 2)} "
          f"z500rms {np.round(row['z_500|rms'], 0)} | SP t850 {np.round(row['t_850|bias|SP'], 1)} tcw {np.round(row['tcw|bias|SP'], 1)} | G2 {'PASS' if ok else 'fail'}")
print('\nforecast from ERA5 (71 starts): model-only systematic z500 (m) day 1/3/5 -- G3: day 5 <= 4.4 m')
for n in ('t1', 'l3', arm):
    f = find(f'figures/fcst_bias/{n}.npz')
    if not os.path.exists(f):
        continue
    d = np.load(f); S, N = d['sum'], int(d['n']); w = np.cos(np.deg2rad(d['lats'])); L = list(d['leads'])
    r = [np.sqrt(np.average(((S[L.index(h), 0] - S[0, 0]) / N) ** 2, weights=w)) for h in (24, 72, 120)]
    print(f"{n:6s} {np.round(r, 2)}  G3 {'PASS' if r[2] <= 4.4 else 'fail'}")
