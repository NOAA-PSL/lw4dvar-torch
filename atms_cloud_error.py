"""ATMS observation-error model, offline fit (notes/integrate_atms_obs.md, "ob errors").

Runs the two-head decoder on the ERA5-paired 2015 files (radianced build_era5_pre.py) for every
scene, the way the 4D-Var calls it (Obs_Time = 0, q_50 = training mean), and bins d = (y_o - BC) - H(x_ERA5):

  cloud (ocean, water_frac >= 0.99, no ice): symmetric cloud amount
      C_clw = (clw(y_o) + clw(H(x))) / 2      GSI/Grody 23.8/31.4 GHz retrieval, applied to both sides
      C_si  = (si(y_o) + si(H(x))) / 2        scattering index Tb(89) - Tb(165), for the 183 GHz channels
  scene: ocean_clear / ocean cloudy / land / snow / seaice (lw4dvar _atms_scene_codes definitions, with
         clw(y_o) for GSI's clw_obs, which has outliers in some cycles)

Prints std / mean of d by bin and channel, and a Geer & Bauer (2011) ramp
  sigma(C) = s_clr for C <= C0, s_cld for C >= C1, linear in between
fitted to the binned std. ERA5 is not the 4D-Var background, so the ratios std(bin) / std(ocean_clear)
are what carry over: they multiply the bundle sigma_o. Writes figures/atms_oberr/cloud_std.png and the
lw4dvar error table figures/atms_oberr/oberr_table.json (observations.atms.oberr_table): per channel the
scene ratios std(scene) / std(ocean_clear) and the C_clw ramp (c0, c1, ratio = s_cld / s_clr).

Usage: python atms_cloud_error.py [N_CYCLES] [TABLE_NAME]  (every 4th cycle of Jan-Apr 2015, default 60;
       table default oberr_table.json). WINDOW=1 also fits the window channels 1-4, 16, 17.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from obs_atms_operator import AtmsOperator  # noqa: E402

# PRE / BUNDLE / QFILL env as in atms_decoder_error.py (other pairs and decoder; QFILL='' for the q70 decoders)
PRE = os.environ.get('PRE', '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/era5_pre_2015')
BUNDLE = os.environ.get('BUNDLE', '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/outputs/bo_twohead_s1/atms_twohead_bo_s1.pt')
QFILL = [n for n in os.environ.get('QFILL', 'spfh-1').split(',') if n]
OUT = 'figures/atms_oberr'
NCYC = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TABLE = sys.argv[2] if len(sys.argv) > 2 else 'oberr_table.json'
WINDOW = [1, 2, 3, 4, 16, 17] if os.environ.get('WINDOW') == '1' else []
# STRAT=1: also ch 10-14 (A1/A2 Phase 2)
CHANNELS = WINDOW + [5, 6, 7, 8, 9] + ([10, 11, 12, 13, 14] if os.environ.get('STRAT') == '1' else []) + [18, 19, 20, 21, 22]
CLW_BINS = np.array([0, .02, .05, .1, .15, .2, .3, .4, .5, .7, 1., 1.5, 3.])
SI_BINS = np.array([-80, -50, -40, -30, -20, -10, 0, 10, 20, 40, 80])
os.makedirs(OUT, exist_ok=True)


def clw(tb1, tb2, zen):
    """GSI ret_amsua cloud liquid water (kg/m2) from 23.8 / 31.4 GHz, ocean"""
    mu = np.cos(np.radians(zen))
    with np.errstate(invalid='ignore', divide='ignore'):
        c = mu * (8.240 - (2.622 - 1.846 * mu) * mu + 0.754 * np.log(285.0 - tb1) - 2.265 * np.log(285.0 - tb2))
    return np.clip(np.nan_to_num(c, nan=3.0), 0.0, 3.0)


op = AtmsOperator.load(BUNDLE)
files = sorted(glob.glob(f"{PRE}/{os.environ.get('FILES', '2015*')}.parquet"))[::4][:NCYC]  # FILES env: e.g. 201609* (a held-out month)
iqf, it0 = [op.input_features.index(n) for n in QFILL], op.input_features.index('Obs_Time')
parts = []
for f in files:
    pre = pd.read_parquet(f)
    x = torch.tensor(pre[op.input_features].to_numpy(np.float32))
    x[:, iqf] = op.x_mean[iqf]
    x[:, it0] = 0.0
    with torch.no_grad():
        hx = op(x)[0].numpy()
    y = pre[[f'ObsBC-{i}' for i in range(22)]].to_numpy()
    wf, ice, snow = pre.water_frac.to_numpy(), pre.ice_frac.to_numpy(), pre.snow_frac.to_numpy()
    zen = pre.Sat_Zenith_Angle.to_numpy()
    p = pd.DataFrame({f'd{c}': y[:, c - 1] - hx[:, c - 1] for c in CHANNELS})
    p['clw_o'], p['clw_b'] = clw(y[:, 0], y[:, 1], zen), clw(hx[:, 0], hx[:, 1], zen)
    p['clw_gsi'] = pre.clw_obs.to_numpy()
    p['si_o'], p['si_b'] = y[:, 15] - y[:, 16], hx[:, 15] - hx[:, 16]
    ocean = (wf >= 0.99) & (ice <= 0)
    land = (wf <= 0.01) & (ice <= 0)
    p['scene'] = np.select([ocean & (p.clw_o.to_numpy() <= 0.05), ocean, land & (snow <= 0), land & (snow > 0), ice >= 0.5],
                           ['ocean_clear', 'ocean_cloudy', 'land', 'snow', 'seaice'], 'other')
    parts.append(p)
    print(os.path.basename(f), len(p), flush=True)
df = pd.concat(parts, ignore_index=True)
df['C_clw'], df['C_si'] = (df.clw_o + df.clw_b) / 2, (df.si_o + df.si_b) / 2
oc = df[df.scene.isin(['ocean_clear', 'ocean_cloudy'])]
print(f'\n{len(df):,} obs from {len(files)} cycles; ocean {len(oc):,} '
      f'({(df.scene == "ocean_cloudy").sum() / len(oc):.0%} of it cloudy, clw(y_o) > 0.05)')
print(f'retrieval check: corr(clw(y_o), GSI clw_obs) = {oc[["clw_o", "clw_gsi"]].corr("spearman").iloc[0, 1]:.3f} (spearman), '
      f'mean {oc.clw_o.mean():.3f} vs {oc.clw_gsi.mean():.3f} kg/m2; clw(H(x)) mean {oc.clw_b.mean():.3f}, '
      f'corr with clw(y_o) {np.corrcoef(oc.clw_o, oc.clw_b)[0, 1]:.3f}')

clr = df[df.scene == 'ocean_clear']
s_clr = {c: clr[f'd{c}'].std() for c in CHANNELS}
print('\nstd(d) by scene, K (ratio to ocean_clear); n per scene in brackets')
print('scene          n        ' + ''.join(f'ch{c:<11d}' for c in CHANNELS))
for sc in ['ocean_clear', 'ocean_cloudy', 'land', 'snow', 'seaice']:
    g = df[df.scene == sc]
    print(f'{sc:13s} {len(g):8d}  ' + ''.join(f'{g[f"d{c}"].std():5.2f}({g[f"d{c}"].std() / s_clr[c]:4.2f}) ' for c in CHANNELS))
table = {c: {sc: float(df.loc[df.scene == sc, f'd{c}'].std() / s_clr[c]) for sc in ('land', 'snow', 'seaice')}
         for c in CHANNELS}
print('bundle sigma_o ' + ' ' * 10 + ''.join(f'{float(op.sigma_o[c - 1]):5.2f}       ' for c in CHANNELS))


def binned(var, bins, chans):
    k = np.digitize(oc[var], bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        g = oc[k == b]
        if len(g) < 200:
            continue
        rows.append({'lo': bins[b], 'hi': bins[b + 1], 'mid': g[var].median(), 'n': len(g),
                     **{f's{c}': g[f'd{c}'].std() for c in chans}, **{f'm{c}': g[f'd{c}'].mean() for c in chans}})
    return pd.DataFrame(rows)


def ramp_fit(t, c):
    """least-squares (C0, C1, s_cld) for sigma(C) on the binned std, s_clr = first bin"""
    s0 = t[f's{c}'].iloc[0]
    best = None
    for c0 in t.mid.iloc[:4]:
        for c1 in t.mid[t.mid > c0]:
            for scl in np.linspace(s0, 3 * t[f's{c}'].max(), 200):
                pred = np.interp(t.mid, [c0, c1], [s0, scl])
                e = (np.log(pred / t[f's{c}']) ** 2 * np.sqrt(t.n)).sum()
                if best is None or e < best[0]:
                    best = (e, c0, c1, scl)
    return s0, best[1], best[2], best[3]


panels = [(None, 'C_clw', CLW_BINS, WINDOW, 'symmetric CLW (kg/m$^2$)')] if WINDOW else []
fig, axes = plt.subplots(1, 3 + len(panels), figsize=(19 + 6 * len(panels), 5))
panels = [(axes[-1],) + q[1:] for q in panels]
for ax, var, bins, chans, label in panels + [(axes[0], 'C_clw', CLW_BINS, [5, 6, 7, 8, 9], 'symmetric CLW (kg/m$^2$)'),
                                    (axes[1], 'C_clw', CLW_BINS, [18, 19, 20, 21, 22], 'symmetric CLW (kg/m$^2$)'),
                                    (axes[2], 'C_si', SI_BINS, [18, 19, 20, 21, 22], 'symmetric scattering index Tb89-Tb165 (K)')]:
    t = binned(var, bins, chans)
    print(f'\nocean, binned by {var}: std(d) K | mean(d) K')
    print('  bin            n      ' + ''.join(f'ch{c:<13d}' for c in chans))
    for _, r in t.iterrows():
        print(f'  {r.lo:5.2f}-{r.hi:5.2f} {int(r.n):8d}  ' + ''.join(f'{r[f"s{c}"]:5.2f} | {r[f"m{c}"]:+5.2f}  ' for c in chans))
    print(f'  ramp fit (Geer & Bauer): sigma = s_clr for {var} <= C0, s_cld for >= C1')
    for c in chans:
        s0, c0, c1, scl = ramp_fit(t, c)
        print(f'    ch{c:2d}: s_clr {s0:.2f} K, C0 {c0:.3f}, C1 {c1:.3f}, s_cld {scl:.2f} K  (ratio {scl / s0:.2f})')
        if var == 'C_clw':
            table[c].update(c0=float(c0), c1=float(c1), ramp_ratio=float(scl / s0))
        ln, = ax.plot(t.mid, t[f's{c}'], 'o', ms=4)
        cc = np.linspace(t.mid.min(), t.mid.max(), 100)
        ax.plot(cc, np.interp(cc, [c0, c1], [s0, scl]), color=ln.get_color(), label=f'ch {c}')
    if var == 'C_clw':
        ax.set_xscale('symlog', linthresh=0.05)
    ax.set_xlabel(label)
    ax.set_ylabel('std(y_o - BC - H(x_ERA5)) (K)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle(f'ATMS O - H(ERA5) spread vs symmetric cloud amount, ocean, {len(files)} cycles Jan-Apr 2015 (dots: bins, lines: ramp fit)')
fig.tight_layout()
fig.savefig(f'{OUT}/' + TABLE.replace('oberr_table', 'cloud_std').replace('.json', '.png'), dpi=140, bbox_inches='tight')
print('wrote', f'{OUT}/' + TABLE.replace('oberr_table', 'cloud_std').replace('.json', '.png'))
with open(f'{OUT}/{TABLE}', 'w') as fh:
    json.dump({'source': f'atms_cloud_error.py, {len(files)} cycles of {PRE}', 'cloud': 'C_clw, kg/m2',
               'channels': {str(c): v for c, v in table.items()}}, fh, indent=1)
print('wrote', f'{OUT}/{TABLE}')
