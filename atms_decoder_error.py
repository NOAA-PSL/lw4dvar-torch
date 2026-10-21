"""ATMS observation error from the decoder on ERA5 pairs (notes/obs_error_tuning.md).

Runs the two-head decoder on the radianced ERA5-paired 2015 files the way the 4D-Var does
(Obs_Time = 0, q_50 = training mean; the obs are up to 3 h off the ERA5 time, as they are off
the 4D-Var's 6-h obs slots). Over land, snow and sea ice the emissivity control z is fitted per
ob to the window channels 1-4, 16, 17 by Gauss-Newton with the surface head's prior, as the
4D-Var's z does; over ocean z = 0. Then, per lw4dvar scene (clw(y_o) <= 0.05 for clear ocean),
for d = (y_o - BC) - H(x_ERA5):
  std and mean per channel, the inter-channel covariance and correlation of the assimilated
  channels 5-7, 18-22 (and the window channels over ocean), and for comparison the replay GSI
  O-F (CRTM on the 6-h replay forecast, adjusted) on the same obs.

ERA5 has errors of its own, so var(d) bounds R from above; it is much tighter than the
4D-Var's O-B (an AIFS 0-5 day background).

Usage: python atms_decoder_error.py [N_CYCLES] [OUT_PREFIX]
       (every 2nd cycle of Jan-Apr 2015, default 120; default figures/obs_error/decoder_era5).
Writes OUT_PREFIX.json (per scene: n, std, mean, cov, corr; channels listed) and OUT_PREFIX_corr.png.
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

PRE = '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/era5_pre_2015'
BUNDLE = '/scratch4/BMC/gsienkf/Sergey.Frolov/atms/outputs/bo_twohead_s1/atms_twohead_bo_s1.pt'
NCYC = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figures/obs_error/decoder_era5'
WIN = [1, 2, 3, 4, 16, 17]
ASSIM = [5, 6, 7, 18, 19, 20, 21, 22]
ALL = WIN + ASSIM
SCENES = ['ocean_clear', 'ocean_cloudy', 'land', 'snow', 'seaice']
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def clw(tb1, tb2, zen):
    """GSI ret_amsua cloud liquid water (kg/m2) from 23.8 / 31.4 GHz, ocean"""
    mu = np.cos(np.radians(zen))
    with np.errstate(invalid='ignore', divide='ignore'):
        c = mu * (8.240 - (2.622 - 1.846 * mu) * mu + 0.754 * np.log(285.0 - tb1) - 2.265 * np.log(285.0 - tb2))
    return np.clip(np.nan_to_num(c, nan=3.0), 0.0, 3.0)


def fit_z(op, x, y, sig_w, iters=4):
    """Per-ob Gauss-Newton fit of z (K PCs) to the window channels:
    min sum_w ((y_w - H_w(x, z)) / sig_w)^2 + sum_k (z_k / sigma_k)^2."""
    iw = [c - 1 for c in WIN]
    with torch.no_grad():
        _, _, log_sig = op(x)
    pinv = torch.exp(-2 * log_sig)  # (n, K)
    z = torch.zeros(x.shape[0], op.K)
    for _ in range(iters):
        zz = z.clone().requires_grad_(True)
        tb = op(x, zz)[0][:, iw]
        # Jacobian dTb_w/dz, (n, W, K): K is small, one backward per PC is cheap via vjp per channel
        J = torch.stack([torch.autograd.grad(tb[:, w].sum(), zz, retain_graph=True)[0] for w in range(len(iw))], 1)
        r = (y[:, iw] - tb.detach()) / sig_w
        Jn = J / sig_w[None, :, None]
        A = Jn.transpose(1, 2) @ Jn + torch.diag_embed(pinv)
        g = (Jn.transpose(1, 2) @ r[..., None])[..., 0] - pinv * z
        z = z + torch.linalg.solve(A, g[..., None])[..., 0]
    return z.detach()


op = AtmsOperator.load(BUNDLE)
files = sorted(glob.glob(f'{PRE}/2015*.parquet'))[::2][:NCYC]
iq50, it0 = op.input_features.index('spfh-1'), op.input_features.index('Obs_Time')
sig_w = op.sigma_o[[c - 1 for c in WIN]].clone()
acc = {sc: {'d': [], 'g': []} for sc in SCENES}
for f in files:
    pre = pd.read_parquet(f)
    x = torch.tensor(pre[op.input_features].to_numpy(np.float32))
    x[:, iq50] = op.x_mean[iq50]
    x[:, it0] = 0.0
    y = torch.tensor(pre[[f'ObsBC-{i}' for i in range(22)]].to_numpy(np.float32))
    wf, ice, snow = pre.water_frac.to_numpy(), pre.ice_frac.to_numpy(), pre.snow_frac.to_numpy()
    zen = pre.Sat_Zenith_Angle.to_numpy()
    ocean = (wf >= 0.99) & (ice <= 0)
    land = (wf <= 0.01) & (ice <= 0)
    c_o = clw(y[:, 0].numpy(), y[:, 1].numpy(), zen)
    scene = np.select([ocean & (c_o <= 0.05), ocean & (c_o <= 0.5), land & (snow <= 0), land & (snow > 0), ice >= 0.5],
                      SCENES, 'other')
    zmask = np.isin(scene, ['land', 'snow', 'seaice'])
    with torch.no_grad():
        hx = op(x)[0]
    if zmask.any():
        zi = torch.as_tensor(np.nonzero(zmask)[0])
        z = fit_z(op, x[zi], y[zi], sig_w)
        with torch.no_grad():
            hx[zi] = op(x[zi], z)[0]
    d = (y - hx).numpy()
    gsi = pre[[f'Obs_Minus_Forecast_adjusted-{i}' for i in range(22)]].to_numpy()
    for sc in SCENES:
        m = scene == sc
        acc[sc]['d'].append(d[m][:, [c - 1 for c in ALL]])
        acc[sc]['g'].append(gsi[m][:, [c - 1 for c in ALL]])
    print(os.path.basename(f), len(pre), {sc: int((scene == sc).sum()) for sc in SCENES}, flush=True)

res = {'channels': ALL, 'bundle_sigma_o': [float(op.sigma_o[c - 1]) for c in ALL], 'scenes': {}}
for sc in SCENES:
    d, g = np.concatenate(acc[sc]['d']), np.concatenate(acc[sc]['g'])
    ok = np.isfinite(d).all(1) & np.isfinite(g).all(1) & (np.abs(d) < 50).all(1)
    d, g = d[ok], g[ok]
    # trim gross outliers per channel at 5 robust std (rain/QC failures) before the covariance
    mad = 1.4826 * np.median(np.abs(d - np.median(d, 0)), 0)
    keep = (np.abs(d - np.median(d, 0)) < 5 * mad).all(1)
    dk, gk = d[keep], g[keep]
    C = np.cov(dk.T)
    sd = np.sqrt(np.diag(C))
    Cg = np.cov(gk.T)
    res['scenes'][sc] = dict(n=int(len(dk)), n_all=int(len(d)), std=sd.tolist(), mean=dk.mean(0).tolist(),
                             std_untrimmed=d.std(0).tolist(), cov=C.tolist(), corr=(C / np.outer(sd, sd)).tolist(),
                             gsi_omf_std=np.sqrt(np.diag(Cg)).tolist(),
                             gsi_omf_corr=(Cg / np.outer(np.sqrt(np.diag(Cg)), np.sqrt(np.diag(Cg)))).tolist())
with open(OUT + '.json', 'w') as fh:
    json.dump(res, fh, indent=1)
print('wrote', OUT + '.json')

np.set_printoptions(precision=2, suppress=True, linewidth=200)
print('\nchannels', ALL)
print('bundle sigma_o ', np.array(res['bundle_sigma_o']))
for sc in SCENES:
    r = res['scenes'][sc]
    print(f'\n{sc}: n {r["n"]} of {r["n_all"]}')
    print('  std(y - H(ERA5))   ', np.array(r['std']))
    print('  untrimmed std      ', np.array(r['std_untrimmed']))
    print('  mean               ', np.array(r['mean']))
    print('  GSI O-F std (replay)', np.array(r['gsi_omf_std']))
    ia = [ALL.index(c) for c in ASSIM]
    print('  corr ch', ASSIM, '\n', np.array(r['corr'])[np.ix_(ia, ia)])

ia = [ALL.index(c) for c in ASSIM]
fig, axes = plt.subplots(2, len(SCENES), figsize=(4 * len(SCENES), 8))
for j, sc in enumerate(SCENES):
    for i, (key, lab) in enumerate((('corr', 'decoder, y - H(ERA5)'), ('gsi_omf_corr', 'replay GSI O-F'))):
        Cm = np.array(res['scenes'][sc][key])[np.ix_(ia, ia)]
        ax = axes[i, j]
        im = ax.imshow(Cm, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(ASSIM)), ASSIM, fontsize=7); ax.set_yticks(range(len(ASSIM)), ASSIM, fontsize=7)
        for a in range(len(ASSIM)):
            for b in range(len(ASSIM)):
                ax.text(b, a, f'{Cm[a, b]:.2f}', ha='center', va='center', fontsize=5.5)
        ax.set_title(f'{sc}: {lab}', fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.6, label='error correlation')
fig.suptitle(f'ATMS inter-channel error correlation, {len(files)} cycles Jan-Apr 2015')
fig.savefig(OUT + '_corr.png', dpi=110, bbox_inches='tight')
print('wrote', OUT + '_corr.png')
