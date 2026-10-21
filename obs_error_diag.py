"""Observation-error diagnostics from a run's saved diags: for each obs group (ATMS channel x scene
x cloud bin, raob variable x level, ps), the assigned sigma_o, O-B / O-A rms and mean, the
Desroziers estimate sigma_D^2 = E[(O-A - mean)(O-B - mean)], the share of obs beyond the Huber
threshold c * sigma at x_b (Gaussian with the right sigma: 13.4% for c = 1.5), and the normalized
innovation and residual variances E[(O-B/sigma)^2], E[(O-A/sigma)^2]. Also the ATMS inter-channel
Desroziers correlation matrices per scene.

Desroziers assumes a converged optimal analysis. Ours is weak (O-A ~ O-B), which pushes sigma_D up
toward rms(O-B); read sigma_D / sigma > 1 as an upper bound, and trust it most where the Huber
share agrees.

Usage: python obs_error_diag.py RUN_DIR OUT_PREFIX [FIRST_CYCLE] [LAST_CYCLE]
       (cycles are 0-based indices into the run's sorted cycles; default 1 .. all, i.e. skip the
       first cycle). Writes OUT_PREFIX.csv, OUT_PREFIX_atms_corr.png and OUT_PREFIX_atms_corr.npz.
"""

import glob
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

run, out = sys.argv[1], sys.argv[2]
c0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
c1 = int(sys.argv[4]) if len(sys.argv) > 4 else None
SUF = '_120h_100it.nc'
dates = sorted(f.split('/')[-1][:13] for f in glob.glob(f'{run}/*_atms_diagnostics{SUF}'))[c0:c1]
print(f'{run}: {len(dates)} cycles {dates[0]} .. {dates[-1]}')

RAOB_ERR = {'t': 1.0, 'u': 2.0, 'v': 2.0}  # raobs_error_std default (long_window_4dvar_utils)
Q_ERR = {1000: 1.4, 925: 1.0, 850: 1.05, 700: 0.95, 600: 0.8, 500: 0.6, 400: 0.33, 300: 0.13}  # g/kg, raobs.q_error_g_kg in the lwatms configs
CLOUD_BINS = [0., 0.05, 0.2, 0.5, 10.]  # cloud_sym, ocean
CORR_CH = [5, 6, 7, 18, 19, 20, 21, 22]


def stats(omb, oma, sig, huber_c=1.5):
    ok = np.isfinite(omb) & np.isfinite(oma) & np.isfinite(sig) & (sig > 0)
    omb, oma, sig = omb[ok], oma[ok], sig[ok]
    if omb.size < 50:
        return None
    sd = np.mean((oma - oma.mean()) * (omb - omb.mean()))
    return dict(n=omb.size, sigma=np.sqrt(np.mean(sig ** 2)), omb_rms=np.sqrt(np.mean(omb ** 2)),
                omb_mean=omb.mean(), oma_rms=np.sqrt(np.mean(oma ** 2)),
                sigma_desroziers=np.sqrt(sd) if sd > 0 else np.nan,
                beyond_huber=np.mean(np.abs(omb / sig) > huber_c),
                nvar_omb=np.mean((omb / sig) ** 2), nvar_oma=np.mean((oma / sig) ** 2))


rows = []
acc = {}


def add(key, omb, oma, sig):
    a = acc.setdefault(key, ([], [], []))
    a[0].append(omb); a[1].append(oma); a[2].append(sig)


corr_acc = {}
for d in dates:
    # ATMS
    ds = xr.open_dataset(f'{run}/{d}_atms_diagnostics{SUF}')
    scenes = dict(s.split('=') for s in ds.attrs['scenes'].split())
    chans = ds.channel.values
    used = ds.used.values == 1
    omb, oma, sig = ds.omb.values, ds.oma.values, ds.sigma_obs.values
    sc, cl = ds.scene.values, ds.cloud_sym.values
    for j, ch in enumerate(chans):
        u = used[..., j]
        if not u.any():
            continue
        for s, sn in scenes.items():
            m = u & (sc == int(s))
            if m.any():
                add(('atms', f'ch{ch}', sn, 'all'), omb[..., j][m], oma[..., j][m], sig[..., j][m])
            if sn.startswith('ocean'):
                for lo, hi in zip(CLOUD_BINS[:-1], CLOUD_BINS[1:]):
                    mb = m & (cl >= lo) & (cl < hi)
                    if mb.any():
                        add(('atms', f'ch{ch}', sn, f'clw {lo:g}-{hi:g}'), omb[..., j][mb], oma[..., j][mb], sig[..., j][mb])
    # inter-channel Desroziers matrices: obs used in all CORR_CH
    jj = [list(chans).index(c) for c in CORR_CH]
    uall = used[..., jj].all(-1)
    for s, sn in scenes.items():
        m = uall & (sc == int(s))
        if m.any():
            ca = corr_acc.setdefault(sn, ([], []))
            ca[0].append(omb[..., jj][m]); ca[1].append(oma[..., jj][m])
    # raobs
    ds = xr.open_dataset(f'{run}/{d}_raobs_diagnostics{SUF}')
    for v in 't', 'u', 'v', 'q':
        lev = ds[f'{v}_omb'].dims[1]
        levels = ds[lev].values if lev in ds.coords else np.arange(ds.sizes[lev])
        for k, p in enumerate(levels):
            u = ds[f'{v}_used'].values[:, k] == 1
            if not u.any():
                continue
            ob, oa = ds[f'{v}_omb'].values[:, k][u], ds[f'{v}_oma'].values[:, k][u]
            if v == 'q':
                ob, oa = ob * 1e3, oa * 1e3  # kg/kg -> g/kg
                s = np.full(ob.shape, Q_ERR.get(int(p), np.nan))
            else:
                s = np.full(ob.shape, RAOB_ERR[v])
            add(('raob', v, str(int(p)), 'all'), ob, oa, s)
    # ps
    ds = xr.open_dataset(f'{run}/{d}_observation_diagnostics{SUF}')
    u = ds.surface_pressure_used.values == 1
    add(('ps', 'ps', 'all', 'all'), ds.surface_pressure_omb.values[u], ds.surface_pressure_oma.values[u],
        ds.surface_pressure_error_std.values[u])

for key, (a, b, c) in acc.items():
    r = stats(np.concatenate(a), np.concatenate(b), np.concatenate(c))
    if r:
        rows.append(dict(zip(('obs', 'var', 'group', 'bin'), key), **r))
df = pd.DataFrame(rows)
df['ratio_D'] = df.sigma_desroziers / df.sigma
df.to_csv(out + '.csv', index=False, float_format='%.4g')
print('wrote', out + '.csv', len(df), 'groups')

# ATMS inter-channel Desroziers correlation, per scene
fig, axes = plt.subplots(1, len(corr_acc), figsize=(4.2 * len(corr_acc), 4))
mats = {}
for ax, (sn, (a, b)) in zip(np.atleast_1d(axes), corr_acc.items()):
    a, b = np.concatenate(a), np.concatenate(b)
    a, b = a - a.mean(0), b - b.mean(0)
    R = (b.T @ a + a.T @ b) / (2 * len(a))  # symmetrized E[(O-A)(O-B)^T]
    sd = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    C = R / np.outer(sd, sd)
    mats[sn] = R
    im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(CORR_CH)), CORR_CH); ax.set_yticks(range(len(CORR_CH)), CORR_CH)
    for i in range(len(CORR_CH)):
        for k in range(len(CORR_CH)):
            ax.text(k, i, f'{C[i, k]:.2f}', ha='center', va='center', fontsize=6)
    ax.set_title(f'{sn} (n={len(a)})')
fig.colorbar(im, ax=axes, shrink=0.8, label='Desroziers error correlation')
fig.suptitle(f'ATMS inter-channel ob-error correlation, {run.rstrip("/").split("/")[-1]}, {dates[0]} .. {dates[-1]}')
fig.savefig(out + '_atms_corr.png', dpi=110, bbox_inches='tight')
np.savez(out + '_atms_corr.npz', channels=CORR_CH, **mats)
print('wrote', out + '_atms_corr.png')
