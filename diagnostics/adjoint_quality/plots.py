"""Figures for the adjoint-quality battery (notes/adjoint_quality.md). CPU only; maps need cartopy (anemoi-dev env).

plot_aq(results, odir)        Q1-Q5 for one model (one JSON per date)
plot_single_ob(ind, odir)     S1-S5 + single_ob_metrics.json for one model's single-ob npz directory
compare(models, odir)         the same numbered figures with one colour per model (medians over dates / cases),
                              models = [(label, aq_results, single_ob_metrics or None)]
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

G0, OM, AE = 9.80665, 7.292e-5, 6.371e6
C = {'-g32': '#2a6fdb', '-gbf': '#e8833a', 'rand': '#888888', 'golden': '#2e9e5b'}
LAB = {'-g32': 'steepest descent (fp32 gradient)', '-gbf': 'steepest descent (reduced-precision gradient)',
       'rand': 'random latent direction', 'golden': 'golden increment direction'}
MODEL_COLORS = ['#2a6fdb', '#e8833a', '#2e9e5b', '#b0302a', '#7a4fbf', '#888888']


# ------------------------------------------------------------------ loading
def load_dir(path):
    """-> (aq results sorted by date, single-ob metrics dict or None) from a results directory or a reference bundle"""
    if os.path.isfile(path):
        b = json.load(open(path))
        return b['aq'], b.get('single_ob')
    aq = []
    for f in sorted(glob.glob(os.path.join(path, '*.json'))):
        r = json.load(open(f))
        r['_maps_file'] = f[:-5] + '_maps.npz'
        if 'taylor' in r and not r.get('smoke') and not r.get('meta', {}).get('settings', {}).get('smoke'):
            aq.append(r)
    aq.sort(key=lambda r: r['date'])
    so = None
    m = glob.glob(os.path.join(path, '**', 'single_ob_metrics.json'), recursive=True)
    if m:
        so = json.load(open(m[0]))
    return aq, so


def _tf32(r):
    """runs made before the tool (no meta) or with --tf32 have TF32 matmuls in 'fp32': eps < 2e-3 is round-off"""
    return r.get('meta', {}).get('tf32', True)


def _lastlead(r):
    return str(max(int(k) for k in r['cos']))


# ------------------------------------------------------------------ Q1-Q5 (one model)
def plot_aq(R, odir, title_model='A2'):
    os.makedirs(odir, exist_ok=True)
    dates = [r['date'][:10] for r in R]
    leads = sorted({k for r in R for k in r['cos']}, key=int)
    # Q1
    fig, ax = plt.subplots(1, len(leads), figsize=(6.5 * len(leads), 4.8), sharey=True, squeeze=False)
    for a, lead in zip(ax[0], leads):
        for name in ('-g32', '-gbf', 'golden'):
            for i, r in enumerate(R):
                t = r['taylor']
                if name not in t or f'fp32_{lead}' not in t[name]:
                    continue
                a.plot(t['eps_frac'], t[name][f'fp32_{lead}'], '-o', ms=3, color=C[name], alpha=0.8,
                       label=LAB[name] if i == 0 else None)
                if name == '-gbf' and f'gbf_{lead}' in t[name]:
                    a.plot(t['eps_frac'], t[name][f'gbf_{lead}'], ':', color=C[name], alpha=0.8,
                           label='same, scored with reduced-precision gradient' if i == 0 else None)
                if name == '-gbf' and f'bf16fwd_{lead}' in t[name]:
                    a.plot(t['eps_frac'], t[name][f'bf16fwd_{lead}'], '--', color='#b0302a', alpha=0.8,
                           label='reduced-precision forward and gradient (what the optimizer sees)' if i == 0 else None)
        a.axhline(1, color='k', lw=0.8)
        if any(_tf32(r) for r in R):
            a.axvspan(7e-5, 2e-3, color='0.9', zorder=0)
            a.text(1.2e-4, 1.85, 'TF32 round-off\n(ignore)', fontsize=8, color='0.4', va='top')
        a.set_xscale('log')
        a.set_ylim(-0.5, 2.0)
        a.set_xlabel('step size / reference increment norm')
        a.set_title(f'J at {lead} h')
    ax[0, 0].set_ylabel('actual change in J / gradient-predicted change')
    ax[0, -1].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'Figure Q1. Taylor test ({title_model}): does the gradient predict how J changes? (1 = exact; {len(R)} windows)')
    fig.tight_layout()
    fig.savefig(f'{odir}/Q1_taylor.png', dpi=110)
    plt.close(fig)
    # Q2
    fig, ax = plt.subplots(1, len(leads), figsize=(6.5 * len(leads), 4.5), sharey=True, squeeze=False)
    for a, lead in zip(ax[0], leads):
        labels = None
        for i, r in enumerate(R):
            c = r['cos'][lead]
            keys = [('bf16_vs_fp32', 'reduced vs fp32'), ('bf16_repeat', 'reduced repeat'), ('fp32_repeat', 'fp32 repeat')]
            vals = [c.get(k, np.nan) for k, _ in keys] + [r['stability'][k][lead] for k in r['stability']]
            labels = [lab for _, lab in keys] + [k.replace('rand_', 'moved by ') + ' x|ref|' if k.startswith('rand')
                                                 else 'at golden u' for k in r['stability']]
            a.plot(range(len(vals)), vals, 'o-', label=dates[i])
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        a.axhline(1, color='k', lw=0.8)
        a.set_ylim(-0.1, 1.05)
        a.set_title(f'gradient of J at {lead} h')
    ax[0, 0].set_ylabel('cosine similarity with fp32 gradient at u=0')
    ax[0, -1].legend(fontsize=8)
    fig.suptitle(f'Figure Q2 ({title_model}). Is the gradient repeatable, precision-robust and smooth? (1 = identical direction)')
    fig.tight_layout()
    fig.savefig(f'{odir}/Q2_cosines.png', dpi=110)
    plt.close(fig)
    # Q3
    P = [('golden_twin', '#2e9e5b', 'golden increment (twin)'), ('golden_cv', '#7a4fbf', 'golden increment (real obs)'),
         ('random', '#888888', 'random latent'), ('-g', '#2a6fdb', 'steepest descent')]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for a, (key, title) in zip(ax, (('norm', 'all prognostic, normalised'), ('z500', 'z500'), ('t850', 't850'))):
        for name, col, lab in P:
            for i, r in enumerate(R):
                s = r['survival']
                if name not in s:
                    continue
                for suf, ls in (('_large', '-'), ('_small', ':')):
                    v = np.array(s[name][key + suf])
                    a.plot(s['lead_h'], v / s[name][key][0], ls, color=col, alpha=0.7,
                           label=(f'{lab}, ' + ('large scale' if suf == '_large' else 'small scale')) if i == 0 else None)
        a.axhline(1, color='k', lw=0.8)
        a.set_yscale('log')
        a.set_xlabel('lead (h)')
        a.set_title(title)
    ax[0].set_ylabel('perturbation size / total size at 6 h')
    ax[2].legend(fontsize=7)
    fig.suptitle(f'Figure Q3 ({title_model}). Do perturbations survive? (>1 grows, <1 damped; split at ~400 km)')
    fig.tight_layout()
    fig.savefig(f'{odir}/Q3_survival.png', dpi=110)
    plt.close(fig)
    # Q4
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, (tag, title) in zip(ax, (('all', 'perturb every prognostic variable'), ('noz', 'perturb all but geopotential'))):
        for i, r in enumerate(R):
            sv = r[f'svd_48h_{tag}']['singular_values']
            ln, = a.plot(range(1, len(sv) + 1), sv, 'o-', label=dates[i])
            a.axhline(np.median(r[f'random_amplification_48h_{tag}']), color=ln.get_color(), ls=':', lw=1)
        a.axhline(1, color='k', lw=0.8)
        a.set_yscale('log')
        a.set_xlabel('singular vector rank')
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel('amplification 6 h -> 48 h (normalised norm)')
    ax[1].legend(fontsize=8)
    fig.suptitle(f'Figure Q4 ({title_model}). How much can the model amplify a 6-h perturbation by 48 h? Leading singular values\n'
                 '(dotted: median amplification of random ~400-km perturbations; 1 = neither grows nor decays)', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{odir}/Q4_singular_values.png', dpi=110)
    plt.close(fig)
    # Q5
    mf = R[0].get('_maps_file')
    if mf and os.path.exists(mf):
        try:
            import cartopy.crs as ccrs
        except ImportError:
            print('cartopy not available: Q5 skipped (use the anemoi-dev env)')
            return
        z = np.load(mf)
        panels = [(k, t) for k, t in (('golden_twin_t850', 'golden increment (twin)'),
                                      ('golden_cv_t850', 'golden increment (real obs)'),
                                      ('-g_t850', 'steepest descent, scaled to the reference norm')) if k in z]
        fig, ax = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), squeeze=False,
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        for a, (k, t) in zip(ax[0], panels):
            f = z[k]
            v = np.nanpercentile(np.abs(f), 99)
            sc = a.scatter(z['lon'], z['lat'], c=f, s=1, cmap='RdBu_r', vmin=-v, vmax=v, transform=ccrs.PlateCarree())
            a.coastlines(linewidth=0.4)
            a.set_title(f'{t}: t850 change at 6 h (K)', fontsize=10)
            fig.colorbar(sc, ax=a, shrink=0.7)
        fig.suptitle(f'Figure Q5 ({title_model}). What the gradient asks for vs what 4D-Var did, {dates[0]}', fontsize=12)
        fig.savefig(f'{odir}/Q5_maps.png', dpi=100, bbox_inches='tight')
        plt.close(fig)


# ------------------------------------------------------------------ single ob: metrics + S1-S5
def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


SOLV = ['G', 'BG', 'W']
SNAME = {'G': 'G: raw adjoint', 'BG': 'BG: B-filtered adjoint (single-ob B solution)', 'W': 'W: AdamW (golden)'}
OBN = {'a': 'z500 +30 m at 6 h', 'b': 'z500 +30 m at 24 h', 'c': 'ps -2 hPa at 6 h'}
INNOV = {'a': 30 * G0, 'b': 30 * G0, 'c': -200.0}


def _key(f):
    """{mmdd}_{loc}_{ob}_{solver}.npz -> (mmdd, loc, ob, solver), else None"""
    parts = os.path.basename(f)[:-4].split('_')
    return tuple(parts) if len(parts) == 4 else None


def plot_single_ob(ind, odir, title_model='A2'):
    from forecast_model import area_weights_from_lats
    os.makedirs(odir, exist_ok=True)
    files, dropped = [], []
    for f in sorted(glob.glob(f'{ind}/*.npz')):
        key = _key(f)
        if key is None:
            continue
        mmdd, loc, ob, sv = key
        summ = f'{ind}/{mmdd}_{loc}_summary.json'
        if ob not in INNOV or not os.path.exists(summ):
            continue
        fit = json.load(open(summ))[f'{ob}_{sv}']['fit'][ob]
        if abs(fit / INNOV[ob] - 1) > 0.25:      # line search failed to fit the ob: exclude
            dropped.append((os.path.basename(f), round(fit / INNOV[ob], 2)))
        else:
            files.append(f)
    print('dropped (ob not fitted, fit/innovation):', dropped)
    z0 = np.load(files[0])
    lat, lon = z0['lat'], z0['lon']
    aw = area_weights_from_lats(lat)
    X = _xyz(lat, lon)
    tree = cKDTree(X)
    _, NB = tree.query(X, k=24)
    glat, glon = np.arange(-89.5, 90, 1.0), np.arange(0.5, 360, 1.0)
    GLO, GLA = np.meshgrid(glon, glat)
    dg, ig = tree.query(_xyz(GLA.ravel(), GLO.ravel()), k=4)
    wg = 1.0 / np.maximum(dg, 1e-9)
    wg /= wg.sum(1, keepdims=True)

    def regrid(f):
        return (f[ig] * wg).sum(1).reshape(GLA.shape)

    def smooth(f):
        for _ in range(3):
            f = f[NB].mean(1)
        return f

    def geo(phi):
        f = 2 * OM * np.sin(np.radians(GLA))
        pp = np.concatenate([phi[:, -1:], phi, phi[:, :1]], 1)
        dlam = (pp[:, 2:] - pp[:, :-2]) / (2 * np.radians(1.0))
        dphi = np.gradient(phi, np.radians(1.0), axis=0)
        return -dphi / (AE * f), dlam / (AE * np.cos(np.radians(GLA)) * f)

    def dist_km(la, lo, olat, olon):
        return 6371.0 * np.arccos(np.clip(_xyz(la, lo) @ _xyz(np.array([olat]), np.array([olon]))[0], -1, 1))

    def balance(phi, u, v, olat, olon):
        ug, vg = geo(regrid(phi))
        U, Vv = regrid(u), regrid(v)
        m = (dist_km(GLA, GLO, olat, olon) < 2000) & (np.abs(GLA) > 20)
        w = np.cos(np.radians(GLA))[m]
        a2 = (w * (U[m] ** 2 + Vv[m] ** 2)).sum()
        g2 = (w * (ug[m] ** 2 + vg[m] ** 2)).sum()
        c = (w * (U[m] * ug[m] + Vv[m] * vg[m])).sum() / np.sqrt(a2 * g2)
        return float(np.sqrt(a2 / g2)), float(c)

    def small_frac(fs, m):
        return float(sum((aw[m] * (f - smooth(f))[m] ** 2).sum() for f in fs) / sum((aw[m] * f[m] ** 2).sum() for f in fs))

    M, cal = {}, {}
    for f in files:
        mmdd, loc, ob, sv = _key(f)
        z = np.load(f)
        olat, olon = float(z['olat']), float(z['olon'])
        dn = dist_km(lat, lon, olat, olon)
        near, far = dn < 2000, dn > 3000
        r = {}
        for h in (6, 24, 48):
            zf, uf, vf = z[f'z_500_{h}'], z[f'u_500_{h}'], z[f'v_500_{h}']
            ratio, corr = balance(zf, uf, vf, olat, olon)
            r[h] = dict(ratio=ratio, corr=corr, small_z=small_frac([zf], near), small_w=small_frac([uf, vf], near),
                        far_z=float((aw * zf ** 2)[far].sum() / (aw * zf ** 2).sum()),
                        far_w=float((aw * (uf ** 2 + vf ** 2))[far].sum() / (aw * (uf ** 2 + vf ** 2)).sum()),
                        z_at_ob=float(z[f'prof_z_{h}'][list(z['levels_z']).index(500)] / G0))
        M[(mmdd, loc, ob, sv)] = r
        if (mmdd, loc) not in cal:
            cal[(mmdd, loc)] = balance(z['bg_z_500_6'], z['bg_u_500_6'], z['bg_v_500_6'], olat, olon)
    out = {'metrics': {'_'.join(k): v for k, v in M.items()}, 'calibration_bg': {'_'.join(k): v for k, v in cal.items()},
           'dropped': dropped}
    json.dump(out, open(f'{odir}/single_ob_metrics.json', 'w'), indent=1)
    print('calibration (background ratio, corr):', {k: tuple(round(x, 2) for x in v) for k, v in cal.items()})
    print(f'{"case":22s} {"h":>3s} {"ratio":>6s} {"corr":>6s} {"smallZ":>6s} {"smallW":>6s} {"farZ":>6s} {"farW":>6s} {"z@ob m":>7s}')
    for k, r in sorted(M.items()):
        for h in (6, 24):
            x = r[h]
            print(f'{"_".join(k):22s} {h:3d} {x["ratio"]:6.2f} {x["corr"]:6.2f} {x["small_z"]:6.2f} {x["small_w"]:6.2f} '
                  f'{x["far_z"]:6.2f} {x["far_w"]:6.2f} {x["z_at_ob"]:7.1f}')
    _single_ob_figs(ind, odir, M, cal, regrid, geo, lat, lon, title_model)
    return out


def _single_ob_figs(ind, odir, M, cal, regrid, geo, lat, lon, title_model):
    S3(odir, [(title_model, {'metrics': {'_'.join(k): v for k, v in M.items()},
                             'calibration_bg': {'_'.join(k): v for k, v in cal.items()}})], per_case=True)
    try:
        import cartopy.crs as ccrs
    except ImportError:
        print('cartopy not available: S1, S2, S4, S5 skipped (use the anemoi-dev env)')
        return
    dates = sorted({k[0] for k in M})
    locs = sorted({k[1] for k in M})
    loc0 = 'sh' if 'sh' in locs else locs[0]
    GLA, GLO = np.meshgrid(np.arange(-89.5, 90, 1.0), np.arange(0.5, 360, 1.0), indexing='ij')

    def regional(ax, z, h, olat, olon, title):
        box = 35
        zf = regrid(z[f'z_500_{h}']) / G0
        U, Vv = regrid(z[f'u_500_{h}']), regrid(z[f'v_500_{h}'])
        ug, vg = geo(regrid(z[f'z_500_{h}']))
        lo = (GLO - olon + 180) % 360 - 180
        m = (np.abs(lo) <= box) & (np.abs(GLA - olat) <= 25)
        cm = ax.scatter(GLO[m], GLA[m], c=zf[m], s=9, marker='s', cmap='RdBu_r', vmin=-30, vmax=30,
                        transform=ccrs.PlateCarree())
        q = m & (GLA.astype(int) % 3 == 0) & (GLO.astype(int) % 3 == 0)
        sc = max(1e-6, np.percentile(np.hypot(ug[m], vg[m]), 98)) * 12
        ax.quiver(GLO[q], GLA[q], ug[q], vg[q], color='tab:red', scale=sc, width=0.004, transform=ccrs.PlateCarree())
        ax.quiver(GLO[q], GLA[q], U[q], Vv[q], color='k', scale=sc, width=0.004, transform=ccrs.PlateCarree())
        ax.plot(olon, olat, 'y*', ms=12, mec='k', transform=ccrs.PlateCarree())
        ax.set_extent([olon - box, olon + box, olat - 25, olat + 25], crs=ccrs.PlateCarree())
        ax.coastlines(lw=0.5)
        ax.set_title(title, fontsize=9)
        return cm

    def map_fig(fn, ob, rows, suptitle):
        fig, ax = plt.subplots(len(rows), 3, figsize=(14, 3.9 * len(rows)), squeeze=False,
                               subplot_kw={'projection': ccrs.PlateCarree(central_longitude=90)})
        cm = None
        for i, (mmdd, h) in enumerate(rows):
            for j, sv in enumerate(SOLV):
                f = f'{ind}/{mmdd}_{loc0}_{ob}_{sv}.npz'
                if not os.path.exists(f) or (mmdd, loc0, ob, sv) not in M:
                    ax[i, j].set_visible(False)
                    continue
                z = np.load(f)
                r = M[(mmdd, loc0, ob, sv)][h]
                cm = regional(ax[i, j], z, h, float(z['olat']), float(z['olon']),
                              f'{SNAME[sv].split(" (")[0]}, {mmdd}, {h} h\n|V|/|Vg| {r["ratio"]:.2f}, corr {r["corr"]:.2f}, '
                              f'small-scale {r["small_z"]:.0%}')
        if cm is None:
            plt.close(fig)
            return
        fig.colorbar(cm, ax=ax, shrink=0.7, label='z500 increment (m)')
        fig.suptitle(suptitle + '\nblack: actual 500-hPa wind increment, red: geostrophic wind of the z500 increment '
                     '(same arrow scale)', fontsize=11)
        fig.savefig(f'{odir}/{fn}', dpi=100, bbox_inches='tight')
        plt.close(fig)

    z0 = np.load(glob.glob(f'{ind}/*_{loc0}_*.npz')[0])
    where = f'{float(z0["olat"]):g}N {float(z0["olon"]):g}E'
    map_fig('S1_maps_ob_a.png', 'a', [(d, 6) for d in dates],
            f'Figure S1 ({title_model}). One z500 ob (+30 m at 6 h, {where}): the 6-h analysis increment per solver')
    map_fig('S2_maps_ob_b.png', 'b', [(dates[0], 6), (dates[0], 24)],
            f'Figure S2 ({title_model}). One z500 ob at 24 h (+30 m, {where}): increment at 6 h (top) and at the ob time 24 h (bottom)')
    # S4
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=90)})
    cm = None
    for a, sv in zip(ax, ('W', 'G')):
        f = f'{ind}/{dates[0]}_{loc0}_a_{sv}.npz'
        if not os.path.exists(f) or (dates[0], loc0, 'a', sv) not in M:
            continue
        z = np.load(f)
        cm = a.scatter(lon, lat, c=z['z_500_6'] / G0, s=2, cmap='RdBu_r', vmin=-10, vmax=10, transform=ccrs.PlateCarree())
        a.plot(float(z['olon']), float(z['olat']), 'y*', ms=12, mec='k', transform=ccrs.PlateCarree())
        a.coastlines(lw=0.5)
        a.set_title(f'{SNAME[sv]}: z500 increment at 6 h, energy beyond 3000 km {M[(dates[0], loc0, "a", sv)][6]["far_z"]:.0%}',
                    fontsize=10)
    if cm is not None:
        fig.colorbar(cm, ax=ax, shrink=0.8, label='m (colour saturates at 10 m; ob increment 30 m)')
    fig.suptitle(f'Figure S4 ({title_model}). Far-field response to one z500 ob at {where} ({dates[0]})', fontsize=11)
    fig.savefig(f'{odir}/S4_farfield.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    # S5
    fig, ax = plt.subplots(1, 5, figsize=(18, 4.8), sharey=True)
    sty = {'G': 'tab:red', 'BG': 'tab:purple', 'W': 'tab:blue'}
    for j, (b, unit, sc) in enumerate([('z', 'm', 1 / G0), ('t', 'K', 1), ('u', 'm/s', 1), ('v', 'm/s', 1), ('q', 'g/kg', 1e3)]):
        a = ax[j]
        for sv in SOLV:
            f = f'{ind}/{dates[0]}_{loc0}_a_{sv}.npz'
            if not os.path.exists(f):
                continue
            z = np.load(f)
            for h, ls in ((6, '-'), (24, ':')):
                a.plot(z[f'prof_{b}_{h}'] * sc, z[f'levels_{b}'], ls, color=sty[sv], label=f'{sv}, {h} h' if j == 0 else None)
        a.axvline(0, color='k', lw=0.5)
        a.set_yscale('log')
        a.set_ylim(1000, 50)
        a.set_yticks([1000, 850, 700, 500, 300, 200, 100, 50])
        a.set_yticklabels(['1000', '850', '700', '500', '300', '200', '100', '50'])
        a.set_title(f'{b} increment at the ob ({unit})', fontsize=10)
        a.grid(alpha=0.3)
    ax[0].set_ylabel('hPa')
    ax[0].legend(fontsize=7)
    fig.suptitle(f'Figure S5 ({title_model}). Vertical structure of the increment at the ob (z500 +30 m at 6 h, {where}, '
                 f'{dates[0]}); solid 6 h, dotted 24 h', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{odir}/S5_profiles.png', dpi=100)
    plt.close(fig)
    print('figures in', odir)


def S3(odir, models, per_case=False):
    """balance metrics at 6 h per solver; per_case: one marker per case (one model), else per-model median + range"""
    keys = [('ratio', '|V| / |V_g| (1 = geostrophic size)'), ('corr', 'pattern corr. with V_g'),
            ('small_z', 'z500 variance at < 400 km'), ('far_z', 'z500 energy beyond 3000 km')]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    mk = {'a': 'o', 'b': 's', 'c': '^'}
    for p, (key, lab) in enumerate(keys):
        a = ax[p]
        for mi, (label, so) in enumerate(models):
            if so is None:
                continue
            cases = {tuple(k.split('_')): v for k, v in so['metrics'].items()}
            if per_case:
                groups = sorted({(k[1], k[2]) for k in cases})
                for gi, (loc, ob) in enumerate(groups):
                    xs, ys = [], []
                    for j, sv in enumerate(SOLV):
                        for di, k in enumerate(sorted(c for c in cases if c[1:] == (loc, ob, sv))):
                            xs.append(j + 0.08 * (gi - len(groups) / 2) + 0.03 * di)
                            ys.append(cases[k]['6'][key] if '6' in cases[k] else cases[k][6][key])
                    a.plot(xs, ys, mk.get(ob, 'o'), color='tab:blue' if loc == 'sh' else 'tab:orange', ms=6, alpha=0.85,
                           label=f'{loc.upper()}, {OBN.get(ob, ob)}' if p == 0 else None)
            else:
                for j, sv in enumerate(SOLV):
                    v = [(c['6'] if '6' in c else c[6])[key] for k, c in cases.items() if k[3] == sv and k[2] == 'a']
                    if v:
                        x = j + 0.12 * (mi - (len(models) - 1) / 2)
                        a.errorbar(x, np.median(v), yerr=[[np.median(v) - min(v)], [max(v) - np.median(v)]], fmt='o',
                                   color=MODEL_COLORS[mi % 6], capsize=3, label=label if (p == 0 and j == 0) else None)
            if key in ('ratio', 'corr') and so.get('calibration_bg'):
                cv = [v[0 if key == 'ratio' else 1] for v in so['calibration_bg'].values()]
                a.axhspan(min(cv), max(cv), color='0.85', zorder=0, label='background itself' if (p == 0 and mi == 0) else None)
        a.set_xticks(range(len(SOLV)))
        a.set_xticklabels([s for s in SOLV])
        a.set_title(lab, fontsize=10)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, loc='upper left')
    what = 'per case' if per_case else 'median and range over dates x locations, ob a (z500 at 6 h)'
    fig.suptitle(f'Figure S3. Is the 6-h increment balanced? Per solver (G raw adjoint, BG B-filtered adjoint, '
                 f'W AdamW golden); within 2000 km of the ob; {what}', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{odir}/S3_metrics.png', dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------ compare models
def _med(arrs):
    a = np.array(arrs, dtype=float)
    return np.nanmedian(a, 0), np.nanmin(a, 0), np.nanmax(a, 0)


def compare(models, odir):
    """models: [(label, aq_results, single_ob_metrics_or_None)] -> Q1-Q4, S3 with one colour per model"""
    os.makedirs(odir, exist_ok=True)
    # Q1: Taylor ratio at the last lead along steepest descent and the golden direction
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, name in zip(ax, ('-g32', 'golden')):
        for mi, (label, R, _) in enumerate(models):
            rows = [r['taylor'][name]['fp32_' + _lastlead(r)] for r in R if name in r['taylor']]
            if not rows:
                continue
            m, lo, hi = _med(rows)
            x = R[0]['taylor']['eps_frac']
            a.plot(x, m, '-o', ms=3, color=MODEL_COLORS[mi % 6], label=label)
            a.fill_between(x, lo, hi, color=MODEL_COLORS[mi % 6], alpha=0.15)
        a.axhline(1, color='k', lw=0.8)
        if any(_tf32(r) for _, R, _ in models for r in R):
            a.axvspan(7e-5, 2e-3, color='0.9', zorder=0)
        a.set_xscale('log')
        a.set_ylim(-0.5, 2.0)
        a.set_xlabel('step size / reference increment norm')
        a.set_title(LAB[name])
    ax[0].set_ylabel('actual / gradient-predicted change in J (last lead)')
    ax[0].legend(fontsize=8)
    fig.suptitle('Figure Q1 (compare). Taylor test: median and range over dates')
    fig.tight_layout()
    fig.savefig(f'{odir}/Q1_taylor.png', dpi=110)
    plt.close(fig)
    # Q2: cosines
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = ['reduced vs fp32', 'reduced repeat', 'fp32 repeat', 'moved 1e-3', 'moved 1e-2', 'moved 1e-1', 'at golden u']
    for mi, (label, R, _) in enumerate(models):
        rows = []
        for r in R:
            c, st = r['cos'][_lastlead(r)], r['stability']
            k = _lastlead(r)
            rows.append([c.get('bf16_vs_fp32', np.nan), c.get('bf16_repeat', np.nan), c.get('fp32_repeat', np.nan)]
                        + [st.get(s, {}).get(k, np.nan) for s in ('rand_0.001', 'rand_0.01', 'rand_0.1', 'at_golden')])
        m, lo, hi = _med(rows)
        x = np.arange(len(labels)) + 0.1 * (mi - (len(models) - 1) / 2)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt='o', capsize=3, color=MODEL_COLORS[mi % 6], label=label)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.axhline(1, color='k', lw=0.8)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel('cosine with fp32 gradient at u=0 (last lead)')
    ax.legend(fontsize=8)
    fig.suptitle('Figure Q2 (compare). Gradient repeatability, precision and smoothness: median and range over dates')
    fig.tight_layout()
    fig.savefig(f'{odir}/Q2_cosines.png', dpi=110)
    plt.close(fig)
    # Q3: survival of golden / steepest descent / random (normalised norm, total)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for a, name in zip(ax, ('golden_twin', '-g', 'random')):
        for mi, (label, R, _) in enumerate(models):
            rows = [np.array(r['survival'][name]['norm']) / r['survival'][name]['norm'][0] for r in R if name in r['survival']]
            if not rows:
                continue
            n = min(len(x) for x in rows)
            m, lo, hi = _med([x[:n] for x in rows])
            lead = R[0]['survival']['lead_h'][:n]
            a.plot(lead, m, color=MODEL_COLORS[mi % 6], label=label)
            a.fill_between(lead, lo, hi, color=MODEL_COLORS[mi % 6], alpha=0.15)
        a.axhline(1, color='k', lw=0.8)
        a.set_yscale('log')
        a.set_xlabel('lead (h)')
        a.set_title({'golden_twin': 'golden increment', '-g': 'steepest descent', 'random': 'random latent'}[name])
    ax[0].set_ylabel('perturbation size / size at 6 h')
    ax[0].legend(fontsize=8)
    fig.suptitle('Figure Q3 (compare). Perturbation survival (all prognostic, normalised): median and range over dates')
    fig.tight_layout()
    fig.savefig(f'{odir}/Q3_survival.png', dpi=110)
    plt.close(fig)
    # Q4
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, tag in zip(ax, ('all', 'noz')):
        for mi, (label, R, _) in enumerate(models):
            rows = [r[f'svd_48h_{tag}']['singular_values'] for r in R if f'svd_48h_{tag}' in r]
            if not rows:
                continue
            n = min(len(x) for x in rows)
            m, lo, hi = _med([x[:n] for x in rows])
            a.plot(range(1, n + 1), m, 'o-', ms=3, color=MODEL_COLORS[mi % 6], label=label)
            a.fill_between(range(1, n + 1), lo, hi, color=MODEL_COLORS[mi % 6], alpha=0.15)
            a.axhline(np.median([np.median(r[f'random_amplification_48h_{tag}']) for r in R]),
                      color=MODEL_COLORS[mi % 6], ls=':', lw=1)
        a.axhline(1, color='k', lw=0.8)
        a.set_yscale('log')
        a.set_xlabel('singular vector rank')
        a.set_title('perturb every prognostic variable' if tag == 'all' else 'perturb all but geopotential', fontsize=10)
    ax[0].set_ylabel('amplification 6 h -> 48 h')
    ax[0].legend(fontsize=8)
    fig.suptitle('Figure Q4 (compare). Leading singular values of the 6 h -> 48 h map (median and range over dates; '
                 'dotted: random ~400-km perturbations)', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{odir}/Q4_singular_values.png', dpi=110)
    plt.close(fig)
    if any(so for _, _, so in models):
        S3(odir, [(label, so) for label, _, so in models])
    print('compare figures in', odir)
