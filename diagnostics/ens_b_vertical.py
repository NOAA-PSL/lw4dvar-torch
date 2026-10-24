"""Vertical structure of single surface-pressure-ob increments with the
static-ensemble latent B (latent_b.EnsembleB): cross-sections through the ob
of the 8 ACE2 layers' T, u, v and total water, plotted against the layers'
mid-point pressure on the section (ak + bk * ps of the background), plus the
rms increment by layer near the ob (< 1000 km) and far from it (2000-4000 km),
to judge whether vertical localization is needed.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_latent_b import utils  # noqa: E402
from check_latent_b_cov import POINTS  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_b import EnsembleB  # noqa: E402

FAMS = [('northward_wind', 'v (m/s)', 'lon'), ('eastward_wind', 'u (m/s)', 'lat'),
        ('air_temperature', 'T (K)', 'lon'), ('specific_total_water', 'q (g/kg)', 'lon')]
NL = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', required=True)
    ap.add_argument('--path_model', required=True)
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--locs', default='1500')
    ap.add_argument('--single_ob_date', default='2015-02-01T00')
    args = ap.parse_args()
    locs = [float(x) for x in args.locs.split(',')]

    path_model = os.path.join(args.path_model, '')
    model = utils.get_model({'model_backend': 'ace2', 'path_model': path_model, 'device': 'cuda'})
    dev = model.device
    d = torch.load(args.samples, weights_only=False)
    meta = d['meta']
    samples = d['samples'].to(dev)
    del d
    C, H, W = model.latent_shape
    lat = np.asarray(model.lats).reshape(H, W)[:, 0]
    lon = np.asarray(model.lons).reshape(H, W)[0]
    ci = model._chan_idx
    ak, bk = model._ak.double().cpu().numpy(), model._bk.double().cpu().numpy()

    def step(s, inc=None):
        return model.advance(s, steps=1, use_checkpoint=False,
                             latent_increment=None if inc is None else inc.to(torch.float32)).state[0].double()

    with open(glob.glob(os.path.join(meta['run_dir'], f'{args.single_ob_date}_control_inputs_120h_50it.pkl'))[0], 'rb') as fh:
        s_b = pickle.load(fh)
    s_b.state = s_b.state.to(dev)
    with torch.no_grad():
        x0 = step(s_b)
    ps0 = x0[ci['PRESsfc']].cpu().numpy()
    d_ob, r_ob = 100.0, 100.0 ** 2
    res = {}
    for loc in locs:
        B = EnsembleB(samples, loc, meta['sht_grid']).to(dev)
        for plat, plon, name in POINTS[:3]:
            i = int(np.abs(lat - plat).argmin())
            j = int(np.abs(lon - plon).argmin())
            v = torch.zeros(B.control_shape, dtype=torch.float64, device=dev, requires_grad=True)
            step(s_b, B(v))[ci['PRESsfc'], i, j].backward()
            g = v.grad.detach()
            hbh = (g * g).sum().item()
            with torch.no_grad():
                dx = (step(s_b, B(g * (d_ob / (hbh + r_ob)))) - x0).cpu().numpy()
            res[(name, loc)] = {'ij': (i, j), 'dx': {f: np.stack([dx[ci[f'{f}_{k}']] for k in range(NL)]) for f, _, _ in FAMS},
                                'dps': dx[ci['PRESsfc']], 'h500': dx[ci['h500']]}
            # rms by layer near / far from the ob (great-circle distance)
            phi0, lam0 = np.deg2rad(lat[i]), np.deg2rad(lon[j])
            phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
            dist = 6371 * np.arccos(np.clip(np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(lam - lam0), -1, 1))
            near, far = dist < 1000, (dist > 2000) & (dist < 4000)
            print(f"\n{name}, loc {loc:.0f} km: dps at ob {dx[ci['PRESsfc']][i, j] / 100:+.2f} hPa; "
                  f"rms increment by layer, near (<1000 km) / far (2000-4000 km); p_mid at ob (hPa)")
            for k in range(NL):
                pm = 0.5 * (ak[k] + bk[k] * ps0[i, j] + ak[k + 1] + bk[k + 1] * ps0[i, j]) / 100
                row = "  ".join(f"{f[:5]} {np.sqrt((res[(name, loc)]['dx'][f][k][near] ** 2).mean()) * (1e3 if f.startswith('spec') else 1):.3f}"
                                f"/{np.sqrt((res[(name, loc)]['dx'][f][k][far] ** 2).mean()) * (1e3 if f.startswith('spec') else 1):.3f}"
                                for f, _, _ in FAMS)
                print(f"  layer {k} ({pm:6.1f}): {row}", flush=True)
    torch.save({'res': res, 'lat': lat, 'lon': lon, 'ak': ak, 'bk': bk, 'ps0': ps0}, args.out + '.pt')
    for loc in locs:
        plot(res, lat, lon, ak, bk, ps0, loc, f"{args.out}_loc{loc:.0f}.png", args.single_ob_date)
    print(f"wrote {args.out}.pt and plots")


def plot(res, lat, lon, ak, bk, ps0, loc, path, date):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pts = POINTS[:3]
    f, axs = plt.subplots(len(pts), len(FAMS), figsize=(5.0 * len(FAMS), 3.9 * len(pts)), squeeze=False)
    for r, (plat, plon, name) in enumerate(pts):
        rec = res[(name, loc)]
        i, j = rec['ij']
        for c, (fam, label, along) in enumerate(FAMS):
            ax = axs[r, c]
            fld = rec['dx'][fam] * (1e3 if fam.startswith('spec') else 1)       # (NL, H, W)
            if along == 'lon':
                jj = (np.arange(-30, 31) + j) % len(lon)
                x = lon[j] + ((lon[jj] - lon[j] + 180) % 360 - 180)
                sec, ps = fld[:, i, jj], ps0[i, jj]
                xl = f'longitude (section at {lat[i]:.1f})'
            else:
                ii = np.arange(max(i - 30, 0), min(i + 31, len(lat)))
                x = lat[ii]
                sec, ps = fld[:, ii, j], ps0[ii, j]
                xl = f'latitude (section at {lon[j]:.1f}E)'
            pmid = 0.5 * (ak[:-1, None] + bk[:-1, None] * ps[None] + ak[1:, None] + bk[1:, None] * ps[None]) / 100
            X = np.broadcast_to(x[None], pmid.shape)
            vmax = np.abs(sec).max()
            cf = ax.contourf(X, pmid, sec, levels=np.linspace(-vmax, vmax, 21), cmap='RdBu_r')
            ax.plot(x, ps / 100, 'k-', lw=0.8)
            for k in range(pmid.shape[0]):
                ax.plot(x, pmid[k], ':', color='gray', lw=0.5)
            ax.axvline(x[np.argmin(np.abs(x - (lon[j] if along == 'lon' else lat[i])))], color='k', lw=0.6, ls='--')
            ax.set_ylim(ps.max() / 100 + 10, 0)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel('layer mid-point pressure (hPa)', fontsize=8)
            plt.colorbar(cf, ax=ax, shrink=0.85)
            ax.set_title(f"{name}: {label} increment", fontsize=9)
    f.suptitle(f'static-ensemble latent B, loc {loc:.0f} km: single PRESsfc ob (O-B +1 hPa, sigma_o 1 hPa), '
               f'cross-sections through the ob, background {date} + 6h (dotted: the 8 ACE2 layer mid-points)')
    plt.tight_layout()
    plt.savefig(path, dpi=90)


if __name__ == '__main__':
    main()
