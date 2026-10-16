"""Check the spectral latent B with channel covariance (latent_b.SpectralChannelB)
against the channel-diagonal version:

1. Band statistics of the channel covariance (eigen-spectrum, effective rank).
2. Decoded random draws vs the real latent errors (as check_latent_b.py):
   area-weighted rms of the one-step physical increment per field and its
   variance fraction at l > 40.
3. Single surface-pressure observation: d = O - B = +1 hPa, sigma_o = 1 hPa,
   at a few points, from one cycled background. Latent increment
   dz = B H^T d / (H B H^T + R), with H^T from autograd through the control
   transform and one ACE2 step (H: PRESsfc at the nearest grid point after
   the first 6h step, where the increment first appears). The decoded
   increment is compared with the geostrophic wind of its PRESsfc increment
   (lowest layer and 10 m), and plotted.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_latent_b import LatentProbe, era5_state, l_power, utils, ace2_ic, th  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_b import SpectralChannelB  # noqa: E402

FIELDS = ['PRESsfc', 'surface_temperature', 'h500', 'air_temperature_2', 'air_temperature_6', 'air_temperature_7',
          'eastward_wind_2', 'eastward_wind_6', 'eastward_wind_7', 'specific_total_water_6', 'TMP2m', 'UGRD10m']
POINTS = [(45.0, 190.0, 'N Pacific 45N 170W'), (40.0, 265.0, 'central US 40N 95W'),
          (-50.0, 90.0, 'S Indian Ocean 50S 90E'), (5.0, 190.0, 'tropical Pacific 5N 170W')]
OMEGA, A_EARTH, RD = 7.292e-5, 6.371e6, 287.05


def geostrophic(dp, lat, lon, rho):
    """Geostrophic wind of a pressure field dp (H, W) in Pa."""
    phi = np.deg2rad(lat)[:, None]
    dpdy = np.gradient(dp, np.deg2rad(lat), axis=0) / A_EARTH
    dlon = np.deg2rad(lon[1] - lon[0])
    dpdx = (np.roll(dp, -1, 1) - np.roll(dp, 1, 1)) / (2 * dlon) / (A_EARTH * np.cos(phi))
    f = 2 * OMEGA * np.sin(phi)
    return -dpdy / (rho * f), dpdx / (rho * f)


def geo_stats(u, v, ug, vg, sel):
    dot = (u * ug + v * vg)[sel].sum()
    crs = (ug * v - vg * u)[sel].sum()
    n2, g2 = (u * u + v * v)[sel].sum(), (ug * ug + vg * vg)[sel].sum()
    return dot / np.sqrt(n2 * g2), np.sqrt(n2 / g2), np.degrees(np.arctan2(crs, dot))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--b_file', required=True)
    ap.add_argument('--path_model', required=True)
    ap.add_argument('--dataset_dir', default='/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer')
    ap.add_argument('--out', required=True, help='output prefix (.pt, .png)')
    ap.add_argument('--n_test', type=int, default=6)
    ap.add_argument('--shrink', type=float, default=0.05)
    ap.add_argument('--single_ob_date', default='2015-02-01T00')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    b = torch.load(args.b_file, weights_only=False)
    meta = b['meta']
    C, H, W = meta['latent_shape']
    L = meta['lmax']
    cross = b['cross']
    w = 2.0 * torch.arange(L, dtype=torch.float64) + 1.0
    diag_err = (torch.diagonal(cross.double(), dim1=1, dim2=2).T - b['sigma2'].double()).abs().max() / b['sigma2'].max()
    print(f"{meta['n_samples']} samples (skip {meta.get('skip')}); diag(cross) vs sigma2 max rel diff {diag_err:.2e}", flush=True)
    Bc = SpectralChannelB(cross, H, W, meta['sht_grid'], shrink=args.shrink, channel_cov=True)
    Bd = SpectralChannelB(cross, H, W, meta['sht_grid'], channel_cov=False)
    print(f"bands {Bc.bands}; shrink {args.shrink}")
    for (lo, hi), e in zip(Bc.bands, Bc.band_eigs):
        f = (e / e.sum()).cumsum(0)
        pr = (e.sum() ** 2 / (e ** 2).sum()).item()
        nsamp = meta['n_samples'] * (w[lo:hi].sum().item())
        print(f"  l {lo:3d}-{hi - 1:3d}: coeff samples {nsamp:9.0f}; top eig {f[0].item():.3f} of var, "
              f"modes for 50/90/99%: {int((f < .5).sum()) + 1}/{int((f < .9).sum()) + 1}/{int((f < .99).sum()) + 1}, "
              f"participation ratio {pr:.1f}, cond {(e[0] / e[-1].clamp(min=1e-30)).item():.2e}")

    path_model = os.path.join(args.path_model, '')
    model = utils.get_model({'model_backend': 'ace2', 'path_model': path_model, 'device': 'cuda'})
    dev = model.device
    Bc, Bd = Bc.to(dev), Bd.to(dev)
    psht = th.RealSHT(H, W, grid=meta['sht_grid'], norm='ortho').to(dev).double()
    hgt = ace2_ic.load_hgtsfc(path_model + 'forcing_data')
    probe = LatentProbe(model)
    assert tuple(model.latent_shape) == (C, H, W) and np.asarray(model.lats).size == H * W
    lat = np.asarray(model.lats).reshape(H, W)[:, 0]
    lon = np.asarray(model.lons).reshape(H, W)[0]
    areaw = torch.as_tensor(model.area_weights, device=dev, dtype=torch.float64).reshape(H, W)
    areaw = areaw / areaw.sum()
    idx = {f: model._chan_idx[f] for f in FIELDS}
    lphys = torch.arange(psht.lmax, device=dev)
    ci = model._chan_idx
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    def step(s, inc=None):
        return model.advance(s, steps=1, use_checkpoint=False,
                             latent_increment=None if inc is None else inc.to(torch.float32)).state[0].double()

    # ---- 2. decoded draws
    run_files = sorted(glob.glob(os.path.join(meta['run_dir'], '*_control_inputs_120h_50it.pkl')))[meta.get('skip') or 0:]
    files = [run_files[i] for i in np.linspace(0, len(run_files) - 1, args.n_test).astype(int)]
    cases = ['real', 'B_diag', 'B_cov']
    rms = {k: {f: [] for f in FIELDS} for k in cases}
    hi = {k: {f: [] for f in FIELDS} for k in cases}
    zr = {k: [] for k in cases}
    for f in files:
        with open(f, 'rb') as fh:
            s_b = pickle.load(fh)
        s_b.state = s_b.state.to(dev)
        s_t = era5_state(model, args.dataset_dir, s_b.date, hgt)
        with torch.no_grad():
            e = probe(s_t) - probe(s_b)
            x0 = step(s_b)
            v = torch.randn(Bc.control_shape, generator=gen, dtype=torch.float64, device=dev)
            incs = {'real': e, 'B_diag': Bd(v), 'B_cov': Bc(v)}
            for k, inc in incs.items():
                dx = step(s_b, inc) - x0
                zr[k].append(inc.pow(2).mean().sqrt().item())
                for fn, i in idx.items():
                    rms[k][fn].append(torch.sqrt((areaw * dx[i] ** 2).sum()).item())
                    p = l_power(psht(dx[i][None]))[0]
                    hi[k][fn].append((p[lphys > 40].sum() / p.sum()).item())
        print(f"{s_b.date:%Y-%m-%dT%H}: latent rms real {zr['real'][-1]:.3f} diag {zr['B_diag'][-1]:.3f} "
              f"cov {zr['B_cov'][-1]:.3f}", flush=True)
    print(f"\ndecoded one-step increments, mean over {len(files)} cycles: rms [frac. variance at l>40]")
    for fn in FIELDS:
        print(f"  {fn:24s} " + "  ".join(f"{k} {np.mean(rms[k][fn]):9.4g} [{np.mean(hi[k][fn]):.2f}]" for k in cases))

    # ---- 3. single surface-pressure observation
    with open(glob.glob(os.path.join(meta['run_dir'], f'{args.single_ob_date}_control_inputs_120h_50it.pkl'))[0], 'rb') as fh:
        s_b = pickle.load(fh)
    s_b.state = s_b.state.to(dev)
    with torch.no_grad():
        x0 = step(s_b)
    rho = (x0[ci['PRESsfc']] / (RD * x0[ci['air_temperature_7']])).cpu().numpy()
    d_ob, r_ob = 100.0, 100.0 ** 2      # +1 hPa innovation, sigma_o 1 hPa (Pa units)
    single = {}
    print(f"\nsingle ob: O-B = +1 hPa, sigma_o = 1 hPa, background {args.single_ob_date} (+6h)")
    for plat, plon, name in POINTS:
        i = int(np.abs(lat - plat).argmin())
        j = int(np.abs(lon - plon).argmin())
        for bname, Bop in (('B_diag', Bd), ('B_cov', Bc)):
            v = torch.zeros(Bop.control_shape, dtype=torch.float64, device=dev, requires_grad=True)
            y = step(s_b, Bop(v))
            y[ci['PRESsfc'], i, j].backward()
            g = v.grad.detach()
            hbh = (g * g).sum().item()
            with torch.no_grad():
                dx = (step(s_b, Bop(g * (d_ob / (hbh + r_ob)))) - x0).cpu().numpy()
            dp = dx[ci['PRESsfc']]
            rec = {'hbh': hbh, 'ij': (i, j), 'lat0': lat[i], 'lon0': lon[j],
                   **{k: dx[ci[k]] for k in ('PRESsfc', 'eastward_wind_7', 'northward_wind_7', 'UGRD10m', 'VGRD10m',
                                             'air_temperature_7', 'h500', 'eastward_wind_2', 'northward_wind_2')}}
            ug, vg = geostrophic(dp, lat, lon, rho)
            box = (np.abs(lat - lat[i])[:, None] <= 20) & (np.abs(((lon - lon[j] + 180) % 360) - 180)[None, :] <= 30)
            sel = box & (np.abs(lat)[:, None] >= 10) & (np.abs(dp) >= 0.1 * np.abs(dp[box]).max())
            st = {}
            for lev, (uk, vk) in {'layer7': ('eastward_wind_7', 'northward_wind_7'), '10m': ('UGRD10m', 'VGRD10m')}.items():
                st[lev] = geo_stats(dx[ci[uk]], dx[ci[vk]], ug, vg, sel) if sel.any() else (np.nan,) * 3
            rec['geo'] = st
            single[(name, bname)] = rec
            spd = np.hypot(dx[ci['eastward_wind_7']], dx[ci['northward_wind_7']])
            print(f"  {name:26s} {bname:6s}: sqrt(HBH^T) {np.sqrt(hbh) / 100:.2f} hPa, dps at ob {dp[i, j] / 100:+.2f} hPa, "
                  f"max |dps| elsewhere {np.abs(dp[box]).max() / 100:.2f}; max |dV| layer7 {spd[box].max():.2f} m/s; "
                  f"geostrophic corr/speed ratio/angle: layer7 {st['layer7'][0]:.2f}/{st['layer7'][1]:.2f}/{st['layer7'][2]:+.0f}deg, "
                  f"10m {st['10m'][0]:.2f}/{st['10m'][1]:.2f}/{st['10m'][2]:+.0f}deg", flush=True)

    torch.save({'rms': rms, 'hi_frac': hi, 'latent_rms': zr, 'fields': FIELDS, 'single': single, 'lat': lat, 'lon': lon,
                'band_eigs': [e.cpu() for e in Bc.band_eigs], 'bands': Bc.bands, 'shrink': args.shrink}, args.out + '.pt')
    plot(single, lat, lon, args.out + '.png', args.single_ob_date)
    print(f"wrote {args.out}.pt, {args.out}.png")


def plot(single, lat, lon, path, date):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(len(POINTS), 2, figsize=(13, 4.2 * len(POINTS)))
    for r, (plat, plon, name) in enumerate(POINTS):
        for c, bname in enumerate(('B_diag', 'B_cov')):
            rec = single[(name, bname)]
            ax = axs[r, c]
            i, j = rec['ij']
            jj = (np.arange(-40, 41) + j) % len(lon)
            ii = np.arange(max(i - 25, 0), min(i + 26, len(lat)))
            x = lon[j] + ((lon[jj] - lon[j] + 180) % 360 - 180)
            y = lat[ii]
            dp = rec['PRESsfc'][np.ix_(ii, jj)] / 100
            vmax = np.abs(dp).max()
            cf = ax.contourf(x, y, dp, levels=np.linspace(-vmax, vmax, 21), cmap='RdBu_r')
            plt.colorbar(cf, ax=ax, label='PRESsfc increment (hPa)')
            u, v = rec['eastward_wind_7'][np.ix_(ii, jj)], rec['northward_wind_7'][np.ix_(ii, jj)]
            s = slice(None, None, 3)
            q = ax.quiver(x[s], y[s], u[s, s], v[s, s], scale_units='xy', angles='xy',
                          scale=np.hypot(u, v).max() / 4 if np.hypot(u, v).max() > 0 else 1)
            ax.quiverkey(q, 0.85, 1.03, np.hypot(u, v).max(), f"{np.hypot(u, v).max():.2f} m/s", labelpos='E')
            ax.plot(lon[j], lat[i], 'k*', ms=12)
            g = rec['geo']['layer7']
            ax.set_title(f"{name}, {bname}\nsqrt(HBH') {np.sqrt(rec['hbh']) / 100:.2f} hPa; geostrophic corr {g[0]:.2f}, "
                         f"|V|/|Vg| {g[1]:.2f}, angle {g[2]:+.0f} deg", fontsize=9)
            ax.set_aspect(1 / np.cos(np.deg2rad(lat[i])))
    f.suptitle(f'single PRESsfc ob (O-B = +1 hPa, sigma_o 1 hPa), lowest-layer wind increment, background {date} + 6h')
    plt.tight_layout()
    plt.savefig(path, dpi=100)


if __name__ == '__main__':
    main()
