"""Check the static-ensemble latent B (latent_b.EnsembleB) for several
localization scales:

1. Single surface-pressure ob (as check_latent_b_cov.py): O-B = +1 hPa,
   sigma_o = 1 hPa, one cycled background; increment B H^T (H B H^T + R)^-1 d
   via autograd through the control transform and one ACE2 step; lowest-layer
   and 10 m wind vs the geostrophic wind of the PRESsfc increment. Plot: rows
   = points, columns = localization scales.
2. Decoded random draws vs the real latent errors: rms of the one-step
   physical increment per field and the fraction of variance at l > 40.
3. Cost: GPU memory of the samples and time for B (forward + backward).
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_latent_b import LatentProbe, era5_state, l_power, utils, ace2_ic, th  # noqa: E402
from check_latent_b_cov import POINTS, RD, FIELDS, geostrophic, geo_stats  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent_b import EnsembleB  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', required=True, help='build_latent_ens_b.py output')
    ap.add_argument('--path_model', required=True)
    ap.add_argument('--dataset_dir', default='/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer')
    ap.add_argument('--out', required=True, help='output prefix (.pt, .png)')
    ap.add_argument('--locs', default='1000,1500,2000,3000', help='localization scales (km), comma separated')
    ap.add_argument('--n_test', type=int, default=6)
    ap.add_argument('--single_ob_date', default='2015-02-01T00')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    locs = [float(x) for x in args.locs.split(',')]

    path_model = os.path.join(args.path_model, '')
    model = utils.get_model({'model_backend': 'ace2', 'path_model': path_model, 'device': 'cuda'})
    dev = model.device
    d = torch.load(args.samples, weights_only=False)
    meta = d['meta']
    ckpt = os.path.join(path_model, 'ace2_era5_ckpt.tar')
    print(f"{meta['n_samples']} samples, checkpoint {meta['checkpoint']} (sha {meta['checkpoint_sha256'][:12]}); "
          f"running {ckpt}", flush=True)
    m0 = torch.cuda.memory_allocated()
    samples = d['samples'].to(dev)
    del d
    print(f"samples on GPU: {(torch.cuda.memory_allocated() - m0) / 1e9:.1f} GB", flush=True)
    C, H, W = model.latent_shape
    grid = meta['sht_grid']
    hgt = ace2_ic.load_hgtsfc(path_model + 'forcing_data')
    probe = LatentProbe(model)
    lat = np.asarray(model.lats).reshape(H, W)[:, 0]
    lon = np.asarray(model.lons).reshape(H, W)[0]
    areaw = torch.as_tensor(model.area_weights, device=dev, dtype=torch.float64).reshape(H, W)
    areaw = areaw / areaw.sum()
    psht = th.RealSHT(H, W, grid=grid, norm='ortho').to(dev).double()
    lphys = torch.arange(psht.lmax, device=dev)
    ci = model._chan_idx
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    Bs = {loc: EnsembleB(samples, loc, grid).to(dev) for loc in locs}

    def step(s, inc=None):
        return model.advance(s, steps=1, use_checkpoint=False,
                             latent_increment=None if inc is None else inc.to(torch.float32)).state[0].double()

    # ---- 3. cost
    for loc, B in Bs.items():
        v = torch.randn(B.control_shape, dtype=torch.float64, device=dev, requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        for _ in range(5):
            z = B(v)
            z.pow(2).sum().backward()
        torch.cuda.synchronize()
        print(f"loc {loc:.0f} km: control {tuple(B.control_shape)} ({v.numel()} values), "
              f"B fwd+bwd {(time.time() - t0) / 5 * 1e3:.1f} ms, peak extra memory "
              f"{(torch.cuda.max_memory_allocated() - base) / 1e9:.2f} GB", flush=True)

    # ---- 1. single ob
    with open(glob.glob(os.path.join(meta['run_dir'], f'{args.single_ob_date}_control_inputs_*.pkl'))[0], 'rb') as fh:
        s_b = pickle.load(fh)
    s_b.state = s_b.state.to(dev)
    with torch.no_grad():
        x0 = step(s_b)
    rho = (x0[ci['PRESsfc']] / (RD * x0[ci['air_temperature_7']])).cpu().numpy()
    d_ob, r_ob = 100.0, 100.0 ** 2
    single = {}
    print(f"\nsingle ob: O-B = +1 hPa, sigma_o = 1 hPa, background {args.single_ob_date} (+6h)")
    for plat, plon, name in POINTS:
        i = int(np.abs(lat - plat).argmin())
        j = int(np.abs(lon - plon).argmin())
        for loc, B in Bs.items():
            v = torch.zeros(B.control_shape, dtype=torch.float64, device=dev, requires_grad=True)
            y = step(s_b, B(v))
            y[ci['PRESsfc'], i, j].backward()
            g = v.grad.detach()
            hbh = (g * g).sum().item()
            with torch.no_grad():
                dx = (step(s_b, B(g * (d_ob / (hbh + r_ob)))) - x0).cpu().numpy()
            dp = dx[ci['PRESsfc']]
            ug, vg = geostrophic(dp, lat, lon, rho)
            box = (np.abs(lat - lat[i])[:, None] <= 20) & (np.abs(((lon - lon[j] + 180) % 360) - 180)[None, :] <= 30)
            sel = box & (np.abs(lat)[:, None] >= 10) & (np.abs(dp) >= 0.1 * np.abs(dp[box]).max())
            st = {lev: geo_stats(dx[ci[u]], dx[ci[vv]], ug, vg, sel) if sel.any() else (np.nan,) * 3
                  for lev, (u, vv) in {'layer7': ('eastward_wind_7', 'northward_wind_7'),
                                       '10m': ('UGRD10m', 'VGRD10m')}.items()}
            spd = np.hypot(dx[ci['eastward_wind_7']], dx[ci['northward_wind_7']])
            single[(name, loc)] = {'hbh': hbh, 'ij': (i, j), 'geo': st,
                                   **{k: dx[ci[k]] for k in ('PRESsfc', 'eastward_wind_7', 'northward_wind_7',
                                                             'UGRD10m', 'VGRD10m', 'air_temperature_7', 'h500')}}
            print(f"  {name:26s} loc {loc:5.0f} km: sqrt(HBH^T) {np.sqrt(hbh) / 100:.2f} hPa, dps at ob {dp[i, j] / 100:+.2f}; "
                  f"max |dV| layer7 {spd[box].max():.2f} m/s; geostrophic corr/speed ratio/angle: "
                  f"layer7 {st['layer7'][0]:.2f}/{st['layer7'][1]:.2f}/{st['layer7'][2]:+.0f}deg, "
                  f"10m {st['10m'][0]:.2f}/{st['10m'][1]:.2f}/{st['10m'][2]:+.0f}deg", flush=True)

    # ---- 2. decoded draws
    run_files = sorted(glob.glob(os.path.join(meta['run_dir'], '*_control_inputs_*.pkl')))[meta['skip']:]
    files = [run_files[i] for i in np.linspace(0, len(run_files) - 1, args.n_test).astype(int)]
    cases = ['real'] + [f'ens{loc:.0f}' for loc in locs]
    rms = {k: {f: [] for f in FIELDS} for k in cases}
    hi = {k: {f: [] for f in FIELDS} for k in cases}
    zr = {k: [] for k in cases}
    for f in files:
        with open(f, 'rb') as fh:
            s_b = pickle.load(fh)
        s_b.state = s_b.state.to(dev)
        s_t = era5_state(model, args.dataset_dir, s_b.date, hgt)
        with torch.no_grad():
            incs = {'real': probe(s_t) - probe(s_b)}
            for loc, B in Bs.items():
                incs[f'ens{loc:.0f}'] = B(torch.randn(B.control_shape, generator=gen, dtype=torch.float64, device=dev))
            x0 = step(s_b)
            for k, inc in incs.items():
                dx = step(s_b, inc) - x0
                zr[k].append(inc.float().pow(2).mean().sqrt().item())
                for fn in FIELDS:
                    di = dx[ci[fn]]
                    rms[k][fn].append(torch.sqrt((areaw * di ** 2).sum()).item())
                    p = l_power(psht(di[None]))[0]
                    hi[k][fn].append((p[lphys > 40].sum() / p.sum()).item())
        print(f"{s_b.date:%Y-%m-%dT%H}: latent rms " + " ".join(f"{k} {zr[k][-1]:.3f}" for k in cases), flush=True)
    print(f"\ndecoded one-step increments, mean over {len(files)} cycles: rms [frac. variance at l>40]")
    for fn in FIELDS:
        print(f"  {fn:24s} " + "  ".join(f"{k} {np.mean(rms[k][fn]):9.4g} [{np.mean(hi[k][fn]):.2f}]" for k in cases))

    torch.save({'single': single, 'rms': rms, 'hi_frac': hi, 'latent_rms': zr, 'locs': locs, 'fields': FIELDS,
                'lat': lat, 'lon': lon, 'samples_meta': meta}, args.out + '.pt')
    plot(single, lat, lon, locs, args.out + '.png', args.single_ob_date, meta['n_samples'])
    print(f"wrote {args.out}.pt, {args.out}.png")


def plot(single, lat, lon, locs, path, date, N):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(len(POINTS), len(locs), figsize=(5.2 * len(locs), 4.0 * len(POINTS)), squeeze=False)
    for r, (plat, plon, name) in enumerate(POINTS):
        for c, loc in enumerate(locs):
            rec = single[(name, loc)]
            ax = axs[r, c]
            i, j = rec['ij']
            jj = (np.arange(-40, 41) + j) % len(lon)
            ii = np.arange(max(i - 25, 0), min(i + 26, len(lat)))
            x = lon[j] + ((lon[jj] - lon[j] + 180) % 360 - 180)
            y = lat[ii]
            dp = rec['PRESsfc'][np.ix_(ii, jj)] / 100
            vmax = np.abs(dp).max()
            cf = ax.contourf(x, y, dp, levels=np.linspace(-vmax, vmax, 21), cmap='RdBu_r')
            plt.colorbar(cf, ax=ax, shrink=0.8)
            u, v = rec['eastward_wind_7'][np.ix_(ii, jj)], rec['northward_wind_7'][np.ix_(ii, jj)]
            s = slice(None, None, 3)
            vm = max(np.hypot(u, v).max(), 1e-6)
            q = ax.quiver(x[s], y[s], u[s, s], v[s, s], scale_units='xy', angles='xy', scale=vm / 4)
            ax.quiverkey(q, 0.8, 1.02, vm, f"{vm:.2f} m/s", labelpos='E', fontproperties={'size': 8})
            ax.plot(lon[j], lat[i], 'k*', ms=10)
            g = rec['geo']['layer7']
            ax.set_title(f"{name}, loc {loc:.0f} km\nsqrt(HBH') {np.sqrt(rec['hbh']) / 100:.2f} hPa; geo corr {g[0]:.2f}, "
                         f"|V|/|Vg| {g[1]:.2f}", fontsize=8)
            ax.set_aspect(1 / np.cos(np.deg2rad(lat[i])))
    f.suptitle(f'static-ensemble latent B ({N} samples): single PRESsfc ob (O-B +1 hPa, sigma_o 1 hPa), '
               f'PRESsfc increment (hPa) and lowest-layer wind, background {date} + 6h')
    plt.tight_layout()
    plt.savefig(path, dpi=90)


if __name__ == '__main__':
    main()
