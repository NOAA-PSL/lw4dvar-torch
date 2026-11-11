"""Single-ob test of NMC static-ensemble B files (build_jax_nmc_b.py) for the
spectral-state JAX backends (ngcm, neuralgcm): the JAX counterpart of
check_ens_b.py's single-ob part for ACE2.

One surface-pressure ob, O-B = +1 hPa, sigma_o = 1 hPa, at a few points, on
an ERA5-initialized background at --date. The increment is applied at the
window start (increment_at_start), so H = surface pressure at the nearest
grid point of the incremented initial state, and

    v = B^1/2' H' d / (H B H' + R),   x_a = x_b + latent_from_grid(B^1/2 v)

with H B^1/2 from autograd through EnsembleB, the grid->latent SHT and the
model's decode. Shown at t0 and after --hours of model integration: surface-
pressure increment (shaded), 850 hPa wind increment (arrows), and the
balance of the 850 hPa wind increment with the geostrophic wind of the
850 hPa height increment (pattern correlation, speed ratio |V|/|Vg|, mean
angle V vs Vg) over the 40 x 60 deg box around the ob, |lat| >= 10, where
|dz850| >= 10% of its box maximum.

Run on a GPU node in the ngcm-torch env, e.g.
  sbatch -t 0:30:00 -J nmc_b_ob -o logs/nmc_b_ob_%j.out run_ngcm.sh \\
      diagnostics/check_jax_nmc_b.py --config config_neuralgcm_4d_2015_t0inc_jb_lr1e-2.yml \\
      --samples B12-6.pt,B36-12.pt --labels "12h-6h,36h-12h" --out /path/nmc_b_single_ob
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POINTS = [(45.0, 190.0, 'N Pacific 45N 170W'), (40.0, 265.0, 'central US 40N 95W'),
          (-50.0, 90.0, 'S Indian Ocean 50S 90E'), (5.0, 190.0, 'tropical Pacific 5N 170W')]
OMEGA, A_EARTH = 7.292e-5, 6.371e6


def geostrophic_from_phi(dphi, lat, lon):
    """Geostrophic wind (m/s) of a geopotential field dphi (H, W) in m^2/s^2
    on a pressure surface; lat may be in either order."""
    phi = np.deg2rad(lat)[:, None]
    dpdy = np.gradient(dphi, np.deg2rad(lat), axis=0) / A_EARTH
    dlon = np.deg2rad(lon[1] - lon[0])
    dpdx = (np.roll(dphi, -1, 1) - np.roll(dphi, 1, 1)) / (2 * dlon) / (A_EARTH * np.cos(phi))
    f = 2 * OMEGA * np.sin(phi)
    return -dpdy / f, dpdx / f


def geo_stats(u, v, ug, vg, sel):
    dot = (u * ug + v * vg)[sel].sum()
    crs = (ug * v - vg * u)[sel].sum()
    n2, g2 = (u * u + v * v)[sel].sum(), (ug * ug + vg * vg)[sel].sum()
    return dot / np.sqrt(n2 * g2), np.sqrt(n2 / g2), np.degrees(np.arctan2(crs, dot))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', help='a solver config whose exp section builds the model')
    ap.add_argument('--samples', help='build_jax_nmc_b.py outputs, comma separated')
    ap.add_argument('--labels', default=None, help='column labels, comma separated')
    ap.add_argument('--loc', type=float, default=1500.0, help='localization scale (km)')
    ap.add_argument('--date', default='2015-02-01T00', help='background (ERA5-initialized) date')
    ap.add_argument('--hours', type=int, default=6, help='integration length for the second panel')
    ap.add_argument('--plev', type=float, default=850.0, help='pressure level (hPa) for wind / balance')
    ap.add_argument('--out', required=True, help='output prefix (.pt, .png)')
    ap.add_argument('--plot_only', action='store_true',
                    help='only redraw OUT.png from OUT.pt (any env with matplotlib; ngcm-torch has none)')
    args = ap.parse_args()
    if args.plot_only:
        r = torch.load(args.out + '.pt', weights_only=False)
        args.loc, args.date = r['loc'], r['date']
        plot(r['single'], r['lat'], r['lon'], r['labels'], (0, r['hours']), args, r['plev'])
        print(f"wrote {args.out}.png")
        return
    import long_window_4dvar_utils as utils
    from latent_b import EnsembleB
    if not (args.config and args.samples):
        ap.error('--config and --samples are required (unless --plot_only)')
    files = args.samples.split(',')
    labels = args.labels.split(',') if args.labels else [os.path.basename(f) for f in files]

    with open(args.config) as f:
        exp = yaml.safe_load(f)['exp']
    model = utils.get_model(exp)
    dev = model.device
    date = datetime.datetime.strptime(args.date, '%Y-%m-%dT%H')
    s_b = model.prepare_initial_state(model.read_inputs(date), date)
    dt_h = model.timestep.total_seconds() / 3600
    nsteps = int(round(args.hours / dt_h))
    plevs = model.pressure_levels('z')
    k = int(np.argmin(np.abs(plevs - args.plev)))
    lats, lons = np.asarray(model.lats), np.asarray(model.lons)
    W = int(np.unique(lons).size)
    H = lats.size // W
    lat, lon = lats.reshape(H, W)[:, 0], lons.reshape(H, W)[0]

    def fields(dec):
        return {'sp': dec['sp'].double().cpu().numpy().reshape(H, W),
                'z': dec['z'][k].double().cpu().numpy().reshape(H, W),
                'u': dec['u'][k].double().cpu().numpy().reshape(H, W),
                'v': dec['v'][k].double().cpu().numpy().reshape(H, W)}

    with torch.no_grad():
        f0 = {0: fields(model.decode_state(s_b)),
              args.hours: fields(model.decode_state(model.advance(s_b, steps=nsteps, use_checkpoint=False)))}

    d_ob, r_ob = 100.0, 100.0 ** 2
    single = {}
    print(f"single ob: O-B = +1 hPa, sigma_o = 1 hPa, background ERA5 {args.date}, localization {args.loc:.0f} km; "
          f"balance at {plevs[k]:.0f} hPa", flush=True)
    for fpath, label in zip(files, labels):
        d = torch.load(fpath, map_location='cpu', weights_only=False)
        meta = d['meta']
        if meta.get('checkpoint_id') != model.checkpoint_id():
            raise ValueError(f"{fpath} built for {meta.get('checkpoint_id')}, not {model.checkpoint_id()}")
        B = EnsembleB(d['samples'].to(dev), args.loc, meta['sht_grid']).to(dev)
        del d
        print(f"\n{label}: {meta['n_samples']} samples ({meta.get('method')})", flush=True)
        for plat, plon, name in POINTS:
            i = int(np.abs(lat - plat).argmin())
            j = int(np.abs(lon - plon).argmin())
            v = torch.zeros(B.control_shape, dtype=torch.float32, device=dev, requires_grad=True)
            sp = model.decode_state(model.apply_increment(s_b, model.latent_from_grid(B(v))))['sp']
            sp[i * W + j].backward()
            g = v.grad.detach()
            hbh = (g.double() * g.double()).sum().item()
            with torch.no_grad():
                inc = model.latent_from_grid(B(g * (d_ob / (hbh + r_ob))))
                s_a = model.apply_increment(s_b, inc)
                fa = {0: fields(model.decode_state(s_a)),
                      args.hours: fields(model.decode_state(model.advance(s_a, steps=nsteps, use_checkpoint=False)))}
            for t in (0, args.hours):
                dx = {n: fa[t][n] - f0[t][n] for n in fa[t]}
                ug, vg = geostrophic_from_phi(dx['z'], lat, lon)
                box = (np.abs(lat - lat[i])[:, None] <= 20) & (np.abs(((lon - lon[j] + 180) % 360) - 180)[None, :] <= 30)
                sel = box & (np.abs(lat)[:, None] >= 10) & (np.abs(dx['z']) >= 0.1 * np.abs(dx['z'][box]).max())
                st = geo_stats(dx['u'], dx['v'], ug, vg, sel) if sel.any() else (np.nan,) * 3
                single[(label, name, t)] = {'hbh': hbh, 'ij': (i, j), 'geo': st, **dx}
                spd = np.hypot(dx['u'], dx['v'])
                print(f"  {name:26s} +{t:2d}h: sqrt(HBH') {np.sqrt(hbh) / 100:.2f} hPa, dps at ob "
                      f"{dx['sp'][i, j] / 100:+.2f} hPa, max |dz{plevs[k]:.0f}| {np.abs(dx['z'][box]).max() / 9.80665:.1f} m, "
                      f"max |dV| {spd[box].max():.2f} m/s; geostrophic corr {st[0]:.2f}, |V|/|Vg| {st[1]:.2f}, "
                      f"angle {st[2]:+.0f} deg", flush=True)
        del B
        torch.cuda.empty_cache()

    torch.save({'single': single, 'labels': labels, 'files': files, 'lat': lat, 'lon': lon, 'loc': args.loc,
                'date': args.date, 'hours': args.hours, 'plev': float(plevs[k])}, args.out + '.pt')
    plot(single, lat, lon, labels, (0, args.hours), args, float(plevs[k]))
    print(f"\nwrote {args.out}.pt, {args.out}.png")


def plot(single, lat, lon, labels, times, args, plev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = [(lab, t) for lab in labels for t in times]
    f, axs = plt.subplots(len(POINTS), len(cols), figsize=(4.6 * len(cols), 3.8 * len(POINTS)), squeeze=False)
    for r, (_, _, name) in enumerate(POINTS):
        for c, (lab, t) in enumerate(cols):
            rec = single[(lab, name, t)]
            ax = axs[r, c]
            i, j = rec['ij']
            jj = (np.arange(-14, 15) + j) % len(lon)
            ii = np.arange(max(i - 9, 0), min(i + 10, len(lat)))
            x = lon[j] + ((lon[jj] - lon[j] + 180) % 360 - 180)
            y = lat[ii]
            dp = rec['sp'][np.ix_(ii, jj)] / 100
            vmax = max(np.abs(dp).max(), 1e-6)
            cf = ax.contourf(x, y, dp, levels=np.linspace(-vmax, vmax, 21), cmap='RdBu_r')
            plt.colorbar(cf, ax=ax, shrink=0.8)
            dz = rec['z'][np.ix_(ii, jj)] / 9.80665
            zm = max(np.abs(dz).max(), 1e-6)
            ax.contour(x, y, dz, levels=np.linspace(-zm, zm, 11)[np.abs(np.linspace(-zm, zm, 11)) > 1e-9],
                       colors='k', linewidths=0.6)
            u, v = rec['u'][np.ix_(ii, jj)], rec['v'][np.ix_(ii, jj)]
            vm = max(np.hypot(u, v).max(), 1e-6)
            q = ax.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=vm / 3, color='0.2')
            ax.quiverkey(q, 0.8, 1.02, vm, f"{vm:.2f} m/s", labelpos='E', fontproperties={'size': 8})
            ax.plot(lon[j], lat[i], 'k*', ms=10)
            g = rec['geo']
            ax.set_title(f"{lab} +{t}h: {name}\nsqrt(HBH') {np.sqrt(rec['hbh']) / 100:.2f} hPa; geo corr {g[0]:.2f}, "
                         f"|V|/|Vg| {g[1]:.2f}, {g[2]:+.0f} deg", fontsize=8)
            ax.set_aspect(1 / np.cos(np.deg2rad(lat[i])))
    f.suptitle(f'NMC static-ensemble B, localization {args.loc:.0f} km: single surface-pressure ob (O-B +1 hPa, '
               f'sigma_o 1 hPa) at window start, background ERA5 {args.date}.\nShaded: ps increment (hPa); '
               f'contours: z{plev:.0f} increment (m); arrows: {plev:.0f} hPa wind increment; '
               f'balance = {plev:.0f} hPa wind vs geostrophic wind of z{plev:.0f}')
    plt.tight_layout()
    plt.savefig(args.out + '.png', dpi=90)


if __name__ == '__main__':
    main()
