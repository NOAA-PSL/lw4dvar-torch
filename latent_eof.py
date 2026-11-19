"""Structure of latent B / Q from the fitted increments saved by fit_latent_to_era5.py --save:
channel (latent-feature) EOFs and the spatial correlation of their amplitudes on the hidden mesh,
toward a structured control_prior (notes/weak_constraint_4dvar.md, plan step 4).

  C = mean over samples and hidden nodes of delta_n delta_n^T   (n_channels x n_channels)
  C = E diag(lam) E^T;  P = delta E  (amplitude of each channel EOF at each node)

Prints the variance spectrum (share in the leading k modes), the rms of the leading modes'
amplitudes by latitude band, and their spatial correlation against great-circle distance
(random node pairs, pooled over samples). Writes OUT.npz: eigvals, eigvecs, lat, lon, the
correlation curves.

--holdout N: estimate C on all but the last N samples and score those N by Gaussian log
likelihood per element (scale fitted on the training samples) under a diagonal prior, the shared
channel covariance, and a channel covariance per node class; shrinkage C + eps*mean(lam)*I. Classes:
latitude band (3), surface (ocean / land / ice-snow: each node's share of land, sea ice and snow cover over the
O96 points nearest it, on the sample's date, from the training zarrs), and band x surface.

Usage: python latent_eof.py SAMPLE_DIR KIND FIT_NPZ OUT [--modes 0,4,19,99] [--pairs 200000] [--holdout 6]
  SAMPLE_DIR/KIND_*.pt: fitted deltas; FIT_NPZ: the fit's output (hidden-node lat / lon)
"""

import argparse
import glob

import numpy as np
from scipy.spatial import cKDTree
import torch

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('samples')
ap.add_argument('kind')
ap.add_argument('fit')
ap.add_argument('out')
ap.add_argument('--modes', default='0,4,19,99')
ap.add_argument('--pairs', type=int, default=200000)
ap.add_argument('--holdout', type=int, default=0)
ap.add_argument('--shrink', default='1e-3', help='comma list of eps for C + eps*mean(lam)*I; the holdout scores each')
ap.add_argument('--save_classes', default='', help='write band x surface channel covariances (all samples, first --shrink eps) '
                'as eigvecs / eigvals (9, ch, ch) / (9, ch) for control_prior q_cov / b_cov')
args = ap.parse_args()
torch.set_num_threads(16)


def surface_classes(files, lat, lon):
    """Per sample file (<kind>_<YYYY-MM-DDTHH>.pt): each hidden node's surface class on that date
    (latent_classes.NodeClasses: 0 ocean, 1 land, 2 ice/snow)."""
    import os
    from latent_classes import NodeClasses
    nc = NodeClasses(lat, lon)
    return {f: nc.surface(os.path.basename(f).split('_', 1)[1][:13]) for f in files}


files = sorted(glob.glob(f'{args.samples}/{args.kind}_*.pt'))
fit = np.load(args.fit)
lat, lon = fit['lat'], fit['lon']
print(f'{len(files)} {args.kind} samples from {args.samples}')

C, n = 0., 0
for f in files:
    d = torch.load(f).double()
    C = C + d.T @ d
    n += d.shape[0]
C = C / n
lam, E = torch.linalg.eigh(C)
lam, E = lam.flip(0), E.flip(1)
frac = (lam.cumsum(0) / lam.sum()).numpy()
print(f'total variance {lam.sum().item():.4g} (rms {lam.sum().sqrt().item() / np.sqrt(len(lam)):.4g} per element)')
print('variance share in the leading k channel EOFs: '
      + ', '.join(f'{k}: {frac[k - 1]:.3f}' for k in (1, 5, 10, 20, 50, 100, 200, 500)))
print(f'modes for 50/90/99%: {np.searchsorted(frac, .5) + 1} / {np.searchsorted(frac, .9) + 1} / '
      f'{np.searchsorted(frac, .99) + 1};  lam[0]/lam[99] {(lam[0] / lam[99]).item():.3g}, '
      f'lam[0]/lam[-1] {(lam[0] / lam[-1]).item():.3g}')

modes = [int(k) for k in args.modes.split(',')]
P = [(torch.load(f).double() @ E[:, modes]).numpy() for f in files]  # (nodes, len(modes)) each
bands = [(-90, -20), (-20, 20), (20, 90)]
print(f"\n{'mode':>5s} {'sqrt(lam)':>9s}  " + ' '.join(f'{a:>4d}..{b:<4d}' for a, b in bands))
for i, k in enumerate(modes):
    amp = np.sqrt(np.mean([p[:, i] ** 2 for p in P], axis=0))
    print(f'{k:5d} {np.sqrt(lam[k].item()):9.4g}  '
          + ' '.join(f'{np.sqrt(np.mean(amp[(lat >= a) & (lat < b)] ** 2)):10.4g}' for a, b in bands))

# spatial correlation of mode amplitudes vs distance (pooled over samples, per-node standardized)
rng = np.random.default_rng(0)
i1 = rng.integers(0, len(lat), args.pairs)
i2 = rng.integers(0, len(lat), args.pairs)
la, lo = np.radians(lat), np.radians(lon)
xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
tree = cKDTree(xyz)
for share, k in ((0.4, 30), (0.3, 600)):  # short-distance pairs: i2 among i1's k nearest nodes
    w = np.nonzero(rng.random(args.pairs) < share)[0]
    _, nb = tree.query(xyz[i1[w]], k=k)
    i2[w] = nb[np.arange(len(w)), rng.integers(1, k, len(w))]
dist = 6371. * np.arccos(np.clip(np.sin(la[i1]) * np.sin(la[i2])
                                 + np.cos(la[i1]) * np.cos(la[i2]) * np.cos(lo[i1] - lo[i2]), -1, 1))
edges = np.array([0, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000, 20100.])
binned = np.digitize(dist, edges) - 1
corr = np.full((len(modes), len(edges) - 1), np.nan)
for i in range(len(modes)):
    z = np.stack([p[:, i] for p in P])  # (samples, nodes)
    z = z / np.sqrt((z ** 2).mean(0, keepdims=True) + 1e-30)
    prod = (z[:, i1] * z[:, i2]).mean(0)
    for b in range(len(edges) - 1):
        sel = binned == b
        if sel.sum() > 50:
            corr[i, b] = prod[sel].mean()
print(f"\nspatial correlation of mode amplitude vs distance (km); n pairs per bin "
      + ' '.join(str(int((binned == b).sum())) for b in range(len(edges) - 1)))
print(f"{'mode':>5s} " + ' '.join(f'{int(a):>6d}' for a in edges[:-1]))
for i, k in enumerate(modes):
    print(f'{k:5d} ' + ' '.join(f'{c:6.2f}' for c in corr[i]))
# all-channel (raw delta) correlation for reference
z = np.stack([torch.load(f).float().numpy() for f in files[:4]])  # (s, nodes, ch)
r = (z[:, i1] * z[:, i2]).sum(-1) / np.sqrt((z[:, i1] ** 2).sum(-1) * (z[:, i2] ** 2).sum(-1))
print(' raw ' + ' '.join(f'{r.mean(0)[binned == b].mean():6.2f}' if (binned == b).sum() > 50 else '   nan'
                         for b in range(len(edges) - 1)))

if args.holdout:
    train, test = files[:-args.holdout], files[-args.holdout:]
    bands3 = [(-90, -20), (-20, 20), (20, 91)]
    masks = [torch.as_tensor((lat >= a) & (lat < b)) for a, b in bands3]

    def cov(fs, mask=None):
        S, m = 0., 0
        for f in fs:
            d = torch.load(f).double()
            d = d if mask is None else d[mask]
            S, m = S + d.T @ d, m + d.shape[0]
        return S / m

    def loglik(Cm, d):
        # mean over elements of log N(d; 0, Cm) (per-node vectors), nats per element
        Lc = torch.linalg.cholesky(Cm)
        z = torch.linalg.solve_triangular(Lc, d.T, upper=False)
        nch = d.shape[1]
        return (-0.5 * (z ** 2).sum(0).mean() - Lc.diagonal().log().sum() - 0.5 * nch * np.log(2 * np.pi)).item() / nch

    surf = surface_classes(files, lat, lon)            # {file: (n_nodes,) 0 ocean, 1 land, 2 ice/snow}
    band = np.digitize(lat, [-20, 20])                 # 0 SH, 1 tropics, 2 NH
    for k, nm in enumerate(('ocean', 'land', 'ice/snow')):
        print(f'surface class {nm}: {np.mean([np.mean(c == k) for c in surf.values()]) * 100:.1f}% of nodes')
    partitions = {'channel C': lambda f: np.zeros(len(lat), int),
                  'channel C by band': lambda f: band,
                  'channel C by surface': lambda f: surf[f],
                  'channel C by band x surface': lambda f: band * 3 + surf[f]}

    def cov_classes(fs, part, ncls):
        S, m = [0.] * ncls, [0] * ncls
        for f in fs:
            d, c = torch.load(f).double(), torch.as_tensor(part(f))
            for k in range(ncls):
                dk = d[c == k]
                S[k], m[k] = S[k] + dk.T @ dk, m[k] + dk.shape[0]
        return [S[k] / m[k] if m[k] > 2000 else None for k in range(ncls)], m

    Ctr = cov(train)
    nch = Ctr.shape[0]
    I = torch.eye(nch, dtype=torch.float64)
    eps_list = [float(e) for e in args.shrink.split(',')]
    covs = {}
    for name, part in partitions.items():
        ncls = 9 if 'x' in name else 3 if 'by' in name else 1
        Cs, m = cov_classes(train, part, ncls)
        for eps in eps_list:
            reg = lambda c: c + eps * c.diagonal().mean() * I  # noqa: E731
            key = name if len(eps_list) == 1 else f'{name} eps {eps:g}'
            covs[key] = [reg(c) if c is not None else reg(Ctr) for c in Cs]  # thin classes fall back to the shared C
        print(f'{name}: training nodes per class {m}')
    partitions = {k: partitions[k.split(' eps ')[0]] for k in covs}
    res = {}
    for f in test:
        d = torch.load(f).double()
        Cd = torch.diag(torch.full((nch,), Ctr.diagonal().mean().item(), dtype=torch.float64))
        res.setdefault('diagonal', []).append(loglik(Cd, d))
        res.setdefault('channel variances (diagonal)', []).append(loglik(torch.diag(Ctr.diagonal()), d))
        for name, part in partitions.items():
            c = torch.as_tensor(part(f))
            ll = sum(loglik(Ck, d[c == k]) * (c == k).sum().item()
                     for k, Ck in enumerate(covs[name]) if (c == k).any()) / len(lat)
            res.setdefault(name, []).append(ll)
    print(f'\nheld-out Gaussian log likelihood, nats per element ({len(train)} training / {len(test)} test samples):')
    for name, v in res.items():
        print(f'  {name:28s} {np.mean(v):8.4f}   (gain over diagonal {np.mean(v) - np.mean(res["diagonal"]):+.4f})')

if args.save_classes:
    surf = surface_classes(files, lat, lon)
    band = np.digitize(lat, [-20, 20])
    eps = float(args.shrink.split(',')[0])
    S, m = [0.] * 9, [0] * 9
    for f in files:
        d, c = torch.load(f).double(), torch.as_tensor(band * 3 + surf[f])
        for k in range(9):
            dk = d[c == k]
            S[k], m[k] = S[k] + dk.T @ dk, m[k] + dk.shape[0]
    Cs = [S[k] / m[k] if m[k] > 2000 else C for k in range(9)]  # thin classes fall back to the shared C
    ev = [torch.linalg.eigh(c + eps * c.diagonal().mean() * torch.eye(c.shape[0], dtype=c.dtype)) for c in Cs]
    np.savez(args.save_classes, eigvals=np.stack([e[0].flip(0).numpy() for e in ev]),
             eigvecs=np.stack([e[1].flip(1).float().numpy() for e in ev]), classes='band_surface',
             nodes_per_class=np.array(m), lat=lat, lon=lon, shrink=eps, n_samples=len(files))
    print('wrote', args.save_classes, 'class rms', ' '.join(f'{np.sqrt(np.mean(e[0].numpy())):.4g}' for e in ev))

np.savez(args.out, eigvals=lam.numpy(), eigvecs=E.float().numpy(), lat=lat, lon=lon, modes=modes,
         corr=corr, edges=edges, n_samples=len(files))
print('wrote', args.out)
