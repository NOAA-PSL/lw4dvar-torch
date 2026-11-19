"""Synthetic analysis-error noise for fine-tuning A2 on imperfect initial states (notes/adjoint_finetune_plan.md,
"Analysis-error noise model").

Statistics are fitted (a37tmp/finetune_noise/build_noise.py) on d = PITA analysis - ERA5 in A2's NORMALISED prognostic
space (model input normaliser, model.pmask_in order). The generator draws

    x = sum_k  a_k(lat) * S^{n_k} w_k / sqrt(wvar[n_k]) * e_k

  e_k     global column EOFs of d (all prognostic variables and levels)
  a_k     rms of d's EOF-k coefficient per 10-deg latitude band, linearly interpolated in latitude
  S^n w   white noise smoothed by n passes of the 13-nearest-neighbour mean on the model grid; n_k is fitted so the
          smoothed noise has the same retained-variance curve as d's EOF-k coefficient field (horizontal scale)
  wvar[n] variance of S^n w for unit white w (so each term has unit variance before a_k)

mode 'resample' instead returns a stored real d (fit set) rotated by a random longitude shift (row by row on the
reduced grid) with a random sign.

    noise = AnalysisNoise('output/finetune/analysis_noise_stats.npz', lats, lons, device='cuda')
    d = noise.sample()                       # (points, n_prognostic), normalised units
    state = noise.apply(state, model)        # packed (1, 2, points, vars): adds d / mul to both time levels, q >= 0
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def neighbours(lat, lon, k=13):
    """(points, k) indices of the k nearest grid points (self included)."""
    g = _xyz(np.asarray(lat), np.asarray(lon))
    return cKDTree(g).query(g, k=k)[1]


def smooth(f, nb, n):
    """n passes of the neighbour mean; f (points, ...) tensor, nb (points, k) long tensor."""
    for _ in range(int(n)):
        f = f[nb].mean(1)
    return f


def smooth_per_column(f, nb, npass):
    """f (points, K), npass (K,) ints: column k gets npass[k] passes."""
    out = f.clone()
    for p in range(int(npass.max()) if len(npass) else 0):
        m = npass > p
        out[:, m] = out[:, m][nb].mean(1)
    return out


def white_variance(nb, nmax, device, seed=0, ncol=64):
    """variance of S^n w for unit white w, n = 0..nmax (averaged over ncol draws)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    w = torch.randn(nb.shape[0], ncol, generator=g).to(device)
    out = []
    for n in range(nmax + 1):
        out.append(float(w.var(0).mean()))
        w = w[nb].mean(1)
    return np.array(out)


class AnalysisNoise:
    def __init__(self, stats_path, lat, lon, device='cpu', mode='gen', which='end', seed=None):
        z = np.load(stats_path, allow_pickle=True)
        self.dev = torch.device(device)
        self.mode, self.which = mode, which
        self.names = list(z['names'])
        self.gen = torch.Generator(device='cpu')
        if seed is not None:
            self.gen.manual_seed(int(seed))
        lat, lon = np.asarray(lat), np.asarray(lon)
        self.nb = torch.as_tensor(neighbours(lat, lon), device=self.dev)
        self.E = torch.as_tensor(z[f'{which}_eofs'], dtype=torch.float32, device=self.dev)        # (P, K)
        self.npass = torch.as_tensor(z[f'{which}_npass'], dtype=torch.long, device=self.dev)    # (K,)
        wvar = z['wvar']
        self.wnorm = torch.as_tensor(1.0 / np.sqrt(wvar[z[f'{which}_npass']]), dtype=torch.float32, device=self.dev)
        bc, amp = z['band_centres'], z[f'{which}_band_rms']                                     # (B,), (K, B)
        A = np.stack([np.interp(lat, bc, amp[k]) for k in range(amp.shape[0])], -1)              # (points, K)
        self.A = torch.as_tensor(A, dtype=torch.float32, device=self.dev)
        self.scale = 1.0
        if mode == 'resample':
            self.D = np.load(str(z['diffs_path']).replace('{which}', which), mmap_mode='r')     # (N, points, P) fp16
            rows = {}
            for i, la in enumerate(np.round(lat, 6)):
                rows.setdefault(la, []).append(i)
            self.rows = [np.array(sorted(ix, key=lambda j: lon[j] % 360.0)) for ix in rows.values()]

    def _rotation(self, frac):
        perm = np.empty(sum(len(r) for r in self.rows), dtype=np.int64)
        for r in self.rows:
            s = int(round(frac * len(r)))
            perm[r] = r[(np.arange(len(r)) - s) % len(r)]
        return torch.as_tensor(perm, device=self.dev)

    def sample(self):
        """(points, P) normalised noise."""
        if self.mode == 'resample':
            i = int(torch.randint(self.D.shape[0], (1,), generator=self.gen))
            frac = float(torch.rand(1, generator=self.gen))
            sign = 1.0 if float(torch.rand(1, generator=self.gen)) < 0.5 else -1.0
            d = torch.as_tensor(np.asarray(self.D[i], dtype=np.float32), device=self.dev)
            return self.scale * sign * d[self._rotation(frac)]
        w = torch.randn(self.A.shape[0], self.A.shape[1], generator=self.gen).to(self.dev)
        c = smooth_per_column(w, self.nb, self.npass) * self.wnorm * self.A
        return self.scale * (c @ self.E.T)

    def apply(self, state, model, d=None, both_levels=True, lag_corr=None):
        """add noise to a packed AIFS state (1, 2, points, vars) in physical units; clamps q >= 0.
        lag_corr: if set, the older time level gets lag_corr * d + sqrt(1 - lag_corr^2) * (an independent draw) --
        the real analysis errors at t-6 h and t correlate only ~0.2-0.3 (build_noise.py 'lag_corr_t-6h')."""
        from model_correction import input_normaliser
        mul, _ = input_normaliser(model)
        pidx = model.pmask_in
        d = self.sample() if d is None else d
        phys = (d / mul[pidx.cpu()].to(d.device)).to(state.dtype)
        out = state.clone()
        out[0, 1][:, pidx] = out[0, 1][:, pidx] + phys
        if both_levels:
            if lag_corr is not None:
                d0 = lag_corr * d + (1 - lag_corr ** 2) ** 0.5 * self.sample()
                phys = (d0 / mul[pidx.cpu()].to(d.device)).to(state.dtype)
            out[0, 0][:, pidx] = out[0, 0][:, pidx] + phys
        qi = [i for n, i in model.var_to_idx.items() if n.startswith('q_')]
        out[..., qi] = out[..., qi].clamp_min(0.0)
        return out
