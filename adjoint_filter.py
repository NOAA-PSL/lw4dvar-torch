"""Adjoint filter for latent 4D-Var (exp.adjoint_filter; notes/plan_20261105.md "Adjoint filter").

The forward model and Jo stay exact; only the GRADIENT flowing backward through the window's model states is
modified -- the analogue of the extra diffusion / simplified physics that traditional 4D-Var puts in its tangent-
linear and adjoint models to suppress unstable, unrealistic modes. Tensor hooks are registered on the packed model
state after each window step (compute_loss_4dvar); the hook sees dJ/d(state at step s), i.e. the adjoint state.

    adjoint_filter:
      diffusion: {scale_km: 300, steps: 1}   # horizontal smoothing of the adjoint state; steps: N = the first N
                                             # window steps (1 = only the injection step), 'all' = every step
      balance: true                          # at the injection step, the transpose of the hydrostatic-z map:
                                             # the z (above the lowest level) adjoint is moved onto t, q by the
                                             # hydrostatic relation (exp.hydrostatic_z's linearisation) and zeroed
      modes: {rank: 24, steps: 8, small_max: 0.5, imbalance_max: 0.7, amp_max: 19.0, cache: true}
                                             # "pole analysis": leading singular vectors of the 6 h -> 6+6*steps h
                                             # tangent-linear map at the background (fp32, randomized SVD, the
                                             # normalised-variance norm of diagnostics/adjoint_quality); modes whose
                                             # input side is small-scale (> small_max of the variance below ~400 km)
                                             # OR hydrostatically inconsistent (> imbalance_max) OR amplify more than
                                             # amp_max (default 10x A2's 48-h balanced growth 1.9) are projected out
                                             # of the injection-step adjoint, in the same metric
Gradient diagnostics are the adjoint-quality tool's (diagnostics/adjoint_quality); AIFS (channels_last) only.
"""
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

RD = 287.05
REARTH_KM = 6371.0


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class AdjointFilter:
    def __init__(self, model, cfg, logger=None):
        if model.state_layout != 'channels_last':
            raise NotImplementedError('adjoint_filter: only channels_last packed states (AIFS family)')
        from model_correction import input_normaliser
        self.model, self.cfg, self.log = model, dict(cfg), (logger.info if logger else print)
        dev = self.dev = model.device
        lat, lon = np.asarray(model.lats), np.asarray(model.lons)
        self.P = model.pmask_in.to(dev)
        self.nP = len(self.P)
        mul, _ = input_normaliser(model)
        w = torch.as_tensor(np.asarray(model.area_weights), dtype=torch.float32, device=dev)
        self.w = w / w.sum()
        # physical -> unit-normalised-norm coordinates (diagnostics/adjoint_quality's metric)
        self.scale = mul[self.P].to(dev).float()[None, :] * (self.w[:, None] / self.nP).sqrt()
        tree = cKDTree(_xyz(lat, lon))
        d13, nb13 = tree.query(_xyz(lat, lon), k=13)
        self.nb = torch.as_tensor(nb13, device=dev)
        # one 13-neighbour mean ~ a Gaussian of per-axis variance mean(d^2)/2 (d: chord ~ arc, km)
        self.sig1 = float(np.sqrt(np.mean((d13 * REARTH_KM) ** 2) / 2))
        self.nb24 = torch.as_tensor(tree.query(_xyz(lat, lon), k=24)[1], device=dev)
        dif = self.cfg.get('diffusion')
        self.dif_passes, self.dif_steps = 0, 0
        if dif:
            self.dif_passes = max(1, int(round((float(dif.get('scale_km', 300)) / self.sig1) ** 2)))
            st = dif.get('steps', 1)
            self.dif_steps = 10 ** 9 if st == 'all' else int(st)
            self.log(f"adjoint_filter: diffusion {dif.get('scale_km', 300)} km = {self.dif_passes} passes of the "
                     f"13-neighbour mean (sigma1 {self.sig1:.0f} km) on window steps <= {st}")
        self.balance = bool(self.cfg.get('balance'))
        self.V = None          # (n_modes_removed, points * nP) orthonormal input-side modes, normalised coords
        self.mode_info = None
        V = model.var_to_idx
        levs = sorted({int(k.split('_')[1]) for k in V if k.startswith('z_') and k.split('_')[1].isdigit()
                       and f"t_{k.split('_')[1]}" in V and f"q_{k.split('_')[1]}" in V}, reverse=True)
        idx = lambda v: torch.as_tensor([V[f'{v}_{p}'] for p in levs], device=dev)  # noqa: E731
        self.hz = {'levels': levs, 'z': idx('z'), 't': idx('t'), 'q': idx('q'),
                   'lnr': torch.as_tensor(np.log(np.array(levs[:-1], float) / np.array(levs[1:], float)),
                                          device=dev, dtype=torch.float32)}
        pos = {int(p): k for k, p in enumerate(self.P.tolist())}
        self.hzP = {v: torch.as_tensor([pos[int(i)] for i in self.hz[v].tolist()], device=dev) for v in 'ztq'}

    # ---------------- operators on the adjoint
    def _smooth(self, f, passes, nb=None):
        nb = self.nb if nb is None else nb
        for _ in range(passes):
            f = f[nb].mean(1)
        return f

    def _balance_T(self, g, x):
        """transpose of the linearised hydrostatic-z map on the newest level: g, x (points, vars)"""
        hz, c = self.hz, RD * 0.5 * self.hz['lnr']
        gz = g[:, hz['z']]
        G = torch.flip(torch.cumsum(torch.flip(gz[:, 1:], [1]), 1), [1])     # G_j = sum_{k > j} g_z_k (layers j)
        cG = c * G
        gtv = torch.zeros_like(g[:, hz['t']])
        gtv[:, :-1] += cG
        gtv[:, 1:] += cG
        t, q = x[:, hz['t']].float(), x[:, hz['q']].float()
        g = g.clone()
        g[:, hz['t']] += gtv * (1 + 0.608 * q)
        g[:, hz['q']] += gtv * 0.608 * t
        g[:, hz['z'][:1]] += gz[:, 1:].sum(1, keepdim=True)
        g[:, hz['z'][1:]] = 0
        return g

    def _project_modes(self, g):
        """remove the unrealistic modes' components from the injection-step adjoint g (points, vars)"""
        gp = g[:, self.P].float()
        gc = (gp / self.scale).reshape(-1)                 # dJ/dc, c = delta * scale
        gc = gc - self.V.T @ (self.V @ gc)
        g = g.clone()
        g[:, self.P] = (gc.reshape(gp.shape) * self.scale).to(g.dtype)
        return g

    def apply(self, state, s):
        """register the hook on `state` (the model state after window step s); returns state"""
        t = state.state
        if not t.requires_grad:
            return state
        dif = self.dif_passes and s <= self.dif_steps
        first = s == 1 and (self.balance or self.V is not None)
        if not (dif or first):
            return state
        xref = t.detach()[0, -1] if (first and self.balance) else None

        def hook(g):
            g = g.clone()
            for lvl in range(g.shape[1]):
                gl = g[0, lvl]
                if dif:
                    gl = self._smooth(gl, self.dif_passes)
                if first and lvl == g.shape[1] - 1:
                    if self.balance:
                        gl = self._balance_T(gl, xref)
                    if self.V is not None:
                        gl = self._project_modes(gl)
                g[0, lvl] = gl
            return g
        t.register_hook(hook)
        return state

    # ---------------- pole analysis
    def compute_modes(self, x0, path=None, adapter=None):
        """leading singular vectors of the 6 h -> 6 h + 6*steps map at the background x0 (window start)"""
        mc = self.cfg.get('modes')
        if not mc:
            return
        if path and mc.get('cache', True) and os.path.exists(path):
            d = torch.load(path, map_location=self.dev, weights_only=False)
            self.V, self.mode_info = d['V'], d['info']
            self.log(f"adjoint_filter: modes from {path}: removed {self.V.shape[0]} of {d['info']['rank']}")
            return
        from diagnostics.adjoint_quality.adapter import make_adapter
        A = adapter or make_adapter(self.model, {'model_backend': 'aifs'})
        L, NS = int(mc.get('rank', 24)), int(mc.get('steps', 8))
        npts, nP, dev = self.scale.shape[0], self.nP, self.dev
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        A.set_precision('fp32')
        try:
            with torch.no_grad():
                x1 = A.advance(x0, 1)
            scale = self.scale
            eps = float(mc.get('eps', 0.004))

            def F(c, grad=False):
                x = A.add_prog(x1, c / scale)
                x = A.advance(x, NS, grad=grad)
                return torch.nan_to_num(A.prog(x) * scale)

            def jvp(v):
                n = v.norm()
                with torch.no_grad():
                    return (F(eps * v / n) - F(-eps * v / n)) / (2 * eps) * n

            def vjp(y):
                c = torch.zeros(npts, nP, device=dev, requires_grad=True)
                g, = torch.autograd.grad((F(c, grad=True) * y).sum(), c)
                return g.detach()

            def orth(cols):
                Q, _ = torch.linalg.qr(torch.stack([c.reshape(-1) for c in cols], 1))
                return [Q[:, j].reshape(npts, nP) for j in range(Q.shape[1])]
            gen = torch.Generator(device='cpu').manual_seed(int(mc.get('seed', 0)))
            om = [self._smooth(torch.randn(npts, nP, generator=gen).to(dev), 3, self.nb24) for _ in range(L)]
            Q = orth([jvp(o / o.norm()) for o in om])
            Z = orth([vjp(q) for q in Q])
            Q = orth([jvp(z) for z in Z])
            Bt = torch.stack([vjp(q).reshape(-1) for q in Q], 1)      # A^T Q
            Uin, sv, _ = torch.linalg.svd(Bt, full_matrices=False)
        finally:
            A.set_precision(A.reduced_precision() or 'fp32')
            torch.backends.cuda.matmul.allow_tf32 = tf32
        # classify the input-side modes
        bg = x1.state[0, -1].float()
        keep, info = [], {'rank': L, 'sv': sv.cpu().tolist(), 'small': [], 'imbalance': [], 'unrealistic': []}
        for j in range(Uin.shape[1]):
            ci = Uin[:, j].reshape(npts, nP)
            cs = self._smooth(ci, 3, self.nb24)
            small = float(((ci - cs) ** 2).sum() / (ci ** 2).sum())
            imb = self._imbalance(ci / scale, bg)
            bad = small > mc.get('small_max', 0.5) or imb > mc.get('imbalance_max', 0.7) \
                or float(sv[j]) > mc.get('amp_max', 19.0)
            info['small'].append(small)
            info['imbalance'].append(imb)
            info['unrealistic'].append(bool(bad))
            if bad:
                keep.append(Uin[:, j])
        self.V = torch.stack(keep, 0).contiguous() if keep else None
        self.mode_info = info
        n_s = sum(1 for s in info['small'] if s > mc.get('small_max', 0.5))
        n_i = sum(1 for s in info['imbalance'] if s > mc.get('imbalance_max', 0.7))
        n_a = sum(1 for s in info['sv'] if s > mc.get('amp_max', 19.0))
        self.log(f"adjoint_filter: {L} modes, singular values {np.round(info['sv'][:6], 1).tolist()}...; "
                 f"unrealistic {len(keep)} (small-scale {n_s}, imbalanced {n_i}, amp > {mc.get('amp_max', 19.0)} "
                 f"{n_a})")
        if path and self.V is not None:
            torch.save({'V': self.V, 'info': info}, path)

    def _imbalance(self, d, bg):
        """hydrostatic imbalance fraction |res|^2/|dthick|^2 of a prognostic perturbation d (points, nP)"""
        hz, hzP = self.hz, self.hzP
        dt, dq, dz = d[:, hzP['t']], d[:, hzP['q']], d[:, hzP['z']]
        t, q = bg[:, hz['t']], bg[:, hz['q']]
        dtv = dt * (1 + 0.608 * q) + 0.608 * t * dq
        dthick = dz[:, 1:] - dz[:, :-1]
        res = dthick - RD * 0.5 * (dtv[:, 1:] + dtv[:, :-1]) * hz['lnr']
        w = self.w[:, None]
        return float(min((w * res * res).sum() / max(float((w * dthick * dthick).sum()), 1e-30), 1e3))
