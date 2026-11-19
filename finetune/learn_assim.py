"""Learning to assimilate: fine-tune A2 so that a short 4D-Var THROUGH A2 recovers the past ERA5 state from future
ERA5 information (notes/adjoint_finetune_plan.md, "Learning to assimilate").

Inner loop (a mini strong-constraint 4D-Var, the same control as long_window_4dvar: a latent increment u injected on
the first 6-h step):
  background  x_b = ERA5(t0) + realistic analysis error (finetune/analysis_noise.py, resampled 2015 PITA - ERA5)
  obs         ERA5 outputs at t0+6h*j, j = j_obs0 .. n_win (default 2..4: the FUTURE, 12-24 h; the 6-h state itself
              is not observed), at a random --obs_frac of grid points (1 = dense), optional Gaussian noise --obs_noise
              (normalised output units)
  J(u)        sum_j masked weighted MSE(y_j(u), obs_j)   (A2's own loss weights; one number per sample)
  update      u_{k+1} = u_k - s_k * g_k / rms(g_k),  g_k = dJ/du at u_k (create_graph=True: differentiable),
              s_k = exp(log_s_k) learnable (init --s0, ~the rms of a golden 4D-Var increment per step)
Outer loss (train):
  L = MSE_fcst (ERA5 -> ERA5 at +6 h, clean start, A2's normal loss: keeps forward skill)
      + beta * [MSE(x_1(u_K), ERA5_6h) - MSE(x_1(0), ERA5_6h)] / MSE(x_1(0), ERA5_6h).detach()
  The bracket is the analysis-minus-background error at the 6-h injection time: improving it requires the gradient
  (the adjoint) to point at the true error. Making the forward model better at denoising lowers both terms equally,
  so it does not pay (NA's failure mode). beta 'auto': the L2A term = --beta_frac of the total at the first step.
Arm LG (gradient alignment): the same background/obs, L = MSE_fcst + beta * (1 - cos(-dJ/dx_b, ERA5(t0) - x_b)) with the
  gradient w.r.t. the PHYSICAL input state (last time level, prognostic, normalised metric).

Arm LF (first-order surrogate of LA1 for K=1, no double backward): u_a = -alpha*g(0) with g = dJ/du detached and alpha the
  per-sample best step against the truth (line search on the 6-h analysis error, detached); the second-order part of
  dL_out/dtheta, -alpha * v . dg/dtheta with v = dL_out/du_a, is replaced by a central finite difference of J_theta along
  v (see L2A.lf). --mode fdcheck compares it with the true double-backward gradient on a 1-step window.

Arm LB (two models, opt-in): A = the frozen original A2 does every forward trajectory, cost, line search and truth
  error; B = a trainable copy (init A2) supplies only the gradient. A is run at u0 = 0 and gives the cotangents
  c_j = dJ/dy_j at its outputs (detached); B is run at the same control and g_B = d/du sum_j <c_j, y^B_j(u)>, i.e. B's
  adjoint applied to A's cotangents. u_a = u0 - alpha g_B, L = (E_a(u_a) - E_b)/E_b with E evaluated by A, so theta_B
  enters only through g_B: dL/dtheta_B = -alpha v . dg_B/dtheta_B, v = dL_out/du_a (from A). With c held fixed (g_B is
  the u-gradient of S_B(u; c) at fixed c) that is exactly d/dtheta of the central difference
  -alpha [S_B(u0 + e v; c) - S_B(u0 - e v; c)] / (2e); c is NOT re-evaluated at u0 +- e v. --anchor lam adds
  lam |theta_B - theta_A|^2. --mode fdcheck --arm LB compares with the true double backward.

Modes: --mode feas  (double-backward check, memory/time for K, n_win); --mode fdcheck; --mode train; --mode eval (held-out 2021:
best-step one-step ratio and K-step ratio, increment cosine with the true error, for --weights A2 | ckpt.pt).
Double backward needs an attention with a twice-differentiable backward: --attention block_sliding_window (SDPA,
forced to the math kernel during the inner loop).
"""
import argparse
import resource
import contextlib
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'finetune'))
from finetune_a2 import A2Train, AugResample, Samples, date_range, H6  # noqa: E402

CK = dict(use_reentrant=False)
MAP_VARS = ['z_500', 'u_250', 'v_250']      # --dump_maps fields


@contextlib.contextmanager
def fp32():
    """fp32 with TF32 off: every torch.autocast region disabled (as diagnostics/adjoint_quality/adapter.py does for A2:
    torch.autocast is replaced by a disabled context), plus the TF32 matmul/cudnn flags that aifs37_model turns on.
    Must stay active through backward too, so checkpoint recompute runs in fp32."""
    sys.path.insert(0, os.path.join(ROOT, 'diagnostics', 'adjoint_quality'))
    try:
        import adapter
    finally:
        sys.path.pop(0)
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    prev, torch.autocast = torch.autocast, adapter._DisabledAutocast      # re-entrant: restore the previous value
    try:
        yield
    finally:
        torch.autocast = prev
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
        torch.set_float32_matmul_precision(old[2])


def log(d, fh=None):
    s = json.dumps(d) if isinstance(d, dict) else str(d)
    print(f'[{time.strftime("%H:%M:%S")}] {s}', flush=True)
    if fh is not None and isinstance(d, dict):
        fh.write(s + '\n')
        fh.flush()


class L2A:
    def __init__(self, T, a, gen):
        self.T, self.a, self.gen = T, a, gen
        self.m = T.model
        from analysis_noise import AnalysisNoise
        self.AN = AugResample(AnalysisNoise, a.noise_stats, T.model.lats, T.model.lons, T.dev, a.seed + 7,
                              tuple(a.amp_range))
        self.log_s = torch.nn.Parameter(torch.full((max(a.K, 1),), math.log(a.s0), device=T.dev))
        # twin truth (arm LB): obs and the 6-h truth come from A's own forward trajectory (zero model error), set in draw
        self.twin, self.tw_y1, self.tw_x1 = bool(a.twin_truth), None, None

    def next_true(self, blk):
        return self.tw_x1 if self.twin else st_next_true(self.T, blk)

    def target1(self, blk):
        return self.tw_y1 if self.twin else self.T.target(blk, 1)

    # one 6-h step with an optional latent increment (set and cleared inside, so checkpoint recompute sees it)
    # net: arm LB runs a step with B instead of A; swapped in here (not around the call) so the checkpoint recompute
    # in backward, which may interleave steps of A and B, uses the right network
    def step_u(self, x, date, blk, j, u, net=None):
        self.m._latent_inc = u
        prev = self.m.interface.model
        if net is not None:
            self.m.interface.model = net
        try:
            return self.T.step(x, date, blk, j)
        finally:
            self.m._latent_inc = None
            self.m.interface.model = prev

    def roll(self, x, t0, blk, n, u, ckpt, net=None):
        ys = []
        for j in range(1, n + 1):
            uj = u if j == 1 else None
            if ckpt:
                x, y = torch.utils.checkpoint.checkpoint(self.step_u, x, t0 + H6 * j, blk, j, uj, net, **CK)
            else:
                x, y = self.step_u(x, t0 + H6 * j, blk, j, uj, net)
            ys.append(y)
        return ys

    def masked_mse(self, y, yt, mask):
        T = self.T
        d = torch.nan_to_num((y.float() - yt) * T.out_mul)
        w = T.aw * mask
        return ((d * d * T.W).mean(-1)[0] * w).sum() / w.sum()

    def draw(self, t0, blk):
        T, a = self.T, self.a
        st = T.initial(t0, blk)
        xb = self.AN.apply(st, T.model, both_levels=True, lag_corr=a.lag_corr)
        npts = len(T.aw)
        obs, masks = {}, {}
        if self.twin:
            with torch.no_grad():          # A (the frozen original) from the ERA5-derived state, no increment
                x_, twin_y = st, {}
                for j in range(1, a.n_win + 1):
                    x_, y_ = self.step_u(x_, t0 + H6 * j, blk, j, None)
                    twin_y[j] = y_.detach().float()
                    if j == 1:
                        self.tw_y1, self.tw_x1 = twin_y[1], x_[0, -1][:, T.P].float()
        for j in range(a.j_obs0, a.n_win + 1):
            yt = twin_y[j] if self.twin else T.target(blk, j)
            if a.obs_noise > 0:
                yt = yt + a.obs_noise * torch.randn(yt.shape, generator=self.gen).to(T.dev) / T.out_mul
            obs[j] = yt
            masks[j] = (torch.rand(npts, generator=self.gen) < a.obs_frac).float().to(T.dev) if a.obs_frac < 1 \
                else torch.ones(npts, device=T.dev)
        return st, xb, obs, masks

    def J(self, ys, obs, masks):
        return sum(self.masked_mse(ys[j - 1], obs[j], masks[j]) for j in obs)

    def inner(self, xb, t0, blk, obs, masks, K, create_graph, ckpt=True, steps=None):
        """K differentiable normalised-gradient steps from u = 0; returns u_K and the J history"""
        u = torch.zeros(self.m.latent_shape, device=self.T.dev, requires_grad=True)
        Js = []
        for k in range(K):
            if self.a.arm == 'LB':
                g, Jk, _ = self.grad_B(xb, t0, blk, obs, masks, u.detach())
            else:
                ys = self.roll(xb, t0, blk, self.a.n_win, u, ckpt)
                Jk = self.J(ys, obs, masks)
                g, = torch.autograd.grad(Jk, u, create_graph=create_graph)
            s = (torch.exp(self.log_s[min(k, len(self.log_s) - 1)]) if steps is None else steps[k])
            u = u - s * g / (g.pow(2).mean().sqrt() + 1e-20)
            Js.append(float(Jk))
        return u, Js

    def analysis_err(self, xb, t0, blk, u, ckpt=True):
        """MSE at the 6-h injection time vs ERA5 for the increment u (output space, A2's loss weights) and the state"""
        x1, y1 = self.step_u(xb, t0 + H6, blk, 1, u) if not ckpt else torch.utils.checkpoint.checkpoint(
            self.step_u, xb, t0 + H6, blk, 1, u, None, **CK)
        return self.T.mse(y1, self.target1(blk)), x1

    def lg_loss(self, xb, st, t0, blk, obs, masks, ckpt=True):
        """1 - cos(-dJ/dx_b, x_true - x_b) on the last input time level, prognostic, normalised metric"""
        T = self.T
        d = torch.zeros(len(T.aw), len(T.P), device=T.dev, requires_grad=True)
        x = T.add_last(xb, d)
        ys = self.roll(x, t0, blk, self.a.n_win, None, ckpt)
        g, = torch.autograd.grad(self.J(ys, obs, masks), d, create_graph=True)
        e = (st - xb)[0, -1][:, T.P].float()
        gz, ez = -g / T.in_mul, e * T.in_mul          # adjoint vector in the normalised metric; error normalised
        w = T.aw[:, None]
        return 1 - (w * gz * ez).sum() / ((w * gz * gz).sum() * (w * ez * ez).sum()).sqrt().clamp_min(1e-30)

    def lf(self, xb, t0, blk, obs, masks, prec=True):
        """Arm LF, first-order surrogate of the K=1 LA1 loss. Returns (l2, pend): l2 = (E_a - E_b)/E_b at the 6-h
        injection time, its graph carrying only the DIRECT theta dependence (u_a held fixed); pend(scale) backprops the
        finite-difference surrogate of the remaining second-order part, scale * S, into the parameter .grad.
          u_a = u0 - alpha g,  g = dJ/du(u0) (theta frozen, detached),  alpha = best step against the truth (detached)
          v = dL_out/du_a (theta frozen),  dL_out/dtheta|via u_a = -alpha v . dg/dtheta
          v . dg/dtheta = d/de J_theta(u0 + e v) at 0  ->  S = -alpha [J(u0 + e v) - J(u0 - e v)] / (2 e),
          e = fd_eps |u0 - u_a| / |v|.
        alpha is treated as theta-independent (its theta dependence is ignored on purpose). prec=True runs the g, v and
        J(u0 +- e v) evaluations in fp32/TF32-off (the central difference cancels, bf16 noise would swamp it)."""
        a = self.a
        with torch.no_grad():
            eb_det = self.analysis_err(xb, t0, blk, None, ckpt=False)[0].detach()
        ctx = fp32() if prec else contextlib.nullcontext()
        with ctx:
            u0 = torch.zeros(self.m.latent_shape, device=self.T.dev, requires_grad=True)
            g, = torch.autograd.grad(self.J(self.roll(xb, t0, blk, a.n_win, u0, True), obs, masks), u0)
            g = g.detach()
            u0 = u0.detach()
            # alpha: line search along -g (geometric grid around the s0 / rms(g) step, then a parabola through the best
            # point and its neighbours); the error is the 6-h analysis error that the outer loss measures
            sref = a.s0 / (g.pow(2).mean().sqrt() + 1e-20)
            grid = [float(sref) * 2.0 ** k for k in range(-4, 6)]
            with torch.no_grad():
                es = [float(self.analysis_err(xb, t0, blk, u0 - al * g, ckpt=False)[0]) for al in grid]
            i = int(np.argmin(es))
            alpha = grid[i]
            if 0 < i < len(grid) - 1:
                x0, x1, x2 = (math.log(grid[i + k]) for k in (-1, 0, 1))
                y0, y1, y2 = es[i - 1], es[i], es[i + 1]
                den = (x0 - x1) * (x0 - x2) * (x1 - x2)
                A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
                B = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
                if A > 0:
                    alpha = math.exp(min(max(-B / (2 * A), x0), x2))
            if es[i] >= float(eb_det):
                alpha = 0.0            # no step improves on the background: no surrogate for this sample
            self.last_lf = {'alpha': alpha, 'alpha_over_ref': alpha / float(sref), 'E_b': float(eb_det)}
            e = v = None
            if alpha > 0:
                ua = (u0 - alpha * g).requires_grad_(True)
                ea, _ = self.analysis_err(xb, t0, blk, ua)
                v, = torch.autograd.grad((ea - eb_det) / eb_det, ua)
                v = v.detach()
                e = a.fd_eps * float((alpha * g).norm()) / (float(v.norm()) + 1e-30)
        ua = u0 - alpha * g
        ea, _ = self.analysis_err(xb, t0, blk, ua)                 # direct dependence of E_a on theta at fixed u_a
        eb, _ = self.analysis_err(xb, t0, blk, None)
        l2 = (ea - eb) / eb_det

        def pend(scale=1.0, fd_eps=None):
            if alpha <= 0:
                return
            e_ = e if fd_eps is None else e * fd_eps / a.fd_eps      # fd_eps override (fdcheck sweeps it)
            with (fp32() if prec else contextlib.nullcontext()):
                for sgn in (1.0, -1.0):         # one at a time: the two graphs never coexist
                    J = self.J(self.roll(xb, t0, blk, a.n_win, u0 + sgn * e_ * v, True), obs, masks)
                    (-alpha * sgn * scale / (2 * e_) * J).backward()
        return l2, pend


    # ---- arm LB (see the module docstring)
    def init_B(self, weights):
        import collections
        import copy
        A, m = self.T.net, self.m
        hooks = A.latent_aggregator._forward_hooks      # the increment hook is bound to the model wrapper: re-register
        A.latent_aggregator._forward_hooks = collections.OrderedDict()
        try:
            self.B = copy.deepcopy(A)
        finally:
            A.latent_aggregator._forward_hooks = hooks
        self.B.latent_aggregator.register_forward_hook(m._add_latent_increment)
        if weights not in (None, '', 'A2'):
            self.B.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False)['state_dict'])
            log(f'loaded theta_B from {weights}')
        self.A = A
        for p in A.parameters():
            p.requires_grad_(False)

    def train_mode(self):
        if self.a.arm == 'LB':
            self.A.eval()
            self.B.train()
        else:
            self.T.net.train()

    def cotangents(self, xb, t0, blk, obs, masks, u):
        """A (frozen) at u: J and c_j = dJ/dy_j at A's outputs, detached"""
        with torch.no_grad():
            ys = self.roll(xb, t0, blk, self.a.n_win, u, False)
        yd = [y.detach().requires_grad_(True) for y in ys]
        Jv = self.J(yd, obs, masks)
        cs = torch.autograd.grad(Jv, yd, allow_unused=True)
        return Jv.detach(), [torch.zeros_like(y) if c is None else c for c, y in zip(cs, yd)]

    def S_B(self, xb, t0, blk, u, cs):
        ys = self.roll(xb, t0, blk, self.a.n_win, u, True, self.B)
        return sum((c * y.float()).sum() for c, y in zip(cs, ys))

    def grad_B(self, xb, t0, blk, obs, masks, u):
        """g_B = d/du sum_j <c_j, y^B_j(u)> with c from A at the same u; returns (g_B detached, J_A, c)"""
        Jv, cs = self.cotangents(xb, t0, blk, obs, masks, u)
        u_ = u.detach().requires_grad_(True)
        g, = torch.autograd.grad(self.S_B(xb, t0, blk, u_, cs), u_)
        return g.detach(), Jv, cs

    def lb(self, xb, t0, blk, obs, masks, prec=True):
        """Arm LB (K=1). Returns (l2, pend): l2 = (E_a - E_b)/E_b at 6 h, all from A, no graph (theta_B has no direct
        dependence); pend(scale) backprops the finite-difference surrogate of -alpha v . dg_B/dtheta_B into B's .grad."""
        a = self.a
        with torch.no_grad():
            eb_det = self.analysis_err(xb, t0, blk, None, ckpt=False)[0].detach()
        with (fp32() if prec else contextlib.nullcontext()):
            u0 = torch.zeros(self.m.latent_shape, device=self.T.dev)
            g, _, cs = self.grad_B(xb, t0, blk, obs, masks, u0)
            sref = a.s0 / (g.pow(2).mean().sqrt() + 1e-20)
            grid = [float(sref) * 2.0 ** k for k in range(-4, 6)]
            with torch.no_grad():
                es = [float(self.analysis_err(xb, t0, blk, u0 - al * g, ckpt=False)[0]) for al in grid]
            i = int(np.argmin(es))
            alpha = grid[i]
            if 0 < i < len(grid) - 1:
                x0, x1, x2 = (math.log(grid[i + k]) for k in (-1, 0, 1))
                y0, y1, y2 = es[i - 1], es[i], es[i + 1]
                den = (x0 - x1) * (x0 - x2) * (x1 - x2)
                A_ = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
                B_ = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
                if A_ > 0:
                    alpha = math.exp(min(max(-B_ / (2 * A_), x0), x2))
            if es[i] >= float(eb_det):
                alpha = 0.0
            self.last_lf = {'alpha': alpha, 'alpha_over_ref': alpha / float(sref), 'E_b': float(eb_det)}
            e = v = None
            if alpha > 0:
                ua = (u0 - alpha * g).requires_grad_(True)
                ea, _ = self.analysis_err(xb, t0, blk, ua)
                v, = torch.autograd.grad((ea - eb_det) / eb_det, ua)
                v = v.detach()
                e = a.fd_eps * float((alpha * g).norm()) / (float(v.norm()) + 1e-30)
            with torch.no_grad():
                l2 = (self.analysis_err(xb, t0, blk, u0 - alpha * g, ckpt=False)[0] - eb_det) / eb_det

        def pend(scale=1.0, fd_eps=None):
            if alpha <= 0:
                return
            e_ = e if fd_eps is None else e * fd_eps / a.fd_eps
            with (fp32() if prec else contextlib.nullcontext()):
                for sgn in (1.0, -1.0):
                    S = self.S_B(xb, t0, blk, u0 + sgn * e_ * v, cs)
                    (-alpha * sgn * scale / (2 * e_) * S).backward()
        return l2, pend

    def lb_true(self, xb, t0, blk, obs, masks, alpha, eb_det):
        """true double-backward gradient of (E_a(u_a) - E_b)/E_b w.r.t. theta_B at fixed alpha (fdcheck)"""
        u0 = torch.zeros(self.m.latent_shape, device=self.T.dev, requires_grad=True)
        _, cs = self.cotangents(xb, t0, blk, obs, masks, u0.detach())
        g, = torch.autograd.grad(self.S_B(xb, t0, blk, u0, cs), u0, create_graph=True)
        ea, _ = self.analysis_err(xb, t0, blk, -alpha * g)
        eb, _ = self.analysis_err(xb, t0, blk, None)
        return torch.autograd.grad((ea - eb) / eb_det, [p for p in self.B.parameters() if p.requires_grad],
                                   allow_unused=True)


def setup(a):
    torch.manual_seed(a.seed)
    gen = torch.Generator(device='cpu').manual_seed(a.seed + 1)
    T = A2Train(a.config, 'cuda')
    if a.weights not in (None, '', 'A2') and a.arm != 'LB':      # LB: the weights are theta_B, A stays the original
        sd = torch.load(a.weights, map_location='cpu', weights_only=False)['state_dict']
        T.net.load_state_dict(sd)
        log(f'loaded weights {a.weights}')
    L = L2A(T, a, gen)
    if a.arm == 'LB':
        L.init_B(a.weights)
    return T, gen, L


def feas(a):
    T, gen, L = setup(a)
    dates = date_range(dt.datetime(2021, 1, 3), dt.datetime(2021, 12, 20))
    S = Samples(dates[::97][:3], T.lagged, 4, T.names)
    out = []
    t0s, blk = S[0]
    t0 = dt.datetime.fromisoformat(t0s)
    T.net.train()
    params = [p for p in T.net.parameters() if p.requires_grad]
    for ckpt in (True, False):
        for K in (1, 2, 4):
            for n_win in (1, 2, 4):
                a.n_win, a.j_obs0 = n_win, (2 if n_win > 1 else 1)
                rec = {'ckpt': ckpt, 'K': K, 'n_win_steps': n_win, 'obs_steps': list(range(a.j_obs0, n_win + 1))}
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t_ = time.time()
                try:
                    with sdpa_kernel([SDPBackend.MATH]):
                        st, xb, obs, masks = L.draw(t0, blk)
                        u, Js = L.inner(xb, t0, blk, obs, masks, K, create_graph=True, ckpt=ckpt)
                        ea, _ = L.analysis_err(xb, t0, blk, u, ckpt=ckpt)
                        eb, _ = L.analysis_err(xb, t0, blk, None, ckpt=ckpt)
                        loss = (ea - eb) / eb.detach()
                        grads = torch.autograd.grad(loss, params + [L.log_s], allow_unused=True)
                    gn = math.sqrt(sum(float(g.pow(2).sum()) for g in grads[:-1] if g is not None))
                    rec.update(ok=True, sec=round(time.time() - t_, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                               J=Js, ratio_minus1=float(loss), grad_norm_weights=gn,
                               grad_log_s=[float(x) for x in grads[-1]] if grads[-1] is not None else None)
                    # how much of the weight gradient comes through the inner gradient (2nd-order path)?
                    if K == 1 and n_win == 4 and ckpt:
                        with sdpa_kernel([SDPBackend.MATH]):
                            u0, _ = L.inner(xb, t0, blk, obs, masks, 1, create_graph=False, ckpt=ckpt)
                            ea0, _ = L.analysis_err(xb, t0, blk, u0.detach(), ckpt=ckpt)
                            l0 = (ea0 - eb.detach()) / eb.detach()
                            g0 = torch.autograd.grad(l0, params, allow_unused=True)
                        dot = sum(float((x * y).sum()) for x, y in zip(grads[:-1], g0) if x is not None and y is not None)
                        n0 = math.sqrt(sum(float(y.pow(2).sum()) for y in g0 if y is not None))
                        rec.update(first_order_only_grad_norm=n0, cos_full_vs_first_order=dot / max(gn * n0, 1e-30))
                except torch.cuda.OutOfMemoryError as e:
                    rec.update(ok=False, err='OOM', sec=round(time.time() - t_, 1))
                except Exception as e:  # noqa: BLE001
                    rec.update(ok=False, err=f'{type(e).__name__}: {str(e)[:300]}')
                log(rec)
                out.append(rec)
                del rec
                st = xb = obs = masks = u = ea = eb = loss = grads = None   # free a failed graph before the next case
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        # LG timing (one double backward w.r.t. the physical state)
    a.n_win, a.j_obs0 = 4, 2
    torch.cuda.reset_peak_memory_stats()
    t_ = time.time()
    try:
        with sdpa_kernel([SDPBackend.MATH]):
            st, xb, obs, masks = L.draw(t0, blk)
            lg = L.lg_loss(xb, st, t0, blk, obs, masks)
            grads = torch.autograd.grad(lg, params, allow_unused=True)
        out.append({'arm': 'LG', 'ok': True, 'sec': round(time.time() - t_, 1),
                    'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1), 'one_minus_cos': float(lg)})
    except Exception as e:  # noqa: BLE001
        out.append({'arm': 'LG', 'ok': False, 'err': f'{type(e).__name__}: {str(e)[:300]}'})
    log(out[-1])
    json.dump(out, open(os.path.join(a.out, 'feas.json'), 'w'), indent=1)

def pgroup(n):
    for k in ('encoder', 'processor', 'decoder'):
        if k in n:
            return k
    return 'other'


def fdcheck(a):
    """Arm LF vs the true double-backward gradient of the same fixed-alpha loss on 2 samples, 1-step window, K=1.
    The true gradient is tried in fp32/TF32-off first (like the surrogate); if that OOMs it falls back to the model's
    normal precision and the surrogate is then also evaluated in that precision, so the pair is comparable.
    Reported per parameter group: cos and |surrogate|/|true| for the full loss (direct + second-order) and for the
    second-order part alone (true - direct vs S)."""
    T, gen, L = setup(a)
    a.n_win, a.j_obs0, a.K = 1, 1, 1
    dates = date_range(dt.datetime(2021, 1, 3), dt.datetime(2021, 12, 20))
    S = Samples(dates[::97][:2], T.lagged, 1, T.names)
    L.train_mode()
    named = [(n, p) for n, p in (L.B if a.arm == 'LB' else T.net).named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    out = []
    offload = a.offload is None or a.offload     # default on for fdcheck

    def flat(gs, idx):
        return torch.cat([(gs[i] if gs[i] is not None else torch.zeros_like(params[i])).flatten().float() for i in idx])

    def compare(gt, gs, tag):
        groups = {'all': list(range(len(named)))}
        for i, (n, _) in enumerate(named):
            groups.setdefault(pgroup(n), []).append(i)
        res = {}
        for k, idx in groups.items():
            x, y = flat(gt, idx), flat(gs, idx)
            res[k] = {'cos': float(x @ y / (x.norm() * y.norm()).clamp_min(1e-30)),
                      'norm_ratio': float(y.norm() / x.norm().clamp_min(1e-30))}
        return {tag: res}

    eps_list = a.fd_eps_list or [a.fd_eps]
    for i in range(len(S)):
        if not a.cases[0] <= i <= a.cases[1]:
            continue
        t0s, blk = S[i]
        t0 = dt.datetime.fromisoformat(t0s)
        free, tot = torch.cuda.mem_get_info()
        log(f'case {i}: GPU free {free / 2**30:.1f} / {tot / 2**30:.1f} GiB, torch allocated '
            f'{torch.cuda.memory_allocated() / 2**30:.1f} reserved {torch.cuda.memory_reserved() / 2**30:.1f}')
        L.gen.manual_seed(100 + i)
        L.AN.gen.manual_seed(100 + i)
        with sdpa_kernel([SDPBackend.MATH]):
            st, xb, obs, masks = L.draw(t0, blk)
            rec = {'t0': t0s}
            for prec in (True, False):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    # surrogate (also fixes alpha for the true gradient)
                    for p in params:
                        p.grad = None
                    with (fp32() if prec else contextlib.nullcontext()):   # the direct term in the same precision as the true gradient
                        l2, pend = (L.lb if a.arm == 'LB' else L.lf)(xb, t0, blk, obs, masks, prec=prec)
                        direct = [None] * len(params) if a.arm == 'LB' else [   # LB: no direct dependence on theta_B
                            None if g_ is None else g_.detach().clone()
                            for g_ in torch.autograd.grad(l2, params, allow_unused=True)]
                    sgrads = {}
                    for eps in eps_list:      # the two J(u0 +- e v) passes are cheap next to the true double backward
                        pend(1.0, eps)
                        sgrads[eps] = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                        for p in params:
                            p.grad = None
                    info = dict(L.last_lf)
                    alpha, eb_det = info['alpha'], info['E_b']
                    del l2, pend
                    torch.cuda.empty_cache()
                    # true double backward with the same fixed alpha
                    # saved tensors of the double-backward graph go to host memory (not pinned: pinned pages count to the cgroup and host-OOM'd at 180 GB)
                    with (fp32() if prec else contextlib.nullcontext()), \
                            (torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else contextlib.nullcontext()):
                        if a.arm == 'LB':
                            tgrad = L.lb_true(xb, t0, blk, obs, masks, alpha, eb_det)
                        else:
                            u0 = torch.zeros(T.model.latent_shape, device=T.dev, requires_grad=True)
                            g, = torch.autograd.grad(L.J(L.roll(xb, t0, blk, 1, u0, True), obs, masks), u0,
                                                     create_graph=True)
                            ea, _ = L.analysis_err(xb, t0, blk, -alpha * g)
                            eb, _ = L.analysis_err(xb, t0, blk, None)
                            tgrad = torch.autograd.grad((ea - eb) / eb_det, params, allow_unused=True)
                    tgrad = [None if g_ is None else g_.detach() for g_ in tgrad]
                    second = [None if t_ is None else t_ - (d_ if d_ is not None else 0) for t_, d_ in zip(tgrad, direct)]
                    # the direct term alone: how much of the true gradient the second-order (dg/dtheta) part carries
                    rec.update(compare(tgrad, direct, f'{"fp32" if prec else "default_precision"}_direct_only'))
                    rec.update(compare(tgrad, second, f'{"fp32" if prec else "default_precision"}_second_only'))
                    tag = 'fp32' if prec else 'default_precision'
                    rec.update(info=info)
                    rec[f'{tag}_peak_gpu_gb'] = torch.cuda.max_memory_allocated() / 2**30
                    for eps, sgrad in sgrads.items():
                        rec.update(compare(tgrad, [None if s_ is None else s_ + (d_ if d_ is not None else 0)
                                                   for s_, d_ in zip(sgrad, direct)], f'{tag}_full_eps{eps:g}'))
                        rec.update(compare(second, sgrad, f'{tag}_second_order_eps{eps:g}'))
                    rec[f'{tag}_ok'] = True
                    del tgrad, second, direct, sgrads
                    if prec:
                        break              # the true fp32 gradient fit: no need for the fallback
                except torch.cuda.OutOfMemoryError as e:
                    rec[('fp32' if prec else 'default_precision') + '_ok'] = 'OOM: ' + str(e)[:200]
                except Exception as e:  # noqa: BLE001
                    rec[('fp32' if prec else 'default_precision') + '_ok'] = f'{type(e).__name__}: {str(e)[:300]}'
                import gc
                u0 = g = ea = eb = None
                gc.collect()
        rec['host_maxrss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        log(rec)
        out.append(rec)
    json.dump(out, open(os.path.join(a.out, 'fdcheck.json'), 'w'), indent=1)
    bad = [r['t0'] for r in out if not any(v is True for k, v in r.items() if k.endswith('_ok'))]
    if bad:
        sys.exit(f'fdcheck: no successful gradient comparison for {bad}')


def evaluate(a, T=None, L=None, tag=None, n=None, twin=None):
    """held-out 2021: background ratio after the inner loop, increment cosine with the true 6-h error"""
    if T is None:
        T, gen, L = setup(a)
    T.net.eval()
    if a.arm == 'LB':
        L.B.eval()
    train_twin = L.twin
    if twin is not None:
        L.twin = twin
    dates = date_range(dt.datetime(2021, 1, 5), dt.datetime(2021, 12, 20))
    n = n or a.n_eval
    S = Samples([dates[i] for i in np.linspace(0, len(dates) - 1, n).astype(int)], T.lagged, a.n_win, T.names)
    L.gen.manual_seed(999)                       # the same backgrounds/obs for every model
    L.AN.gen.manual_seed(999)
    for p in L.AN.pools:
        p.gen.manual_seed(1999) if hasattr(p, 'gen') and isinstance(p.gen, torch.Generator) else None
    grid = [0.005, 0.01, 0.02, 0.04, 0.08]
    res = []
    maps = {}
    for i in range(len(S)):
        t0s, blk = S[i]
        t0 = dt.datetime.fromisoformat(t0s)
        st, xb, obs, masks = L.draw(t0, blk)
        with torch.no_grad():
            eb, xb1 = L.analysis_err(xb, t0, blk, None, ckpt=False)
            eb = float(eb)
        # one gradient, best step along it (direction quality, independent of the step size)
        u = torch.zeros(T.model.latent_shape, device=T.dev, requires_grad=True)
        if a.arm == 'LB':
            g, _, _ = L.grad_B(xb, t0, blk, obs, masks, u)
        else:
            ys = L.roll(xb, t0, blk, a.n_win, u, ckpt=True)
            g, = torch.autograd.grad(L.J(ys, obs, masks), u)
        g = g / (g.pow(2).mean().sqrt() + 1e-20)
        rec = {'t0': t0s, 'eb': eb}
        best = (9.9, None)
        with torch.no_grad():
            for s in grid:
                ea, xa1 = L.analysis_err(xb, t0, blk, -s * g, ckpt=False)
                r = float(ea) / eb
                rec[f'r1_s{s}'] = r
                if r < best[0]:
                    best = (r, xa1)
            rec['r1_best'] = best[0]
            inc = (best[1] - xb1)[0, -1][:, T.P].float() * T.in_mul
            err = (L.next_true(blk) - xb1[0, -1][:, T.P].float()) * T.in_mul
            w = T.aw[:, None]
            rec['cos1_best'] = float((w * inc * err).sum() / ((w * inc * inc).sum() * (w * err * err).sum()).sqrt())
            if a.dump_maps:      # physical-unit 6-h fields: truth, forecast from the noisy x_b, its best-step analysis, clean start
                vi = [T.prog_names.index(v) for v in MAP_VARS]
                _, xc1 = L.analysis_err(st, t0, blk, None, ckpt=False)
                for k, f in (('truth', L.next_true(blk)), ('xb6', xb1[0, -1][:, T.P].float()),
                             ('xa6', best[1][0, -1][:, T.P].float()), ('clean6', xc1[0, -1][:, T.P].float())):
                    maps[f'c{i}_{k}'] = f[:, vi].cpu().numpy().astype(np.float32)
        # K-step normalised descent with the model's learned / default steps
        steps = [float(torch.exp(L.log_s[min(k, len(L.log_s) - 1)])) for k in range(a.K_eval)]
        uK, Js = L.inner(xb, t0, blk, obs, masks, a.K_eval, create_graph=False, ckpt=True, steps=steps)
        with torch.no_grad():
            ea, xa1 = L.analysis_err(xb, t0, blk, uK.detach(), ckpt=False)
            rec['rK'] = float(ea) / eb
            rec['J'] = Js
            inc = (xa1 - xb1)[0, -1][:, T.P].float() * T.in_mul
            err = (L.next_true(blk) - xb1[0, -1][:, T.P].float()) * T.in_mul
            w = T.aw[:, None]
            rec['cosK'] = float((w * inc * err).sum() / ((w * inc * inc).sum() * (w * err * err).sum()).sqrt())
        res.append(rec)
        log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != 'J'})
    if a.dump_maps:
        np.savez_compressed(a.dump_maps, lat=np.asarray(T.model.lats), lon=np.asarray(T.model.lons), vars=np.array(MAP_VARS),
                            t0=np.array([r['t0'] for r in res]), **maps)
    summ = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k not in ('t0', 'J')}
    summ['n'] = len(res)
    summ['tag'] = tag or a.tag
    L.twin = train_twin
    log({'eval_summary': summ})
    return summ


def obsbase(a):
    """Observing-system baseline with the UNCHANGED model: a converged inner 4D-Var (Adam on the latent increment, as
    production) from ERA5 + analysis noise against DENSE / sparse ERA5 pseudo-obs, settings:
      a6     dense obs at t0+6h only (the increment acts directly: tests the control + optimizer, an upper bound)
      b24    dense obs at t0+24h only (the adjoint must carry the information back 18 h)
      b48    dense obs at t0+48h only (--b48 cases)
      c5     random 5% of points at 6, 12, 18, 24 h
    Scores at the 6-h injection time: E_a/E_b (analysis / background error, normalised all-variable, A2 input norm)
    and the cosine of the increment with the true error, for z500, u250, q700 and all variables."""
    T, gen, L = setup(a)
    T.net.eval()
    dates = date_range(dt.datetime(2021, 1, 5), dt.datetime(2021, 12, 20))
    sel = [dates[i] for i in np.linspace(0, len(dates) - 1, a.n_eval).astype(int)]
    S = Samples(sel, T.lagged, 8, T.names)
    VARS = {'z500': 'z_500', 'u250': 'u_250', 'q700': 'q_700'}
    vi = {k: T.prog_names.index(v) for k, v in VARS.items()}
    w = T.aw[:, None]
    settings = [s for s in a.settings]
    res = {k: [] for k in settings}
    for i in range(a.cases[0], min(a.cases[1], len(S))):
        t0s, blk = S[i]
        t0 = dt.datetime.fromisoformat(t0s)
        L.gen.manual_seed(1000 + i)
        L.AN.gen.manual_seed(2000 + i)
        for p_ in L.AN.pools:
            p_.gen.manual_seed(3000 + i)
        st = T.initial(t0, blk)
        xb = L.AN.apply(st, T.model, both_levels=True, lag_corr=a.lag_corr)
        with torch.no_grad():
            xb1 = L.step_u(xb, t0 + H6, blk, 1, None)[0][0, -1][:, T.P].float()
        err = (st_next_true(T, blk) - xb1) * T.in_mul
        npts = len(T.aw)
        for name in settings:
            if name == 'b48' and i >= a.b48:
                continue
            if name == 'a6':
                js, frac = [1], 1.0
            elif name == 'b24':
                js, frac = [4], 1.0
            elif name == 'b48':
                js, frac = [8], 1.0
            else:
                js, frac = [1, 2, 3, 4], 0.05
            g2 = torch.Generator(device='cpu').manual_seed(4000 + i)
            masks = {j: ((torch.rand(npts, generator=g2) < frac).float().to(T.dev) if frac < 1 else
                         torch.ones(npts, device=T.dev)) for j in js}
            obs = {j: T.target(blk, j) for j in js}
            u = torch.zeros(T.model.latent_shape, device=T.dev, requires_grad=True)
            opt = torch.optim.Adam([u], lr=a.inner_lr)
            n = max(js)
            Js = []
            t_ = time.time()
            for it in range(a.inner_iters):
                for g_ in opt.param_groups:
                    g_['lr'] = a.inner_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / a.inner_iters)))
                ys = L.roll(xb, t0, blk, n, u, ckpt=True)
                Jk = L.J(ys, obs, masks)
                opt.zero_grad()
                Jk.backward()
                opt.step()
                Js.append(float(Jk))
            with torch.no_grad():
                xa1 = L.step_u(xb, t0 + H6, blk, 1, u.detach())[0][0, -1][:, T.P].float()
            inc = (xa1 - xb1) * T.in_mul
            rec = {'t0': t0s, 'J0': Js[0], 'Jend': Js[-1], 'sec': round(time.time() - t_, 1)}
            for k, sl in [('all', slice(None))] + [(k, [v]) for k, v in vi.items()]:
                e, d = err[:, sl], inc[:, sl]
                ee, dd, ed = (w * e * e).sum(), (w * d * d).sum(), (w * e * d).sum()
                rec[f'cos_{k}'] = float(ed / (ee * dd).sqrt().clamp_min(1e-30))
                rec[f'ratio_{k}'] = float(((w * (e - d) ** 2).sum() / ee).sqrt())   # rms(analysis err)/rms(bg err)
            if a.dump_maps_obs:      # physical-unit 6-h fields: true background error, analysis increment, residual
                vv = {'z500': 'z_500', 'u250': 'u_250', 'q700': 'q_700'}
                D = {}
                for k_, v_ in vv.items():
                    j_ = T.prog_names.index(v_)
                    e_p = (err[:, j_] / T.in_mul[j_]).cpu().numpy().astype(np.float32)
                    i_p = ((xa1 - xb1)[:, j_]).cpu().numpy().astype(np.float32)
                    D.update({f'err_{k_}': e_p, f'inc_{k_}': i_p, f'resid_{k_}': e_p - i_p})
                os.makedirs(a.dump_maps_obs, exist_ok=True)
                np.savez_compressed(os.path.join(a.dump_maps_obs, f'obsmaps_{a.tag}_case{i}_{name}.npz'),
                                    lat=np.asarray(T.model.lats), lon=np.asarray(T.model.lons), aw=T.aw.cpu().numpy(),
                                    t0=t0s, setting=name, case=i, cos_all=rec['cos_all'], ratio_all=rec['ratio_all'], **D)
            res[name].append(rec)
            log({'setting': name, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}})
    summ = {}
    for name, rs in res.items():
        if rs:
            summ[name] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k != 't0'}
            summ[name]['n'] = len(rs)
    log({'obsbase_summary': summ, 'weights': a.weights})
    json.dump({'summary': summ, 'cases': res, 'weights': a.weights}, open(os.path.join(a.out, f'obsbase_{a.tag}.json'), 'w'), indent=1)


def st_next_true(T, blk):
    """ERA5 at t0+6h on the prognostic input variables (physical units), from the block"""
    i = len(T.lagged)               # dates: lagged inputs, then t0+6h ...
    names = T.prog_names
    return torch.as_tensor(np.stack([blk[i, T.pos[n]] for n in names], -1), device=T.dev).float()


def evaluate_both(a, T, L, tag):
    """twin-truth training: eval against the twin truth (tag_twin) and against real ERA5 (tag_real); else one eval"""
    if not a.twin_truth:
        return evaluate(a, T, L, tag=tag, n=a.n_eval)
    st = evaluate(a, T, L, tag=f'{tag}_twin', n=a.n_eval, twin=True)
    sr = evaluate(a, T, L, tag=f'{tag}_real', n=a.n_eval, twin=False)
    return {'twin': st, 'real': sr}


def train(a):
    T, gen, L = setup(a)
    assert not a.twin_truth or a.arm == 'LB', '--twin_truth is implemented for arm LB only'
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, 'history.jsonl'), 'a')
    log({'config': vars(a)}, fh)
    kmax = max(a.n_win, 1)
    tr_dates = [d for d in date_range(dt.datetime(1979, 1, 2), dt.datetime(2020, 12, 31, 18))
                if d + H6 * kmax <= dt.datetime(2020, 12, 31, 18)]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(tr_dates))[:a.steps * a.accum]
    S = Samples([tr_dates[i] for i in order], T.lagged, kmax, T.names)
    dl = torch.utils.data.DataLoader(S, batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=2)
    params = [p for p in (L.B if a.arm == 'LB' else T.net).parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': params, 'lr': a.lr, 'weight_decay': 0.01},
                             {'params': [L.log_s], 'lr': a.lr_s, 'weight_decay': 0.0}], betas=(0.9, 0.95))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        f = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 3e-7 + 0.5 * (a.lr - 3e-7) * (1 + math.cos(math.pi * f))

    beta = None if a.beta == 'auto' else float(a.beta)
    it = iter(dl)
    L.train_mode()
    t_start = time.time()
    if a.eval_every:
        evaluate_both(a, T, L, 'step0')
        L.train_mode()
    for s in range(a.steps):
        opt.param_groups[0]['lr'] = lr_at(s)
        tot = {'mse': 0.0, 'l2a': 0.0}
        for b in range(a.accum):
            t0s, blk = next(it)
            t0 = dt.datetime.fromisoformat(t0s)
            blk = np.asarray(blk)
            if a.arm == 'LB':
                lm = torch.zeros((), device=T.dev)                   # B never forecasts: A does, and A is frozen
            else:
                st = T.initial(t0, blk)
                _, y = T.step(st, t0 + H6, blk, 1)                   # normal forecast loss (clean ERA5 start)
                lm = T.mse(y, T.target(blk, 1))
            if a.arm in ('LF', 'LB'):                                # first order: no MATH-kernel restriction
                st, xb, obs, masks = L.draw(t0, blk)
                l2, pend = (L.lb if a.arm == 'LB' else L.lf)(xb, t0, blk, obs, masks)
            else:
                with sdpa_kernel([SDPBackend.MATH]):
                    st, xb, obs, masks = L.draw(t0, blk)
                    if a.arm == 'LG':
                        l2 = L.lg_loss(xb, st, t0, blk, obs, masks)
                    else:
                        u, _ = L.inner(xb, t0, blk, obs, masks, a.K, create_graph=True)
                        ea, _ = L.analysis_err(xb, t0, blk, u)
                        eb, _ = L.analysis_err(xb, t0, blk, None)
                        l2 = (ea - eb) / eb.detach()
            if beta is None and a.arm == 'LB':
                beta = 1.0                 # Adam is scale-free; there is no forecast loss to balance against
            if beta is None:
                beta = a.beta_frac / (1 - a.beta_frac) * float(lm) / max(abs(float(l2)), 1e-6)
                log({'beta_auto': beta, 'mse0': float(lm), 'l2a0': float(l2)}, fh)
            lnoisy = 0.0
            if a.noisy_fcst_w > 0 and a.arm == 'LF':      # the clean-start MSE does not constrain the response to analysis noise in x_b
                _, yn = T.step(xb, t0 + H6, blk, 1)
                lnoisy = a.noisy_fcst_w * T.mse(yn, T.target(blk, 1))
            if a.arm != 'LB':              # LB: no graph here, the whole gradient comes from pend
                ((lm + lnoisy + beta * l2) / a.accum).backward()
            if a.arm in ('LF', 'LB'):
                pend(beta / a.accum)       # alpha is held fixed (its theta dependence is ignored on purpose)
                log({'lf': L.last_lf}, fh) if (s % 10 == 0 or a.smoke) else None
            tot['mse'] += float(lm) / a.accum
            tot['l2a'] += float(l2) / a.accum
        if a.arm == 'LB' and a.anchor > 0:
            for pA, pB in zip(L.A.parameters(), L.B.parameters()):
                if pB.grad is not None:
                    pB.grad += 2 * a.anchor * (pB.detach() - pA)
        torch.nn.utils.clip_grad_value_(params, 32.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or a.smoke:
            log({'step': s + 1, 'lr': lr_at(s), 'train_mse': tot['mse'], 'train_l2a(ratio-1)': tot['l2a'],
                 's_k': [round(float(torch.exp(x)), 5) for x in L.log_s], 'sec_per_step': (time.time() - t_start) / (s + 1),
                 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 2**30,
                 'host_maxrss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}, fh)
        if a.eval_every and (s + 1) % a.eval_every == 0:
            summ = evaluate_both(a, T, L, f'step{s + 1}')
            log({'eval': summ, 'step': s + 1}, fh)
            L.train_mode()
        if (s + 1) in a.save_at or (s + 1) == a.steps:
            path = os.path.join(a.out, f'step_{s + 1}.pt')
            torch.save({'state_dict': (L.B if a.arm == 'LB' else T.net).state_dict(), 'log_s': L.log_s.detach().cpu(), 'step': s + 1,
                        'arm': a.arm, 'args': vars(a)}, path)
            log({'saved': path}, fh)
    fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['feas', 'fdcheck', 'train', 'eval', 'obsbase'], required=True)
    ap.add_argument('--arm', default='LA1', choices=['LA1', 'LA2', 'LG', 'LF', 'LB'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='a37tmp/l2a/config_l2a.yml')
    ap.add_argument('--weights', default='A2')
    ap.add_argument('--tag', default='')
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--K_eval', type=int, default=4)
    ap.add_argument('--n_win', type=int, default=4, help='window length in 6-h steps (obs at steps j_obs0..n_win)')
    ap.add_argument('--j_obs0', type=int, default=2)
    ap.add_argument('--obs_frac', type=float, default=1.0)
    ap.add_argument('--obs_noise', type=float, default=0.0)
    ap.add_argument('--fd_eps', type=float, default=1e-3, help='arm LF: FD step e = fd_eps |u0-u_a| / |v|')
    ap.add_argument('--fd_eps_list', type=float, nargs='+', default=None, help='fdcheck: sweep fd_eps over these values')
    ap.add_argument('--anchor', type=float, default=0.0, help='arm LB: lam |theta_B - theta_A|^2')
    ap.add_argument('--s0', type=float, default=0.02)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--lr_s', type=float, default=1e-2)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--beta', default='auto')
    ap.add_argument('--beta_frac', type=float, default=0.3)
    ap.add_argument('--save_at', type=int, nargs='+', default=[500, 1000, 1500])
    ap.add_argument('--eval_every', type=int, default=500)
    ap.add_argument('--n_eval', type=int, default=12)
    ap.add_argument('--noise_stats', default='output/finetune/analysis_noise_stats.npz')
    ap.add_argument('--lag_corr', type=float, default=0.27)
    ap.add_argument('--amp_range', type=float, nargs=2, default=[0.5, 1.5])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--settings', nargs='+', default=['a6', 'b24', 'c5', 'b48'])
    ap.add_argument('--cases', type=int, nargs=2, default=[0, 99])
    ap.add_argument('--inner_iters', type=int, default=150)
    ap.add_argument('--inner_lr', type=float, default=0.003)
    ap.add_argument('--b48', type=int, default=4, help='obsbase: cases with the 48-h setting')
    ap.add_argument('--offload', action=argparse.BooleanOptionalAction, default=None,
                    help='fdcheck: save the true double-backward graph on host memory (default on for fdcheck)')
    ap.add_argument('--dump_maps', default='', help='eval: npz of per-case 6-h fields (z500, u250, v250) for the maps')
    ap.add_argument('--dump_maps_obs', default='', help='obsbase: dir for per-case/setting npz of 6-h err, inc, resid (z500, u250, q700)')
    ap.add_argument('--noisy_fcst_w', type=float, default=0.0,
                    help='train: weight of the 6-h forecast MSE from the NOISY x_b vs ERA5 (extra term, normalised like MSE_fcst)')
    ap.add_argument('--twin_truth', action='store_true',
                    help='arm LB: obs and 6-h truth from A\'s own forward trajectory (zero model error); evals report both')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.accum, a.save_at, a.eval_every, a.n_eval, a.warmup = 4, 1, [4], 2, 2, 1
    os.makedirs(a.out, exist_ok=True)
    {'feas': feas, 'fdcheck': fdcheck, 'train': train, 'obsbase': obsbase, 'eval': lambda a_: json.dump(evaluate(a_), open(os.path.join(a_.out, f'eval_{a_.tag}.json'), 'w'), indent=1)}[a.mode](a)


if __name__ == '__main__':
    main()
