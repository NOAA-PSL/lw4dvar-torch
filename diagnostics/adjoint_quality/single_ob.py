"""Single-observation balance test (notes/adjoint_quality.md, part `single_ob`).

When 4D-Var through the model fits ONE mass observation, is the 6-h analysis increment balanced (a geostrophically
consistent wind of realistic size, broad synoptic shape) or a small-scale unbalanced patch, and how far does it spread?
Background x0 = the config's first guess; latent increment u injected on step 1 (as 4D-Var does); the model's
production precision.
Obs (innovations relative to the background): a = z500 +30 m at 6 h (sigma 10 m), b = the same at 24 h,
c = ps -2 hPa at 6 h (sigma 1 hPa). H = Gaussian-weighted mean of grid values within 200 km (e-folding 100 km).
Solvers: G  = the raw adjoint direction dH/du, scaled by a line search to fit the ob;
         BG = the B-filtered adjoint direction (L L^T g, class latent covariance; only with --b_cov), line search;
         W  = AdamW with the config's (golden) settings, `iters` iterations.
Saves analysis - background at 6, 24, 48 h for selected fields plus column profiles at the ob, one npz per
(date, location, ob, solver), and a summary JSON per (date, location).
"""
import json
import os

import numpy as np
import torch

from .battery import ROOT, xyz

G0 = 9.80665
LOCS = {'sh': (-50.0, 90.0), 'nh': (45.0, 180.0)}
OBS = {'a': ('z_500', 1, 30 * G0, 10 * G0), 'b': ('z_500', 4, 30 * G0, 10 * G0), 'c': ('sp', 1, -200.0, 100.0)}
SAVE_F = ['z_500', 'z_250', 't_500', 't_850', 'u_500', 'v_500', 'u_250', 'v_250', 'sp', 'q_700']
SAVE_STEPS = [1, 4, 8]


def parse_loc(loc):
    if loc in LOCS:
        return loc, LOCS[loc]
    la, lo = (float(x) for x in loc.split(','))
    return f'{la:g}N{lo:g}E'.replace('-', 'm'), (la, lo)      # no '_': file names are {mmdd}_{loc}_{ob}_{solver}


def run(S, outdir, loc, obs=('a', 'b', 'c'), iters=200, b_cov=None, sigma_b=0.07, adjoint_filter=None):
    """adjoint_filter: an exp.adjoint_filter dict (adjoint_filter.py) applied to every gradient (G and W solvers)"""
    A, exp, dev, log = S.A, S.exp, S.dev, S.log
    os.makedirs(outdir, exist_ok=True)
    tag_loc, (olat, olon) = parse_loc(loc)
    V = A.var_to_idx
    lat, lon = S.lat, S.lon
    PROF = {b: sorted([(int(n.split('_')[1]), i) for n, i in V.items()
                       if n.startswith(b + '_') and n.split('_')[1].isdigit()], reverse=True) for b in ('t', 'z', 'u', 'v', 'q')}
    d_km = 6371.0 * np.arccos(np.clip(xyz(lat, lon) @ xyz(np.array([olat]), np.array([olon]))[0], -1, 1))
    sel = np.flatnonzero(d_km < 200.0)
    wts = np.exp(-(d_km[sel] / 100.0) ** 2)
    wts /= wts.sum()
    SEL = torch.as_tensor(sel, device=dev)
    WTS = torch.as_tensor(wts, device=dev, dtype=torch.float32)
    log(f'ob at {olat},{olon}: {len(sel)} grid points, nearest {d_km.min():.0f} km')
    OB = {k: OBS[k] for k in obs}
    nmax = max(SAVE_STEPS)
    latent_scale = float(exp.get('latent_scale', 1.0))
    if A.reduced_precision():
        A.set_precision(A.reduced_precision())      # production precision

    def H(state, name):
        return (A.field(state, name)[SEL].float() * WTS).sum()

    AF = None
    if adjoint_filter:
        from adjoint_filter import AdjointFilter
        AF = AdjointFilter(S.model, adjoint_filter, S.logger)
        AF.compute_modes(S.x0, adapter=A)
        if A.reduced_precision():
            A.set_precision(A.reduced_precision())

    def rollout(u, nsteps, grad=False):
        x = A.advance(S.x0, 1, grad=grad, latent_increment=u)
        if grad and AF is not None:
            x = AF.apply(x, 1)
        out = {1: x}
        for s in range(2, nsteps + 1):
            x = A.advance(x, 1, grad=grad)
            if grad and AF is not None:
                x = AF.apply(x, s)
            out[s] = x
        return out

    with torch.no_grad():
        bg = rollout(None, nmax)
    bg_y = {k: float(H(bg[s], n)) for k, (n, s, _, _) in OB.items()}
    log(f'background H: {bg_y}')

    B_of = None
    if b_cov:      # class B, replicating compute_optimal._prior_map for a per-class b_cov
        from latent_classes import NodeClasses
        ev = np.load(b_cov if os.path.isabs(b_cov) else os.path.join(ROOT, b_cov))
        lam = torch.as_tensor(ev['eigvals'], dtype=torch.float64).clamp(min=0)
        E = torch.as_tensor(ev['eigvecs'], dtype=torch.float64)
        cls = NodeClasses(ev['lat'], ev['lon']).band_surface(exp['date'])
        n_per = np.bincount(cls, minlength=E.shape[0])
        m_ = float((lam.mean(1).numpy() * n_per).sum() / n_per.sum())
        s_ = (lam / m_).sqrt()
        Lt = [((E[k] * s_[k]).T * sigma_b * latent_scale).to(device=dev, dtype=torch.float32) for k in range(E.shape[0])]
        IDX = [torch.as_tensor(np.flatnonzero(cls == k), device=dev) for k in range(E.shape[0])]

        def apply(t, Ms, transpose=False):
            out = torch.zeros_like(t)
            for k, ik in enumerate(IDX):
                if len(ik):
                    out = out.index_copy(-2, ik, t.index_select(-2, ik) @ (Ms[k].T if transpose else Ms[k]))
            return out

        def B_of(g):
            return apply(apply(g, Lt, True), Lt)

    def grad_H(name, step):
        u = torch.zeros(A.latent_shape, device=dev, requires_grad=True)
        y = H(rollout(u / latent_scale, step, grad=True)[step], name)
        g, = torch.autograd.grad(y, u)
        return g

    def line_fit(d, name, step, target, lin):
        """scale a so that H(x(a d)) - H(x0) = target (secant on a)"""
        bgv = float(H(bg[step], name))

        def f(a):
            with torch.no_grad():
                return float(H(rollout(a * d / latent_scale, step)[step], name)) - bgv - target
        a0, a1 = 0.0, target / lin
        f0, f1 = -target, f(a1)
        for _ in range(8):
            if abs(f1) < 0.02 * abs(target) or f1 == f0:
                break
            a0, a1, f0 = a1, a1 - f1 * (a1 - a0) / (f1 - f0), f1
            f1 = f(a1)
        return a1, f1 + target

    def adam_solve(name, step, y, sig):
        lr, wd = exp['learn_rate'], exp['weight_decay']
        v = torch.zeros(A.latent_shape, device=dev, requires_grad=True)
        opt = torch.optim.AdamW([v], lr=lr, weight_decay=wd, betas=(exp.get('beta1', 0.9), exp.get('beta2', 0.99)))
        ws = min(exp['warmup_steps'], max(1, iters // 2))
        sch = torch.optim.lr_scheduler.SequentialLR(opt, [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=exp['start_factor'], end_factor=1.0, total_iters=ws),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters - ws), eta_min=lr * exp['end_factor'])],
            milestones=[ws])
        best, hist = (1e30, None), []
        for it in range(iters):
            r = (H(rollout(v / latent_scale, step, grad=True)[step], name) - y) / sig
            J = r * r
            if float(J.detach()) < best[0]:
                best = (float(J), v.detach().clone())
            opt.zero_grad()
            J.backward()
            opt.step()
            sch.step()
            hist.append((float(J), float(r)))
            if it % 20 == 0 or it == iters - 1:
                log(f'  it {it}: J {float(J):.4g} r {float(r):.3f}')
        return (best[1] / latent_scale).detach(), hist

    def save(tag, inj, extra):
        with torch.no_grad():
            an = rollout(inj, nmax)
        D = {}
        for s in SAVE_STEPS:
            for f in SAVE_F:
                D[f'{f}_{6 * s}'] = (A.field(an[s], f) - A.field(bg[s], f)).float().cpu().numpy()
            for b, lv in PROF.items():
                D[f'prof_{b}_{6 * s}'] = np.array([float(((A.field(an[s], f'{b}_{p}') - A.field(bg[s], f'{b}_{p}'))[SEL].float()
                                                          * WTS).sum()) for p, _ in lv])
        for f in ('z_500', 'u_500', 'v_500'):
            D[f'bg_{f}_6'] = A.field(bg[1], f).float().cpu().numpy()
        fit = {k: float(H(an[s], n)) - bg_y[k] for k, (n, s, _, _) in OB.items()}
        np.savez_compressed(os.path.join(outdir, f'{S.mmdd}_{tag_loc}_{tag}.npz'), lat=lat, lon=lon, olat=olat, olon=olon,
                            **{f'levels_{b}': np.array([p for p, _ in lv]) for b, lv in PROF.items()},
                            latent_norm=float(inj.norm()), **D)
        R = dict(extra, fit=fit, latent_norm=float(inj.norm()))
        log(f'{tag}: fit {fit}, |inj| {R["latent_norm"]:.3g}')
        return R

    results = {'date': S.date, 'loc': tag_loc, 'olat': olat, 'olon': olon, 'bg_H': bg_y}
    for ok, (name, step, innov, sig) in OB.items():
        y = bg_y[ok] + innov
        log(f'ob {ok}: {name} step {step} innovation {innov} sigma {sig}')
        g = grad_H(name, step)
        sols = [('G', g)] + ([('BG', B_of(g))] if B_of is not None else [])
        for tag, d in sols:
            a, got = line_fit(d, name, step, innov, float((g * d).sum()))
            results[f'{ok}_{tag}'] = save(f'{ok}_{tag}', a * d / latent_scale, {'scale': a, 'H_change': got})
        inj, hist = adam_solve(name, step, y, sig)
        results[f'{ok}_W'] = save(f'{ok}_W', inj, {'J_first_last': [hist[0], hist[-1]]})
        torch.cuda.empty_cache()
    return results
