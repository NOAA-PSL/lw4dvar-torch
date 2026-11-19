"""Perturbation-family amplification (notes/adjoint_finetune_plan.md, step 0).

How strongly does the model amplify each KIND of input perturbation? Each family is a perturbation of the 6-h state
(the last time level after the first step from the 4D-Var background, where 4D-Var's increment acts), built in
physical units, rescaled to a common size in the normalised all-prognostic norm, and run forward in fp32 next to an
unperturbed control. Growth = |difference at lead| / |perturbation| at 6, 24, 48 h after the perturbation.

Families (random ones: smoothed / high-passed Gaussian noise; 'large' = 3 passes of a 200-nearest-neighbour mean,
roughly > 1000 km; 'small' = white noise minus 3 passes of a 24-neighbour mean, roughly < 400 km):
  hyd_z_large / hyd_z_small   z on all levels, no matching t       (hydrostatic imbalance)
  hyd_t_large                 t on all levels, z unchanged          (hydrostatic imbalance)
  geo_wind_large              u, v only, no matching mass field     (geostrophic imbalance)
  small_all                   small-scale noise in t, z, u, v, q, sp/msl, per-variable background-error sizes
  nmc_full / nmc_large        A2 lagged-forecast difference (fg24 - fg12 at the same valid time): balanced, B-like
  a2err_full / a2err_small    A2's own 6-h forecast error (A2 6 h from ERA5 minus ERA5): what rollout training sees
Two sizes (linear check). Per-variable sizes for the random families: t 0.5 K, z 5 m, u/v 1 m/s, q 0.3 g/kg,
sp/msl 50 Pa.
"""
import numpy as np
import torch

import long_window_4dvar_utils as utils

SIG = {'t': 0.5, 'z': 5 * 9.80665, 'u': 1.0, 'v': 1.0, 'q': 3e-4, 'sp': 50.0, 'msl': 50.0}
FAMILIES = ('hyd_z_large', 'hyd_z_small', 'hyd_t_large', 'geo_wind_large', 'small_all',
            'nmc_full', 'nmc_large', 'a2err_full', 'a2err_small')


def _base(name):
    b, _, lv = name.rpartition('_')
    return b if (b and lv.isdigit()) else name


def run(S, sizes=(0.02, 0.005), leads_h=(6, 24, 48), n_real=2, seed=0):
    A = S.A
    A.set_precision('fp32')
    dev, npts, nP = S.dev, len(S.lat), S.nP
    names = A.prog_names
    base = [_base(n) for n in names]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    ts = int(round(S.model.timestep.total_seconds() / 3600))
    steps = [h // ts for h in leads_h]
    nstep = max(steps)
    nb_large = None

    def smooth_large(f):
        nonlocal nb_large
        if nb_large is None:
            from scipy.spatial import cKDTree
            from .battery import xyz
            _, nb = cKDTree(xyz(S.lat, S.lon)).query(xyz(S.lat, S.lon), k=200)
            nb_large = torch.as_tensor(nb, device=dev)
        for _ in range(3):
            f = f[nb_large].mean(1)
        return f

    def noise():
        return torch.randn(npts, nP, generator=gen).to(dev)

    def var_mask(keep):
        return torch.tensor([1.0 if b in keep else 0.0 for b in base], device=dev)

    sig = torch.tensor([SIG.get(b, 0.0) for b in base], device=dev)

    def random_family(fam):
        e = noise()
        if fam.endswith('_large'):
            e = smooth_large(e)
        else:
            e = e - S.smooth(e)
        e = e / e.std(0, keepdim=True).clamp_min(1e-12)
        keep = {'hyd_z_large': {'z'}, 'hyd_z_small': {'z'}, 'hyd_t_large': {'t'}, 'geo_wind_large': {'u', 'v'},
                'small_all': {'t', 'z', 'u', 'v', 'q', 'sp', 'msl'}}[fam]
        return e * sig * var_mask(keep)

    with torch.no_grad():
        x1 = A.advance(S.x0, 1)
        # balanced lagged-forecast difference at the 6-h valid time
        xb24, _ = utils.get_input(dict(S.exp, nhrs_back=S.exp.get('nhrs_back', 12) + 12, restart_window=False),
                                  S.model, S.logger)
        nmc = A.prog(A.advance(xb24, 1)) - A.prog(x1)
        # A2's own 6-h forecast error at the 6-h valid time
        xe0, _ = utils.get_input(dict(S.exp, nhrs_back=0, restart_window=False), S.model, S.logger)
        xe1, _ = utils.get_input(dict(S.exp, nhrs_back=0, restart_window=False,
                                      date=utils.add_hours(S.date, ts)), S.model, S.logger)
        a2err = A.prog(A.advance(xe0, 1)) - A.prog(xe1)
        nmc, a2err = torch.nan_to_num(nmc), torch.nan_to_num(a2err)
        fixed = {'nmc_full': nmc, 'nmc_large': smooth_large(nmc), 'a2err_full': a2err,
                 'a2err_small': a2err - S.smooth(a2err)}
        S.log(f'native sizes (normalised norm): nmc {S.nnorm(nmc):.4f}, a2err {S.nnorm(a2err):.4f}')

        ctrl, x = {}, x1
        for s in range(1, nstep + 1):
            x = A.advance(x, 1)
            if s in steps:
                ctrl[s] = A.prog(x).clone()

        def response(d):
            x, out = A.add_prog(x1, d), {}
            for s in range(1, nstep + 1):
                x = A.advance(x, 1)
                if s in steps:
                    out[s] = A.prog(x) - ctrl[s]
            return out

        def comp(r):
            e = (torch.nan_to_num(r).double() * S.mul.double()) ** 2
            e = (e * S.w[:, None]).sum(0)
            e = e / e.sum()
            g = {}
            for b, f in zip(base, e.tolist()):
                g[b] = g.get(b, 0.0) + f
            return {k: round(v, 4) for k, v in sorted(g.items(), key=lambda t: -t[1]) if v > 0.01}

        R = {'leads_h': list(leads_h), 'sizes': list(sizes), 'native_norm': {'nmc': S.nnorm(nmc), 'a2err': S.nnorm(a2err)},
             'families': {}}
        for fam in FAMILIES:
            reals = [fixed[fam]] if fam in fixed else [random_family(fam) for _ in range(n_real)]
            res = {str(sz): [] for sz in sizes}
            small_frac, out_comp = [], None
            for d in reals:
                n0 = S.nnorm(d)
                for sz in sizes:
                    o = response(d * (sz / n0))
                    res[str(sz)].append([S.nnorm(o[s]) / sz for s in steps])
                    if sz == sizes[0]:
                        rl = o[steps[-1]]
                        rs = rl - S.smooth(torch.nan_to_num(rl))
                        small_frac.append(S.nnorm(rs) ** 2 / max(S.nnorm(rl) ** 2, 1e-30))
                        if out_comp is None:
                            out_comp = comp(rl)
            g = {sz: np.mean(v, 0).tolist() for sz, v in res.items()}
            R['families'][fam] = {'growth': g, 'in_comp': comp(reals[0]), 'out_comp_48h': out_comp,
                                  'out_small_frac_48h': float(np.mean(small_frac)),
                                  'per_step_48h': float(g[str(sizes[0])][-1] ** (1.0 / steps[-1]))}
            S.log(f'{fam:15s} growth 6/24/48 h {np.round(g[str(sizes[0])], 2).tolist()} '
                  f'(size {sizes[1]}: {np.round(g[str(sizes[1])], 2).tolist()}), out small {np.mean(small_frac):.2f}')
    return R


def plot(results_dir, out_png, label='A2'):
    """Figure F1: growth vs lead per family (mean over dates, min-max band), from families_{mmdd}.json"""
    import glob
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = [json.load(open(f)) for f in sorted(glob.glob(f'{results_dir}/families_????.json'))]
    leads = [0] + R[0]['leads_h']
    sz = str(R[0]['sizes'][0])
    style = {'hyd_z_large': ('#c0392b', '-', 'z only, large scale (hydrostatic imbalance)'),
             'hyd_z_small': ('#c0392b', '--', 'z only, small scale'),
             'hyd_t_large': ('#e67e22', '-', 't only, large scale (hydrostatic imbalance)'),
             'geo_wind_large': ('#2980b9', '-', 'wind only, large scale (geostrophic imbalance)'),
             'small_all': ('#2980b9', '--', 'small-scale noise, all variables'),
             'nmc_large': ('#27ae60', '-', 'balanced: lagged-forecast difference, large scale'),
             'nmc_full': ('#27ae60', '--', 'balanced: lagged-forecast difference, all scales'),
             'a2err_full': ('#555555', '-', "A2's own 6-h forecast error"),
             'a2err_small': ('#555555', '--', "A2's own 6-h error, small scales")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fam, (c, ls, lab) in style.items():
        g = np.array([[1.0] + r['families'][fam]['growth'][sz] for r in R])
        ax.plot(leads, g.mean(0), color=c, ls=ls, lw=2, marker='o', ms=4, label=lab)
        ax.fill_between(leads, g.min(0), g.max(0), color=c, alpha=0.12, lw=0)
    ax.axhline(1, color='k', lw=0.8)
    ax.set_yscale('log')
    ax.set_yticks([0.5, 0.7, 1, 2, 3, 5])
    ax.set_yticklabels(['0.5', '0.7', '1', '2', '3', '5'])
    ax.set_xticks(leads)
    ax.set_xlabel('hours after the perturbation (applied at the 6-h state, as 4D-Var does)')
    ax.set_ylabel('perturbation size / initial size (normalised norm)')
    ax.set_title(f'Figure F1. Which kinds of input perturbation does {label} amplify? '
                 f'({len(R)} dates of 2022, band = range)', fontsize=11)
    ax.legend(fontsize=8, loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print('saved', out_png)
