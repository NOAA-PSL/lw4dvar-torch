"""ERA5-following first guess for a window: start the model from ERA5 at T and fit the weak-constraint
nudges (latent jumps at 24, 48, 72, 96 h, injected as compute_loss_4dvar injects them) that keep the
trajectory on ERA5 at every 6-h step through the window. Toward a "warm start from ERA5" 4D-Var
(notes/plan_20261105.md): the fitted nudges become the jumps' first guess and prior mean, and the
window then looks for a trajectory that fits the obs better than ERA5 does.

Misfit: sum over the window's 6-h steps of the area-weighted mean square of the normalized state
difference (the model's input normalization, all variables with finite values), as fit_latent_to_era5.py.
L-BFGS from zero on scaled controls (minimum-norm fit), checkpointed rollout.

Writes OUT/<T>_nudges.pt in the weak-constraint file layout ({'hours', 'steps', 'forcing',
'latent_scale'}, as *_wc_forcing_*.pt) and prints the misfit per lead: free forecast vs nudged.
Score the result against obs with diagnostics/forecast_skill.py SOURCE = nudged:OUT (and era5self for ERA5).

Usage (GPU): python fit_window_nudges.py CONFIG OUT T [T ...] [--hours 120] [--every 24] [--iters 60] [--lr 0.05]
"""

import argparse
import os
from datetime import datetime, timedelta

import torch

import long_window_4dvar_utils as utils


def _pre(model):
    """input normalizer; A1/A2 (aifs37) keep their processors in a ModuleDict keyed by dataset"""
    p = model.interface.pre_processors
    ds = getattr(model, 'dataset', None)
    return p[ds] if ds is not None and isinstance(p, __import__('torch').nn.ModuleDict) else p

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('config')
ap.add_argument('out')
ap.add_argument('dates', nargs='+')
ap.add_argument('--hours', type=int, default=120)
ap.add_argument('--every', type=int, default=24)
ap.add_argument('--iters', type=int, default=60)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--weights', choices=['none', 'training'], default='none',
                help="per-variable misfit weights: none = equal; training = the A1/A2 training loss scalers "
                     "(pressure levels by layer thickness, floor 0.1 x mean, 6.5 per variable; 10u/10v 0.5), "
                     "rescaled to mean 1 over the fitted variables")
ap.add_argument('--control', choices=['latent', 'state'], default='latent',
                help="latent = jumps added to the encoder's hidden-mesh output (as the solver); state = a direct "
                     "increment to the prognostic variables of the newest input level at each jump time, in "
                     "normalized units (diagnostic: is a latent jump the limit?)")
ap.add_argument('--sequential', action='store_true',
                help='fit the jumps one at a time in time order, each over the --horizon steps it starts, earlier '
                     'jumps frozen (instead of all jumps jointly over the whole window)')
ap.add_argument('--horizon', type=int, default=4, help='--sequential: 6-h steps each jump is fitted over')
ap.add_argument('--max_jumps', type=int, default=0, help='--sequential: fit only the first N jumps (0 = all)')
args = ap.parse_args()

logger = utils.get_logger()
exp, windows = utils.load_config(args.config)
exp = utils.get_window(exp, list(windows)[0], windows)
model = utils.get_model(exp)
areaw = torch.as_tensor(model.area_weights, dtype=torch.float32, device=model.device)
areaw = areaw / areaw.sum()
os.makedirs(args.out, exist_ok=True)
n_steps = args.hours // 6
jump_hours = list(range(args.every, args.hours, args.every))
jump_steps = [h // 6 + 1 for h in jump_hours]     # the jump at h acts on the h -> h + 6 step (as compute_optimal)


def var_weights():
    """(n_input_vars,) misfit weights in input-tensor order (see --weights)"""
    w = torch.ones(len(model.var_to_idx), device=model.device)
    if args.weights == 'none':
        return w
    lev = {}
    for name, i in model.var_to_idx.items():
        base, _, l = name.rpartition('_')
        if base and l.isdigit():
            lev.setdefault(base, []).append((float(l), i))
        elif name in ('10u', '10v'):
            w[i] = 0.5
    for base, li in lev.items():          # anemoi ThicknessVariableLevelScaler(floor=0.1, total=6.5)
        li.sort()
        p = torch.tensor([l for l, _ in li], dtype=torch.float64)
        edges = torch.cat([p.new_zeros(1), (p[1:] + p[:-1]) / 2, p[-1:]])
        dp = edges.diff()
        wt = torch.maximum(dp, 0.1 * dp.mean())
        wt = wt * 6.5 / wt.sum()
        for (_, i), x in zip(li, wt.tolist()):
            w[i] = x
    return w


def norm_mul():
    """(n_input_vars,) normalized = physical * mul + add, from the input normalizer (dataset-sized buffers
    indexed by _input_idx, as InputNormalizer.transform)"""
    for m in _pre(model).modules():
        if hasattr(m, '_norm_mul') and hasattr(m, '_input_idx'):
            mul = m._norm_mul[m._input_idx]
            assert mul.numel() == len(model.var_to_idx), (mul.numel(), len(model.var_to_idx))
            return mul.to(model.device)
    raise RuntimeError('no InputNormalizer in the pre-processors')


def state_jump(x, inc):
    """x + inc on the prognostic variables of the newest level (inc normalized, (points, n_prog))"""
    pidx = model.pmask_in
    st = x.state.clone()
    st[:, -1, :, pidx] = st[:, -1, :, pidx] + inc / norm_mul()[pidx]
    return type(x)(st, x.date)


def control_shape():
    return (len(model.lats), len(model.pmask_in)) if args.control == 'state' else tuple(model.latent_shape)


def fmt(T):
    return T.strftime('%Y-%m-%dT%H')


def era5(T):
    state, _ = utils.get_input(dict(exp, date=fmt(T), nhrs_back=0, restart_window=False), model, logger)
    return state


def norm_last(state):
    x = state.state[:, -1:, None, ...]
    return _pre(model)(x, in_place=False)[0, 0, 0]


for d in args.dates:
    T = datetime.strptime(d, '%Y-%m-%dT%H')
    ofile = os.path.join(args.out, f'{d}_nudges.pt')
    if os.path.exists(ofile):
        print('exists', ofile, flush=True)
        continue
    x0 = era5(T)
    with torch.no_grad():
        tgts = [norm_last(era5(T + timedelta(hours=6 * s))).detach() for s in range(1, n_steps + 1)]
    ok = torch.stack([torch.isfinite(t).all(0) for t in tgts]).all(0)
    vw = var_weights()[ok]
    vw = vw / vw.mean()

    def step(x, s, inc, grad):
        """advance one 6-h step (step number s), applying the control inc (or None) on it"""
        if args.control == 'state':
            return model.advance(state_jump(x, inc) if inc is not None else x, steps=1, use_checkpoint=grad)
        return model.advance(x, steps=1, latent_increment=inc, use_checkpoint=grad)

    def mis(x, s):
        return (areaw[:, None] * vw * (norm_last(x) - tgts[s - 1])[:, ok] ** 2).sum() / ok.sum()

    def misfits(eta, grad=True):
        """per-step misfits (list of tensors) of the rollout from ERA5(T) with jumps eta (n_jumps, *latent)"""
        out, x = [], x0
        for s in range(1, n_steps + 1):
            k = jump_steps.index(s) if s in jump_steps else None
            inc = eta[k] if (eta is not None and k is not None) else None
            x = step(x, s, inc, grad)
            out.append(mis(x, s))
        return out

    with torch.no_grad():
        free = [m.item() for m in misfits(None, grad=False)]
    hist = []
    if not args.sequential:
        scale = args.lr * 0.8 * (len(jump_hours) * torch.Size(control_shape()).numel())
        v = torch.zeros((len(jump_hours),) + control_shape(), device=model.device, requires_grad=True)
        opt = torch.optim.LBFGS([v], lr=1.0, max_iter=args.iters, history_size=20, line_search_fn='strong_wolfe',
                                tolerance_grad=0., tolerance_change=0.)

        def closure():
            opt.zero_grad()
            j = sum(misfits(scale * v))
            j.backward()
            hist.append(j.item())
            return j
        opt.step(closure)
        eta = (scale * v).detach()
    else:
        eta = torch.zeros((len(jump_hours),) + control_shape(), device=model.device)
        scale = args.lr * 0.8 * torch.Size(control_shape()).numel()
        x, s_done = x0, 0                         # state after step s_done, earlier jumps applied
        nfit = len(jump_steps) if args.max_jumps <= 0 else min(args.max_jumps, len(jump_steps))
        for k in range(nfit):
            js = jump_steps[k]
            with torch.no_grad():
                for s in range(s_done + 1, js):
                    kk = jump_steps.index(s) if s in jump_steps else None
                    x = step(x, s, eta[kk] if kk is not None else None, False)
                s_done = js - 1
            xs = x
            s_end = min(js + args.horizon - 1, n_steps)
            v = torch.zeros(control_shape(), device=model.device, requires_grad=True)
            opt = torch.optim.LBFGS([v], lr=1.0, max_iter=args.iters, history_size=20,
                                    line_search_fn='strong_wolfe', tolerance_grad=0., tolerance_change=0.)
            hk = []

            def closure():
                opt.zero_grad()
                xx, j = xs, 0.
                for s in range(js, s_end + 1):
                    xx = step(xx, s, scale * v if s == js else None, True)
                    j = j + mis(xx, s)
                j.backward()
                hk.append(j.item())
                return j
            opt.step(closure)
            eta[k] = (scale * v).detach()
            hist.append(hk)
            print(f'  jump {k} (step {js}, fitted over steps {js}-{s_end}): {hk[0]:.4f} -> {hk[-1]:.4f} '
                  f'in {len(hk)} evals; rms {eta[k].pow(2).mean().sqrt().item():.3g}', flush=True)
    with torch.no_grad():
        fit = [m.item() for m in misfits(eta, grad=False)]
    torch.save({'hours': jump_hours, 'steps': jump_steps, 'forcing': eta.half().cpu(), 'latent_scale': 1.0,
                'control': args.control, 'weights': args.weights,
                'sequential': args.sequential, 'horizon': args.horizon, 'max_jumps': args.max_jumps,
                'misfit_free': free, 'misfit_nudged': fit, 'lbfgs_hist': hist}, ofile)
    print(f'{d}: total misfit free {sum(free):.4g} -> nudged {sum(fit):.4g} '
          f'({100 * (1 - sum(fit) / sum(free)):.0f}% removed); jump rms '
          + ' '.join(f'{e.pow(2).mean().sqrt().item():.3g}' for e in eta), flush=True)
    print('   lead h      ' + ' '.join(f'{6 * s:6d}' for s in range(1, n_steps + 1, 2)), flush=True)
    print('   free        ' + ' '.join(f'{free[s]:6.3f}' for s in range(0, n_steps, 2)), flush=True)
    print('   nudged      ' + ' '.join(f'{fit[s]:6.3f}' for s in range(0, n_steps, 2)), flush=True)
