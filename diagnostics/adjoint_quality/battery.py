"""Adjoint-quality battery, parts `gradient` and `survival` (notes/adjoint_quality.md).

Twin setting: background x0 = the config's first guess (utils.get_input, e.g. A2 fg12); truth = the model's own
forecast from the analysis at the window start (utils.get_input with nhrs_back=0), exactly as
long_window_4dvar_utils.make_synthetic_obs builds the twin truth.
J_L(u) = 0.5 * area-weighted mean over points and prognostic vars of ((x_L(u) - truth_L) * mul)^2, u = latent increment
injected on the first step (as 4D-Var does).

gradient: precision cosines (reduced vs fp32, repeatability), Taylor test, gradient stability, 6-h response maps.
survival: growth of perturbations to the last lead (large / small scale), randomized SVD of the 6 h -> 48 h map.

Step sizes are relative to a reference latent norm |u_ref|: the norm of a 4D-Var increment for this model and date
(`golden`, e.g. A2's twin PITA run), or, if none is given, the norm at which the steepest-descent direction's 6-h
response equals `target_resp6` (A2's golden 6-h response, 0.040 normalised units).
"""
import glob
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.spatial import cKDTree

import long_window_4dvar_utils as utils

from .adapter import make_adapter

SCHEMA = 1
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULTS = dict(leads=(24, 120), eps=(1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0), stab=(1e-3, 1e-2, 1e-1),
                svd_rank=24, svd_steps=7, target_resp6=0.040, knn=24, seed=1)
SMOKE = dict(leads=(24, 24), eps=(1e-3, 1e-2, 0.1), stab=(1e-2,), svd_rank=6, svd_steps=2)


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def git_hash():
    try:
        h = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--', 'diagnostics/adjoint_quality'],
                               capture_output=True, text=True).stdout.strip()
        return h + ('+dirty' if dirty else '')
    except OSError:
        return 'unknown'


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


class Log:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        print(f'[{time.time() - self.t0:7.0f}s] {msg}', flush=True)


class Setup:
    """model, adapter, background, norms, smoother -- shared by every part"""

    def __init__(self, config, tf32=False):
        self.log = Log()
        torch.set_float32_matmul_precision('highest')
        self.logger = utils.get_logger()
        exp, windows = utils.load_config(config)
        exp['date'] = exp['sdate']
        self.exp = utils.get_window(exp, list(windows)[0], windows)
        self.config = config
        self.date = self.exp['date']
        self.mmdd = self.date[5:7] + self.date[8:10]
        self.model = utils.get_model(self.exp)
        # backends switch TF32 on in __init__; the battery's "fp32" means true fp32 unless --tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        self.tf32 = bool(tf32)
        self.A = make_adapter(self.model, self.exp)
        self.dev = self.model.device
        self.lat, self.lon = self.A.lat, self.A.lon
        w = torch.as_tensor(self.A.area_weights, device=self.dev, dtype=torch.float64)
        self.w = w / w.sum()
        self.mul = self.A.mul
        self.nP = len(self.A.prog_names)
        self.log(f'{self.date}: backend {self.A.backend}, reduced precision {self.A.reduced_precision()}, '
                 f'latent {self.A.latent_shape}, {len(self.lat)} points, {self.nP} prognostic vars')
        with torch.no_grad():
            self.x0, _ = utils.get_input(dict(self.exp, restart_window=False), self.model, self.logger)

    def knn(self, k):
        if getattr(self, '_nb', None) is None:
            _, nb = cKDTree(xyz(self.lat, self.lon)).query(xyz(self.lat, self.lon), k=k)
            self._nb = torch.as_tensor(nb, device=self.dev)
        return self._nb

    def smooth(self, f, k=24):
        """mean over k nearest neighbours, 3 passes (~300-500 km on O96)"""
        nb = self.knn(k)
        for _ in range(3):
            f = f[nb].mean(1)
        return f

    def nnorm(self, d):
        z = torch.nan_to_num(d.double() * self.mul.double())
        return float(((z * z).mean(1) * self.w).sum().sqrt())

    def rms(self, f):
        f = torch.nan_to_num(f.double())
        return float((f * f * self.w).sum().sqrt())

    def meta(self, part, settings):
        return {'schema': SCHEMA, 'tool': 'diagnostics/adjoint_quality', 'part': part, 'git': git_hash(),
                'created': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'config': self.config,
                'date': self.date, 'model': self.A.model_id(self.exp), 'tf32': self.tf32,
                'settings': {k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()}}


def load_increment(pattern, mmdd, date, dev):
    """latest file matching `pattern` ({mmdd} / {date} substituted) -> injected latent increment"""
    files = sorted(glob.glob(pattern.format(mmdd=mmdd, date=date[:10])))
    if not files:
        raise FileNotFoundError(f'no increment matches {pattern!r} for {date}')
    d = torch.load(files[-1], map_location='cpu', weights_only=False)
    return (d['increment'] / d.get('latent_scale', 1.0)).to(dev).float(), files[-1]


class Battery:
    def __init__(self, S, settings, golden=None, golden_cv=None):
        self.S, self.A, self.set = S, S.A, settings
        dev = S.dev
        ts = int(round(S.model.timestep.total_seconds() / 3600))
        self.leads = {str(h): h // ts for h in settings['leads']}
        self.nstep = max(self.leads.values())
        self.precisions = ['fp32'] + ([S.A.reduced_precision()] if S.A.reduced_precision() else [])
        self.red = S.A.reduced_precision()
        gen = torch.Generator(device='cpu').manual_seed(settings['seed'])
        self.gen = gen
        rhat = torch.randn(S.A.latent_shape, generator=gen).to(dev)
        self.rhat = rhat / rhat.norm()
        self.zero = torch.zeros(S.A.latent_shape, device=dev)
        self.ug = self.uc = None
        self.R = {'date': S.date}
        if golden:
            self.ug, self.R['golden_file'] = load_increment(golden, S.mmdd, S.date, dev)
        if golden_cv:
            self.uc, self.R['cv_file'] = load_increment(golden_cv, S.mmdd, S.date, dev)
        # twin truth
        self.A.set_precision('fp32')
        with torch.no_grad():
            xt, _ = utils.get_input(dict(S.exp, nhrs_back=0, restart_window=False), S.model, S.logger)
            self.truth = {}
            x = xt
            for s in range(1, self.nstep + 1):
                x = self.A.advance(x, 1)
                if s in self.leads.values():
                    self.truth[s] = self.A.prog(x).clone()
        S.log('twin truth done')
        self._g32 = None

    # ---- rollouts and costs
    def rollout(self, u, grad=False, keep_all=False):
        x = self.A.advance(self.S.x0, 1, grad=grad, latent_increment=u)
        outs = {1: self.A.prog(x)}
        for s in range(2, self.nstep + 1):
            x = self.A.advance(x, 1, grad=grad)
            if keep_all or s in self.leads.values():
                outs[s] = self.A.prog(x)
        return outs

    def costs(self, outs):
        m = self.S.mul.double()
        J = {}
        for k, s in self.leads.items():
            z = torch.nan_to_num((outs[s] - self.truth[s]).double() * m)
            J[k] = 0.5 * ((z * z).mean(1) * self.S.w).sum()
        return J

    def grads(self, u0, prec):
        self.A.set_precision(prec)
        u = u0.detach().clone().requires_grad_(True)
        J = self.costs(self.rollout(u, grad=True))
        ks = list(self.leads)
        g = {}
        for i, k in enumerate(ks):
            g[k], = torch.autograd.grad(J[k], u, retain_graph=i < len(ks) - 1)
            g[k] = g[k].float()
        return g, {k: float(v) for k, v in J.items()}

    def Jval(self, u, prec):
        self.A.set_precision(prec)
        with torch.no_grad():
            return {k: float(v) for k, v in self.costs(self.rollout(u)).items()}

    @property
    def klast(self):
        return str(max(self.set['leads']))

    def g32(self):
        if self._g32 is None:
            self._g32, self._J0_32 = self.grads(self.zero, 'fp32')
        return self._g32

    def ref_norm(self):
        """|u_ref|: the golden increment's norm, else calibrated on the steepest-descent 6-h response"""
        if getattr(self, '_ng', None) is not None:
            return self._ng
        if self.ug is not None:
            self._ng, self.R['ref_norm_source'] = float(self.ug.norm()), 'golden'
        else:
            d = -self.g32()[self.klast]
            d = d / d.norm()
            self.A.set_precision('fp32')
            with torch.no_grad():
                b1 = self.A.prog(self.A.advance(self.S.x0, 1))
                n = 1.0
                for _ in range(4):        # secant on the (nearly linear) 6-h response size
                    r = self.S.nnorm(self.A.prog(self.A.advance(self.S.x0, 1, latent_increment=n * d)) - b1)
                    n = n * self.set['target_resp6'] / max(r, 1e-30)
            self._ng, self.R['ref_norm_source'] = float(n), f"calibrated: -g 6-h response = {self.set['target_resp6']}"
        self.R['golden_norm'] = self._ng
        if self.uc is not None:
            self.R['cv_norm'] = float(self.uc.norm())
        return self._ng

    # ================= gradient =================
    def gradient(self):
        S, R = self.S, self.R
        g32 = self.g32()
        J0_32 = self._J0_32
        g32b, _ = self.grads(self.zero, 'fp32')
        R['J0'] = {'fp32': J0_32}
        R['cos'] = {k: {'fp32_repeat': cos(g32[k], g32b[k]), 'gnorm_fp32': float(g32[k].norm())} for k in self.leads}
        del g32b
        gred = None
        if self.red:
            gred, J0_red = self.grads(self.zero, self.red)
            gredb, _ = self.grads(self.zero, self.red)
            R['J0'][self.red] = J0_red
            for k in self.leads:
                # keys keep the 'bf16' spelling of the reference runs whatever the reduced dtype is
                R['cos'][k].update({'bf16_vs_fp32': cos(gred[k], g32[k]), 'bf16_repeat': cos(gred[k], gredb[k]),
                                    'gnorm_bf16': float(gred[k].norm())})
            del gredb
        ng = self.ref_norm()
        if self.ug is not None:
            for k in self.leads:
                R['cos'][k]['g_vs_golden'] = cos(g32[k], self.ug)
        S.log(f"cos: {json.dumps(R['cos'])}")

        kl = self.klast
        dirs = {'-g32': -g32[kl] / g32[kl].norm()}
        if gred is not None:
            dirs['-gbf'] = -gred[kl] / gred[kl].norm()
        dirs['rand'] = self.rhat
        if self.ug is not None:
            dirs['golden'] = self.ug / ng
        self.dirs = dirs
        R['taylor'] = {'eps_frac': list(self.set['eps'])}
        for name, d in dirs.items():
            res = {f'{p}_{k}': [] for p in ('fp32', 'gbf') for k in self.leads}
            res['dJ_' + kl] = []
            for f in self.set['eps']:
                e = f * ng
                J = self.Jval(e * d, 'fp32')
                for k in self.leads:
                    dJ = J[k] - J0_32[k]
                    res[f'fp32_{k}'].append(dJ / (e * float((g32[k] * d).sum())))
                    if gred is not None:
                        res[f'gbf_{k}'].append(dJ / (e * float((gred[k] * d).sum())))
                res['dJ_' + kl].append(J[kl] - J0_32[kl])
            if name == '-gbf':                          # what the reduced-precision optimizer itself sees
                res[f'bf16fwd_{kl}'] = []
                for f in self.set['eps']:
                    e = f * ng
                    J = self.Jval(e * d, self.red)
                    res[f'bf16fwd_{kl}'].append((J[kl] - R['J0'][self.red][kl]) / (e * float((gred[kl] * d).sum())))
            R['taylor'][name] = {k: v for k, v in res.items() if v}
            S.log(f'taylor {name}: fp32_{kl} {np.round(res["fp32_" + kl], 3).tolist()}')
        R['stability'] = {}
        for s in self.set['stab']:
            gs, _ = self.grads(s * ng * self.rhat, 'fp32')
            R['stability'][f'rand_{s}'] = {k: cos(gs[k], g32[k]) for k in self.leads}
        if self.ug is not None:
            gs, _ = self.grads(self.ug, 'fp32')
            R['stability']['at_golden'] = {k: cos(gs[k], g32[k]) for k in self.leads}
        S.log(f"stability {json.dumps(R['stability'])}")
        return R

    # ================= survival =================
    def survival(self, out_base):
        S, A, R = self.S, self.A, self.R
        ng = self.ref_norm()
        g32 = self.g32()
        kl = self.klast
        ghat = -g32[kl] / g32[kl].norm()
        A.set_precision('fp32')
        P = A.prog_pos
        with torch.no_grad():
            ctrl = self.rollout(self.zero, keep_all=True)
            perts = {}
            if self.ug is not None:
                perts['golden_twin'] = self.ug
            if self.uc is not None:
                perts['golden_cv'] = self.uc
            perts['random'] = ng * self.rhat
            perts['-g'] = ng * ghat
            iz, it8, iu, iv = P['z_500'], P['t_850'], P['u_250'], P['v_250']
            R['survival'] = {'lead_h': [6 * s for s in range(1, self.nstep + 1)]}
            maps = {}
            for name, u in perts.items():
                o = self.rollout(u, keep_all=True)
                cur = {k: [] for k in ('norm', 'norm_large', 'norm_small', 'z500', 'z500_large', 'z500_small',
                                       't850', 't850_large', 't850_small', 'ws250')}
                for s in range(1, self.nstep + 1):
                    d = o[s] - ctrl[s]
                    dl = S.smooth(torch.nan_to_num(d), self.set['knn'])
                    cur['norm'].append(S.nnorm(d))
                    cur['norm_large'].append(S.nnorm(dl))
                    cur['norm_small'].append(S.nnorm(d - dl))
                    for nm, i in (('z500', iz), ('t850', it8)):
                        cur[nm].append(S.rms(d[:, i]))
                        cur[nm + '_large'].append(S.rms(dl[:, i]))
                        cur[nm + '_small'].append(S.rms(d[:, i] - dl[:, i]))
                    cur['ws250'].append(float(torch.sqrt(((torch.nan_to_num(d[:, iu]).double() ** 2
                                                           + torch.nan_to_num(d[:, iv]).double() ** 2) * S.w).sum())))
                    if s == 1 and name in ('golden_twin', '-g', 'golden_cv'):
                        maps[f'{name}_t850'] = d[:, it8].cpu().numpy()
                        maps[f'{name}_z500'] = d[:, iz].cpu().numpy()
                R['survival'][name] = cur
                S.log(f'survival {name}: norm {np.round(np.array(cur["norm"]) / cur["norm"][0], 2).tolist()}')
            R['response6h_at_equal_latent_norm'] = {n: R['survival'][n]['norm'][0] for n in perts if n != 'golden_cv'}
            if 'golden_cv' in perts:
                R['response6h_cv'] = R['survival']['golden_cv']['norm'][0]
            del ctrl
        np.savez(out_base + '_maps.npz', lat=S.lat, lon=S.lon, **maps)
        self._svd(out_base)
        return R

    def _svd(self, out_base):
        S, A, R = self.S, self.A, self.R
        nP, npts, dev, gen = S.nP, len(S.lat), S.dev, self.gen
        NS, L = self.set['svd_steps'], self.set['svd_rank']
        A.set_precision('fp32')
        with torch.no_grad():
            x1 = A.advance(S.x0, 1)
        scale = S.mul[None, :] * (S.w.float()[:, None] / nP).sqrt()   # physical -> unit-normalised-norm coordinates
        state = {'mask': None}

        def F(c, grad=False):
            c = c * state['mask'] if state['mask'] is not None else c
            x = A.add_prog(x1, c / scale)
            x = A.advance(x, NS, grad=grad)
            return torch.nan_to_num(A.prog(x) * scale)

        # FD step: 10% of the 6-h response of a reference-size increment (A2: of the golden increment)
        resp6 = R['survival'].get('golden_twin', R['survival']['-g'])['norm'][0]
        eps_fd = 0.1 * resp6

        def jvp(v, eps=eps_fd):
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

        names = A.prog_names

        def composition(c):
            e = (c.double() ** 2).sum(0).cpu().numpy()
            e = e / e.sum()
            byv, byl = {}, {}
            for nm, f in zip(names, e):
                b, _, lv = nm.rpartition('_')
                if b and lv.isdigit():
                    lv = int(lv)
                    band = 'p<100' if lv < 100 else '100-300' if lv < 300 else '300-700' if lv < 700 else '>=700'
                else:
                    b, band = nm, 'surface'
                byv[b] = byv.get(b, 0) + float(f)
                byl[band] = byl.get(band, 0) + float(f)
            return {'by_var': byv, 'by_level': byl, 'top': sorted(zip(names, e.tolist()), key=lambda t: -t[1])[:6]}

        def rough(c):
            cs = S.smooth(c, self.set['knn'])
            return float(((c - cs) ** 2).sum() / (c ** 2).sum())

        def run(tag, mask):
            state['mask'] = mask
            om = [S.smooth(torch.randn(npts, nP, generator=gen).to(dev), self.set['knn']) * mask for _ in range(L)]
            om = [o / o.norm() for o in om]
            v1 = om[0]
            a1, a2 = jvp(v1), jvp(v1, 0.1 * eps_fd)
            lin = {'eps': [eps_fd, 0.1 * eps_fd], 'Av_norm': [float(a1.norm()), float(a2.norm())], 'cos': cos(a1, a2)}
            ytest = S.smooth(torch.randn(npts, nP, generator=gen).to(dev), self.set['knn'])
            dot = {'<Av,y>': float((a1 * ytest).sum()), '<v,ATy>': float((v1 * vjp(ytest)).sum())}
            Y = [a1] + [jvp(o) for o in om[1:]]
            R[f'random_amplification_48h_{tag}'] = [float(y.norm()) for y in Y]
            S.log(f'[{tag}] linearity {lin}; dot test {dot}')
            Q = orth(Y)
            Z = orth([vjp(q) for q in Q])
            Q = orth([jvp(z) for z in Z])
            Bt = torch.stack([vjp(q).reshape(-1) for q in Q], 1)       # A^T Q
            Uin, sv, Wt = torch.linalg.svd(Bt, full_matrices=False)
            outv = torch.stack([q.reshape(-1) for q in Q], 1) @ Wt.T
            sv = sv.cpu().numpy()
            lead = []
            for j in range(min(3, len(sv))):
                ci, co = Uin[:, j].reshape(npts, nP), outv[:, j].reshape(npts, nP)
                lead.append({'in': composition(ci), 'out': composition(co), 'in_small_frac': rough(ci),
                             'out_small_frac': rough(co)})
                if j == 0:
                    k = int(np.argmax([(ci[:, i] ** 2).sum().item() for i in range(nP)]))
                    np.savez(out_base + f'_sv1_{tag}.npz', lat=S.lat, lon=S.lon, var=names[k],
                             vin=(ci[:, k] / scale[:, k]).cpu().numpy(), vout=(co[:, k] / scale[:, k]).cpu().numpy())
            R[f'svd_48h_{tag}'] = {'singular_values': sv.tolist(), 'linearity': lin, 'dot_test': dot, 'eps_fd': eps_fd,
                                   'n_steps': NS, 'leading': lead}
            S.log(f'[{tag}] singular values: {np.round(sv, 2).tolist()}')

        run('all', torch.ones(nP, device=dev))
        run('noz', torch.tensor([0. if nm.startswith('z_') else 1. for nm in names], device=dev))
