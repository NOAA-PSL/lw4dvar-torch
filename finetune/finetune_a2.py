"""Lean fine-tuning of the 37-level AIFS model A2 for a better-behaved adjoint (notes/adjoint_finetune_plan.md).

Reuses the aifs37 backend's differentiable step (pre/post processors, forcings, bf16 autocast) and A2's training
zarrs (aifs37_model.ERA5Zarr). Single GPU per arm, gradient accumulation. Years: train 1979-2020, validate 2021,
2022 untouched (the PITA test year).

Arms (--arm):
  C0  continue training unchanged (single-step MSE): the control for the effect of fine-tuning itself
  R   rollout curriculum (--rollout 2 3 4 over equal thirds of the steps), loss averaged over the rollout steps
  N   denoising: with prob --noise_p, add a perturbation from --noise_fam (diagnostics/adjoint_quality/families.py
      kinds) of normalised size --noise_size to the last input time level; the target is unchanged
  J   excess-growth hinge: per optimizer step, on --j_samples samples, L_J = relu(G_K - gamma)^2 with G_K the
      K-step (--j_K) growth |f_K(x+v) - f_K(x)| / |v| of a perturbation v from --noise_fam of size --j_eps;
      lambda = --j_lambda, or 'auto' (L_J = 10% of the MSE at step 0)
  P   ERA5 MSE + GEFS member-pair loss (finetune/pair_loss.py): per optimizer step --pair_n pairs,
      || [A2(x_m) - A2(x_c)] - [GEFS_m - GEFS_c](+6h) ||^2, weighted so it is --pair_frac of the total at step 0
  PU  P + the J hinge on the unphysical families (--noise_fam default hyd_z_large hyd_z_small hyd_t_large small_all)
  PT  P with the exact tangent-linear: L_TL = || M'(x_c)(x_m - x_c) - dG ||^2 replaces the finite-difference pair term,
      and the hinge uses the exact JVP, L_J^TL = relu(||M'(x) v|| / ||v|| - --j_gamma)^2 (K = 1, one 6-h step, v of
      normalised size 1 from --noise_fam as PU). M' by forward-mode AD (pair_loss.jvp_step); the PT terms run in fp32
      with TF32 off and the saved tensors offloaded to host (--pt_offload). Needs
      ANEMOI_INFERENCE_GRAPHTRANSFORMER_ATTENTION_BACKEND=pyg (the Triton kernel has no double derivative).
Loss: MSE of normalised outputs (the checkpoint's own output normaliser), area-weighted (model.area_weights), with
A2's general_variable weights and a pressure-level weight max(0.1, 0.2 + 0.001 p) (approximates anemoi's
ThicknessVariableLevelScaler; the same loss is used for every arm and for validation, so arms compare fairly).
Optimizer as A2's training: AdamW betas (0.9, 0.95), weight decay 0.01, grad clip by value 32; lr warmup to --lr then
cosine to 3e-7.

Usage (anemoi-rocm env, MI300X): sbatch finetune/run_finetune.sh --arm C0 --out output/finetune/C0 [--smoke]
Checkpoints {'state_dict', 'step', 'arm', 'args'} at --save_at; evaluate them with any tool through the config key
exp.model_weights (long_window_4dvar_utils.get_model).
"""
import argparse
import contextlib
import datetime as dt
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import long_window_4dvar_utils as utils  # noqa: E402

GV = {'q': 0.6, 't': 6, 'u': 0.8, 'v': 0.5, 'w': 0.001, 'z': 12, 'sp': 10, '10u': 0.5, '10v': 0.5, '2d': 0.5,
      'tp': 0.025, 'cp': 0.0025}
PL = ('q', 't', 'u', 'v', 'w', 'z')
SIG = {'t': 0.5, 'z': 5 * 9.80665, 'u': 1.0, 'v': 1.0, 'q': 3e-4, 'sp': 50.0, 'msl': 50.0}
H6 = dt.timedelta(hours=6)


def log(msg, fh=None):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _base(name):
    b, _, lv = name.rpartition('_')
    return (b, int(lv)) if (b and lv.isdigit()) else (name, None)


# ------------------------------------------------------------------ data
class Samples(torch.utils.data.Dataset):
    """t0 -> (fields at the lagged input dates, fields at t0 + 6h ... t0 + 6h*k) restricted to `names`"""

    def __init__(self, dates, lagged, k, names):
        self.dates, self.lagged, self.k, self.names = list(dates), list(lagged), k, list(names)
        self.z = None

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, i):
        if self.z is None:
            import aifs37_model
            self.z = aifs37_model.ERA5Zarr(cache_size=8)
        t0 = self.dates[i]
        ds = [t0 + h for h in self.lagged] + [t0 + H6 * j for j in range(1, self.k + 1)]
        out = []
        for d in ds:
            f = self.z.fields(d)
            out.append(np.stack([f[n] for n in self.names]).astype(np.float32))
        return t0.isoformat(), np.stack(out)       # (n_dates, n_names, points)


def date_range(a, b):
    d, out = a, []
    while d <= b:
        out.append(d)
        d += H6
    return out


# ------------------------------------------------------------------ model pieces
class A2Train:
    def __init__(self, config, device):
        exp, windows = utils.load_config(config)
        exp['date'] = exp['sdate']
        exp = utils.get_window(exp, list(windows)[0], windows)
        exp['device'] = device
        self.model = m = utils.get_model(exp)
        self.net = m.interface.model
        for p in self.net.parameters():
            p.requires_grad_(True)
        self.dev = m.device
        md = m.metadata
        self.lagged = sorted(md.lagged)
        n_out = len(m.idx_to_var)
        self.out_names = [m.idx_to_var[i] for i in range(n_out)]
        self.loaded = list(m._loaded)
        self.dyn_loaded = list(m._dyn_loaded)
        names = sorted(set(self.loaded) | set(self.out_names) | set(self.dyn_loaded))
        self.names = names
        self.pos = {n: i for i, n in enumerate(names)}
        # output normaliser scale
        post = m.interface.post_processors[m.dataset]
        mul = None
        for mod in post.modules():
            if hasattr(mod, '_norm_mul') and hasattr(mod, '_output_idx'):
                mul = mod._norm_mul[mod._output_idx].float()
        if mul is None or len(mul) != n_out:
            raise RuntimeError('could not find the output normaliser')
        self.out_mul = mul.to(self.dev)
        W = []
        for n in self.out_names:
            b, lv = _base(n)
            w = GV.get(b, 1.0)
            if lv is not None and b in PL:
                w *= max(0.1, 0.2 + 0.001 * lv)
            W.append(w)
        self.W = torch.tensor(W, device=self.dev)
        aw = torch.as_tensor(np.asarray(m.area_weights), device=self.dev, dtype=torch.float32)
        self.aw = aw / aw.sum()
        # input-side prognostic scale (for perturbation sizes, as in diagnostics/adjoint_quality)
        from model_correction import input_normaliser
        imul, _ = input_normaliser(m)
        self.P = m.pmask_in
        self.in_mul = imul[self.P].to(self.dev).float()
        inv = {i: n for n, i in m.var_to_idx.items()}
        self.prog_names = [inv[int(i)] for i in self.P.tolist()]
        self.prog_base = [_base(n)[0] for n in self.prog_names]
        self.sig = torch.tensor([SIG.get(b, 0.0) for b in self.prog_base], device=self.dev)
        self._nb = {}

    # ---- state construction and one differentiable step returning the full output
    def initial(self, t0, block):
        """block: (n_dates, n_names, points) numpy; the first len(lagged) dates are the inputs"""
        m = self.model
        fields = {v: np.stack([block[j, self.pos[v]] for j in range(len(self.lagged))]) for v in self.loaded}
        st = {'date': t0, 'latitudes': m.lats, 'longitudes': m.lons, 'fields': fields}
        return m.prepare_initial_state(st, t0).state

    def target(self, block, j):
        """physical output-space target at t0 + 6h*j: (1, points, n_out)"""
        i = len(self.lagged) - 1 + j
        y = torch.as_tensor(np.stack([block[i, self.pos[n]] for n in self.out_names], -1), device=self.dev)
        return y[None]

    def step(self, state, date, block=None, j=None):
        m = self.model
        y = m._forward(state)
        new_last = state[:, -1].clone()
        new_last[:, :, m.pmask_in] = y[:, :, m.pmask_out]
        fstate = {'date': date, 'latitudes': m.lats, 'longitudes': m.lons, 'fields': {}}
        for src in m._dyn_computed:
            arr = src.load_forcings_array([date], fstate)
            new_last[0, :, torch.as_tensor(src.mask, device=self.dev, dtype=torch.long)] = torch.as_tensor(
                np.ascontiguousarray(arr[:, 0].T), dtype=new_last.dtype, device=self.dev)
        if len(self.dyn_loaded):
            i = len(self.lagged) - 1 + j
            new_last[0, :, m._dyn_loaded_idx] = torch.as_tensor(
                np.stack([block[i, self.pos[v]] for v in self.dyn_loaded], -1), dtype=new_last.dtype, device=self.dev)
        return torch.cat([state[:, 1:], new_last[:, None]], dim=1), y

    def mse(self, y, yt):
        d = torch.nan_to_num((y.float() - yt) * self.out_mul)
        return ((d * d * self.W).mean(-1)[0] * self.aw).sum()

    # ---- perturbations (families as in diagnostics/adjoint_quality/families.py)
    def knn(self, k):
        if k not in self._nb:
            from scipy.spatial import cKDTree
            la, lo = np.radians(np.asarray(self.model.lats)), np.radians(np.asarray(self.model.lons))
            xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
            _, nb = cKDTree(xyz).query(xyz, k=k)
            self._nb[k] = torch.as_tensor(nb, device=self.dev)
        return self._nb[k]

    def smooth(self, f, k):
        nb = self.knn(k)
        for _ in range(3):
            f = f[nb].mean(1)
        return f

    def nnorm(self, d):
        z = d.float() * self.in_mul
        return ((z * z).mean(-1) * self.aw).sum().sqrt()

    def perturbation(self, fam, size, gen):
        npts, nP = len(self.aw), len(self.prog_names)
        e = torch.randn(npts, nP, generator=gen, device='cpu').to(self.dev)
        e = self.smooth(e, 200) if fam.endswith('_large') else e - self.smooth(e, 24)
        e = e / e.std(0, keepdim=True).clamp_min(1e-12)
        keep = {'hyd_z_large': {'z'}, 'hyd_z_small': {'z'}, 'hyd_t_large': {'t'}, 'geo_wind_large': {'u', 'v'},
                'small_all': {'t', 'z', 'u', 'v', 'q', 'sp', 'msl'}}[fam]
        mask = torch.tensor([1.0 if b in keep else 0.0 for b in self.prog_base], device=self.dev)
        d = e * self.sig * mask
        return d * (size / self.nnorm(d))

    def add_last(self, state, d):
        st = state.clone()
        st[0, -1, :, self.P] = st[0, -1, :, self.P] + d.to(st.dtype)
        return st


class AugResample:
    """Resampled real 2015 analysis errors with augmentation (coordinator spec, 2026-11-07): each draw takes a
    random pool (window start / end), two random real fields f1, f2 (each with a random longitude shift and sign, from
    analysis_noise.AnalysisNoise 'resample'), mixes them as cos(th) f1 + sin(th) f2 (variance kept), and scales by
    U(amp_range). apply() is AnalysisNoise.apply with this sample(): the older time level gets
    lag_corr * d + sqrt(1 - lag_corr^2) * (an independent draw)."""

    def __init__(self, cls, stats, lat, lon, dev, seed, amp_range):
        self.pools = [cls(stats, lat, lon, device=dev, mode='resample', which=w, seed=seed + i)
                      for i, w in enumerate(('start', 'end'))]
        self.gen = torch.Generator(device='cpu').manual_seed(seed + 100)
        self.amp = amp_range
        self._apply = cls.apply

    def _pool(self):
        return self.pools[int(torch.randint(len(self.pools), (1,), generator=self.gen))]

    def sample(self):
        th = float(torch.rand(1, generator=self.gen)) * 2 * math.pi
        s = self.amp[0] + (self.amp[1] - self.amp[0]) * float(torch.rand(1, generator=self.gen))
        return s * (math.cos(th) * self._pool().sample() + math.sin(th) * self._pool().sample())

    def apply(self, state, model, d=None, both_levels=True, lag_corr=None):
        return self._apply(self, state, model, d=d, both_levels=both_levels, lag_corr=lag_corr)


# ------------------------------------------------------------------ training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['C0', 'R', 'N', 'J', 'NA', 'NR', 'P', 'PU', 'PT'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='a37tmp/adjoint/config_pita22_ot_s_0104.yml',
                    help='any A2 4D-Var config (only the model block is used)')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--lr_min', type=float, default=3e-7)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--save_at', type=int, nargs='+', default=[1000, 2000, 4000])
    ap.add_argument('--val_every', type=int, default=500)
    ap.add_argument('--n_val', type=int, default=24)
    ap.add_argument('--rollout', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--noise_fam', nargs='+', default=['hyd_z_large', 'hyd_z_small', 'hyd_t_large'])
    ap.add_argument('--noise_size', type=float, default=0.02)
    ap.add_argument('--noise_p', type=float, default=0.5)
    ap.add_argument('--noise_stats', default='output/finetune/analysis_noise_stats.npz',
                    help='NA/NR: finetune/analysis_noise.py statistics (2015 PITA analysis - ERA5)')
    ap.add_argument('--noise_mode', default='resample_aug', choices=['gen', 'resample', 'resample_aug'])
    ap.add_argument('--lag_corr', type=float, default=0.27, help='NA/NR: correlation of the older input level noise')
    ap.add_argument('--amp_range', type=float, nargs=2, default=[0.5, 1.5])
    ap.add_argument('--val_noise_mode', default='gen', choices=['gen', 'resample'],
                    help='noise of the noisy validation (kept the same for every arm)')
    ap.add_argument('--j_K', type=int, default=4)
    ap.add_argument('--j_eps', type=float, default=0.01)
    ap.add_argument('--j_gamma', type=float, default=2.2)
    ap.add_argument('--j_lambda', default='auto')
    ap.add_argument('--j_samples', type=int, default=1)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pair_prefixes', nargs='+', default=[], help='P/PU: GEFS month prefixes (convert_month.py)')
    ap.add_argument('--pair_n', type=int, default=2)
    ap.add_argument('--pair_frac', type=float, default=0.2)
    ap.add_argument('--pair_workers', type=int, default=3)
    ap.add_argument('--pt_offload', action=argparse.BooleanOptionalAction, default=True,
                    help='PT: save the forward-over-reverse graph on host memory')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.arm in ('PU', 'PT') and '--noise_fam' not in sys.argv:
        a.noise_fam = ['hyd_z_large', 'hyd_z_small', 'hyd_t_large', 'small_all']
    JH = a.arm in ('J', 'PU', 'PT')    # arms with the excess-growth hinge
    PT = a.arm == 'PT'
    if PT:
        a.j_K = 1                      # the exact JVP is of the single 6-h step
    if a.smoke:
        a.steps, a.accum, a.save_at, a.val_every, a.n_val, a.warmup = 6, 2, [6], 3, 2, 2
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    gen = torch.Generator(device='cpu').manual_seed(a.seed + 1)
    T = A2Train(a.config, 'cuda')
    if PT:      # flex_attention has no forward-mode rule; the block (sdpa) sliding window is the same function
        import aifs_attention
        log(f'PT: attention -> block sliding window in {aifs_attention.use_sliding_window_attention(T.net, "block")} layers')
    kmax = max(a.rollout) if a.arm in ('R', 'NR') else 1
    if JH:
        kmax = max(kmax, 1)
    train_dates = [d for d in date_range(dt.datetime(1979, 1, 2), dt.datetime(2020, 12, 31, 18))
                   if d + H6 * max(kmax, a.j_K if JH else 1) <= dt.datetime(2020, 12, 31, 18)]
    val_dates = date_range(dt.datetime(2021, 1, 3), dt.datetime(2021, 12, 25))
    val_dates = [val_dates[i] for i in np.linspace(0, len(val_dates) - 1, a.n_val).astype(int)]
    kload = max(kmax, a.j_K) if JH else kmax
    order = np.random.permutation(len(train_dates))
    need = a.steps * a.accum
    order = np.concatenate([order] * (need // len(order) + 1))[:need]
    tr = Samples([train_dates[i] for i in order], T.lagged, kload, T.names)
    va = Samples(val_dates, T.lagged, 4, T.names)
    dl = torch.utils.data.DataLoader(tr, batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=4,
                                     persistent_workers=False)
    PL_ = None
    if a.arm in ('P', 'PU', 'PT'):
        sys.path.insert(0, os.path.join(ROOT, 'finetune'))
        from pair_loss import GEFSPairs, PairLoss, jvp_step
        pds = GEFSPairs(a.pair_prefixes, T.lagged, T.names, seed=a.seed)
        PL_ = PairLoss(T, pds.fields)
        record_later = {'pairs': len(pds), 'pair_months': len(a.pair_prefixes), 'pair_fields_scored': len(PL_.out_sel)}
        need_p = a.steps * a.pair_n
        pidx = np.arange(need_p) % len(pds)
        pdl = torch.utils.data.DataLoader(torch.utils.data.Subset(pds, pidx.tolist()), batch_size=None, shuffle=False,
                                          num_workers=a.pair_workers, prefetch_factor=2)
        pit = iter(pdl)
        lam_p = None
    opt = torch.optim.AdamW([p for p in T.net.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        f = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * f))

    AN = AV = None
    if a.arm in ('NA', 'NR') or os.path.exists(a.noise_stats):
        sys.path.insert(0, os.path.join(ROOT, 'finetune'))
        from analysis_noise import AnalysisNoise
        if os.path.exists(a.noise_stats):
            if a.noise_mode == 'resample_aug':
                AN = AugResample(AnalysisNoise, a.noise_stats, T.model.lats, T.model.lons, T.dev, a.seed + 7,
                                 a.amp_range)
            else:
                AN = AnalysisNoise(a.noise_stats, T.model.lats, T.model.lons, device=T.dev, mode=a.noise_mode,
                                   seed=a.seed + 7)
            # fixed draws for the noisy validation (same for every arm and step)
            AV = AnalysisNoise(a.noise_stats, T.model.lats, T.model.lons, device=T.dev, mode=a.val_noise_mode,
                               seed=12345)
            val_noise = [AV.sample() for _ in range(a.n_val)]
        elif a.arm in ('NA', 'NR'):
            raise FileNotFoundError(a.noise_stats)

    hist = open(os.path.join(a.out, 'history.jsonl'), 'a')

    def record(d):
        hist.write(json.dumps(d) + '\n')
        hist.flush()
        log(json.dumps(d))

    def validate(s):
        T.net.eval()
        l1, l4, z5 = [], [], []
        iz = T.out_names.index('z_500')
        with torch.no_grad():
            for i in range(len(va)):
                t0s, blk = va[i]
                t0 = dt.datetime.fromisoformat(t0s)
                st = T.initial(t0, blk)
                for j in range(1, 5):
                    st, y = T.step(st, t0 + H6 * j, blk, j)
                    yt = T.target(blk, j)
                    if j == 1:
                        l1.append(float(T.mse(y, yt)))
                    if j == 4:
                        l4.append(float(T.mse(y, yt)))
                        z5.append(float(((y[0, :, iz].float() - yt[0, :, iz]) ** 2 * T.aw).sum().sqrt() / 9.80665))
        rec = {'step': s, 'val_mse_6h': float(np.mean(l1)), 'val_mse_24h': float(np.mean(l4)),
               'val_z500_rmse_24h_m': float(np.mean(z5))}
        if AV is not None:          # robustness: the same forecasts from ERA5 + fixed analysis-noise draws
            n1, n4 = [], []
            with torch.no_grad():
                for i in range(len(va)):
                    t0s, blk = va[i]
                    t0 = dt.datetime.fromisoformat(t0s)
                    st = AV.apply(T.initial(t0, blk), T.model, d=val_noise[i], both_levels=True)
                    for j in range(1, 5):
                        st, y = T.step(st, t0 + H6 * j, blk, j)
                        if j == 1:
                            n1.append(float(T.mse(y, T.target(blk, j))))
                        if j == 4:
                            n4.append(float(T.mse(y, T.target(blk, j))))
            rec.update(val_noisy_mse_6h=float(np.mean(n1)), val_noisy_mse_24h=float(np.mean(n4)))
        record(rec)

    # ---- check: our step == the backend's advance (same weights, no grad)
    with torch.no_grad():
        t0s, blk = va[0]
        t0 = dt.datetime.fromisoformat(t0s)
        st = T.initial(t0, blk)
        s1, _ = T.step(st, t0 + H6, blk, 1)
        from aifs_model import AIFSState
        s2 = T.model.advance(AIFSState(st, t0), steps=1, use_checkpoint=False).state
        diff = float((s1 - s2).abs().max())
        record({'check_step_vs_backend_maxabs': diff})
    validate(0)

    lam = None
    if JH:
        if a.j_lambda == 'auto':
            lam = None
        else:
            lam = float(a.j_lambda)

    def pt_ctx():
        """PT: fp32 / TF32 off / math sdpa, saved tensors of the second derivative on the host"""
        from learn_assim import fp32
        from torch.nn.attention import SDPBackend, sdpa_kernel
        st_ = contextlib.ExitStack()
        st_.enter_context(fp32())
        st_.enter_context(sdpa_kernel([SDPBackend.MATH]))
        if a.pt_offload:
            st_.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=False))
        return st_

    def growth_loss_tl(st0):
        fam = a.noise_fam[int(torch.randint(len(a.noise_fam), (1,), generator=gen))]
        v = T.perturbation(fam, 1.0, gen)                       # unit normalised size: M'v is linear in v
        _, dy = jvp_step(T, st0, T.add_last(torch.zeros_like(st0), v))
        G = T.nnorm(dy[0][:, T.model.pmask_out])
        return torch.relu(G - a.j_gamma) ** 2, float(G)

    def growth_loss(st0, t0, blk):
        fam = a.noise_fam[int(torch.randint(len(a.noise_fam), (1,), generator=gen))]
        v = T.perturbation(fam, a.j_eps, gen)
        x, xp = st0, T.add_last(st0, v)
        for j in range(1, a.j_K + 1):
            x, _ = torch.utils.checkpoint.checkpoint(T.step, x, t0 + H6 * j, blk, j, use_reentrant=False)
            xp, _ = torch.utils.checkpoint.checkpoint(T.step, xp, t0 + H6 * j, blk, j, use_reentrant=False)
        G = T.nnorm((xp - x)[0, -1][:, T.P]) / a.j_eps
        return torch.relu(G - a.j_gamma) ** 2, float(G)

    T.net.train()
    it = iter(dl)
    t_start = time.time()
    for s in range(a.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(s)
        k_now = 1
        if a.arm in ('R', 'NR'):
            k_now = a.rollout[min(len(a.rollout) - 1, s * len(a.rollout) // a.steps)]
        tot_mse, tot_j, Gs = 0.0, 0.0, []
        for b in range(a.accum):
            t0s, blk = next(it)
            t0 = dt.datetime.fromisoformat(t0s)
            blk = np.asarray(blk)
            st = T.initial(t0, blk)
            if a.arm in ('NA', 'NR') and float(torch.rand(1, generator=gen)) < a.noise_p:
                lc = a.lag_corr if a.noise_mode == 'resample_aug' else None
                st = AN.apply(st, T.model, both_levels=True, lag_corr=lc)   # ERA5 + analysis noise; target stays ERA5
            if a.arm == 'N' and float(torch.rand(1, generator=gen)) < a.noise_p:
                fam = a.noise_fam[int(torch.randint(len(a.noise_fam), (1,), generator=gen))]
                st = T.add_last(st, T.perturbation(fam, a.noise_size, gen))
            loss = 0.0
            x = st
            for j in range(1, k_now + 1):
                if k_now > 1:
                    x, y = torch.utils.checkpoint.checkpoint(T.step, x, t0 + H6 * j, blk, j, use_reentrant=False)
                else:
                    x, y = T.step(x, t0 + H6 * j, blk, j)
                loss = loss + T.mse(y, T.target(blk, j)) / k_now
            tot_mse += float(loss.detach()) / a.accum
            if PT and b < a.j_samples:
                with pt_ctx():
                    lj, G = growth_loss_tl(st)
                    Gs.append(G)
                    if lam is None and float(lj) > 0:
                        lam = 0.1 * float(loss) / float(lj)
                        record({'j_lambda_auto': lam, 'G0': G, 'mse0': float(loss)})
                    if lam is not None and float(lj) > 0:      # an inactive hinge has no gradient
                        (lam * lj / a.j_samples / a.accum).backward()
                    tot_j += float(lj) / a.j_samples
            elif JH and b < a.j_samples:
                lj, G = growth_loss(st, t0, blk)
                Gs.append(G)
                if lam is None and float(lj) > 0:   # auto: L_J = 10% of the MSE at the first active hinge
                    lam = 0.1 * float(loss) / float(lj)
                    record({'j_lambda_auto': lam, 'G0': G, 'mse0': float(loss)})
                if lam is not None:
                    loss = loss + lam * lj * a.accum / a.j_samples
                tot_j += float(lj) / a.j_samples
            (loss / a.accum).backward()
        tot_p = 0.0
        if PL_ is not None:
            for _ in range(a.pair_n):
                tin, pblk, an = next(pit)
                with (pt_ctx() if PT else contextlib.nullcontext()):    # PT: backward inside the fp32 / offload context
                    lp = PL_.tl(tin, np.asarray(pblk), an) if PT else PL_(tin, np.asarray(pblk), an)
                    if lam_p is None:          # pair term = pair_frac of the total at the first step
                        lam_p = a.pair_frac / (1 - a.pair_frac) * tot_mse / max(float(lp), 1e-12)
                        record({'pair_lambda': lam_p, 'pair_loss0': float(lp), 'mse0': tot_mse, **record_later})
                    (lam_p * lp / a.pair_n).backward()
                tot_p += float(lp) / a.pair_n
        torch.nn.utils.clip_grad_value_(T.net.parameters(), 32.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 20 == 0 or a.smoke:
            rec = {'step': s + 1, 'lr': lr_at(s), 'train_mse': tot_mse, 'k': k_now, 'train_pair': tot_p,
                   'sec_per_step': (time.time() - t_start) / (s + 1)}
            if JH:
                rec.update(train_LJ=tot_j, G=float(np.mean(Gs)) if Gs else None, lam=lam)
            record(rec)
        if (s + 1) % a.val_every == 0 or (s + 1) == a.steps:
            validate(s + 1)
            T.net.train()
        if (s + 1) in a.save_at:
            path = os.path.join(a.out, f'step_{s + 1}.pt')
            torch.save({'state_dict': T.net.state_dict(), 'step': s + 1, 'arm': a.arm, 'args': vars(a)}, path)
            record({'saved': path})
    hist.close()


if __name__ == '__main__':
    main()
