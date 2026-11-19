"""GEFS member-pair loss for fine-tuning A2 toward a physical tangent-linear (notes/adjoint_finetune_plan.md, arms P/PU).

Data: a37tmp/gefs/convert_month.py output per month: PREFIX.npy (n_t0, n_fhr, n_member, n_field, points) float16 =
GEFS - ERA5 at the valid time on A2's grid/levels, PREFIX.json (t0, fhr, member, fields).
A pair sample = (month, t0, start lead L in {6, 24}, member m >= 1): inputs at leads L-6, L, target at L+6.
  x_c / x_m = ERA5 state at the input valid times + the control / member anomaly on the GEFS upper-air fields (both input
  levels); surface fields: control = ERA5, member = ERA5 + (member - control) (GEFS-ERA5 surface offsets are orographic);
  msl += dsp * msl / sp, tcw += sum_p dq |dp| / g (the GEFS fields that A2 needs but GEFS lacks, adjusted consistently).
  loss = weighted MSE( [A2(x_m) - A2(x_c)] - [G_m - G_c](L+6) ) over the output variables that GEFS provides,
  with the trainer's output normaliser, variable/level weights and area weights.
  Arm PT: tl() replaces the finite difference by the exact tangent-linear M'(x_c)(x_m - x_c) (forward-mode JVP).
"""
import contextlib
import datetime as dt
import json

import numpy as np
import torch

H6 = dt.timedelta(hours=6)
G0 = 9.80665
_REAL_CKPT = []
_IN_DUAL = False       # inside the forward-AD level of jvp_step (see _dual_checkpoint)


class GEFSPairs(torch.utils.data.Dataset):
    def __init__(self, prefixes, lagged, names, starts=(6, 24), seed=0):
        self.prefixes, self.lagged, self.names = list(prefixes), list(lagged), list(names)
        self.meta = [json.load(open(p + '.json')) for p in self.prefixes]
        self.fields = self.meta[0]['fields']
        self.index = []
        for mi, m in enumerate(self.meta):
            for ti in range(len(m['t0'])):
                if m['t0'][ti] in m.get('exclude_t0', []):
                    continue
                for L in starts:
                    if all(f in m['fhr'] for f in (L - 6, L, L + 6)):
                        for k, mem in enumerate(m['member']):
                            if mem != 0:
                                self.index.append((mi, ti, L, k))
        rng = np.random.default_rng(seed)
        rng.shuffle(self.index)
        self.arr = [None] * len(self.prefixes)
        self.z = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        mi, ti, L, k = self.index[i % len(self.index)]
        m = self.meta[mi]
        if self.arr[mi] is None:
            self.arr[mi] = np.load(self.prefixes[mi] + '.npy', mmap_mode='r')
        A = self.arr[mi]
        if self.z is None:
            import aifs37_model
            self.z = aifs37_model.ERA5Zarr(cache_size=8)
        t0 = dt.datetime.fromisoformat(m['t0'][ti])
        tin = t0 + dt.timedelta(hours=L)                      # latest input time
        ds = [tin + h for h in self.lagged] + [tin + H6]
        blk = np.stack([np.stack([self.z.fields(d)[n] for n in self.names]).astype(np.float32) for d in ds])
        leads = [L - 6, L, L + 6]
        jj = [m['fhr'].index(f) for f in leads]
        ci = m['member'].index(0)
        an = np.stack([np.stack([np.asarray(A[ti, j, ci], np.float32), np.asarray(A[ti, j, k], np.float32)])
                       for j in jj])                         # (3 leads, 2 [control, member], n_field, points)
        return tin.isoformat(), blk, an


class PairLoss:
    def __init__(self, T, fields):
        m = T.model
        self.T = T
        V = m.var_to_idx
        self.fields = list(fields)
        self.in_pos = torch.as_tensor([i for i, f in enumerate(self.fields) if f in V], device=T.dev)
        self.in_idx = torch.as_tensor([V[f] for f in self.fields if f in V], device=T.dev)   # A2 has no q above ~50 hPa
        self.out_sel = [(i, self.fields.index(n)) for i, n in enumerate(T.out_names) if n in self.fields]
        self.oi = torch.as_tensor([a for a, _ in self.out_sel], device=T.dev)
        self.fi = torch.as_tensor([b for _, b in self.out_sel], device=T.dev)
        self.msl, self.sp, self.tcw = V.get('msl'), V.get('sp'), V.get('tcw')
        qf = [(i, int(f.split('_')[1])) for i, f in enumerate(self.fields) if f.startswith('q_')]
        ps = np.array([p for _, p in qf], float) * 100
        dp = np.abs(np.gradient(ps))
        self.qi = torch.as_tensor([i for i, _ in qf], device=T.dev)
        self.qdp = torch.as_tensor(dp / G0, device=T.dev, dtype=torch.float32)
        self.q_in = torch.as_tensor([V[f] for f in self.fields if f.startswith('q_') and f in V], device=T.dev)
        self.sfc = torch.as_tensor([i for i, f in enumerate(self.fields) if '_' not in f or not f.split('_')[-1].isdigit()],
                                   device=T.dev)
        self.spf = self.fields.index('sp') if 'sp' in self.fields else None

    def inject(self, state, an_levels):
        """state (1, n_lag, points, vars) physical; an_levels (n_lag, n_field, points) anomalies"""
        st = state.clone()
        for lvl in range(st.shape[1]):
            a = an_levels[lvl].T                                   # (points, n_field)
            x = st[0, lvl]
            if self.spf is not None and self.msl is not None:
                dsp = a[:, self.spf]
                x[:, self.msl] = x[:, self.msl] + dsp * x[:, self.msl] / x[:, self.sp]
            if self.tcw is not None:
                x[:, self.tcw] = x[:, self.tcw] + (a[:, self.qi] * self.qdp).sum(-1)
            x[:, self.in_idx] = x[:, self.in_idx] + a[:, self.in_pos].to(x.dtype)
            x[:, self.q_in] = x[:, self.q_in].clamp_min(0)
            st[0, lvl] = x
        return st

    def _states(self, tin_iso, blk, an):
        T = self.T
        tin = dt.datetime.fromisoformat(tin_iso)
        an = torch.as_tensor(np.asarray(an), device=T.dev)         # (3, 2, n_field, points)
        base = T.initial(tin, blk)
        # surface fields (sp, 2t, 2d, 10u, 10v) carry GEFS-vs-ERA5 orography/climate offsets (sp ~400 Pa rms):
        # the control keeps ERA5 there and the member gets ERA5 + (member - control); upper air gets full anomalies
        ac, am = an[:2, 0].clone(), an[:2, 1].clone()
        am[:, self.sfc] = am[:, self.sfc] - ac[:, self.sfc]
        ac[:, self.sfc] = 0
        xc = self.inject(base, ac)
        xm = self.inject(base, am)
        return tin, xc, xm, an

    def _loss(self, dA, an):
        T = self.T
        dG = (an[2, 1] - an[2, 0]).T[:, self.fi]
        d = (dA - dG) * T.out_mul[self.oi]
        return ((d * d * T.W[self.oi]).mean(-1) * T.aw).sum()

    def __call__(self, tin_iso, blk, an):
        T = self.T
        tin, xc, xm, an = self._states(tin_iso, blk, an)
        _, yc = T.step(xc, tin + H6, blk, 1)
        _, ym = T.step(xm, tin + H6, blk, 1)
        return self._loss((ym - yc)[0][:, self.oi].float(), an)    # (points, n_sel)

    def tl(self, tin_iso, blk, an):
        """Arm PT: || M'(x_c)(x_m - x_c) - dG ||^2, M' by forward-mode AD (exact tangent-linear of the 6-h step)."""
        _, xc, xm, an = self._states(tin_iso, blk, an)
        _, dy = jvp_step(self.T, xc, xm - xc)
        return self._loss(dy[0][:, self.oi].float(), an)


def jvp_step(T, state, delta):
    """(y, M'(state) delta) of the 6-h map's full output (1, points, n_out). Forward-mode through the model, so
    backward of the tangent is the forward-over-reverse second derivative w.r.t. the weights (caller sets fp32 etc.)."""
    import torch.autograd.forward_ad as fwAD
    from anemoi.models.distributed import graph
    # the single-process model-parallel sections are identities in forward but have no jvp rule
    cls = [c for n, c in vars(graph).items() if n.endswith('ParallelSection')]
    for c in cls:       # installed for good: the checkpoint recompute in backward needs them too
        c.jvp = staticmethod(lambda ctx, *t: t[0].view_as(t[0]))
    _install_safe_ops()
    from anemoi.models.layers import utils as lutils
    if lutils.checkpoint is not _dual_checkpoint:
        _REAL_CKPT.append(lutils.checkpoint)
        lutils.checkpoint = _dual_checkpoint
    return _dual(T, state, delta, fwAD)


def _install_safe_ops():
    """torch's forward-AD formulas of layer_norm, softmax and the math sdpa are wrong or modify saved tensors in place
    (wrong weight gradients, and save_on_cpu hides the version-counter error). Inside forward-AD regions swap in plain
    differentiable compositions (verified by the directional-derivative check a37tmp/pt_check.py)."""
    import torch.nn.functional as F
    if getattr(F, '_pt_safe', False):
        return
    ln, sdpa = F.layer_norm, F.scaled_dot_product_attention

    def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
        if not _IN_DUAL or len(normalized_shape) != 1:
            return ln(x, normalized_shape, weight, bias, eps)
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        y = xc * torch.rsqrt((xc * xc).mean(-1, keepdim=True) + eps)
        y = y if weight is None else y * weight
        return y if bias is None else y + bias

    def attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if not _IN_DUAL or dropout_p or is_causal or kw:
            return sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        sc = q.shape[-1] ** -0.5 if scale is None else scale
        s_ = (q @ k.transpose(-1, -2)) * sc
        if attn_mask is not None:
            s_ = s_.masked_fill(~attn_mask, float('-inf')) if attn_mask.dtype == torch.bool else s_ + attn_mask
        e = torch.exp(s_ - s_.detach().amax(-1, keepdim=True))
        return (e / e.sum(-1, keepdim=True)) @ v

    F.layer_norm, F.scaled_dot_product_attention, F._pt_safe = layer_norm, attention, True


def _dual_checkpoint(func, *args, use_reentrant=None, **kwargs):
    """torch.utils.checkpoint for dual (forward-AD) tensors: the recompute in backward has no dual level, so each
    checkpointed segment gets (primal, tangent) as explicit inputs, re-enters its own dual level inside the
    checkpointed function and returns (primal, tangent) outputs, which are re-attached to the caller's level."""
    import torch.autograd.forward_ad as fwAD
    from torch.utils._pytree import tree_flatten, tree_unflatten
    checkpoint = _REAL_CKPT[0]
    if not _IN_DUAL:       # ordinary (non-JVP) forward passes of the same process
        return checkpoint(func, *args, **kwargs, use_reentrant=False)
    flat, spec = tree_flatten((args, kwargs))
    tidx = [i for i, x in enumerate(flat) if isinstance(x, torch.Tensor)]
    prim, tang = [], []
    for i in tidx:
        pr, tg = fwAD.unpack_dual(flat[i])
        prim.append(pr)
        tang.append(tg)
    has_t = [t is not None for t in tang]
    tin = [t for t in tang if t is not None]

    def seg(*xs):
        global _IN_DUAL
        _IN_DUAL, was = True, _IN_DUAL      # `was`: already inside the forward pass's dual level
        try:
            return _seg(was, *xs)
        finally:
            _IN_DUAL = was

    def _seg(nested, *xs):
        n = len(prim)
        pr, tg = list(xs[:n]), iter(xs[n:])
        fl = list(flat)
        with (contextlib.nullcontext() if nested else fwAD.dual_level()):   # nested levels are unsupported
            for k, i in enumerate(tidx):
                fl[i] = fwAD.make_dual(pr[k], next(tg)) if has_t[k] else pr[k]
            a_, k_ = tree_unflatten(fl, spec)
            out = func(*a_, **k_)
            o_flat, o_spec = tree_flatten(out)
            res = []
            for o in o_flat:
                if isinstance(o, torch.Tensor):
                    po, to = fwAD.unpack_dual(o)
                    if to is None and po.is_floating_point():
                        to = torch.zeros_like(po)
                    res.append((po, to))
                else:
                    res.append((o, None))
        return o_spec, res

    box = {}

    def seg_t(*xs):
        o_spec, res = seg(*xs)
        box['spec'] = o_spec
        box['kind'] = ['plain' if not isinstance(r[0], torch.Tensor) else 'int' if r[1] is None else 'dual' for r in res]
        box['nontensor'] = [r[0] for r in res if not isinstance(r[0], torch.Tensor)]
        return tuple(x for r in res if isinstance(r[0], torch.Tensor) for x in ((r[0], r[1]) if r[1] is not None else (r[0],)))

    out = checkpoint(seg_t, *prim, *tin, use_reentrant=False)
    it, nt, o_flat = iter(out), iter(box['nontensor']), []
    for kind in box['kind']:
        o_flat.append(next(nt) if kind == 'plain' else next(it) if kind == 'int' else fwAD.make_dual(next(it), next(it)))
    return tree_unflatten(o_flat, box['spec'])


def _dual(T, state, delta, fwAD):
    global _IN_DUAL
    with fwAD.dual_level():
        _IN_DUAL = True
        try:
            y = T.model._forward(fwAD.make_dual(state, delta))
            y, dy = fwAD.unpack_dual(y)
        finally:
            _IN_DUAL = False
    return y, dy
