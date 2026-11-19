"""State-dependent model-error size for weak-constraint 4D-Var (notes/colearn_plan.md, plan_20261105.md "B0x").

The golden jumps' direction is not predictable from the state, but their SIZE is (top / bottom decile rms ratio
2.7-12x vs 1.1-2.4x for band x surface classes). JumpSigma is a per-column heteroscedastic MLP: from the state at the
jump time (column t, q, u, v at LEVELS + surface / time features, the 6-h tendency, and the mean / std over the 8
nearest columns) it predicts log sigma^2 of the jump in each of t, q, u, v at LEVELS, 2t, 2d, sp (normalised units).

node_scale(): per hidden-mesh node, S = sqrt(mean over its 4 nearest grid columns of s^2), with s^2 the column's
predicted variance relative to the global mean, averaged over variables; S is normalised to mean S^2 = 1 (the
total jump budget is unchanged) and clipped. Used by long_window_4dvar_utils.compute_optimal through
    weak_constraint: {..., node_scale: {path: output/colearn/jump_sigma.pt, clip: [0.25, 4.0]}}
which injects S * eta instead of eta: under AdamW's uniform weight decay that is a prior spread proportional to S.

Train: python jump_sigma.py train JUMPS.pt OUT.pt   (JUMPS.pt from a37tmp/colearn/jump_predict2.py)
"""
import sys

import numpy as np
import torch
from torch import nn

from model_correction import LEVELS, IN_PROFILE, IN_SINGLE, input_normaliser

OUT_PROFILE = ('t', 'q', 'u', 'v')
OUT_SINGLE = ('2t', '2d', 'sp')
IN_NAMES = [f'{v}_{p}' for v in IN_PROFILE for p in LEVELS] + list(IN_SINGLE)
DYN_NAMES = [f'{v}_{p}' for v in IN_PROFILE for p in LEVELS] + ['sp', 'skt', 'tcw']
OUT_NAMES = [f'{v}_{p}' for v in OUT_PROFILE for p in LEVELS] + list(OUT_SINGLE)


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _mlp(nin, nout, hidden=256, depth=3):
    layers, n = [], nin
    for _ in range(depth):
        layers += [nn.Linear(n, hidden), nn.SiLU()]
        n = hidden
    return nn.Sequential(*layers, nn.Linear(n, nout))


class Features:
    """Feature builder on the model grid: col (normalised g inputs), tend, neighbour mean - self, neighbour std."""

    def __init__(self, model, knn=8):
        from scipy.spatial import cKDTree
        dev = model.device
        mul, add = input_normaliser(model)
        V = model.var_to_idx
        self.iidx = torch.as_tensor([V[n] for n in IN_NAMES], device=dev)
        self.didx = torch.as_tensor([V[n] for n in DYN_NAMES], device=dev)
        self.imul, self.iadd = mul[self.iidx].to(dev), add[self.iidx].to(dev)
        self.dmul, self.dadd = mul[self.didx].to(dev), add[self.didx].to(dev)
        lat, lon = np.asarray(model.lats), np.asarray(model.lons)
        _, nb = cKDTree(_xyz(lat, lon)).query(_xyz(lat, lon), k=knn + 1)
        self.nb = torch.as_tensor(nb[:, 1:], device=dev)

    def __call__(self, state):
        """state: packed (1, 2, points, vars) -> (points, nfeat) float32"""
        x1, x0 = state[0, -1], state[0, -2]
        z1 = torch.nan_to_num(x1[:, self.didx].float() * self.dmul + self.dadd)
        z0 = torch.nan_to_num(x0[:, self.didx].float() * self.dmul + self.dadd)
        zn = z1[self.nb]
        col = torch.nan_to_num(x1[:, self.iidx].float() * self.imul + self.iadd)
        return torch.cat([col, z1 - z0, zn.mean(1) - z1, zn.std(1)], -1)


class JumpSigma(nn.Module):
    def __init__(self, nin, nout=len(OUT_NAMES)):
        super().__init__()
        self.net = _mlp(nin, nout)
        self.register_buffer('mean_var', torch.ones(nout))   # global mean predicted variance per output (training)

    def forward(self, f):
        return self.net(f)                                   # log sigma^2 (normalised by the training target std)

    def column_var(self, f):
        """relative predicted variance per column, averaged over outputs: (points,)"""
        return (torch.exp(self.net(f)) / self.mean_var).mean(-1)

    def save(self, path, **extra):
        torch.save({'state_dict': self.state_dict(), 'nin': self.net[0].in_features, **extra}, path)

    @classmethod
    def load(cls, path, device):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m = cls(ck['nin'])
        m.load_state_dict(ck['state_dict'])
        return m.to(device).eval()


def node_scale(model, states, path, clip=(0.25, 4.0), k_nodes=4, logger=None):
    """states: list of packed model states (one per jump, the state BEFORE the jump step) ->
    (n_jumps, n_nodes, 1) float32 per-node scale S, mean S^2 = 1 per jump."""
    from scipy.spatial import cKDTree
    dev = model.device
    sig = JumpSigma.load(path, dev)
    feats = Features(model)
    m = model.interface.model
    nlat, nlon = np.degrees(m._graph_data[m._graph_name_hidden]['x'].cpu().numpy()).T
    _, near = cKDTree(_xyz(np.asarray(model.lats), np.asarray(model.lons))).query(_xyz(nlat, nlon), k=k_nodes)
    near = torch.as_tensor(near, device=dev)
    out = []
    with torch.no_grad():
        for st in states:
            cv = sig.column_var(feats(st))                     # (points,)
            s2 = cv[near].mean(1)                              # (nodes,)
            s2 = s2 / s2.mean()
            S = s2.sqrt().clamp(*clip)
            S = S / S.pow(2).mean().sqrt()
            out.append(S)
            if logger:
                q = torch.quantile(S, torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()
                logger.info(f'node_scale: S p10/p50/p90 {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}, max {S.max():.2f}')
    return torch.stack(out)[..., None].float()


def train(jumps_f, out_f, epochs=6, seed=0):
    D = torch.load(jumps_f, weights_only=False)
    X = torch.cat([D[k].float() for k in ('col', 'tend', 'nbm', 'nbs')], -1).reshape(-1, sum(
        D[k].shape[-1] for k in ('col', 'tend', 'nbm', 'nbs')))
    Y = D['Y'].float()
    ysd = Y.reshape(-1, Y.shape[-1]).std(0).clamp_min(1e-6)
    Y = (Y / ysd).reshape(-1, Y.shape[-1])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    sig = JumpSigma(X.shape[-1], Y.shape[-1]).to(dev)
    opt = torch.optim.Adam(sig.parameters(), lr=1e-3)
    nb = 8192
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (X.shape[0] // nb))
    for ep in range(epochs):
        perm = torch.randperm(X.shape[0])
        tot = 0.
        for b in range(X.shape[0] // nb):
            i = perm[b * nb:(b + 1) * nb]
            lv = sig(X[i].to(dev))
            y = Y[i].to(dev)
            loss = (lv + y * y * torch.exp(-lv)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        print(f'epoch {ep}: nll {tot / (X.shape[0] // nb):.4f}', flush=True)
    with torch.no_grad():
        mv = torch.zeros(Y.shape[-1], device=dev)
        n = 0
        for b in range(0, X.shape[0], 65536):
            mv += torch.exp(sig(X[b:b + 65536].to(dev))).sum(0)
            n += min(65536, X.shape[0] - b)
        sig.mean_var.copy_(mv / n)
    sig.save(out_f, out_names=OUT_NAMES, trained_on=jumps_f, n_jumps=int(D['Y'].shape[0]))
    print('saved', out_f)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(sys.argv[2], sys.argv[3])
