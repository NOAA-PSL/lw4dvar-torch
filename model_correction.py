"""Learned model correction g_theta for the A1/A2 (aifs37) backend: x_{k+1} = M(x_k) + g_theta(x_k)
(notes/colearn_plan.md).

A column model: the same small MLP is applied to every grid column (like a physics parameterisation), so its
capacity is far too small to memorise a window. Inputs are the current state's columns (t, q, u, v, w at
LEVELS, plus surface and time-of-day / season features), normalised with the model's own input normaliser;
outputs are increments of t and q at LEVELS (in normalised units, converted back to physical). The last layer
starts at zero, so an untrained correction reproduces the uncorrected model exactly.

Hooked in backends/aifs/aifs37_model.py (`_advance_one_step`): every rollout -- the 4D-Var, forecast_skill,
the AMSU-A evaluator -- picks it up when `model.model_correction` is set. Config:
    exp:
      model_correction: {path: output/colearn/g_A/best.pt}     # load a trained correction
Train: a37tmp/colearn/train_offline.py (step A, against ERA5).
"""

import torch
from torch import nn

LEVELS = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
          250, 225, 200)
IN_PROFILE = ('t', 'q', 'u', 'v')   # A2 has no w input
OUT_PROFILE = ('t', 'q')
IN_SINGLE = ('sp', 'skt', 'tcw', 'lsm', 'cos_local_time', 'sin_local_time', 'cos_julian_day', 'sin_julian_day',
             'insolation')


def input_normaliser(model):
    """(mul, add) over the model's input variables: normalised = physical * mul + add"""
    ds = getattr(model, 'dataset', None)
    p = model.interface.pre_processors
    p = p[ds] if ds is not None and isinstance(p, nn.ModuleDict) else p
    for m in p.modules():
        if hasattr(m, '_norm_mul') and hasattr(m, '_input_idx'):
            return m._norm_mul[m._input_idx].float(), m._norm_add[m._input_idx].float()
    raise RuntimeError('no InputNormalizer in the pre-processors')


class ColumnCorrection(nn.Module):
    def __init__(self, var_to_idx, mul, add, levels=LEVELS, hidden=128, depth=2, out_scale=1e-3):
        super().__init__()
        self.levels = tuple(int(x) for x in levels)
        self.in_names = [f'{v}_{p}' for v in IN_PROFILE for p in self.levels] + list(IN_SINGLE)
        self.out_names = [f'{v}_{p}' for v in OUT_PROFILE for p in self.levels]
        missing = [n for n in self.in_names + self.out_names if n not in var_to_idx]
        if missing:
            raise ValueError(f'model_correction: variables not in the model input: {missing}')
        in_idx = torch.as_tensor([var_to_idx[n] for n in self.in_names], dtype=torch.long)
        out_idx = torch.as_tensor([var_to_idx[n] for n in self.out_names], dtype=torch.long)
        self.register_buffer('in_idx', in_idx)
        self.register_buffer('out_idx', out_idx)
        self.register_buffer('in_mul', mul[in_idx].clone())
        self.register_buffer('in_add', add[in_idx].clone())
        self.register_buffer('out_mul', mul[out_idx].clone())
        layers, n = [], len(self.in_names)
        for _ in range(depth):
            layers += [nn.Linear(n, hidden), nn.SiLU()]
            n = hidden
        last = nn.Linear(n, len(self.out_names))
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)
        self.out_scale = float(out_scale)   # normalised increment per unit net output (keeps Adam steps small)
        self.hparams = {'levels': self.levels, 'hidden': hidden, 'depth': depth, 'out_scale': self.out_scale}

    def forward(self, x):
        """x: (..., n_points, n_input_vars) physical state -> (..., n_points, n_out) physical increments"""
        z = x[..., self.in_idx].float() * self.in_mul + self.in_add
        z = torch.nan_to_num(z)
        return self.net(z) * self.out_scale / self.out_mul

    def save(self, path, **extra):
        torch.save({'state_dict': self.state_dict(), 'hparams': self.hparams, **extra}, path)

    @classmethod
    def for_model(cls, model, **hparams):
        mul, add = input_normaliser(model)
        return cls(model.var_to_idx, mul.cpu(), add.cpu(), **hparams).to(model.device)

    @classmethod
    def load(cls, path, model):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        g = cls.for_model(model, **ck['hparams'])
        g.load_state_dict(ck['state_dict'])
        return g


class StaticCorrection(nn.Module):
    """Fixed per-point, per-variable increment added every step: x_{k+1} = M(x_k) + c (offline mean tendency
    correction; a37tmp/strat_drift/tend_corr.py writes the npz: names (n_var,), mean (n_points, n_var), physical
    units of ERA5(t+6h) - M(ERA5)(t+6h)). Config:
        exp:
          model_correction: {static: output/strat_tend/t1_2014jfm.npz, vars: [t], scale: 1.0}
    `vars` keeps only those bases (t -> t_1000..t_1; sp, 2t, ... by full name); default all in the file."""

    def __init__(self, var_to_idx, names, field, vars=None, scale=1.0):
        super().__init__()
        def base(n):
            b, _, lev = n.rpartition('_')
            return b if b and lev.isdigit() else n
        keep = [i for i, n in enumerate(names) if vars is None or base(n) in vars or n in vars]
        if not keep:
            raise ValueError(f'StaticCorrection: no variables left for vars={vars}')
        self.out_names = [names[i] for i in keep]
        missing = [n for n in self.out_names if n not in var_to_idx]
        if missing:
            raise ValueError(f'StaticCorrection: variables not in the model input: {missing}')
        self.register_buffer('out_idx', torch.as_tensor([var_to_idx[n] for n in self.out_names], dtype=torch.long))
        self.register_buffer('field', torch.as_tensor(field[:, keep], dtype=torch.float32) * float(scale))

    def forward(self, x):
        """x: (..., n_points, n_input_vars) -> (..., n_points, n_out), independent of x"""
        return self.field.expand(*x.shape[:-1], self.field.shape[-1])

    @classmethod
    def load(cls, path, model, vars=None, scale=1.0):
        import numpy as np
        d = np.load(path)
        return cls(model.var_to_idx, [str(n) for n in d['names']], d['mean'], vars, scale).to(model.device)
