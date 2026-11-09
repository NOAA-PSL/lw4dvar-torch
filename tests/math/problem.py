"""A one-window 4D-Var problem on the toy model, built in memory: synthetic
surface-pressure obs every 3 h over a 24 h window (so obs fall between model
steps and exercise the time interpolation), and `cost(increment)`, the
solver's own `compute_loss_4dvar`.

QC is frozen (no gross check, no orography rejection), so the cost is a
smooth function of the increment and finite differences are meaningful.
"""

import numpy as np
import torch

import long_window_4dvar_utils as utils

utils._ensure_backend_on_path('fcn3')
import fcn3_grid  # noqa: E402

from toy_model import ToyModel  # noqa: E402

N_OBS = 24        # real obs per slot
N_PAD = 4         # padding slots (unavailable), as get_psobs leaves them
BASE_EXP = {
    'dt_verif': 6, 'n_verif': 4, 'dt_obs': 3,
    'ps_operator': 'logpinterp',
    'nobs_max': N_OBS + N_PAD,
    'zthresh': 1e10, 'zconst': 1e-3, 'bg_check': 1e10,
    'checkpoint_stride': 1,
}


class Problem:
    def __init__(self, model=None, seed=0, **exp):
        self.model = model if model is not None else ToyModel()
        self.exp = {**BASE_EXP, **exp}
        self.state = self.model.initial_state()
        rng = np.random.default_rng(seed)
        m = self.model
        self.grid_interp = fcn3_grid.BilinearGridInterpolator(m.lons, m.lats)
        n_slots = self.exp['n_verif'] * self.exp['dt_verif'] // self.exp['dt_obs'] + 1
        n_steps = self.exp['n_verif'] * self.exp['dt_verif'] // 6
        nmax = self.exp['nobs_max']

        # smooth orography (m) and stations near (but not on) it
        lat = np.radians(m.lats)
        lon = np.radians(m.lons)
        orog = 400.0 + 300.0 * np.cos(lat) * np.sin(2 * lon)
        self.verif_ic = {'geopotential_at_surface': torch.as_tensor(utils.GRAV * orog)}

        def f64(a):
            return torch.as_tensor(a, dtype=torch.float64)

        traj = {k: torch.full((n_slots, nmax), 1e10, dtype=torch.float64)
                for k in ('lon', 'lat', 'oberr', 'oberr_qc')}
        traj['elev'] = torch.zeros((n_slots, nmax), dtype=torch.float64)
        traj['ob'] = torch.zeros((n_slots, nmax), dtype=torch.float64)
        for j in range(n_slots):
            traj['lon'][j, :N_OBS] = f64(rng.uniform(0, 360, N_OBS))
            # half the obs crowd into one 10-degree box, so obs-density
            # weights differ from 1
            la = rng.uniform(-75, 75, N_OBS)
            la[:N_OBS // 2] = rng.uniform(40, 49, N_OBS // 2)
            traj['lon'][j, :N_OBS // 2] = f64(rng.uniform(100, 109, N_OBS // 2))
            traj['lat'][j, :N_OBS] = f64(la)
            traj['elev'][j, :N_OBS] = f64(rng.uniform(0, 900, N_OBS))
            traj['oberr'][j, :N_OBS] = 1.0
            traj['oberr_qc'][j, :N_OBS] = 1.0
        traj['n_valid'] = torch.full((n_slots,), N_OBS, dtype=torch.int32)
        step_lo, alpha = [], []
        for j in range(n_slots):
            t = j * self.exp['dt_obs']
            s, a = t // 6, (t % 6) / 6
            if s >= n_steps:
                s, a = n_steps - 1, 1.0
            step_lo.append(s)
            alpha.append(a)
        traj['step_lo'] = torch.as_tensor(step_lo)
        traj['alpha'] = torch.as_tensor(alpha, dtype=torch.float64)
        traj['n_steps'] = n_steps
        idx, wts = zip(*(self.grid_interp.weights(traj['lon'][j].numpy(), traj['lat'][j].numpy())
                         for j in range(n_slots)))
        traj['interp_idx'] = torch.stack(idx)
        traj['interp_wts'] = torch.stack(wts).to(torch.float64)
        self.psobs_traj = traj
        self.interp_specs = utils._resolve_loss_interp_specs(m, self.exp)

        # obs: the background's own H(x) plus a few hPa of noise, so the
        # innovations are O(1) obs errors
        traj['ob'][:, :N_OBS] = self.background_hx()[:, :N_OBS] + f64(rng.normal(0, 3.0, (n_slots, N_OBS)))

        self.keep_mask = self.ref_state1 = None
        if self.exp.get('control_variables'):
            self.keep_mask = utils._resolve_control_mask(m, self.exp, len(m.mean.flatten()), m.device)
            with torch.no_grad():
                self.ref_state1 = m.advance(self.state, steps=1, use_checkpoint=False)
        self.jc_ref = None
        if self.exp.get('jc_ps_weight'):
            # as compute_optimal builds it
            with torch.no_grad():
                bg1 = m.advance(self.state, steps=1, use_checkpoint=False)
                bg2 = m.advance(bg1, steps=1, use_checkpoint=False)
                p1 = m.decode_state(bg1, only=['sp'])['sp']
                p2 = m.decode_state(bg2, only=['sp'])['sp']
            w = torch.as_tensor(m.area_weights)
            factor = float(self.exp['jc_ps_weight']) * int(traj['n_valid'].sum())
            self.jc_ref = ('sp', p1, p2, w / w.sum(), factor)

    def background_hx(self):
        """H(x) of the background at every slot, (n_slots, nobs_max)."""
        traj, m = self.psobs_traj, self.model
        out = []
        with torch.no_grad():
            states = [self.state]
            for _ in range(traj['n_steps']):
                states.append(m.advance(states[-1], steps=1, use_checkpoint=False))
            decoded = [m.decode_state(s) for s in states]
            for j in range(len(traj['step_lo'])):
                s, a = int(traj['step_lo'][j]), float(traj['alpha'][j])
                d = utils._interp_decoded(decoded[s], decoded[s + 1], a, self.interp_specs)
                out.append(utils._compute_ps_observation_diagnostics_at_time(
                    m, d, self.verif_ic, traj, self.exp, j, self.grid_interp)[1])
        return torch.stack(out)

    def random_increment(self, seed, scale=0.3):
        g = torch.Generator().manual_seed(seed)
        return scale * torch.randn(self.model.latent_shape, generator=g, dtype=torch.float64)

    def cost(self, increment):
        """The solver's Jo (+ Jc) for a latent increment."""
        J, _ = utils.compute_loss_4dvar(
            self.model, increment, 1.0, self.state, self.verif_ic, self.psobs_traj, 1, self.exp,
            self.grid_interp, -1, self.interp_specs, self.keep_mask, self.ref_state1, self.jc_ref)
        return J

    def cost_and_grad(self, increment):
        x = increment.detach().clone().requires_grad_(True)
        J = self.cost(x)
        (g,) = torch.autograd.grad(J, x)
        return J.detach(), g
