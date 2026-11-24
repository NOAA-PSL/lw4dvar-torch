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
    def __init__(self, model=None, seed=0, raobs=False, raobs_bg_check=None, **exp):
        self.model = model if model is not None else ToyModel()
        # a None value drops the key, so the solver applies its own default
        self.exp = {k: v for k, v in {**BASE_EXP, **exp}.items() if v is not None}
        if raobs:
            # Must be set before self.interp_specs below:
            # _resolve_loss_interp_specs only folds raobs's own variables
            # into the sub-6h time-interpolation set when it sees these
            # (PR #29's review -- a toy t/q raobs trajectory, to give the
            # raobs cost term independent finite-difference/checkpointing
            # coverage through the real compute_loss_4dvar path, not just
            # an isolated unit test of the diagnostics function).
            self.exp['raobs_enabled'] = True
            self.exp['raobs_variables'] = ['temperature', 'specific_humidity']
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
            if m.increment_at_start:
                self.ref_state1 = self.state
            else:
                with torch.no_grad():
                    self.ref_state1 = m.advance(self.state, steps=1, use_checkpoint=False)
        self.jc_ref = None
        if self.exp.get('jc_ps_weight'):
            # as compute_optimal builds it
            with torch.no_grad():
                bg1 = m.advance(self.state, steps=1, use_checkpoint=False)
                if m.increment_at_start:
                    bg1, bg2 = self.state, bg1
                else:
                    bg2 = m.advance(bg1, steps=1, use_checkpoint=False)
                p1 = m.decode_state(bg1, only=['sp'])['sp']
                p2 = m.decode_state(bg2, only=['sp'])['sp']
            w = torch.as_tensor(m.area_weights)
            factor = float(self.exp['jc_ps_weight']) * int(traj['n_valid'].sum())
            self.jc_ref = ('sp', p1, p2, w / w.sum(), factor)
        self.jcz_ref = None
        if self.exp.get('jc_z_weight'):
            # as compute_optimal builds it
            w = torch.as_tensor(m.area_weights)
            self.jcz_ref = (w / w.sum(), float(self.exp['jc_z_weight']) * int(traj['n_valid'].sum()))

        self.raobs_traj = self._build_raobs_traj(rng, raobs_bg_check) if raobs else None

    def _build_raobs_traj(self, rng, bg_check):
        """A toy t/q radiosonde trajectory sharing psobs's own obs-slot
        grid (step_lo/alpha -- same convention get_raobs uses in
        production), so it automatically includes a between-step slot
        (dt_obs=3 < dt_verif=6, same as psobs's own obs). Values are the
        background's own H(x) plus noise (an O(1) innovation, like
        background_hx's psobs counterpart); one (level, station) cell per
        slot is masked out at "read time" to confirm a masked raobs
        observation contributes nothing to the cost or its gradient.
        `bg_check`: None (no gross check at all) or a threshold, so both
        the background-check-off and background-check-on code paths get
        independent finite-difference/checkpointing coverage through the
        real compute_loss_4dvar, not just the isolated unit tests of
        _compute_raobs_observation_diagnostics_at_time in
        tests/unit/test_solver_helpers.py (PR #29's review)."""
        m = self.model
        bases = ('t', 'q')
        levels_hpa = {b: np.asarray(m.pressure_levels(b), dtype=np.float64) for b in bases}
        capacity = 4
        n_slots = len(self.psobs_traj['step_lo'])
        step_lo = self.psobs_traj['step_lo']
        alpha = self.psobs_traj['alpha']
        idx, wts = self.grid_interp.weights(
            rng.uniform(0., 360., capacity), rng.uniform(-70., 70., capacity))
        traj = {
            'variables': list(bases),
            'levels_hpa': {b: torch.as_tensor(levels_hpa[b]) for b in bases},
            'interp_idx': torch.stack([idx] * n_slots),
            'interp_wts': torch.stack([wts] * n_slots).to(torch.float64),
            'n_steps': self.psobs_traj['n_steps'],
            'step_lo': step_lo,
            'alpha': alpha,
            'vertical_interpolation': 'exact',
            'tuv_bg_check': bg_check,
            'q_bg_check': bg_check,
        }
        with torch.no_grad():
            states = [self.state]
            for _ in range(traj['n_steps']):
                states.append(m.advance(states[-1], steps=1, use_checkpoint=False))
            decoded = [m.decode_state(s) for s in states]
            for base in bases:
                n_levels = levels_hpa[base].size
                values = torch.zeros((n_slots, n_levels, capacity), dtype=torch.float64)
                mask = torch.ones((n_slots, n_levels, capacity), dtype=torch.bool)
                noise_std = 0.5 if base == 't' else 0.0005
                for j in range(n_slots):
                    s, a = int(step_lo[j]), float(alpha[j])
                    field = (1. - a) * decoded[s][base] + a * decoded[s + 1][base]
                    bg = self.grid_interp.interp(idx, wts, field)  # (n_levels, capacity)
                    noise = torch.as_tensor(rng.normal(0., noise_std, (n_levels, capacity)))
                    values[j] = bg + noise
                mask[:, 0, 0] = False  # one masked-at-read-time cell, every slot
                traj['values_'+base] = values
                traj['mask_'+base] = mask
                if base == 'q':
                    traj['default_error_q'] = torch.full(
                        (n_slots, n_levels, capacity), 0.0005, dtype=torch.float64)
                else:
                    traj['error_t'] = 0.5

        # Attach the frozen background mask the SAME way compute_optimal
        # does in production (compute_ps_observation_hx, step_start=0, on
        # the uncorrected background) -- otherwise compute_loss_4dvar's
        # frozen_qc stays False for this trajectory (no 'bg_used' key)
        # and every gate below only ever exercises the LIVE-recompute QC
        # branch, never the frozen one compute_optimal actually uses
        # (PR #29's review, follow-up on the previous round's gate
        # coverage).
        self.model._prime_noise()  # no-op for ToyModel; matches compute_optimal's own call
        _, _, raobs_background_hx = utils.compute_ps_observation_hx(
            self.model, self.state, self.verif_ic, self.psobs_traj, self.exp, self.grid_interp,
            self.interp_specs, step_start=0, raobs_traj=traj,
        )
        traj['bg_used'] = {base: torch.as_tensor(raobs_background_hx[base]['used']) for base in bases}
        return traj

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
        J, _, _ = utils.compute_loss_4dvar(
            self.model, increment, 1.0, self.state, self.verif_ic, self.psobs_traj, 1, self.exp,
            self.grid_interp, -1, self.interp_specs, self.keep_mask, self.ref_state1, self.jc_ref,
            self.jcz_ref, self.raobs_traj)
        return J

    def cost_and_grad(self, increment):
        x = increment.detach().clone().requires_grad_(True)
        J = self.cost(x)
        (g,) = torch.autograd.grad(J, x)
        return J.detach(), g
