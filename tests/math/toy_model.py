"""A tiny, nonlinear, float64 `LatentForecastModel` for the math gates.

Not a weather model: just enough structure that the solver's cost function
(`compute_loss_4dvar`) runs through its real code paths -- the logpinterp
forward operator on pressure-level geopotential, the bilinear interpolator,
time interpolation between steps, `control_variables` masking and
`checkpoint_stride` -- with a forecast step that is nonlinear in the latent
increment, so the gradient, Taylor and dot-product tests are not trivially
satisfied the way they would be for a linear model.

State: channels-first `(1, 10, NLAT, NLON)`, geopotential (m^2/s^2) on
1000/850/500 hPa, temperature (K) and specific humidity (kg/kg) on the same
levels, then surface pressure (Pa). One step (6 h): encode the normalized
state with a fixed random channel mixing and tanh, add the latent increment
(first step only), advect one grid point east, mix neighbouring latitudes,
and decode a tanh of the latent back to a state tendency.
"""

import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.checkpoint

from forecast_model import LatentForecastModel

NLAT, NLON = 8, 16
LEVELS = np.array([1000.0, 850.0, 500.0])   # surface first, like the real backends
FAMILIES = {'z': [0, 1, 2], 't': [3, 4, 5], 'q': [6, 7, 8], 'sp': [9]}
ALIASES = {'z': 'geopotential', 't': 'temperature', 'q': 'specific_humidity', 'sp': 'surface_pressure'}
# typical values and spreads per channel, for normalization
MEAN = np.array([1100.0, 14500.0, 56000.0, 288.0, 280.0, 255.0, 8e-3, 5e-3, 1e-3, 1.0e5])
SPREAD = np.array([600.0, 600.0, 800.0, 8.0, 8.0, 8.0, 3e-3, 2e-3, 5e-4, 1.2e3])
N_LATENT = 6


@dataclass
class ToyState:
    state: torch.Tensor
    date: datetime.datetime


class ToyModel(LatentForecastModel):
    state_layout = 'channels_first'

    def __init__(self, seed=0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        n = len(MEAN)
        self.dtype = dtype
        self.mean = torch.as_tensor(MEAN, dtype=dtype).view(1, n, 1, 1)
        self.spread = torch.as_tensor(SPREAD, dtype=dtype).view(1, n, 1, 1)
        self.w_enc = torch.randn(N_LATENT, n, generator=g, dtype=dtype) / n ** 0.5
        self.w_dec = 0.3 * torch.randn(n, N_LATENT, generator=g, dtype=dtype) / N_LATENT ** 0.5
        lat1d = np.linspace(-78.75, 78.75, NLAT)
        lon1d = np.arange(NLON) * 360.0 / NLON
        self._lats = np.repeat(lat1d, NLON)
        self._lons = np.tile(lon1d, NLAT)

    # fixed properties
    @property
    def timestep(self):
        return datetime.timedelta(hours=6)

    @property
    def device(self):
        return torch.device('cpu')

    @property
    def lats(self):
        return self._lats

    @property
    def lons(self):
        return self._lons

    @property
    def latent_shape(self):
        return (N_LATENT, NLAT, NLON)

    def pressure_levels(self, base):
        if base not in ('z', 't', 'q'):
            raise KeyError(base)
        return LEVELS.copy()

    def resolve_columns(self, name):
        for base, alias in ALIASES.items():
            if name in (base, alias):
                return list(FAMILIES[base])
        raise KeyError(name)

    # states
    def wrap_state(self, tensor, date):
        return ToyState(tensor, date)

    def initial_state(self, date=datetime.datetime(2015, 1, 1), seed=1):
        """A smooth, roughly hydrostatic-looking state with some structure."""
        g = torch.Generator().manual_seed(seed)
        lat = torch.as_tensor(np.radians(self._lats), dtype=self.dtype).view(NLAT, NLON)
        lon = torch.as_tensor(np.radians(self._lons), dtype=self.dtype).view(NLAT, NLON)
        a = torch.randn(len(MEAN), 3, generator=g, dtype=self.dtype)
        pattern = (a[:, 0, None, None] * torch.cos(lat)[None] * torch.sin(lon + a[:, 1, None, None])
                   + a[:, 2, None, None] * torch.sin(2 * lat)[None])
        state = self.mean + 0.5 * self.spread * pattern[None]
        return ToyState(state, date)

    def prepare_initial_state(self, input_state, date):
        return ToyState(torch.as_tensor(input_state['state'], dtype=self.dtype), date)

    def prime_from_state(self, state):
        return state

    def _prime_noise(self):
        pass

    def decode_state(self, state, time_index=-1, only=None):
        x = state.state[0].reshape(len(MEAN), NLAT * NLON)
        out = {}
        for base, cols in FAMILIES.items():
            if only is not None and base not in only and ALIASES[base] not in only:
                continue
            v = x[cols] if len(cols) > 1 else x[cols[0]]
            out[base] = v
            out[ALIASES[base]] = v
        return out

    # dynamics
    def _step(self, x, increment):
        h = torch.tanh(torch.einsum('lc,bchw->blhw', self.w_enc, (x - self.mean) / self.spread))
        if increment is not None:
            h = h + increment[None]
        h = torch.roll(h, shifts=1, dims=-1)                              # advect east
        h = 0.5 * h + 0.25 * (torch.roll(h, 1, dims=-2) + torch.roll(h, -1, dims=-2))
        dx = torch.einsum('cl,blhw->bchw', self.w_dec, torch.tanh(h))
        return x + self.spread * dx

    def advance(self, state, steps=1, use_checkpoint=True, latent_increment=None):
        x = state.state
        for s in range(steps):
            inc = latent_increment if s == 0 else None
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self._step, x, inc, use_reentrant=False)
            else:
                x = self._step(x, inc)
        return ToyState(x, state.date + steps * self.timestep)
