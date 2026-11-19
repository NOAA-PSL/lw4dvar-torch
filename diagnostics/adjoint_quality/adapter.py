"""Backend adapter for the adjoint-quality battery (notes/adjoint_quality.md).

Everything the battery needs from a forecast model goes through `Adapter`:
  - precision: `set_precision('bf16'|'fp16'|'fp32')`; fp32 disables every `torch.autocast` region (backend-agnostic:
    `torch.autocast` is replaced by a disabled context for the duration), the reduced mode restores it;
  - the prognostic variables: names, indices into the packed state, and a per-variable scale `mul` such that
    `physical * mul` is in normalised (unit-variance) units -- the norm every test uses;
  - `prog(state)` -> (points, n_prog) last time level; `add_prog(state, d)` -> new state with `d` added there;
  - `field(state, name)` -> (points,) one packed column, last time level;
  - grid (lat, lon, area weights -- `model.area_weights`, never cos lat), `latent_shape`, `advance`.

Implemented and verified: the AIFS family (`aifs` backend: AIFS-single 1.x / 2.x, and `aifs37` A1 / A2), whose packed
state is (1, multi_step, points, vars) ('channels_last') and whose normaliser is anemoi's InputNormalizer.
Other backends raise NotImplementedError naming what is missing (see `make_adapter`).
"""
import contextlib

import numpy as np
import torch

_ORIG_AUTOCAST = torch.autocast


class _DisabledAutocast(_ORIG_AUTOCAST):
    """torch.autocast(...) that is always disabled (fp32 everywhere a backend asked for autocast)."""

    def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
        super().__init__(device_type, enabled=False)


class Adapter:
    backend = None

    def __init__(self, model):
        self.model = model

    # ---- precision
    def reduced_precision(self):
        """name of the model's reduced-precision autocast mode ('bf16' / 'fp16'), or None if it runs in fp32"""
        dt = getattr(self.model, '_autocast_dtype', None)
        return {torch.bfloat16: 'bf16', torch.float16: 'fp16'}.get(dt)

    def set_precision(self, mode):
        if mode == 'fp32':
            torch.autocast = _DisabledAutocast
        elif mode == self.reduced_precision():
            torch.autocast = _ORIG_AUTOCAST
        else:
            raise ValueError(f'precision {mode!r}: this model supports fp32 and {self.reduced_precision()}')

    # ---- grid / latent
    @property
    def lat(self):
        return np.asarray(self.model.lats)

    @property
    def lon(self):
        return np.asarray(self.model.lons)

    @property
    def area_weights(self):
        return np.asarray(self.model.area_weights)

    @property
    def latent_shape(self):
        return tuple(self.model.latent_shape)

    def advance(self, state, steps=1, grad=False, latent_increment=None):
        return self.model.advance(state, steps=steps, use_checkpoint=grad, latent_increment=latent_increment)

    def model_id(self, exp):
        return {'backend': exp.get('model_backend'), 'aifs37': bool(exp.get('aifs37', False)),
                'model_name': exp.get('model_name'), 'path_model': exp.get('path_model'),
                'autocast': self.reduced_precision() or 'fp32', 'latent_shape': list(self.latent_shape),
                'n_points': int(len(self.lat)), 'n_prog': int(len(self.prog_names))}


class AIFSAdapter(Adapter):
    """AIFS family: packed (1, multi_step, points, vars); prognostics = `pmask_in`; scale = anemoi InputNormalizer."""
    backend = 'aifs'

    def __init__(self, model):
        super().__init__(model)
        from model_correction import input_normaliser
        dev = model.device
        self.P = model.pmask_in.to(dev)
        mul, _ = input_normaliser(model)
        self.mul = mul[self.P].to(dev).float()
        self.var_to_idx = dict(model.var_to_idx)
        inv = {i: n for n, i in self.var_to_idx.items()}
        self.prog_names = [inv[int(i)] for i in self.P.tolist()]
        self.prog_pos = {n: k for k, n in enumerate(self.prog_names)}

    def prog(self, state):
        return state.state[0, -1][:, self.P]

    def add_prog(self, state, d):
        st = state.state.clone()
        st[0, -1, :, self.P] = st[0, -1, :, self.P] + d
        return type(state)(st, state.date)

    def field(self, state, name):
        return state.state[0, -1][:, self.var_to_idx[name]]


def make_adapter(model, exp):
    backend = exp.get('model_backend')
    if backend == 'aifs':
        return AIFSAdapter(model)
    missing = {
        'fcn3': "FCN3 packs the state channels-first (1, n_vars, H, W) and has no anemoi InputNormalizer: needs "
                "prog/add_prog/field for that layout and a per-channel normalisation scale (e.g. the model's own "
                "in/out std), plus a check that advance(latent_increment=...) matches its latent_shape (3-D)",
        'sfno': 'SFNO shares the FCN3 layout (channels-first) and the same gaps as fcn3',
        'ace2': "ACE2 packs channels-first with prognostics first (model._prognostic) and normalises inside the "
                "stepper: needs the per-channel scale exposed and a channels-first prog/add_prog/field",
        'aurora': "Aurora is channels_last like AIFS but has no pmask_in / InputNormalizer: needs its prognostic "
                  "channel list and per-variable scale (Aurora's own normalisation stats), and an fp32 check of "
                  "its internal autocast (fp16 default)",
    }
    raise NotImplementedError(f'adjoint-quality adapter for backend {backend!r} not implemented: '
                              + missing.get(backend, 'unknown backend'))


@contextlib.contextmanager
def precision(adapter, mode):
    adapter.set_precision(mode)
    try:
        yield
    finally:
        torch.autocast = _ORIG_AUTOCAST
