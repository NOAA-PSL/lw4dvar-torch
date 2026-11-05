"""
The official NeuralGCM (`neuralgcm.PressureLevelModel`, e.g. the published
deterministic_2_8_deg.pkl) behind the torch 4D-Var solver, via
backends/jax_bridge.py -- the model of the original JAX pressure-obs
experiments (lw4dvar-jax / long-window-4dvar-psobs), for reproducing them
with this solver.

Facts this depends on (backends/neuralgcm/probe_official_state.py, 2.8 deg):

- 1 h step. `encode` gives a ModelState: `state` (modal vorticity,
  divergence, temperature_variation, cloud ice / liquid and specific
  humidity tracers, all (32, 127, 65); log_surface_pressure (1, 127, 65);
  sim_time scalar), `memory` (the same structure: the previous state), and
  diagnostics (two nodal (128, 64), one modal (127, 65)) plus an int32
  randomness leaf.
- Nondimensional units (dinosaur DEFAULT_SCALE: time 1 / (2 Omega)).
- `decode` gives geopotential, temperature, humidity, u, v, cloud water on
  37 pressure levels (1..1000 hPa), (lon, lat) on the TL63 Gaussian grid,
  latitudes south-to-north. No surface pressure: ps obs use `logpinterp`
  (the original experiments' operator); decode_state adds 'sp' from the
  state's ln ps (for the jc_ps_weight balance penalty).
- Inputs and forcings: per-time ERA5 netCDFs (`neuralgcm_input_dir`,
  YYYY-MM-DDTHH.nc, 6-hourly; the original runs' `path_input`). Each step
  gets the single-time forcings of the nearest file, as `unroll` would use
  (fixed structure, so the jitted functions do not retrace).

Packed torch state (1, 389, 128, 65): every float leaf of the ModelState
(state, memory, diagnostics) as channels, modal (127, 65) and nodal
(128, 64) padded to (128, 65), the two sim_time scalars in a last channel
(carried exactly in float32, as the model itself stores them). The int
randomness leaf comes from the first encoded state.

Latent control: the 193 channels of `state` (not memory or diagnostics),
like the original's `state.state` control; the increment is divided by
`state_scales` ({family: factor}, the original convention; default
{tracer_specific_humidity: 1e5}). As in the original, tracers not named in
state_scales (cloud ice / liquid by default) are frozen: their increment
scale is 0.
"""

import datetime
import os
import pickle

import numpy as np
import torch
import xarray as xr

import forecast_model
from backends import jax_bridge

import jax  # noqa: E402
import jax._src.array as jax_array  # noqa: E402
import jax.numpy as jnp  # noqa: E402

GRAV = 9.80616  # dinosaur's

# control families (state.state leaves, in ModelState flatten order).
FAMILIES = ("vorticity", "divergence", "temperature_variation", "log_surface_pressure",
            "tracer_cloud_ice", "tracer_cloud_liquid", "tracer_specific_humidity")
DEFAULT_STATE_SCALES = {"tracer_specific_humidity": 1.0e5}
# decode outputs -> solver names (pressure-level families, surface first).
DECODED = {"z": "geopotential", "t": "temperature", "q": "specific_humidity",
           "u": "u_component_of_wind", "v": "v_component_of_wind"}
SPATIAL = (128, 65)


def _reconstruct_array(fun, args, arr_state, aval_state):
    # the checkpoints pickle jax arrays with an aval state (`named_shape`)
    # current jax no longer accepts; numpy arrays are all from_checkpoint needs.
    del aval_state
    value = fun(*args)
    value.__setstate__(arr_state)
    return np.asarray(value)


def _np_datetime(date: datetime.datetime) -> np.datetime64:
    if date.tzinfo is not None:
        date = date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return np.datetime64(date, "s")


class NeuralGCMState:
    def __init__(self, state: torch.Tensor, date: datetime.datetime):
        self.state = state
        self.date = date

    def __copy__(self) -> "NeuralGCMState":
        return NeuralGCMState(self.state, self.date)


class NeuralGCMModel(forecast_model.LatentForecastModel):
    def __init__(self, checkpoint_path: str, input_dir: str, device: str = "cuda",
                 state_scales: "dict[str, float] | None" = None,
                 ngcm_train_path: "str | None" = None):
        import sys

        if ngcm_train_path:
            # dinosaur's default GPU float32 dot algorithm is miscompiled with
            # jax <= 0.11.1 (neuralgcm/dinosaur#147): use HIGHEST, before any jit.
            if ngcm_train_path not in sys.path:
                sys.path.insert(0, ngcm_train_path)
            from ngcm_train import precision  # noqa: E402
            precision.use_highest_precision()
        import neuralgcm  # noqa: E402

        jax_array._reconstruct_array = _reconstruct_array
        with open(checkpoint_path, "rb") as f:
            self.model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(f))
        self._device = torch.device(device)
        self._input_dir = input_dir
        self._timestep = datetime.timedelta(seconds=int(self.model.timestep / np.timedelta64(1, "s")))
        grid = self.model._structure.specs.aux_features["xarray_dataset"]
        lat1d = np.asarray(grid["latitude"].values, np.float64)
        lon1d = np.asarray(grid["longitude"].values, np.float64)
        self._nlat, self._nlon = lat1d.size, lon1d.size
        self._lats = np.repeat(lat1d, self._nlon)
        self._lons = np.tile(lon1d, self._nlat)
        levels = np.asarray(grid["level"].values, np.float32)
        self._order = np.argsort(-levels)  # surface first
        self._plevs = levels[self._order]

        self._layout = None  # set by the first encode
        self._n_control = None
        scales = dict(DEFAULT_STATE_SCALES if state_scales is None else state_scales)
        unknown = set(scales) - set(FAMILIES)
        if unknown:
            raise KeyError(f"state_scales families {sorted(unknown)} not in {list(FAMILIES)}")
        self._scale_by_family = scales
        self._channel_scale = None
        self._files = {}
        self._levels_by_base = {b: [(int(p), None) for p in self._plevs] for b in DECODED}
        self._single_by_base = {"z500": None, "sp": None}

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> datetime.timedelta:
        return self._timestep

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def lats(self) -> np.ndarray:
        return self._lats

    @property
    def lons(self) -> np.ndarray:
        return self._lons

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        self._require_layout()
        return (self._n_control, *SPATIAL)

    @property
    def diagnostic_columns(self) -> list[int]:
        """Everything but the controlled state.state channels: memory,
        diagnostics and the sim_time channel. The solver's control mask
        never resets these to the uncorrected background (it would leave
        the memory -- the previous state, which the network differences
        against the state -- unperturbed while the state carries the
        increment)."""
        self._require_layout()
        return list(range(self._n_control, self._scalar_channel + 1))

    @property
    def state_layout(self) -> str:
        return "channels_first"

    def resolve_columns(self, name: str) -> list[int]:
        self._require_layout()
        if name in self._family_channels:
            return list(self._family_channels[name])
        raise KeyError(f"{name!r} is not a NeuralGCM state family (one of {list(FAMILIES)})")

    def is_known_variable(self, name: str) -> bool:
        return name in FAMILIES or name in DECODED or name in ("z500", "sp")

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        if base not in DECODED:
            raise KeyError(f"{base!r} is not a pressure-level family")
        return self._plevs.copy()

    def wrap_state(self, state: torch.Tensor, date: datetime.datetime) -> NeuralGCMState:
        return NeuralGCMState(state, date)

    def _prime_noise(self) -> None:
        """No-op: deterministic."""

    def _require_layout(self):
        if self._layout is None:
            raise RuntimeError("NeuralGCMModel: no state encoded yet (prepare_initial_state first)")

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _file(self, t: np.datetime64) -> xr.Dataset:
        if t not in self._files:
            if len(self._files) > 64:
                self._files.clear()
            path = os.path.join(self._input_dir, str(t.astype("datetime64[h]")) + ".nc")
            self._files[t] = xr.open_dataset(path).load()
        return self._files[t]

    def _forcings(self, date: datetime.datetime):
        """Single-time forcings ([level, lon, lat], `data_from_xarray`) from
        the 6-hourly file nearest `date`: `advance` and `decode` both take
        one time (advance prepends its own time axis), and NeuralGCM's
        `unroll` likewise uses the nearest forcing time (jnp.round, half to
        even: offsets 0-3 h use the earlier file, 4-5 h the later)."""
        t = _np_datetime(date).astype("datetime64[h]")
        offset = int(t.astype(int) % 6)
        t = t - offset + (6 if offset > 3 else 0)
        return self.model.data_from_xarray(self._file(t).isel(time=0))[1]

    def read_inputs(self, date: datetime.datetime):
        """(inputs, forcings) at `date` from the per-time input file."""
        ds = self._file(_np_datetime(date).astype("datetime64[h]"))
        return self.model.data_from_xarray(ds.isel(time=0))

    def read_verif(self, date: datetime.datetime) -> dict[str, torch.Tensor]:
        """ERA5 z500 and surface geopotential (m^2/s^2) at `date`, flattened
        latitude-major, from the per-time input file."""
        ds = self._file(_np_datetime(date).astype("datetime64[h]")).isel(time=0)
        to = lambda a: torch.as_tensor(np.asarray(a.transpose("latitude", "longitude").values).reshape(-1),
                                       dtype=torch.float32, device=self._device)
        return {"z500": to(ds["geopotential"].sel(level=500)),
                "geopotential_at_surface": to(ds["geopotential_at_surface"])}

    # ------------------------------------------------------------------
    # packing
    # ------------------------------------------------------------------

    def _set_layout(self, model_state) -> None:
        """Channel layout of every float leaf; control = the state.state
        leaves (the first n_state leaves of the flatten order)."""
        leaves, treedef = jax.tree_util.tree_flatten(model_state)
        n_state = len(jax.tree.leaves(model_state.state))
        layout, scalars, ints = [], [], {}
        c = 0
        for i, leaf in enumerate(leaves):
            a = np.asarray(leaf)
            if not np.issubdtype(a.dtype, np.floating):
                ints[i] = a
                continue
            if a.ndim == 0:
                scalars.append(i)
                continue
            shape = a.shape if a.ndim == 3 else (1, *a.shape)
            if shape[1] > SPATIAL[0] or shape[2] > SPATIAL[1]:
                raise ValueError(f"leaf {i} shape {a.shape} exceeds the packed {SPATIAL}")
            layout.append((i, a.shape, c, shape[0]))
            c += shape[0]
        # state.state is the ModelState's first child, so its leaves come
        # first: the control is the leading channels.
        self._n_control = sum(n for i, _, _, n in layout if i < n_state)
        self._scalar_channel = c
        self._layout = (treedef, len(leaves), layout, scalars, ints)
        # control families: state.state leaves in order (tracers sorted by name).
        names = {"specific_cloud_ice_water_content": "tracer_cloud_ice",
                 "specific_cloud_liquid_water_content": "tracer_cloud_liquid",
                 "specific_humidity": "tracer_specific_humidity"}
        paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(model_state.state)[0]]
        self._family_channels = {}
        fields = ["vorticity", "divergence", "temperature_variation", "log_surface_pressure"]
        k = 0
        for i, shape, c0, n in layout:
            if i >= n_state:
                break
            path = paths[i]
            family = next((v for key, v in names.items() if key in path), None)
            if family is None:
                family = fields[k]
                k += 1
            self._family_channels[family] = list(range(c0, c0 + n))
        scale = np.ones(self._n_control, np.float32)
        for family, factor in self._scale_by_family.items():
            scale[self._family_channels[family]] = 1.0 / float(factor)
        # the original's mask_gradient: tracers not named in state_scales
        # (or with factor <= 1) are frozen. Excluding them via the solver's
        # control_variables is NOT enough: that only resets them after the
        # first step, so their increments still get gradients through it and
        # AdamW moves them by ~lr per element (~50x the cloud-ice rms: the
        # cost then rose instead of falling).
        for family in FAMILIES:
            if family.startswith("tracer_") and self._scale_by_family.get(family, 1.0) <= 1.0001:
                scale[self._family_channels[family]] = 0.0
        self._channel_scale = torch.as_tensor(scale, device=self._device)[:, None, None]
        self._build_functions()

    def _pack(self, model_state) -> jax.Array:
        _, _, layout, scalars, _ = self._layout
        leaves = jax.tree_util.tree_leaves(model_state)
        parts = []
        for i, shape, _, n in layout:
            a = leaves[i].reshape((n, *shape[-2:]))
            parts.append(jnp.pad(a, ((0, 0), (0, SPATIAL[0] - shape[-2]), (0, SPATIAL[1] - shape[-1]))))
        s = jnp.zeros(SPATIAL, jnp.float32).at[0, :len(scalars)].set(
            jnp.stack([leaves[i].astype(jnp.float32) for i in scalars]))
        return jnp.concatenate(parts + [s[None]], axis=0)

    def _unpack(self, packed: jax.Array):
        treedef, n_leaves, layout, scalars, ints = self._layout
        leaves = [None] * n_leaves
        for i, shape, c0, n in layout:
            leaves[i] = packed[c0:c0 + n, :shape[-2], :shape[-1]].reshape(shape)
        for j, i in enumerate(scalars):
            leaves[i] = packed[self._scalar_channel, 0, j]
        for i, a in ints.items():
            leaves[i] = jnp.asarray(a)
        return jax.tree_util.tree_unflatten(treedef, leaves)

    def _build_functions(self) -> None:
        model = self.model

        def advance(forcings, packed):
            return self._pack(model.advance(self._unpack(packed), forcings))

        horizontal = model.model_coords.horizontal
        pa = float(model.from_nondim_units(1.0, "Pa"))

        def decode(forcings, packed):
            model_state = self._unpack(packed)
            out = model.decode(model_state, forcings)
            fields = jnp.stack([out[DECODED[b]] for b in DECODED])  # (5, levels, lon, lat)
            # surface pressure (Pa) from the state's ln ps (decode has none).
            lnps = horizontal.to_nodal(model_state.state.log_surface_pressure)
            return fields, pa * jnp.exp(lnps[0])  # (lon, lat)

        self._pack_jit = jax.jit(self._pack)
        self._encode = jax.jit(model.encode)
        self._advance_fn = jax_bridge.torch_function(advance)
        self._decode_fn = jax_bridge.torch_function(decode)

    # ------------------------------------------------------------------
    # state construction / decoding
    # ------------------------------------------------------------------

    def prepare_initial_state(self, input_state, date: datetime.datetime) -> NeuralGCMState:
        inputs, forcings = input_state
        encode = self._encode if self._layout is not None else self.model.encode
        model_state = encode(inputs, forcings)
        if self._layout is None:
            self._set_layout(model_state)
        tensor = jax_bridge.to_torch(self._pack_jit(model_state)).to(self._device)[None]
        return NeuralGCMState(tensor, date)

    def prime_from_state(self, state: NeuralGCMState) -> NeuralGCMState:
        if self._layout is None:
            self.prepare_initial_state(self.read_inputs(state.date), state.date)
        return state

    def decode_state(self, state: NeuralGCMState, time_index: int = -1,
                     only: "list[str] | None" = None) -> dict[str, torch.Tensor]:
        """'z'/'t'/'q'/'u'/'v' on the 37 pressure levels, surface first,
        (n_levels, n_points) latitude-major; 'z500'; 'sp' (Pa, from the
        state's ln ps -- decode has no surface pressure; used by
        jc_ps_weight); canonical aliases."""
        raw, ps = self._decode_fn(state.state[0], constants=self._forcings(state.date))
        raw = raw[:, torch.as_tensor(self._order, device=raw.device)]  # surface first
        flat = raw.transpose(2, 3).reshape(raw.shape[0], raw.shape[1], -1)
        out = {b: flat[k] for k, b in enumerate(DECODED)}
        out["sp"] = out["surface_pressure"] = ps.T.reshape(-1)
        out["z500"] = out["z"][int(np.argmin(np.abs(self._plevs - 500.0)))]
        for base, alias in (("z", "geopotential"), ("t", "temperature"), ("q", "specific_humidity")):
            out[alias] = out[base]
        return out

    # ------------------------------------------------------------------
    # differentiable rollout
    # ------------------------------------------------------------------

    def advance(self, state: NeuralGCMState, steps: int = 1, use_checkpoint: bool = True,
                latent_increment: "torch.Tensor | None" = None) -> NeuralGCMState:
        """`steps` 1 h steps; `latent_increment` (latent_shape), divided by
        the family's state_scales factor, is added to the state.state channels
        before the first. One bridge crossing per step (recomputed in
        backward, so use_checkpoint has nothing further to do)."""
        del use_checkpoint
        x = state.state[0]
        if latent_increment is not None:
            n = self._n_control
            x = torch.cat([x[:n] + latent_increment * self._channel_scale, x[n:]], 0)
        date = state.date
        for _ in range(steps):
            x = self._advance_fn(x, constants=self._forcings(date))
            date = date + self._timestep
        return NeuralGCMState(x[None], date)
