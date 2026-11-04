"""
NeuralGCM-style JAX model (ngcm-train's TL63 semi-Lagrangian hybrid model)
behind the torch 4D-Var solver, via backends/jax_bridge.py.

Same forecast_model.LatentForecastModel contract as the torch backends. The
model is built and restored with ngcm-train's own code
(`ngcm_train.inference.load_model`, from the run's checkpoint metadata) and run
through terrax's `InferenceModel` (assimilate / advance / observe). Facts this
depends on, from probing a restored r72 checkpoint
(backends/ngcm/probe_ngcm_state.py):

- State (terrax pytree after `assimilate`): modal vorticity, divergence,
  temperature (32, 128, 65) and log_surface_pressure (128, 65); NODAL
  specific_humidity and specific_total_condensate (32, 128, 64) (ngcm-train
  keeps tracers nodal); a jax_datetime `time` (two int32 leaves). 6 h step.
- Forcings: sea_surface_temperature and sea_ice_cover from the training
  store's surface.zarr, read per step as a 2-entry [t, t + 6 h] window.
- Nodal grid: Gaussian 128 x 64 (longitude, latitude), latitudes SOUTH-TO-
  NORTH; 32 hybrid layers, interfaces a + b * ps with a in hPa.

Packed torch state, channels-first `(1, 161, 128, 65)`: vorticity (32),
divergence (32), temperature (32), log_surface_pressure (1),
specific_humidity (32), specific_total_condensate (32), the nodal tracers
zero-padded from 64 to 65 latitudes. The time is not packed: it is rebuilt
from `ModelState.date` at every call.

Latent: the packed modal/nodal state itself -- `latent_increment` (shape
`latent_shape` = (161, 128, 65)) is added to the state before the first
step, divided by a per-family factor (`state_scales`, {family: factor}, the
lw4dvar-jax NeuralGCM convention: the optimizer works on factor * state).
Default (DEFAULT_STATE_SCALES): the original NeuralGCM experiments' setup
in this model's units -- vorticity/divergence 1 / (2 Omega) ~ 6857 (NeuralGCM's
nondimensional time unit), temperature and ln ps 1, tracers 1e5. The
original runs did not optimize condensate (a tracer not named in
state_scales); leave it out of control_variables to match.
This is a classic spectral-space 4D-Var control, not a network-internal
latent.

Decoding (differentiable, through the bridge): `observe` gives nodal
hybrid-layer T, q, condensate, u, v and surface pressure; geopotential and
the pressure-level families come from backends/hybrid_levels.py
(hydrostatic, 32 layers) in torch. Native hybrid fields are exposed under
names distinct from the solver's canonical pressure-level aliases
('air_temperature', 'hybrid_specific_humidity', ...).
"""

import datetime
import os
import sys

import numpy as np
import torch
import xarray as xr

import forecast_model
from backends import hybrid_levels
from backends import jax_bridge

import coordax as cx  # noqa: E402  (JAX side; ngcm env only)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax_datetime as jdt  # noqa: E402

GRAV = hybrid_levels.GRAV

MODAL_KEYS = ("vorticity", "divergence", "temperature")
LNPS_KEY = "log_surface_pressure"
TRACER_KEYS = ("specific_humidity", "specific_total_condensate")
# observe() queries on hybrid layers, in decode order.
LAYER_QUERIES = ("temperature", "specific_humidity", "specific_total_condensate",
                 "u_component_of_wind", "v_component_of_wind")
# packed-state channel family names (control_variables) -> state keys.
CHANNEL_FAMILIES = {
    "vorticity": "vorticity",
    "divergence": "divergence",
    "air_temperature": "temperature",
    "log_surface_pressure": LNPS_KEY,
    "hybrid_specific_humidity": "specific_humidity",
    "specific_total_condensate": "specific_total_condensate",
}
# decoded native hybrid-layer families (nodal) -> observe() names.
LAYER_FAMILIES = {
    "air_temperature": "temperature",
    "hybrid_specific_humidity": "specific_humidity",
    "specific_total_condensate": "specific_total_condensate",
    "eastward_wind": "u_component_of_wind",
    "northward_wind": "v_component_of_wind",
}
DERIVED = ("z", "t", "q", "u", "v")
LOWEST_LEVEL_KEYS = ("t_lowest", "q_lowest", "p_lowest")


class NGCMState:
    def __init__(self, state: torch.Tensor, date: datetime.datetime):
        self.state = state
        self.date = date

    def __copy__(self) -> "NGCMState":
        return NGCMState(self.state, self.date)


# default `state_scales` (the optimizer works on factor * state): the
# original NeuralGCM experiments' setup (lw4dvar-jax / long-window-4dvar-psobs,
# state_scales {specific_humidity: 1e5}) translated to this model's units.
# NeuralGCM's state is nondimensionalized with dinosaur's DEFAULT_SCALE
# (time unit 1 / (2 Omega)), ngcm-train's with terrax's SI_SCALE, so modal
# vorticity and divergence (s^-1 here) get the factor 1 / (2 Omega) to match;
# temperature (K), ln ps increments and humidity are the same in both.
OMEGA = 7.292e-5  # s^-1, dinosaur.scales.ANGULAR_VELOCITY
DEFAULT_STATE_SCALES = {
    "vorticity": 1.0 / (2.0 * OMEGA),
    "divergence": 1.0 / (2.0 * OMEGA),
    "hybrid_specific_humidity": 1.0e5,
    "specific_total_condensate": 1.0e5,
}


def _import_ngcm_train(path: str):
    """ngcm-train's inference (build/restore from a run's checkpoint
    metadata, the same code its own scoring uses) and model modules."""
    if path not in sys.path:
        sys.path.insert(0, path)
    from ngcm_train import inference  # noqa: E402
    from ngcm_train import model as model_lib  # noqa: E402
    return inference, model_lib


def _np_datetime(date: datetime.datetime) -> np.datetime64:
    if date.tzinfo is not None:
        date = date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return np.datetime64(date, "s")


class NGCMModel(forecast_model.LatentForecastModel):
    def __init__(
        self,
        exp_dir: str,
        step: int,
        data_dir: str,
        ngcm_train_path: str,
        device: str = "cuda",
        use_ema: bool = True,
        plevs=hybrid_levels.DEFAULT_PLEVS,
        state_scales: "dict[str, float] | None" = None,
    ):
        from terrax.core import api, coordinates, xarray_utils  # noqa: E402

        self._coordinates, self._xarray_utils = coordinates, xarray_utils
        inference, model_lib = _import_ngcm_train(ngcm_train_path)
        self._model_lib = model_lib
        self._device = torch.device(device)
        self._hybrid = xr.open_zarr(os.path.join(data_dir, "hybrid.zarr"))
        self._surface = xr.open_zarr(os.path.join(data_dir, "surface.zarr"))
        self._pressure = xr.open_zarr(os.path.join(data_dir, "pressure.zarr"))
        static = xr.open_zarr(os.path.join(data_dir, "static.zarr")).load()
        model, levels = inference.load_model(exp_dir, step, data_dir, use_ema)
        self._tracer_names = tuple(model.tracer_names)
        if self._tracer_names != TRACER_KEYS:
            raise ValueError(f"expected tracers {TRACER_KEYS}, got {self._tracer_names}")
        self._timestep = datetime.timedelta(seconds=int(model.timestep / np.timedelta64(1, "s")))
        self._grid = model.model_ylm_map.lon_lat_grid
        self._dynamic_spec = model.dynamic_inputs_spec
        self._inference = api.InferenceModel.from_model_api(model)
        self._levels = levels

        lat1d = self._hybrid.latitude.values
        lon1d = self._hybrid.longitude.values
        self._nlat, self._nlon = lat1d.size, lon1d.size
        self._lats = np.repeat(lat1d, self._nlon)
        self._lons = np.tile(lon1d, self._nlat)
        self._nlay = len(levels.a_boundaries) - 1
        self._ak = torch.as_tensor(np.asarray(levels.a_boundaries, np.float64) * 100.0,
                                   dtype=torch.float32, device=self._device)  # hPa -> Pa
        self._bk = torch.as_tensor(np.asarray(levels.b_boundaries, np.float64),
                                   dtype=torch.float32, device=self._device)
        self._plevs = np.asarray(plevs, dtype=np.float32)

        # surface geopotential on the model's spectral truncation (what its
        # dycore sees, before any learned correction).
        ylm_map = model.model_ylm_map
        phis = cx.field(np.asarray(static["geopotential_at_surface"].transpose("longitude", "latitude").values,
                                   np.float32), self._grid)
        phis = np.asarray(ylm_map.to_nodal(ylm_map.to_modal(phis)).data)  # (lon, lat)
        self._phi_sfc = torch.as_tensor(phis.T.reshape(-1), device=self._device)

        # channel layout of the packed state.
        n = self._nlay
        self._channels = {}
        c = 0
        for k in MODAL_KEYS:
            self._channels[k] = list(range(c, c + n))
            c += n
        self._channels[LNPS_KEY] = [c]
        c += 1
        for k in TRACER_KEYS:
            self._channels[k] = list(range(c, c + n))
            c += n
        self._n_channels = c
        scales = dict(DEFAULT_STATE_SCALES if state_scales is None else state_scales)
        unknown = set(scales) - set(CHANNEL_FAMILIES)
        if unknown:
            raise KeyError(f"state_scales families {sorted(unknown)} not in {list(CHANNEL_FAMILIES)}")
        inverse = np.ones(c, np.float32)
        for family, factor in scales.items():
            inverse[self._channels[CHANNEL_FAMILIES[family]]] = 1.0 / float(factor)
        self._channel_scale = torch.as_tensor(inverse, device=self._device)[:, None, None]

        self._levels_by_base = {b: [(i, None) for i in range(n)] for b in LAYER_FAMILIES}
        for b in DERIVED:
            self._levels_by_base[b] = [(int(p), None) for p in self._plevs]
        self._single_by_base = {"sp": None, "z500": None, **{k: None for k in LOWEST_LEVEL_KEYS}}

        self._template = None  # (treedef, leaf index by key, time leaf indices), set by the first assimilate
        self._advance_fn = None
        self._decode_fn = None
        self._forcing_cache = {}
        self._assimilate = None

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
        return (self._n_channels, self._nlon, self._nlat + 1)

    @property
    def surface_geopotential(self) -> torch.Tensor:
        """(n_points,) m^2/s^2, the model's T63-truncated orography."""
        return self._phi_sfc

    @property
    def state_layout(self) -> str:
        return "channels_first"

    def resolve_columns(self, name: str) -> list[int]:
        if name in CHANNEL_FAMILIES:
            return list(self._channels[CHANNEL_FAMILIES[name]])
        raise KeyError(f"{name!r} is not a packed NGCM state family (one of {list(CHANNEL_FAMILIES)})")

    def is_known_variable(self, name: str) -> bool:
        return (name in CHANNEL_FAMILIES or name in LAYER_FAMILIES or name in DERIVED
                or name in ("sp", "z500") or name in LOWEST_LEVEL_KEYS)

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        if base not in DERIVED:
            raise KeyError(f"{base!r} is not a derived pressure-level family")
        return self._plevs.copy()

    def wrap_state(self, state: torch.Tensor, date: datetime.datetime) -> NGCMState:
        return NGCMState(state, date)

    def _prime_noise(self) -> None:
        """No-op: deterministic."""

    # ------------------------------------------------------------------
    # JAX side: data, packing, bridged functions
    # ------------------------------------------------------------------

    def _forcings(self, date: datetime.datetime) -> dict:
        """Dynamic inputs for one step from `date`: a 2-entry [t, t + dt]
        window (fixed structure, so the jitted functions never retrace)."""
        t0 = _np_datetime(date)
        if t0 in self._forcing_cache:
            return self._forcing_cache[t0]
        times = np.array([t0, t0 + np.timedelta64(self._timestep)], "datetime64[ns]")
        timedelta = self._coordinates.TimeDelta((times - times[0]).astype("timedelta64[s]"))
        dynamic = {}
        for key, specs in self._dynamic_spec.items():
            window = self._surface[[k for k in specs if k != "time"]].sel(time=times).load()
            dynamic[key] = {k: cx.field(window[k].transpose("time", "longitude", "latitude").values,
                                        timedelta, self._grid) for k in window}
            dynamic[key]["time"] = cx.field(jdt.to_datetime(times), timedelta)
        if len(self._forcing_cache) > 64:
            self._forcing_cache.clear()
        self._forcing_cache[t0] = dynamic
        return dynamic

    def _time_leaves(self, date: datetime.datetime) -> list:
        return jax.tree.leaves(jdt.to_datetime(_np_datetime(date)))

    def _set_template(self, state) -> None:
        leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(state)
        index, time_index = {}, []
        for i, (path, _) in enumerate(leaves_with_path):
            key = jax.tree_util.keystr(path)
            if "['time']" in key:
                time_index.append(i)
                continue
            for name in (*MODAL_KEYS, LNPS_KEY, *TRACER_KEYS):
                if f"['{name}']" in key:
                    index[name] = i
        missing = set((*MODAL_KEYS, LNPS_KEY, *TRACER_KEYS)) - set(index)
        if missing or len(time_index) != 2:
            raise ValueError(f"unexpected NGCM state structure: missing {missing}, {len(time_index)} time leaves")
        leaves = [leaf for _, leaf in leaves_with_path]
        self._template = (treedef, index, time_index, leaves)
        self._build_functions()

    def _pack(self, state) -> jax.Array:
        _, index, _, _ = self._template
        leaves = jax.tree.leaves(state)
        pad = lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, 1)))
        parts = [leaves[index[k]] for k in MODAL_KEYS]
        parts.append(leaves[index[LNPS_KEY]][None])
        parts += [pad(leaves[index[k]]) for k in TRACER_KEYS]
        return jnp.concatenate(parts, axis=0)

    def _unpack(self, packed: jax.Array, time_leaves: list):
        treedef, index, time_index, template = self._template
        leaves = list(template)
        span = lambda k: slice(self._channels[k][0], self._channels[k][-1] + 1)
        for k in MODAL_KEYS:
            leaves[index[k]] = packed[span(k)]
        leaves[index[LNPS_KEY]] = packed[self._channels[LNPS_KEY][0]]
        for k in TRACER_KEYS:
            leaves[index[k]] = packed[span(k), :, :-1]
        for i, leaf in zip(time_index, time_leaves):
            leaves[i] = leaf
        return jax.tree_util.tree_unflatten(treedef, leaves)

    def _build_functions(self) -> None:
        inference = self._inference
        grid, levels = self._grid, self._levels
        model_lib = self._model_lib
        queries = {
            model_lib.HYBRID_KEY: {k: cx.coords.compose(levels, grid) for k in LAYER_QUERIES},
            model_lib.SURFACE_KEY: {"surface_pressure": grid},
        }

        def advance(constants, packed):
            state = self._unpack(packed, constants["time"])
            return self._pack(inference.advance(state, constants["dynamic"]))

        def decode(constants, packed):
            state = self._unpack(packed, constants["time"])
            obs = inference.observe(state, queries, constants["dynamic"])
            layers = [obs[model_lib.HYBRID_KEY][k].data for k in LAYER_QUERIES]
            ps = obs[model_lib.SURFACE_KEY]["surface_pressure"].data
            return jnp.concatenate(layers + [ps[None]], axis=0)  # (5 * nlay + 1, lon, lat)

        self._pack_jit = jax.jit(self._pack)
        self._advance_fn = jax_bridge.torch_function(advance)
        self._decode_fn = jax_bridge.torch_function(decode)

    def _constants(self, date: datetime.datetime) -> dict:
        return {"time": self._time_leaves(date), "dynamic": self._forcings(date)}

    # ------------------------------------------------------------------
    # state construction / decoding
    # ------------------------------------------------------------------

    def read_inputs(self, date: datetime.datetime) -> dict:
        """ERA5 hybrid-layer inputs at `date` from the training store (local
        zarr -- no prefetch step needed, unlike the other backends)."""
        t = _np_datetime(date)
        model_lib, xarray_utils = self._model_lib, self._xarray_utils

        def sel(ds, names):
            fields = xarray_utils.from_xarray(ds[names].sel(time=t).load())
            fields["time"] = cx.field(jdt.to_datetime(t))
            return fields

        return {
            model_lib.HYBRID_KEY: sel(self._hybrid, ["u_component_of_wind", "v_component_of_wind", "temperature",
                                                     *self._tracer_names, "geopotential"]),
            model_lib.SURFACE_KEY: sel(self._surface, ["surface_pressure"]),
        }

    def read_z500(self, date: datetime.datetime) -> torch.Tensor:
        """(n_points,) ERA5 500 hPa geopotential (m^2/s^2) on the model grid,
        from the store's pressure.zarr (verification for the z500 diagnostic)."""
        z = self._pressure["geopotential"].sel(time=_np_datetime(date), pressure=500)
        return torch.as_tensor(z.transpose("latitude", "longitude").values.reshape(-1),
                               dtype=torch.float32, device=self._device)

    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> NGCMState:
        if self._assimilate is None:
            self._assimilate = jax.jit(self._inference.assimilate)
        state = self._assimilate(input_state, self._forcings(date))
        if self._template is None:
            self._set_template(state)
        packed = self._pack_jit(state)
        tensor = jax_bridge.to_torch(packed).to(self._device)[None]
        return NGCMState(tensor, date)

    def prime_from_state(self, state: NGCMState) -> NGCMState:
        if self._template is None:
            # restart path: build the template from the store at the
            # state's date (structure only; the values come from `state`).
            self.prepare_initial_state(self.read_inputs(state.date), state.date)
        return state

    def decode_state(self, state: NGCMState, time_index: int = -1,
                     only: "list[str] | None" = None) -> dict[str, torch.Tensor]:
        """Native hybrid families (nodal, (32, n_points) top -> bottom),
        derived pressure-level 'z'/'t'/'q'/'u'/'v' ((n_plevs, n_points),
        surface first), 'sp' (Pa), 'z500', the lowest-layer
        't_lowest'/'q_lowest'/'p_lowest' (ps_operator='ps_native'), and the
        canonical aliases. `time_index` is ignored (one time level)."""
        raw = self._decode_fn(state.state[0], constants=self._constants(state.date))
        # (fields, lon, lat) -> (fields, lat * lon), latitude-major like lats/lons.
        flat = raw.transpose(1, 2).reshape(raw.shape[0], -1)
        n = self._nlay
        layer = {name: flat[i * n:(i + 1) * n] for i, name in enumerate(LAYER_FAMILIES)}
        ps = flat[-1]
        want = None if only is None else set(only)
        need = lambda *names: want is None or any(x in want for x in names)
        out = {k: v for k, v in layer.items() if need(k)}
        canonical = {"z": "geopotential", "t": "temperature", "q": "specific_humidity"}
        if need(*DERIVED, *canonical.values(), "z500"):
            plev = hybrid_levels.to_pressure_levels(
                ps, layer["air_temperature"], layer["hybrid_specific_humidity"], self._phi_sfc,
                self._ak, self._bk, self._plevs,
                others={"u": layer["eastward_wind"], "v": layer["northward_wind"]},
                want_z=need("z", "geopotential", "z500"))
            out.update(plev)
            for base, alias in canonical.items():
                if base in plev:
                    out[alias] = plev[base]
            if "z" in plev:
                out["z500"] = plev["z"][int(np.argmin(np.abs(self._plevs - 500.0)))]
        out["sp"] = ps
        out["surface_pressure"] = ps
        if need(*LOWEST_LEVEL_KEYS):
            out["t_lowest"] = layer["air_temperature"][-1]
            out["q_lowest"] = layer["hybrid_specific_humidity"][-1]
            out["p_lowest"] = 0.5 * ((self._ak[-2] + self._bk[-2] * ps) + (self._ak[-1] + self._bk[-1] * ps))
        return out

    # ------------------------------------------------------------------
    # differentiable rollout
    # ------------------------------------------------------------------

    def advance(self, state: NGCMState, steps: int = 1, use_checkpoint: bool = True,
                latent_increment: "torch.Tensor | None" = None) -> NGCMState:
        """`steps` 6 h steps; `latent_increment` (latent_shape) is added to the
        packed state, divided by its family's `state_scales` factor, before
        the first. Each step
        crosses the bridge once; the bridge recomputes the step in backward,
        so `use_checkpoint` has nothing further to do (the saved per-step
        input is the ~5 MB packed state)."""
        del use_checkpoint
        x = state.state[0]
        if latent_increment is not None:
            x = x + latent_increment * self._channel_scale
        date = state.date
        for _ in range(steps):
            x = self._advance_fn(x, constants=self._constants(date))
            date = date + self._timestep
        return NGCMState(x[None], date)
