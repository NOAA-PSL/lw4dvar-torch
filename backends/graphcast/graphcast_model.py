"""
GraphCast (DeepMind's `graphcast` package; GraphCast_small, 1 degree, 13
pressure levels, 6 h step) behind the torch 4D-Var solver via
backends/jax_bridge.py -- the model of lw4dvar-jax's `graphcast` backend
(Bo Huang's port, August 2026), for running it with this solver.

Facts this depends on (graphcast_small.npz's task_config):

- Inputs are TWO times, t - 6 h and t (input_duration 12 h): the 11 target
  variables (5 surface: 2m_temperature, mean_sea_level_pressure, 10 m u/v,
  total_precipitation_6hr; 6 on 13 levels: temperature, geopotential, u, v,
  vertical_velocity, specific_humidity), the 5 forcings (TOA incident solar
  radiation, year/day progress sin/cos -- pure functions of the date,
  computed here with graphcast.data_utils, not read) and 2 static fields
  (geopotential_at_surface, land_sea_mask). It predicts the 11 targets at
  t + 6 h.
- 1 degree regular grid INCLUDING the pole rows (181 x 360), latitudes
  south-to-north in the input files; levels 50..1000 hPa.
- Inputs: lw4dvar-jax's per-time ERA5 netCDFs (`graphcast_input_dir`,
  YYYY-MM-DDTHH.nc, 6-hourly, dims (time, [level], lon, lat)), which also
  give the z500 truth and the orography. Normalization statistics
  (`graphcast_stats_dir`: {mean,stddev,diffs_stddev}_by_level.nc).

Packed torch state, channels-first (1, 2 n_t, 181, 360), n_t = 83: the t
slice of the target variables (task order, levels 50..1000) in channels
[0, n_t), the t - 6 h slice in [n_t, 2 n_t). Latitudes south-to-north as
in the files. Forcings and the static fields are not in the state: each
step gets them, with the parameters, as the bridge's non-differentiated
`constants`.

Latent control: the t slice (increment_at_start: the analysis is valid at
window start), increment = latent x the variable's climatological standard
deviation per level (stddev_by_level -- lw4dvar-jax's normalization of the
control, so learning rates carry over). As in lw4dvar-jax's defaults, the
t - 6 h slice is held at the background (it is not in the latent at all
here) and specific humidity is frozen (`graphcast_frozen`; unconstrained
AdamW steps drove q / T negative and the loss to NaN there). One
difference: AdamW's weight decay pulls the latent toward 0, i.e. toward the
background, not toward the climatological mean as lw4dvar-jax's offset
normalization did.

decode_state needs no model call: the t slice IS the physical state.
Pressure-level families are returned surface first (1000 -> 50 hPa), the
order get_surface_pressure assumes. No surface pressure: ps obs use
`logpinterp` (lw4dvar-jax's operator); jc_ps_weight uses 'msl'.
"""

import datetime
import hashlib
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

import forecast_model
from backends import jax_bridge

import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from graphcast import autoregressive, casting, checkpoint, data_utils, normalization, xarray_jax  # noqa: E402
from graphcast import graphcast as gc  # noqa: E402

STEP = np.timedelta64(6, "h")
# graphcast name -> the solver's short name (pressure-level families first)
SHORT = {"geopotential": "z", "temperature": "t", "specific_humidity": "q",
         "u_component_of_wind": "u", "v_component_of_wind": "v", "vertical_velocity": "w",
         "2m_temperature": "t2m", "mean_sea_level_pressure": "msl",
         "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
         "total_precipitation_6hr": "tp6"}
LONG = {s: v for v, s in SHORT.items()}
STATIC = ("geopotential_at_surface", "land_sea_mask")
DEFAULT_FROZEN = ("specific_humidity",)


def _np_datetime(date: datetime.datetime) -> np.datetime64:
    if date.tzinfo is not None:
        date = date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return np.datetime64(date, "ns")


class GraphCastState:
    def __init__(self, state: torch.Tensor, date: datetime.datetime):
        self.state = state
        self.date = date

    def __copy__(self) -> "GraphCastState":
        return GraphCastState(self.state, self.date)


class GraphCastModel(forecast_model.LatentForecastModel):
    def __init__(self, checkpoint_path: str, stats_dir: str, input_dir: str, device: str = "cuda",
                 frozen: "list[str] | None" = None):
        with open(checkpoint_path, "rb") as f:
            ckpt = checkpoint.load(f, gc.CheckPoint)
        self._device = torch.device(device)
        self._input_dir = input_dir
        self.task_config = tc = ckpt.task_config
        self.model_config = ckpt.model_config
        if pd.Timedelta(tc.input_duration) != pd.Timedelta(2 * STEP):
            raise ValueError(f"expected two input times 6 h apart, task_config.input_duration={tc.input_duration}")
        self._params = jax.device_put(ckpt.params)
        stats = {k: xr.load_dataset(os.path.join(stats_dir, f"{k}_by_level.nc"))
                 for k in ("diffs_stddev", "mean", "stddev")}
        self._stats = stats

        self._levels = np.asarray(tc.pressure_levels, np.int64)            # 50..1000 (task order)
        self._plevs = self._levels[::-1].astype(np.float32)                  # surface first
        # t-slice channel layout: target variables in task order
        self._slices, c = {}, 0
        for v in tc.target_variables:
            if v not in SHORT:
                raise KeyError(f"GraphCast target {v!r} has no short name here")
            n = self._levels.size if v in ("geopotential", "temperature", "specific_humidity",
                                            "u_component_of_wind", "v_component_of_wind",
                                            "vertical_velocity") else 1
            self._slices[v] = (c, n)
            c += n
        self._n_t = c

        # increment scale: stddev_by_level per channel; frozen variables 0
        frozen = set(DEFAULT_FROZEN if frozen is None else frozen)
        unknown = frozen - set(self._slices)
        if unknown:
            raise KeyError(f"graphcast_frozen {sorted(unknown)} not GraphCast targets ({list(self._slices)})")
        scale = np.zeros(self._n_t, np.float32)
        for v, (c0, n) in self._slices.items():
            if v in frozen:
                continue
            s = stats["stddev"][v]
            s = s.sel(level=list(self._levels)).values if n > 1 else np.atleast_1d(s.values)
            scale[c0:c0 + n] = s
        self._frozen = sorted(frozen)
        self._channel_scale = torch.as_tensor(scale, device=self._device)[:, None, None]

        # grid (from the first input file found at use; fixed 1 degree here)
        self._lat1d = np.arange(-90.0, 90.01, 1.0)
        self._lon1d = np.arange(0.0, 360.0, 1.0)
        nlat, nlon = self._lat1d.size, self._lon1d.size
        self._shape = (nlat, nlon)
        self._lats = np.repeat(self._lat1d, nlon)
        self._lons = np.tile(self._lon1d, nlat)
        self._levels_by_base = {SHORT[v]: [(int(p), None) for p in self._plevs]
                                for v, (_, n) in self._slices.items() if n > 1}
        self._single_by_base = {SHORT[v]: None for v, (_, n) in self._slices.items() if n == 1}
        self._single_by_base["z500"] = None

        self._files = {}
        self._forcing_cache = {}
        self._forcing_dims = None
        self._static = None
        h = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        self._ckpt_id = f"graphcast:sha256:{h.hexdigest()}"
        self._build_functions()

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> datetime.timedelta:
        return datetime.timedelta(hours=6)

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
        return (self._n_t, *self._shape)

    @property
    def diagnostic_columns(self) -> list[int]:
        """The t - 6 h slice: never reset to the background by the control
        mask (after a step it holds the previous step's t, analysis
        included)."""
        return list(range(self._n_t, 2 * self._n_t))

    @property
    def state_layout(self) -> str:
        return "channels_first"

    def resolve_columns(self, name: str) -> list[int]:
        """t-slice channels of a target variable (graphcast or short name)."""
        v = LONG.get(name, name)
        if v in self._slices:
            c0, n = self._slices[v]
            return list(range(c0, c0 + n))
        raise KeyError(f"{name!r} is not a GraphCast target variable ({list(self._slices)} or {list(LONG)})")

    def is_known_variable(self, name: str) -> bool:
        return name in LONG or name in self._slices or name == "z500"

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        if base not in self._levels_by_base:
            raise KeyError(f"{base!r} is not a pressure-level family")
        return self._plevs.copy()

    def wrap_state(self, state: torch.Tensor, date: datetime.datetime) -> GraphCastState:
        return GraphCastState(state, date)

    # the latent is the t slice of the state: the analysis is valid at
    # window start
    increment_at_start = True

    def apply_increment(self, state: GraphCastState, latent_increment: torch.Tensor) -> GraphCastState:
        """t slice + increment x stddev_by_level (frozen variables 0); the
        t - 6 h slice unchanged."""
        x = state.state[0]
        n = self._n_t
        x = torch.cat([x[:n] + latent_increment * self._channel_scale, x[n:]], 0)
        return GraphCastState(x[None], state.date)

    def checkpoint_id(self) -> str:
        return self._ckpt_id

    def _prime_noise(self) -> None:
        """No-op: deterministic."""

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _file(self, t: np.datetime64) -> xr.Dataset:
        key = t.astype("datetime64[h]")
        if key not in self._files:
            if len(self._files) > 64:
                self._files.clear()
            path = os.path.join(self._input_dir, str(key) + ".nc")
            self._files[key] = xr.open_dataset(path).load()
        return self._files[key]

    def _read_slice(self, t: np.datetime64) -> np.ndarray:
        """(n_t, lat, lon) target variables at `t`, task layout."""
        ds = self._file(t)
        if not (np.allclose(ds["lat"].values, self._lat1d) and np.allclose(ds["lon"].values, self._lon1d)):
            raise ValueError(f"{t}: input grid is not the 1-degree (-90..90, 0..359) grid GraphCast_small uses")
        if self._static is None:
            self._static = {v: jnp.asarray(ds[v].transpose("lat", "lon").values, jnp.float32) for v in STATIC}
        parts = []
        for v, (_, n) in self._slices.items():
            da = ds[v].isel(time=0)
            if n > 1:
                parts.append(da.sel(level=list(self._levels)).transpose("level", "lat", "lon").values)
            else:
                parts.append(da.transpose("lat", "lon").values[None])
        return np.concatenate(parts, 0).astype(np.float32)

    def read_inputs(self, date: datetime.datetime):
        """{'t': (n_t, lat, lon), 'tm1': ...} at `date` and `date` - 6 h."""
        t = _np_datetime(date)
        return {"t": self._read_slice(t), "tm1": self._read_slice(t - STEP)}

    def read_verif(self, date: datetime.datetime) -> dict[str, torch.Tensor]:
        """ERA5 z500 and surface geopotential (m^2/s^2) at `date`, flattened
        latitude-major (south-to-north), from the per-time input file."""
        ds = self._file(_np_datetime(date))
        to = lambda a: torch.as_tensor(np.asarray(a.transpose("lat", "lon").values).reshape(-1),
                                       dtype=torch.float32, device=self._device)
        return {"z500": to(ds["geopotential"].isel(time=0).sel(level=500)),
                "geopotential_at_surface": to(ds["geopotential_at_surface"])}

    def _forcings(self, date: datetime.datetime) -> dict:
        """The 5 forcings at (t - 6 h, t, t + 6 h), t = `date`, from
        graphcast.data_utils (what lw4dvar-jax's precompute_forcings
        computes): {name: jnp (3, *spatial)}; the spatial dims per variable
        (year progress none, day progress lon, TISR lat x lon) are fixed."""
        t = _np_datetime(date)
        if t not in self._forcing_cache:
            if len(self._forcing_cache) > 256:
                self._forcing_cache.clear()
            times = np.array([t - STEP, t, t + STEP])
            ds = xr.Dataset(coords={"lat": self._lat1d, "lon": self._lon1d, "batch": [0],
                                    "time": np.arange(3), "datetime": (("batch", "time"), times[None])})
            data_utils.add_derived_vars(ds)
            data_utils.add_tisr_var(ds)
            out, dims = {}, {}
            for v in self.task_config.forcing_variables:
                da = ds[v].isel(batch=0)
                spatial = tuple(d for d in ("lat", "lon") if d in da.dims)
                dims[v] = spatial
                out[v] = jnp.asarray(da.transpose("time", *spatial).values, jnp.float32)
            if self._forcing_dims is None:
                self._forcing_dims = dims
            self._forcing_cache[t] = out
        return self._forcing_cache[t]

    # ------------------------------------------------------------------
    # the model step
    # ------------------------------------------------------------------

    def _build_functions(self) -> None:
        tc, model_config, stats = self.task_config, self.model_config, self._stats

        @hk.transform_with_state
        def run_forward(inputs, targets_template, forcings):
            predictor = gc.GraphCast(model_config, tc)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor, diffs_stddev_by_level=stats["diffs_stddev"],
                mean_by_level=stats["mean"], stddev_by_level=stats["stddev"])
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        lat, lon, levels = self._lat1d, self._lon1d, self._levels
        step_ns = STEP.astype("timedelta64[ns]")
        in_coords = {"lat": lat, "lon": lon, "level": levels, "batch": [0],
                     "time": np.array([-step_ns, np.timedelta64(0, "ns")])}
        out_coords = dict(in_coords, time=np.array([step_ns]))
        slices = self._slices
        n_t = self._n_t

        def dims_of(n):
            return ("batch", "time", "level", "lat", "lon") if n > 1 else ("batch", "time", "lat", "lon")

        def advance(constants, packed):
            params, static, forc = constants
            fdims = self._forcing_dims
            t_blk, tm1_blk = packed[:n_t], packed[n_t:]
            data = {}
            for v, (c0, n) in slices.items():
                pair = jnp.stack([tm1_blk[c0:c0 + n], t_blk[c0:c0 + n]])        # (2, n, lat, lon)
                data[v] = (dims_of(n), (pair if n > 1 else pair[:, 0])[None])
            for v, a in forc.items():
                data[v] = (("batch", "time", *fdims[v]), a[:2][None])
            for v, a in static.items():
                data[v] = (("lat", "lon"), a)
            inputs = xarray_jax.Dataset(data, coords=in_coords)
            forcings = xarray_jax.Dataset(
                {v: (("batch", "time", *fdims[v]), a[2:3][None]) for v, a in forc.items()}, coords=out_coords)
            template = xarray_jax.Dataset(
                {v: (dims_of(n), jnp.zeros((1, 1, *((levels.size,) if n > 1 else ()), lat.size, lon.size), jnp.float32))
                 for v, (_, n) in slices.items()}, coords=out_coords)
            pred, _ = run_forward.apply(params, {}, jax.random.PRNGKey(0),
                                        inputs=inputs, targets_template=template, forcings=forcings)
            new = []
            for v, (_, n) in slices.items():
                a = xarray_jax.unwrap_data(pred[v].transpose(*dims_of(n)), require_jax=True)[0, 0]
                new.append(a if n > 1 else a[None])
            return jnp.concatenate(new + [t_blk], 0).astype(packed.dtype)

        self._advance_fn = jax_bridge.torch_function(advance)

    def _constants(self, date: datetime.datetime):
        if self._static is None:
            self._read_slice(_np_datetime(date))
        return (self._params, self._static, self._forcings(date))

    # ------------------------------------------------------------------
    # state construction / decoding
    # ------------------------------------------------------------------

    def prepare_initial_state(self, input_state, date: datetime.datetime) -> GraphCastState:
        x = np.concatenate([input_state["t"], input_state["tm1"]], 0)
        return GraphCastState(torch.as_tensor(x, device=self._device)[None], date)

    def prime_from_state(self, state: GraphCastState) -> GraphCastState:
        if self._static is None:
            self._read_slice(_np_datetime(state.date))
        return state

    def decode_state(self, state: GraphCastState, time_index: int = -1,
                     only: "list[str] | None" = None) -> dict[str, torch.Tensor]:
        """The t slice: 'z'/'t'/'q'/'u'/'v'/'w' on the 13 levels, surface
        first, (13, n_points) latitude-major; 't2m', 'msl', 'u10', 'v10',
        'tp6' (n_points,); 'z500'; canonical aliases."""
        x = state.state[0, :self._n_t].reshape(self._n_t, -1)
        out = {}
        for v, (c0, n) in self._slices.items():
            f = x[c0:c0 + n]
            out[SHORT[v]] = f.flip(0) if n > 1 else f[0]
        out["z500"] = out["z"][int(np.argmin(np.abs(self._plevs - 500.0)))]
        for base, alias in (("z", "geopotential"), ("t", "temperature"), ("q", "specific_humidity"),
                            ("msl", "mean_sea_level_pressure")):
            out[alias] = out[base]
        return out

    # ------------------------------------------------------------------
    # differentiable rollout
    # ------------------------------------------------------------------

    def advance(self, state: GraphCastState, steps: int = 1, use_checkpoint: bool = True,
                latent_increment: "torch.Tensor | None" = None) -> GraphCastState:
        """`steps` 6 h steps; `latent_increment` (latent_shape) x
        stddev_by_level is added to the t slice before the first. One bridge
        crossing per step (recomputed in backward)."""
        del use_checkpoint
        if latent_increment is not None:
            state = self.apply_increment(state, latent_increment)
        x = state.state[0]
        date = state.date
        for _ in range(steps):
            x = self._advance_fn(x, constants=self._constants(date))
            date = date + self.timestep
        return GraphCastState(x[None], date)
