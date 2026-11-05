"""
Differentiable wrapper for the 37-level AIFS checkpoints A1 (top 1 hPa) and A2
(top 10 hPa), see notes/integrate_aifs37_models.md.

These checkpoints need the `anemoi-dev` env (anemoi-models 0.19 fork with
`GlobalDryMassFixer`, anemoi-inference 0.12), whose APIs differ from the pins
`aifs_model.AIFSModel` was written against:

- the model is multi-dataset: `forward()` takes and returns `{dataset: tensor}`,
  pre/post processors are `ModuleDict`s keyed by dataset, and the output is
  `(batch, time, ensemble, grid, vars)`;
- `encoder`/`decoder` are `ModuleDict`s with graph providers, so the old
  hand-unrolled encoder -> processor -> decoder path no longer applies. The
  latent increment is instead added to the latent aggregator's output by a
  forward hook, which is exactly where the old code added it (before the
  processor and its latent skip);
- the runner keeps per-dataset `TensorHandler`s instead of the old runner-level
  tensor methods.

Initial conditions and ERA5 truth come from the training zarrs (2015 is inside
1979-2023), joined the way training joined them (`ERA5Zarr`): Bo's v8 zarr for
the surface/forcing fields, the 37-level zarr for pressure levels and the new
surface fields, and the stl1 zarr. Same O96 grid and order as the checkpoint, no
GRIB, no network.

The time-varying loaded forcings (cvh, cvl, lai_hv, lai_lv, slt, fsr) are read
from the zarr at every step; the computed ones (insolation, julian day, local
time) come from anemoi-inference's providers, as before.

Everything else (decode_state, resolve_columns, pressure_levels, advance with
per-step checkpointing) is inherited from AIFSModel.
"""

import datetime
from typing import Optional, Union

import numpy as np
import torch

# aifs_model imports get_shard_shapes (anemoi-models 0.9) or get_shape_shards
# (0.5); 0.19 renamed it get_shard_sizes. Alias it before importing aifs_model
# so that file stays untouched for the pinned envs.
import anemoi.models.distributed.shapes as _shapes
if not hasattr(_shapes, "get_shard_shapes"):
    _shapes.get_shard_shapes = _shapes.get_shard_sizes

from anemoi.inference.config.run import RunConfiguration  # noqa: E402
from anemoi.inference.runners import create_runner  # noqa: E402

import aifs_model  # noqa: E402

ZARR_DIR = "/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/"
ZARR_V8 = "/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/data/training/era5-o96-1979-2023-6h-v8.zarr"
ZARR_37 = ZARR_DIR + "era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr"
ZARR_STL1 = ZARR_DIR + "era5-o96-stl1-1979-01-01T00-2023-12-31T18.zarr"
# variables taken from the v8 zarr in the A1/A2 training join (training.yaml)
V8_SELECT = ("10u", "10v", "2d", "2t", "cos_julian_day", "cos_latitude", "cos_local_time", "cos_longitude",
             "cp", "insolation", "lsm", "msl", "sdor", "sin_julian_day", "sin_latitude", "sin_local_time",
             "sin_longitude", "skt", "slor", "sp", "tcw", "tp", "z")


class ERA5Zarr:
    """ERA5 on the O96 grid from the A1/A2 training zarrs, one date at a time:
    `fields(date)` -> {variable: (n_points,) float32}. Later zarrs in the join
    do not override the v8 selection (the training join has no overlaps)."""

    def __init__(self, cache_size: int = 64):
        import zarr
        self._src = []
        for path, select in ((ZARR_V8, V8_SELECT), (ZARR_37, None), (ZARR_STL1, None)):
            z = zarr.open(path, "r")
            names = list(z.attrs["variables"])
            keep = [i for i, n in enumerate(names) if (n in select if select else n not in V8_SELECT)]
            start = datetime.datetime.fromisoformat(str(z["dates"][0]).replace("T", " ")[:19])
            self._src.append((z["data"], [names[i] for i in keep], np.array(keep), start))
        self.latitudes = np.asarray(zarr.open(ZARR_V8, "r")["latitudes"][:])
        self.longitudes = np.asarray(zarr.open(ZARR_V8, "r")["longitudes"][:])
        self._cache: dict = {}
        self._cache_size = cache_size

    def fields(self, date: datetime.datetime) -> dict:
        date = date.replace(tzinfo=None)
        if date not in self._cache:
            out = {}
            for data, names, keep, start in self._src:
                i, r = divmod((date - start).total_seconds(), 6 * 3600)
                if r or i < 0 or i >= data.shape[0]:
                    raise ValueError(f"{date} is not a 6-hourly date in the training zarr")
                block = np.asarray(data[int(i), :, 0, :])  # one chunk: (n_vars, n_points)
                out.update({n: block[k] for n, k in zip(names, keep)})
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[date] = out
        return self._cache[date]


class AIFS37Model(aifs_model.AIFSModel):
    """AIFSModel for the anemoi-models 0.19 / anemoi-inference 0.12 checkpoints
    (single dataset `data`). Same constructor arguments as AIFSModel."""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda",
                 autocast_dtype: Optional[Union[str, torch.dtype]] = None,
                 compile_wrapper: bool = False, attention: Optional[str] = None):
        if compile_wrapper:
            raise NotImplementedError("compile_wrapper is not implemented for AIFS37Model")
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        config = RunConfiguration.load(config_path, [f"device={device}", f"checkpoint={checkpoint_path}"])
        impls = {"block_sliding_window": "block", "flex_sliding_window": "flex"}
        if attention is not None and attention not in impls:
            raise ValueError(f"attention={attention!r}: expected None or one of {sorted(impls)}")
        if attention is not None:
            import aifs_attention
            aifs_attention.install_flash_attn_stub()
        self.runner = create_runner(config)
        if len(self.runner.dataset_names) != 1:
            raise NotImplementedError(f"multi-dataset checkpoint: {self.runner.dataset_names}")
        self.dataset = self.runner.dataset_names[0]
        self.handler = self.runner.tensor_handlers[self.dataset]
        md = self.handler.metadata
        self.metadata = md
        if attention is not None:
            n_swapped = aifs_attention.use_sliding_window_attention(self.runner.model, impls[attention])
            print(f"attention: {attention} in {n_swapped} attention layers")
        self.runner.model.eval()
        for p in self.runner.model.parameters():  # gradients only w.r.t. our controls
            p.requires_grad_(False)

        self.checkpoint = self.runner.checkpoint
        self._device = torch.device(self.runner.device)
        if autocast_dtype is None:
            self._autocast_dtype = self.runner.autocast
        elif isinstance(autocast_dtype, str):
            self._autocast_dtype = getattr(torch, autocast_dtype)
        else:
            self._autocast_dtype = autocast_dtype
        self.interface = self.runner.model
        m = self.interface.model
        self.multi_step = md.multi_step_input
        if m.n_step_output != 1 or md.multi_step_output != 1:
            raise NotImplementedError(f"n_step_output={m.n_step_output}: only single-step output is supported")
        self._timestep: datetime.timedelta = md.timestep

        self.var_to_idx = md.variable_to_input_tensor_index
        self.idx_to_var = md.output_tensor_index_to_variable
        self._lats = np.asarray(md.latitudes)
        self._lons = np.asarray(md.longitudes)
        self.pmask_in = torch.as_tensor(md.prognostic_input_mask, device=self.device, dtype=torch.long)
        self.pmask_out = torch.as_tensor(md.prognostic_output_mask, device=self.device, dtype=torch.long)
        self._reset = np.zeros(md.number_of_input_features, dtype=bool)
        for name, i in self.var_to_idx.items():
            if md.typed_variables[name].is_constant_in_time:
                self._reset[i] = True

        # same grouping as AIFSModel (see its comments)
        self._levels_by_base: dict[str, list[tuple[int, int]]] = {}
        self._single_by_base: dict[str, int] = {}
        for name, i in self.var_to_idx.items():
            base, _, lev = name.rpartition("_")
            if base and lev.isdigit():
                self._levels_by_base.setdefault(base, []).append((int(lev), i))
            elif name == "z":
                self._single_by_base["geopotential_at_surface"] = i
            else:
                self._single_by_base[name] = i
        for base in self._levels_by_base:
            self._levels_by_base[base].sort(key=lambda t: t[0], reverse=True)

        # forcings refreshed every step: computed ones from anemoi's providers,
        # loaded (coupled) ones from the zarr
        computed = md.select_variables_and_masks(include=["computed"], exclude=["constant"])[0]
        self._dyn_computed = [p for p in self.handler.dynamic_forcings_providers
                              if set(p.variables) <= set(computed)]
        self._dyn_loaded = [v for p in self.handler.dynamic_forcings_providers
                            if p not in self._dyn_computed for v in p.variables]
        self._dyn_loaded_idx = torch.as_tensor([self.var_to_idx[v] for v in self._dyn_loaded],
                                               device=self.device, dtype=torch.long)
        self._loaded = [v for v, i in self.var_to_idx.items() if v not in
                        md.select_variables_and_masks(include=["computed"])[0]]

        self.era5 = ERA5Zarr()
        if not (np.allclose(self.era5.latitudes, self._lats) and np.allclose(self.era5.longitudes, self._lons)):
            raise ValueError("training-zarr grid differs from the checkpoint grid")

        self._latent_inc = None
        m.latent_aggregator.register_forward_hook(self._add_latent_increment)

    # ------------------------------------------------------------------
    # ERA5 from the training zarr (replaces aifs_ic.py for these checkpoints)
    # ------------------------------------------------------------------

    def read_single_date_fields(self, date: datetime.datetime) -> dict:
        """{variable: (n_points,)} ERA5 at `date` (inputs, diagnostics and
        anything else in the zarrs, e.g. w and the upper q levels)."""
        return self.era5.fields(date)

    def build_input_state(self, date: datetime.datetime) -> dict:
        """anemoi State at `date` with the lagged levels [date - 6 h, date] for
        every non-computed input variable."""
        dates = [date + h for h in self.metadata.lagged]
        per = [self.era5.fields(d) for d in dates]
        fields = {v: np.stack([p[v] for p in per]) for v in self._loaded}
        return {"date": date, "latitudes": self._lats, "longitudes": self._lons, "fields": fields}

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> aifs_model.AIFSState:
        tensor_np = self.handler.prepare_input_tensor(input_state)  # (multi_step, n_vars, n_points)
        tensor_np = np.swapaxes(tensor_np, -2, -1)[np.newaxis, ...]
        state = torch.from_numpy(np.ascontiguousarray(tensor_np, dtype=np.float32)).to(self.device)
        return aifs_model.AIFSState(state, date)

    def prime_from_state(self, aifs_state: aifs_model.AIFSState) -> aifs_model.AIFSState:
        """Nothing to rebuild: `_advance_one_step` uses no runner bookkeeping."""
        return aifs_model.AIFSState(aifs_state.state.to(self.device), aifs_state.date)

    @property
    def latent_shape(self) -> tuple[int, int]:
        m = self.interface.model
        return (int(m.node_attributes.num_nodes[m._graph_name_hidden]), int(m.processor.num_channels))

    # ------------------------------------------------------------------
    # Differentiable rollout
    # ------------------------------------------------------------------

    def _add_latent_increment(self, module, args, output):
        return output if self._latent_inc is None else output + self._latent_inc

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """pre_processors -> forward -> post_processors without predict_step's
        no_grad; (1, multi_step, n_points, n_vars) -> (1, n_points, n_out)."""
        ds = self.dataset
        xin = x[:, 0: self.multi_step, None, ...]  # add ensemble dim
        xin = self.interface.pre_processors[ds](xin, in_place=False)
        with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype):
            y = self.interface.model.forward({ds: xin}, model_comm_group=None, grid_shard_sizes=None)[ds]
        y = self.interface.post_processors[ds](y.float(), in_place=False)
        return y[:, 0, 0]  # drop time (n_step_output 1) and ensemble

    def _predict_step_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def _predict_step_with_grad_latent(self, x: torch.Tensor, latent_increment: torch.Tensor) -> torch.Tensor:
        # Set and cleared inside the (checkpointed) step, so the recompute in
        # backward sees the same increment.
        self._latent_inc = latent_increment
        try:
            return self._forward(x)
        finally:
            self._latent_inc = None

    def _advance_one_step(
        self, state: torch.Tensor, date: datetime.datetime, latent_increment: torch.Tensor = None
    ) -> torch.Tensor:
        if latent_increment is None:
            y_pred = self._predict_step_with_grad(state)
        else:
            y_pred = self._predict_step_with_grad_latent(state, latent_increment)

        # roll the two input levels (multi_step 2, advance_map inin [(1, 0)],
        # outin [(0, 1)]) without in-place writes into the autograd input
        new_last = state[:, -1].clone()
        new_last[:, :, self.pmask_in] = y_pred[:, :, self.pmask_out]
        fstate = {"date": date, "latitudes": self.lats, "longitudes": self.lons, "fields": {}}
        for src in self._dyn_computed:
            arr = src.load_forcings_array([date], fstate)  # (variables, dates, values)
            new_last[0, :, torch.as_tensor(src.mask, device=self.device, dtype=torch.long)] = torch.as_tensor(
                np.ascontiguousarray(arr[:, 0].T), dtype=new_last.dtype, device=self.device)
        if len(self._dyn_loaded):
            f = self.era5.fields(date)
            new_last[0, :, self._dyn_loaded_idx] = torch.as_tensor(
                np.stack([f[v] for v in self._dyn_loaded], axis=-1), dtype=new_last.dtype, device=self.device)
        return torch.cat([state[:, 1:], new_last[:, None]], dim=1)
