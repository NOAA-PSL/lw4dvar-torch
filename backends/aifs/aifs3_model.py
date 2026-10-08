"""
AIFS backend for the multi-dataset anemoi API (anemoi-models >= 0.19,
anemoi-inference >= 0.12; the `aifs3` conda env) -- e.g. the user-trained
1-degree checkpoint `aifs2-1deg-ic1`. Selected automatically by
long_window_4dvar_utils.get_model() when the active env has that API; configs
still say `model_backend: aifs`.

What changed upstream, and how this class handles it (see
integrate_multiple_backends.md for the investigation):

  - The model is multi-dataset: `forward()` takes and returns
    `{dataset_name: tensor}`, pre/post-processors are per-dataset
    ModuleDicts, and per-dataset latents are merged by `latent_aggregator`.
    This checkpoint has a single dataset ('data').
  - The anemoi-inference tensor bookkeeping (`prepare_input_tensor`,
    `copy_prognostic_fields_to_input_tensor`, `add_dynamic_forcings_to_input_tensor`)
    moved from the Runner to a per-dataset `TensorHandler`, and the per-variable
    metadata from `Checkpoint` to `handler.metadata`.
  - `predict_step` still wraps everything in `torch.no_grad()`, so
    `_predict_step_with_grad` re-implements its (single-GPU, unsharded) body:
    add ensemble dim -> spatial pre-processor (if any) -> normalize ->
    `forward()` -> de-normalize. `forward()` itself is called unchanged.
  - The latent increment is injected by a forward hook on
    `model.latent_aggregator` (output + increment), instead of manually
    unrolling encoder/processor/decoder as AIFSModel does. With one dataset
    the aggregator returns the encoder's hidden-mesh latent unchanged
    (SumAggregator), so this is the same injection point as AIFSModel: the
    processor AND the latent skip both see the incremented latent.

Everything else (decode_state, resolve_columns, pressure_levels, the
AIFSState wrapper, advance()'s checkpointed step loop) is inherited from
AIFSModel unchanged.
"""

import datetime
from typing import Optional, Union

import numpy as np
import torch

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner

from aifs_model import AIFSModel, AIFSState


class AIFS3Model(AIFSModel):
    """AIFSModel for the multi-dataset anemoi API (single-dataset checkpoints)."""

    ic_module = "aifs3_ic"  # IC/verification reader used by long_window_4dvar_utils

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda",
                 autocast_dtype: Optional[Union[str, torch.dtype]] = None,
                 compile_wrapper: bool = False):
        if compile_wrapper:
            raise NotImplementedError("compile_wrapper is not implemented for AIFS3Model")
        if device == "cuda":
            # same reasoning as AIFSModel.__init__
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        config = RunConfiguration.load(config_path, [f"device={device}", f"checkpoint={checkpoint_path}"])
        self.runner = create_runner(config)
        if len(self.runner.dataset_names) != 1:
            raise NotImplementedError(
                f"AIFS3Model supports single-dataset checkpoints only, got {self.runner.dataset_names}"
            )
        self._dataset = self.runner.dataset_names[0]
        self.handler = self.runner.tensor_handlers[self._dataset]
        md = self.handler.metadata
        if self.runner.checkpoint.multi_step_output != 1:
            raise NotImplementedError("AIFS3Model supports one output time level per step only")

        self.runner.model.eval()
        for p in self.runner.model.parameters():  # gradients only w.r.t. the increment
            p.requires_grad_(False)

        self.checkpoint = self.runner.checkpoint
        self._device = torch.device(self.runner.device)
        if autocast_dtype is None:
            self._autocast_dtype = self.runner.autocast
        elif isinstance(autocast_dtype, str):
            self._autocast_dtype = getattr(torch, autocast_dtype)
        else:
            self._autocast_dtype = autocast_dtype
        self.interface = self.runner.model  # AnemoiModelInterface
        self.multi_step = self.interface.n_step_input
        self._timestep: datetime.timedelta = self.checkpoint.timestep
        self._init_variable_index(md)

        # Latent increment injection point (see module docstring). The hook
        # is a no-op unless _pending_increment is set, which only
        # _predict_step_with_grad_latent does (and always clears).
        self._pending_increment = None
        self.interface.model.latent_aggregator.register_forward_hook(self._add_pending_increment)

    def _add_pending_increment(self, module, args, output):
        if self._pending_increment is None:
            return output
        return output + self._pending_increment

    @property
    def latent_shape(self) -> tuple[int, int]:
        """(num_hidden_mesh_nodes, num_channels) of the aggregated latent."""
        m = self.interface.model
        return (int(m.node_attributes.num_nodes[m._graph_name_hidden]), int(m.latent_aggregator.hidden_dim))

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> AIFSState:
        """Packed `(1, multi_step, n_points, n_vars)` tensor from an anemoi
        `State` dict (see aifs3_ic.py). Constant forcings already present in
        `input_state['fields']` (lsm/sdor/slor/z from the ERA5 GRIB) are used
        as-is; computed forcings come from the handler's own providers."""
        input_state = dict(input_state, fields=dict(input_state["fields"]))
        tensor_np = self.handler.prepare_input_tensor(input_state)  # (multi_step, n_vars, n_points)
        tensor_np = np.swapaxes(tensor_np, -2, -1)[np.newaxis, ...]  # (1, multi_step, n_points, n_vars)
        state = torch.from_numpy(np.ascontiguousarray(tensor_np, dtype=np.float32)).to(self.device)
        return AIFSState(state, date)

    # prime_from_state is inherited: it rebuilds a State dict from the packed
    # tensor and calls prepare_initial_state, which re-primes the handler.

    # ------------------------------------------------------------------
    # Differentiable rollout
    # ------------------------------------------------------------------

    def _predict_step_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """`AnemoiModelInterface.predict_step` (anemoi-models 0.19,
        models/base.py) without its `torch.no_grad()`, single GPU. Returns
        `(batch, time, ensemble, n_points, n_vars)`, the same layout
        `predict_step` returns and `TensorHandler` expects."""
        name = self._dataset
        xin = x[:, 0 : self.multi_step, None, ...]  # add ensemble dim
        spatial = getattr(self.interface, "spatial_pre_processors", None)
        if spatial and name in spatial:
            xin, _ = spatial[name](xin, model_comm_group=None, grid_shard_sizes=None)
        xin = self.interface.pre_processors[name](xin, in_place=False)
        with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype):
            y = self.interface.model.forward({name: xin}, model_comm_group=None, grid_shard_sizes=None)[name]
        return self.interface.post_processors[name](y.float(), in_place=False)

    def _predict_step_with_grad_latent(self, x: torch.Tensor, latent_increment: torch.Tensor) -> torch.Tensor:
        self._pending_increment = latent_increment
        try:
            return self._predict_step_with_grad(x)
        finally:
            self._pending_increment = None

    def _advance_one_step(
        self, state: torch.Tensor, date: datetime.datetime, latent_increment: torch.Tensor = None
    ) -> torch.Tensor:
        """One step, mirroring anemoi-inference's own `Runner.forecast` loop:
        copy prognostic outputs into the next input, then refresh dynamic
        (and boundary) forcings at the new valid `date`."""
        if latent_increment is None:
            y_pred = self._predict_step_with_grad(state)
        else:
            y_pred = self._predict_step_with_grad_latent(state, latent_increment)

        check = self._reset.copy()
        # TensorHandler writes into its input in place; clone so the previous
        # state (possibly needed by autograd) is left untouched.
        new_state = self.handler.copy_prognostic_fields_to_input_tensor(state.clone(), y_pred, check)
        forcing_state = {"date": date, "latitudes": self.lats, "longitudes": self.lons, "fields": {}}
        new_state = self.handler.add_dynamic_forcings_to_input_tensor(new_state, forcing_state, [date], check)
        new_state = self.handler.add_boundary_forcings_to_input_tensor(new_state, forcing_state, [date], check)

        if not check.all():
            mapping = {v: k for k, v in self.var_to_idx.items()}
            missing = [mapping[i] for i in range(len(check)) if not check[i]]
            raise ValueError(f"Missing variables in input tensor after step: {sorted(missing)}")
        return new_state
