"""
Differentiable Microsoft Aurora (AuroraV1p5) model wrapper for the 4D-Var
solver.

Architecture notes (from reading the real `microsoft-aurora==2.0.1` source
directly, and confirmed empirically via probe_aurora_latent.py -- not
assumed from general knowledge):

- `AuroraV1p5.forward()` runs `x = self.encoder(...)`, `x = self.backbone(
  ..., patch_res=..., rollout_step=...)`, `pred = self.decoder(x, ...)` --
  the same encoder/processor/decoder split AIFS/FCN3 already have, but with
  two real simplifications neither of those needed a workaround for: there
  is no hardcoded `torch.no_grad()` anywhere in `forward()` (AIFS's
  `predict_step` has one), and there is no stochastic noise state at all
  (FCN3's diffusion-noise re-priming has no Aurora analog -- the
  noise/ensemble variant is a wholly separate class+checkpoint,
  `AuroraV1p5Ensemble`, not used here).
- Control-injection point: a `torch.nn.Module.register_forward_pre_hook`
  on `self._model.decoder`, adding `latent_increment` to its first
  positional argument (the backbone's output). A hook is sufficient here
  specifically because there's no no_grad to bypass and no stateful
  side-effect to worry about splitting around (contrast AIFS's manual
  `_predict_step_with_grad_latent` unroll and FCN3's manual `_pure_forward`
  split) -- confirmed to preserve real end-to-end gradients in
  probe_aurora_latent.py. The hook is registered/used/removed entirely
  inside `_advance_one_step` (never left registered on `self`, and the
  increment is passed to `_advance_one_step` as a genuine function
  argument, never read off `self`) -- necessary for `torch.utils.checkpoint`
  correctness: checkpoint recomputes `_advance_one_step` during backward by
  replaying the exact arguments it was called with, which only works if the
  increment is one of those arguments and not a mutable side-channel that
  could have changed by the time a later step's backward triggers an
  earlier step's recompute.
- Latent shape (backbone output == decoder input), confirmed empirically
  in probe_aurora_latent.py: `(259200, 1024)` -- a flat Swin-transformer
  token sequence (259200 = 180x360 patches x 4 latent_levels, embed_dim
  1024), NOT excluding the batch dim (matching FCN3Model.latent_shape's own
  convention -- broadcasts against the real `(1, 259200, 1024)` tensor).
- Needs two lagged time levels as input (`max_history_size=2`), like AIFS,
  not self-starting like FCN3 -- so the packed state here follows AIFS's
  `(1, multi_step=2, n_points, n_vars)` channels-last convention exactly
  (not FCN3's channels-first `(1, n_vars, H, W)`), so `state_layout`
  reuses the existing 'channels_last' branch in long_window_4dvar_utils.py
  with no new shared-driver code needed.
- Grid: same 0.25deg regular 721x1440 ERA5 grid as FCN3 (confirmed via the
  bundled static-field pickle's implicit shape) -- `fcn3_grid.
  BilinearGridInterpolator` should work unchanged for Aurora's ps-obs
  interpolation once wired into get_grid_interpolator.
- Real native `sp` (surface pressure) field, unlike FCN3 -- unlike FCN3,
  `ps_operator: 'ps'` (AIFS's native-sp forward operator) could in
  principle work for Aurora, though that hasn't been wired up or tested.
- **Real, non-obvious pitfall found while validating this** (see
  probe_aurora_latent.py for the full debugging trail): `AuroraV1p5` runs
  its encoder/backbone/decoder under `torch.autocast(dtype=torch.float16)`
  by default. A loss computed on the model's raw, UNNORMALISED physical-
  unit output (e.g. msl ~1e5 Pa) produces gradients many orders of
  magnitude outside fp16's dynamic range, overflowing to inf/nan during
  the autocast-region backward -- looks exactly like a broken model but
  isn't. Any diagnostic/loss code here must stay scale-aware (compute on
  normalised quantities, or on innovation/oberr-normalised terms the way
  the real ps-obs loss already is) rather than assume raw decoded output
  is safe to square-and-sum directly.
- 26 surface variables (richer than FCN3's 7 or AIFS's set), 7 of which
  are output-only (predicted by the model but never present in real ERA5
  input -- `i10fg, blh, uvb_1h, ssrd_1h, ttr_1h, scaled_tp_1h,
  scaled_sf_1h`). Aurora's own `_pre_encoder_hook` unconditionally
  zero-pads these before every encoder call regardless of what's actually
  stored for them in the packed state, so the roll-forward step in
  `_advance_one_step` can simply store the model's own real predictions
  for them without any special-casing -- they're self-correcting on the
  next step's input.
- `insolation` is a normal (non-output-only) surface variable, but
  Aurora's own `_post_unnorm_hook`/`_update_insolation` unconditionally
  overwrites the model's predicted value with the true recomputed value
  (from `aurora.insolation.insolation()`, using `pred.metadata.time`) after
  every forward call -- so, like the output-only variables, no special
  roll-forward handling is needed here either: `pred.surf_vars['insolation']`
  is already the correct real value by the time `_pack_pred` reads it. Only
  the very first (IC) insolation value, supplied by aurora_ic.py (not yet
  written), needs to be computed correctly from scratch.
"""

import copy
import datetime
import pickle

import numpy as np
import torch

from aurora import AuroraV1p5, Batch, Metadata

import forecast_model

# Canonical, descending (surface-first) level order -- used consistently
# for Metadata.atmos_levels, the packed tensor's level axis, and
# _levels_by_base's column ordering, matching AIFS/FCN3's own convention.
_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
_ATMOS_VARS = ("z", "u", "v", "t", "q")
_SURF_VARS = (
    "2t", "10u", "10v", "msl", "2d", "tcwv", "tcc", "100u", "100v", "sp", "lcc", "mcc",
    "hcc", "skt", "stl1", "swvl1", "ci", "scaled_sd", "i10fg", "blh", "uvb_1h",
    "ssrd_1h", "ttr_1h", "scaled_tp_1h", "scaled_sf_1h", "insolation",
)
_OUTPUT_ONLY_SURF_VARS = (
    "i10fg", "blh", "uvb_1h", "ssrd_1h", "ttr_1h", "scaled_tp_1h", "scaled_sf_1h",
)
# Aurora's own Batch.crop(patch_size=4) drops the LAST latitude row
# (v[..., :-1, :]) whenever H isn't a multiple of patch_size -- ERA5's
# native 721 rows isn't (721 % 4 == 1), so Aurora silently crops the
# South Pole row (lat=-90 exactly) on every real call, and its OWN output
# is therefore only ever 720 rows. Rather than carry a 721-row state
# where row 720 would go stale (never actually predicted, silently
# truncated by Aurora and never written back), this backend's OWN native
# grid is 720 rows, matching exactly what the checkpoint actually
# computes -- confirmed empirically (a first attempt at 721 rows failed
# with a real-vs-cropped shape mismatch when packing `pred` back). This
# needs no changes to fcn3_grid.BilinearGridInterpolator, which derives
# nlat/nlon from whatever `model.lats`/`model.lons` actually contain.
_NLAT, _NLON = 720, 1440

_CHECKPOINT_NAME = "aurora-0.25-v1.5.ckpt"
_STATIC_NAME = "aurora-0.25-v1.5-static.pickle"
_CHECKPOINT_REVISION = "a96afd7ee6d65e3bd2d476f3be798a25a56f2296"


class AuroraState:
    """Thin wrapper whose `.state` attribute *is* the packed
    `(1, 2, n_points, n_vars)` tensor -- same convention as AIFSState (see
    that class's docstring)."""

    __slots__ = ("state", "date")

    def __init__(self, state: torch.Tensor, date: datetime.datetime):
        self.state = state
        self.date = date

    def __copy__(self) -> "AuroraState":
        return AuroraState(self.state, self.date)


class AuroraModel(forecast_model.LatentForecastModel):
    """Wraps a loaded AuroraV1p5 checkpoint for differentiable rollouts.

    `model.wrapper` (the real `AuroraV1p5` instance) is a public,
    undeclared escape hatch, same rationale as `AIFSModel.runner` /
    `FCN3Model.wrapper`.
    """

    def __init__(self, package_root: str, device: str = "cuda"):
        """
        Parameters
        ----------
        package_root : str
            Directory containing `aurora-0.25-v1.5.ckpt` and
            `aurora-0.25-v1.5-static.pickle` -- the Hugging Face cache
            snapshot directory `aurora_prefetch_checkpoint.py` fetches
            into, analogous to FCN3Model's `package_root` (a directory IS
            the package here, unlike AIFS's separate checkpoint_path/
            config_path).
        """
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._device = torch.device(device)

        with open(f"{package_root}/{_STATIC_NAME}", "rb") as f:
            static_raw = pickle.load(f)
        # Crop to 720 rows (drop the South Pole row), matching Aurora's own
        # Batch.crop() exactly -- see _NLAT's comment above.
        self._static_vars = {
            k: torch.from_numpy(v[:-1, :]).to(self._device) for k, v in static_raw.items()
        }

        self.wrapper = AuroraV1p5()
        self.wrapper.load_checkpoint_local(f"{package_root}/{_CHECKPOINT_NAME}")
        self.wrapper = self.wrapper.to(self._device)
        self.wrapper.eval()
        # Gradients only ever w.r.t. our own control increment, never
        # Aurora's pretrained weights -- torch.load doesn't set
        # requires_grad=False on its own. Same lesson AIFSModel/FCN3Model's
        # __init__ document at length.
        for p in self.wrapper.parameters():
            p.requires_grad_(False)

        self._timestep = self.wrapper.timestep

        # linspace over the FULL 721-point axis, then drop the last (South
        # Pole) row -- NOT linspace(90, -90, 720), which would give a
        # different (non-0.25deg-exact) spacing across the same range.
        lat1d = np.linspace(90.0, -90.0, 721, dtype=np.float64)[:-1]
        lon1d = np.linspace(0.0, 360.0, _NLON, endpoint=False, dtype=np.float64)
        # Row-major flatten (lat slowest, lon fastest) -- matches
        # fcn3_grid.py's convention (same 0.25deg grid, minus Aurora's
        # cropped South Pole row), so fcn3_grid.BilinearGridInterpolator is
        # directly reusable (it derives nlat/nlon from the data itself).
        self._lats = np.repeat(lat1d, _NLON)
        self._lons = np.tile(lon1d, _NLAT)
        self._lat1d_t = torch.from_numpy(lat1d).to(self._device)
        self._lon1d_t = torch.from_numpy(lon1d).to(self._device)

        self._levels_by_base: dict[str, list[tuple[int, int]]] = {}
        self._single_by_base: dict[str, int] = {}
        col = 0
        for base in _ATMOS_VARS:
            self._levels_by_base[base] = []
            for lev in _LEVELS:
                self._levels_by_base[base].append((lev, col))
                col += 1
        for name in _SURF_VARS:
            self._single_by_base[name] = col
            col += 1
        self._n_vars = col  # 5*13 + 26 = 91

    # ------------------------------------------------------------------
    # forecast_model.LatentForecastModel properties
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
    def state_layout(self) -> str:
        """See AuroraState/module docstring: packed `(1, 2, n_points,
        n_vars)`, n_vars trailing -- same convention as AIFSModel."""
        return "channels_last"

    @property
    def latent_shape(self) -> tuple[int, int]:
        """`(259200, 1024)` -- confirmed empirically in
        probe_aurora_latent.py, not computed from a config formula (Aurora
        exposes no equivalent of FCN3's config.json-derived embed-dim
        formula; hardcoding the checkpoint-confirmed shape here matches
        how this was actually determined)."""
        return (259200, 1024)

    def resolve_columns(self, name: str) -> list[int]:
        if name in self._levels_by_base:
            return [i for _, i in self._levels_by_base[name]]
        if name in self._single_by_base:
            return [self._single_by_base[name]]
        raise KeyError(f"{name!r} is not a known Aurora variable or family")

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        return np.array(_LEVELS, dtype=np.float32)

    def is_known_variable(self, name: str) -> bool:
        return name in self._levels_by_base or name in self._single_by_base

    # ------------------------------------------------------------------
    # Packing: AuroraState.state (1, 2, n_points, n_vars) <-> aurora.Batch
    # ------------------------------------------------------------------

    def _unpack(self, state: torch.Tensor, date: datetime.datetime) -> Batch:
        """Build a real `aurora.Batch` for the two time levels packed in
        `state`, valid at `date` (the LATEST of the two levels -- the
        older one is implicitly `date - self.timestep`, matching how
        `Metadata.time` records only the single most recent time even
        though `surf_vars`/`atmos_vars` carry both levels' data)."""
        t = state.reshape(1, 2, _NLAT, _NLON, self._n_vars)
        surf_vars = {name: t[..., col] for name, col in self._single_by_base.items()}
        atmos_vars = {
            base: torch.stack([t[..., col] for _, col in cols], dim=2)  # (1, 2, L, H, W)
            for base, cols in self._levels_by_base.items()
        }
        return Batch(
            surf_vars=surf_vars,
            static_vars=self._static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=self._lat1d_t,
                lon=self._lon1d_t,
                time=(date,),
                atmos_levels=_LEVELS,
            ),
        )

    def _pack_pred(self, pred: Batch, old_state: torch.Tensor) -> torch.Tensor:
        """Roll `old_state`'s two time levels forward by one: level 0
        becomes the old level 1, and level 1 becomes `pred` (Aurora's
        one-step-ahead prediction, already carrying a leading size-1 time
        dim -- see `forward()`'s "Insert history dimension in prediction").
        No special-casing needed for output-only vars or `insolation` --
        see module docstring."""
        new_state = old_state.reshape(1, 2, _NLAT, _NLON, self._n_vars).clone()
        new_state[:, 0] = new_state[:, 1]
        for name, col in self._single_by_base.items():
            new_state[:, 1, :, :, col] = pred.surf_vars[name][:, 0]
        for base, cols in self._levels_by_base.items():
            arr = pred.atmos_vars[base][:, 0]  # (1, L, H, W)
            for k, (_, col) in enumerate(cols):
                new_state[:, 1, :, :, col] = arr[:, k]
        return new_state.reshape(1, 2, _NLAT * _NLON, self._n_vars)

    def decode_state(
        self, aurora_state: AuroraState, time_index: int = -1, only: "list[str] | None" = None
    ) -> dict[str, torch.Tensor]:
        """See AIFSModel.decode_state's docstring -- same slicing
        convention, adapted to Aurora's packing. `only` restricts decoding
        to just these base names, same rationale as the other two backends
        (skip the unused-column tensor ops in compute_loss_4dvar's hot
        loop)."""
        tensor = aurora_state.state[:, time_index, :, :][0]  # (n_points, n_vars)
        out: dict[str, torch.Tensor] = {}
        only_set = None if only is None else set(only)
        levels_items = (
            self._levels_by_base.items() if only_set is None
            else ((b, self._levels_by_base[b]) for b in only_set if b in self._levels_by_base)
        )
        for base, levels in levels_items:
            cols = [i for _, i in levels]
            out[base] = tensor[:, cols].transpose(0, 1)  # (n_levels, n_points)
        single_items = (
            self._single_by_base.items() if only_set is None
            else ((n, self._single_by_base[n]) for n in only_set if n in self._single_by_base)
        )
        for name, i in single_items:
            out[name] = tensor[:, i]
        if "z" in out:
            out["geopotential"] = out["z"]
        if "t" in out:
            out["temperature"] = out["t"]
        if "q" in out:
            out["specific_humidity"] = out["q"]
        if "sp" in out:
            out["surface_pressure"] = out["sp"]
        # Static orography, exposed the same way AIFS's own (checkpoint-
        # native, constant-in-time) 'z' input column is -- NOT verified yet
        # that Aurora's bundled static 'z' is in the same m^2/s^2
        # geopotential units ERA5/AIFS/FCN3 use (unlike FCN3's own
        # orography, which WAS directly checked against ERA5's z); treat
        # as an assumption to confirm empirically before trusting ps-obs
        # QC results that depend on it.
        if only_set is None or "geopotential_at_surface" in only_set:
            out["geopotential_at_surface"] = self._static_vars["z"].reshape(-1)
        return out

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> AuroraState:
        """`input_state`: `{"fields": {name: (2, 721, 1440) array}}` for
        every name in `_single_by_base`/`_ATMOS_VARS x _LEVELS` (as
        `f"{base}{level}"`-keyed... no: atmos fields are keyed by plain
        base name with a leading (2, len(_LEVELS), 721, 1440) array; see
        aurora_ic.py). `date` is the LATEST of the two time levels."""
        fields = input_state["fields"]
        t = torch.zeros(1, 2, _NLAT, _NLON, self._n_vars, device=self._device)
        for name, col in self._single_by_base.items():
            t[..., col] = torch.as_tensor(fields[name], dtype=torch.float32, device=self._device)
        for base, cols in self._levels_by_base.items():
            arr = torch.as_tensor(fields[base], dtype=torch.float32, device=self._device)  # (2, L, H, W)
            for k, (_, col) in enumerate(cols):
                t[..., col] = arr[:, k]
        state = t.reshape(1, 2, _NLAT * _NLON, self._n_vars)
        return AuroraState(state, date)

    def prime_from_state(self, aurora_state: AuroraState) -> AuroraState:
        """No per-process bookkeeping to rebuild (unlike AIFS's anemoi
        Runner cache) -- the packed tensor is self-contained. Returned
        as-is for interface parity with AIFSModel/FCN3Model."""
        return aurora_state

    # ------------------------------------------------------------------
    # Differentiable rollout
    # ------------------------------------------------------------------

    def _advance_one_step(
        self, state: torch.Tensor, date: datetime.datetime, latent_increment: torch.Tensor = None
    ) -> torch.Tensor:
        batch = self._unpack(state, date - self._timestep)
        # NOTE: `_unpack` is called with the state's PREVIOUS date (level-1's
        # date is `date` here, since `date` passed in is already the NEW,
        # post-step date -- see `advance()`, which increments `date` before
        # calling this, matching AIFS's convention (dynamic forcings/
        # insolation describe the state being CONSTRUCTED, not the input)).
        lead_hours = self._timestep.total_seconds() / 3600
        lead_times = torch.full((1,), lead_hours, device=self._device)

        def _hook(module, args, kwargs):
            x = args[0]
            if latent_increment is not None:
                x = x + latent_increment
            return (x,) + args[1:], kwargs

        handle = self.wrapper.decoder.register_forward_pre_hook(_hook, with_kwargs=True)
        try:
            pred = self.wrapper(batch, lead_times=lead_times)
        finally:
            handle.remove()

        return self._pack_pred(pred, state)

    def advance(
        self,
        aurora_state: AuroraState,
        steps: int = 1,
        use_checkpoint: bool = True,
        latent_increment: torch.Tensor = None,
    ) -> AuroraState:
        """Differentiable multi-step rollout -- same per-step
        `torch.utils.checkpoint` pattern as AIFSModel/FCN3Model.
        `latent_increment` (shape == `latent_shape`) applied ONLY on the
        first of `steps` internal advances, same rationale as the other
        two backends (Aurora re-encodes from the physical state on every
        step; there is no persistent latent state to keep perturbing)."""
        state = aurora_state.state
        date = aurora_state.date
        for i in range(steps):
            date = date + self._timestep
            step_increment = latent_increment if i == 0 else None
            if use_checkpoint:
                state = torch.utils.checkpoint.checkpoint(
                    self._advance_one_step, state, date, step_increment, use_reentrant=False
                )
            else:
                state = self._advance_one_step(state, date, step_increment)
        return AuroraState(state, date)
