"""
Differentiable FourCastNet3 (FCN3) model wrapper for the 4D-Var solver.

Mirrors the AIFS v2 port's aifs_model.py in spirit (same
forecast_model.LatentForecastModel contract, same "reuse the real
checkpoint-loading library's own submodules/helper methods rather than
reimplementing normalization/forcing bookkeeping" philosophy), but FCN3's
own architecture is different enough that the mechanics are genuinely new
code, not a search-and-replace port:

- FCN3's state is a SINGLE physical-space time level `(1, 72, 721, 1440)`
  (`n_history == 0` is hard-enforced by the checkpoint) -- no AIFS-style
  lagged multi_step packing.
- FCN3 has a real `encode()` -> `process()` -> `decode()` split
  (`makani.models.networks.fourcastnet3.AtmoSphericNeuralOperatorNet`),
  directly analogous to AIFS's encoder/processor/decoder mesh split, but
  reached through `makani`'s own `ModelWrapper`/`SingleStepWrapper`/
  `Preprocessor2D` layers (checkpoint loading, normalization, the solar
  zenith-angle forcing, static orography/land-mask features, and FCN3's
  own stochastic diffusion-noise input) rather than anemoi's `Runner`.
- FCN3 is an ensemble/diffusion model: 8 stochastic noise channels are
  concatenated into every input, drawn from a per-instance
  `torch.Generator`-backed, temporally-correlated (AR(1)-in-spectral-space)
  process (`makani.models.noise.DiffusionNoiseS2`). This wrapper runs it
  DETERMINISTICALLY by design (see `_prime_noise`): a fixed seed, reset once
  per fresh episode (`prepare_initial_state`/`prime_from_state`), then
  allowed to continue its autoregressive sequence one step per
  `advance()` call -- see `_advance_one_step`'s docstring for why the noise
  update must run OUTSIDE the `torch.utils.checkpoint` boundary.

See this repo's CLAUDE.md "FourCastNet3 architecture notes" section for the
config.json-derived facts (channel schema, latent shape formula, big_skip/
bias_correction/history_normalization_mode all being no-ops for this
checkpoint) this implementation depends on -- established by direct
`makani` source inspection, not documentation.
"""

import datetime
import re

import numpy as np
import torch

from makani.models.model_package import LocalPackage, load_model_package
from makani.models.stepper import _assert_checkpoint_safe

import forecast_model

# FCN3's pressure-level channel names are the bare concatenation of a
# lowercase base letter and the level in hPa (e.g. "u500", "z1000") with NO
# separator -- unlike AIFS's "u_500" underscore convention. A single-level
# name that happens to end in digits preceded by more letters (e.g. "u10m",
# "t2m") does NOT fullmatch this (the trailing "m" breaks the "all digits to
# the end" requirement), so it correctly falls through to the single-level
# path instead of being misparsed as level "10"/"2" of a bogus one-member
# "u10m"/"t2m" family.
_FAMILY_RE = re.compile(r"^([a-z]+)(\d+)$")


def _chunked_encode(net, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Replicates `AtmoSphericNeuralOperatorNet.encode` exactly, except
    `net.atmo_encoder` is called on `chunk_size`-sized slices of the
    (batchdims * n_atmo_groups)-folded leading dimension instead of all 13
    pressure-level groups in one call. See FCN3Model.__init__'s
    `atmo_chunk_size` docstring and this repo's CLAUDE.md "decode() memory
    investigation" for why -- kept as a free function (not a method) since
    it operates purely on `net`'s own submodules/buffers, mirroring exactly
    how the unmodified method is written, so a future upstream fix is easy
    to diff against.

    Only the loop over `net.atmo_encoder` is new; `net.surf_encoder`'s
    input is already small (7 single-level channels, not level-folded), so
    it is left as a single unchunked call, exactly as `encode()` does it.
    """
    batchdims = x.shape[:-3]
    x_atmo = x[..., net.atmo_channels, :, :].contiguous().reshape(-1, net.n_atmo_chans, *x.shape[-2:])
    n = x_atmo.shape[0]
    if chunk_size >= n:
        x_out = net.atmo_encoder(x_atmo)
    else:
        x_out = torch.cat([net.atmo_encoder(x_atmo[i : i + chunk_size]) for i in range(0, n, chunk_size)], dim=0)
    x_out = x_out.reshape(*batchdims, net.n_atmo_groups * net.atmo_embed_dim, *x_out.shape[-2:])

    if hasattr(net, "surf_encoder"):
        x_surf = x[..., net.surf_channels, :, :].contiguous()
        x_surf = net.surf_encoder(x_surf)
        x_out = torch.cat((x_out, x_surf), dim=-3)

    x_out = x_out.reshape(*batchdims, net.total_embed_dim, *x_out.shape[-2:])
    return x_out


def _chunked_decode(net, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Replicates `AtmoSphericNeuralOperatorNet.decode` exactly, except
    `net.atmo_decoder` is called on `chunk_size`-sized slices instead of all
    13 pressure-level groups in one call -- see `_chunked_encode`'s
    docstring. This is the call that actually matters for the ~20GiB
    single-call memory finding: `decode()`'s DISCO transpose-convolution
    upsamples to NATIVE 721x1440 resolution (encode()'s analogous
    convolution only needs to produce the downsampled 360x720 hidden grid,
    ~4x fewer elements), so this is where chunking buys the most headroom.
    """
    batchdims = x.shape[:-3]
    x_atmo_in = x[..., : (net.n_atmo_groups * net.atmo_embed_dim), :, :].reshape(-1, net.atmo_embed_dim, *x.shape[-2:])
    n = x_atmo_in.shape[0]
    if chunk_size >= n:
        x_atmo = net.atmo_decoder(x_atmo_in)
    else:
        x_atmo = torch.cat([net.atmo_decoder(x_atmo_in[i : i + chunk_size]) for i in range(0, n, chunk_size)], dim=0)

    x_out = torch.zeros(*batchdims, net.n_out_chans, *x_atmo.shape[-2:], dtype=x.dtype, device=x.device)
    x_out[..., net.atmo_channels, :, :] = x_atmo.reshape(*batchdims, -1, *x_atmo.shape[-2:])

    if hasattr(net, "surf_decoder"):
        x_surf = x[..., -net.surf_embed_dim :, :, :]
        x_surf = net.surf_decoder(x_surf)
        x_out[..., net.surf_channels, :, :] = x_surf.reshape(*batchdims, -1, *x_surf.shape[-2:])

    return x_out


class FCN3State:
    """Thin wrapper whose `.state` attribute *is* the packed
    `(1, 72, 721, 1440)` physical-space tensor -- FCN3 has no history to
    pack (n_history == 0 is hard-enforced by the checkpoint), so this is
    simpler than AIFS's two-lagged-time-level `AIFSState`. Mirrors it
    structurally anyway (same `__slots__`/`__copy__` shape) so any solver-
    core code written against `ModelState`'s `copy.copy(...)` + `.state = `
    idiom keeps working unchanged.
    """

    __slots__ = ("state", "date")

    def __init__(self, state: torch.Tensor, date: datetime.datetime):
        self.state = state
        self.date = date

    def __copy__(self) -> "FCN3State":
        return FCN3State(self.state, self.date)


class FCN3Model(forecast_model.LatentForecastModel):
    """Wraps a loaded FourCastNet3 checkpoint for differentiable rollouts.

    Implements `forecast_model.LatentForecastModel` -- see that module for
    the formal contract. `model.wrapper` (the real `makani` `ModelWrapper`)
    stays a public escape hatch, same rationale as AIFS's `model.runner`:
    a future IC-fetching module for this backend will likely need to reach
    past this class into `wrapper.params`/`wrapper.model.preprocessor`
    directly, and that coupling is real, not something worth hiding behind
    a thin wrapper prematurely (see forecast_model.InitialConditionProvider).
    """

    def __init__(
        self,
        package_root: str,
        device: str = "cuda",
        noise_seed: int = 333,
        atmo_chunk_size: int = 2,
    ):
        """
        Parameters
        ----------
        package_root : str
            Path to the vendored HF checkpoint clone (this repo's
            `fourcastnet3/` directory) -- resolved via `LocalPackage`, which
            expects `config.json`, `mins.npy`/`maxs.npy`,
            `global_means.npy`/`global_stds.npy`, `orography.nc`,
            `land_mask.nc`, and `training_checkpoints/best_ckpt_mp0.tar`
            beneath it (all present in the vendored clone).
        noise_seed : int
            Fixed seed for FCN3's stochastic diffusion-noise input --
            see `_prime_noise`. Exposed here (not hardcoded) so a future
            ensemble extension could vary it per member; the initial 4D-Var
            implementation uses one fixed value throughout, by design (see
            module docstring).
        atmo_chunk_size : int
            How many of the 13 atmospheric pressure-level groups
            `_chunked_encode`/`_chunked_decode` (module-level functions
            below) push through `net.atmo_encoder`/`net.atmo_decoder` in one
            call, instead of all 13 at once the way the unmodified
            `AtmoSphericNeuralOperatorNet.encode`/`.decode` do. See this
            repo's CLAUDE.md "decode() memory investigation" section for why
            this exists: a single unchunked `decode()` call needs ~20GiB
            (all 13 levels batched through one DISCO transpose-convolution
            at native 721x1440 resolution), which alone makes a 2-step
            differentiable rollout exceed a 93GiB H100. Default `2`: the
            measured memory/time sweep (see CLAUDE.md) found peak memory for
            a 2-step checkpointed rollout drops from OOM (chunk_size=13,
            unchunked) to 78.18 / 67.64 / 64.08 / 62.05 / 62.05 GiB at
            chunk_size 7 / 4 / 3 / 2 / 1 respectively, with NO measurable
            wall-clock cost at any chunk size (~3.5s/step regardless) --
            `2` sits at the memory floor (`1` gave no further reduction)
            without paying `1`'s extra, pointless kernel-launch overhead.
            `13` (or anything >= 13) reproduces the original unchunked
            behavior exactly -- useful for a direct correctness comparison
            (see verify_chunked_atmo.py), not for real use.
        """
        self._atmo_chunk_size = atmo_chunk_size
        self._device = torch.device(device)
        if self._device.type == "cuda":
            # Same rationale as AIFSModel.__init__: nothing in this pipeline
            # runs at full fp32 precision anyway once bf16/kernel-level
            # nondeterminism is in play, so TF32 is a free speedup for
            # whatever matmuls fall outside any autocast region.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.wrapper = load_model_package(LocalPackage(package_root), pretrained=True, device=device, multistep=False)
        # load_model_package's own model_registry.get_model(...).to(device) call
        # happens BEFORE ModelWrapper.__init__ registers its own in/out
        # bias+scale buffers, so those specific buffers are NOT guaranteed to
        # have already been moved by that earlier .to(device) -- move
        # everything again here, unconditionally, rather than assume.
        self.wrapper.to(self._device)
        self.wrapper.eval()

        # We only ever want gradients w.r.t. OUR latent increment, never
        # w.r.t. FCN3's pretrained weights -- torch.load doesn't set
        # requires_grad=False on its own. Same lesson AIFSModel.__init__
        # documents at length (CLAUDE.md's frozen-weights graph-reuse bug).
        for p in self.wrapper.parameters():
            p.requires_grad_(False)

        self._step_wrapper = self.wrapper.model  # SingleStepWrapper
        self._net = self._step_wrapper.model  # AtmoSphericNeuralOperatorNet
        self._preprocessor = self._step_wrapper.preprocessor  # Preprocessor2D

        # Defensive check, not just an assumption: confirms the network
        # submodules actually checkpointed in _advance_one_step (net.encode_process
        # / net.decode / net.clamp_water_channels) carry no private-generator
        # (rng_cpu/rng_gpu) submodule of their own. FCN3's stochastic noise
        # module lives on self._preprocessor (a sibling, not a descendant of
        # self._net), so this is expected to pass -- see _advance_one_step's
        # docstring for why it would matter if it didn't. Mirrors makani's own
        # _assert_checkpoint_safe guard in stepper.py's MultiStepWrapper.
        _assert_checkpoint_safe(self._net)

        params = self.wrapper.params
        self._channel_names: list[str] = list(params.channel_names)
        self._noise_seed = noise_seed
        self._timestep = datetime.timedelta(hours=int(params.dt) * int(params.dhours))

        lat1d = np.asarray(params.lat, dtype=np.float64)
        lon1d = np.asarray(params.lon, dtype=np.float64)
        self._nlat = lat1d.size
        self._nlon = lon1d.size
        # Row-major flatten convention (lat slowest, lon fastest), matching
        # tensor.reshape(C, -1) on a (C, nlat, nlon) tensor -- decode_state
        # and fcn3_grid.GridInterpolator must agree on this ordering.
        self._lats = np.repeat(lat1d, self._nlon)
        self._lons = np.tile(lon1d, self._nlat)

        self._levels_by_base: dict[str, list[tuple[int, int]]] = {}
        self._single_by_base: dict[str, int] = {}
        for i, name in enumerate(self._channel_names):
            m = _FAMILY_RE.fullmatch(name)
            if m:
                base, lev = m.group(1), int(m.group(2))
                self._levels_by_base.setdefault(base, []).append((lev, i))
            else:
                self._single_by_base[name] = i
        for base in self._levels_by_base:
            # surface/high-pressure first, matching AIFS's _levels_by_base
            # convention (see forecast_model.py's pressure_levels docstring
            # for why callers, not this method, own any further re-ordering
            # a specific use needs).
            self._levels_by_base[base].sort(key=lambda t: t[0], reverse=True)

        n_atmo_groups = len({lev for levels in self._levels_by_base.values() for lev, _ in levels})
        h = int(params.img_shape_x // params.scale_factor)
        w = int(params.img_shape_y // params.scale_factor)
        total_embed_dim = n_atmo_groups * int(params.atmo_embed_dim) + int(params.surf_embed_dim)
        self._latent_shape = (total_embed_dim, h, w)

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
    def latent_shape(self) -> tuple[int, int, int]:
        """(total_embed_dim, h, w) -- see this repo's CLAUDE.md
        "FourCastNet3 architecture notes" for the formula
        (n_atmo_groups * atmo_embed_dim + surf_embed_dim, img_shape // scale_factor)
        this is computed from at __init__ time, so it stays correct if the
        checkpoint is ever swapped for one with different embed dims."""
        return self._latent_shape

    def resolve_columns(self, name: str) -> list[int]:
        """See forecast_model.LatentForecastModel.resolve_columns. Checks
        `_levels_by_base` (pressure-level families) BEFORE
        `_single_by_base` -- same precedence lesson AIFSModel.resolve_columns
        documents (CLAUDE.md's retired `state_scales` lookup-order bug).
        FCN3 has no separate raw-checkpoint-mapping fallback tier the way
        AIFS's `var_to_idx` is (there is no anemoi `Checkpoint` object here),
        so this is a two-tier, not three-tier, lookup.
        """
        if name in self._levels_by_base:
            return [i for _, i in self._levels_by_base[name]]
        if name in self._single_by_base:
            return [self._single_by_base[name]]
        raise KeyError(f"{name!r} is not a known FCN3 variable or family")

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        return np.array([lev for lev, _ in self._levels_by_base[base]], dtype=np.float32)

    @property
    def state_layout(self) -> str:
        """'channels_first': the packed state is `(1, n_vars, H, W)` --
        `n_vars` is dim 1, unlike AIFSModel's `(1, multi_step, n_points,
        n_vars)` ('channels_last', n_vars trailing). Lets backend-agnostic
        solver-core code (`_apply_control_mask`/`_resolve_control_mask` in
        long_window_4dvar_utils.py) broadcast a per-variable mask correctly
        without hardcoding either backend's packing."""
        return "channels_first"

    def wrap_state(self, state: torch.Tensor, date: datetime.datetime) -> FCN3State:
        """Construct this backend's state wrapper -- lets solver-core code
        build a new state without importing/naming FCN3State directly."""
        return FCN3State(state, date)

    # ------------------------------------------------------------------
    # State construction / decoding
    # ------------------------------------------------------------------

    def decode_state(
        self, state: FCN3State, time_index: int = -1, only: "list[str] | None" = None
    ) -> dict[str, torch.Tensor]:
        """Slice the packed `(1, 72, 721, 1440)` tensor into {base_name: tensor}.

        `time_index` is accepted for interface compatibility but ignored:
        FCN3 has no history dimension to index into (n_history == 0), so
        there is only ever one time level.

        Does NOT provide 'surface_pressure' -- FCN3 has no native `sp`
        channel at all (see this repo's CLAUDE.md "Variables" note); the
        ps-obs forward-operator design this would feed is an explicitly
        deferred decision, not resolved by this model-backend pass.
        """
        tensor = state.state[0]  # (72, nlat, nlon)
        flat = tensor.reshape(tensor.shape[0], -1)  # (72, n_points)
        out: dict[str, torch.Tensor] = {}
        only_set = None if only is None else set(only)
        levels_items = (
            self._levels_by_base.items()
            if only_set is None
            else ((b, self._levels_by_base[b]) for b in only_set if b in self._levels_by_base)
        )
        for base, levels in levels_items:
            cols = [i for _, i in levels]
            out[base] = flat[cols, :]  # (n_levels, n_points)
        single_items = (
            self._single_by_base.items()
            if only_set is None
            else ((n, self._single_by_base[n]) for n in only_set if n in self._single_by_base)
        )
        for name, i in single_items:
            out[name] = flat[i, :]  # (n_points,)
        # NeuralGCM/AIFS-style aliases used by the (not-yet-ported) loss/QC code.
        if "z" in out:
            out["geopotential"] = out["z"]
        if "t" in out:
            out["temperature"] = out["t"]
        if "q" in out:
            out["specific_humidity"] = out["q"]
        return out

    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> FCN3State:
        """Build the packed `(1, 72, 721, 1440)` tensor from a generic
        `{"fields": {channel_name: array}}` dict (`array` either already
        `(nlat, nlon)` or flat `(n_points,)` in this class's row-major
        convention) -- no anemoi-`Runner`-style forcings assembly needed:
        FCN3's own aux/forcing channels (zenith, noise, orography,
        land-mask) are all derived fresh inside `_advance_one_step` from
        `makani`'s own `Preprocessor2D`, not part of the persistent state.

        Also primes the stochastic noise state fresh for a new episode --
        see `_prime_noise`.

        `date` MUST carry `tzinfo=datetime.timezone.utc` -- the zenith-angle
        forcing (`ModelWrapper._prepare_input` -> `cos_zenith_angle`,
        called from `_advance_one_step`) subtracts a tz-aware UTC epoch
        internally and raises `TypeError: can't subtract offset-naive and
        offset-aware datetimes` on a naive datetime. AIFS's own IC-fetching
        code happens not to hit this (anemoi handles tz internally); a
        future fcn3_ic.py must produce UTC-aware dates throughout.
        """
        fields = input_state["fields"]
        arrs = []
        for name in self._channel_names:
            a = np.asarray(fields[name], dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(self._nlat, self._nlon)
            arrs.append(a)
        tensor_np = np.stack(arrs, axis=0)[np.newaxis, ...]  # (1, 72, nlat, nlon)
        state_t = torch.from_numpy(np.ascontiguousarray(tensor_np)).to(self._device)
        self._prime_noise()
        return FCN3State(state_t, date)

    def _prime_noise(self) -> None:
        """Reset FCN3's stochastic diffusion-noise input to a fixed, fresh
        episode: reseed its private generator (`Preprocessor2D.set_rng`,
        which also zeros the spectral state) and then draw one fresh
        "stationary distribution" history (`ModelWrapper.update_state`,
        `replace_state=True`) rather than leave it at the unphysical
        all-zero state `set_rng`'s reset leaves behind -- see
        `makani.models.noise.DiffusionNoiseS2.update`'s docstring for why
        `replace_state=True` (draw a whole fresh correlated history from
        the stationary distribution) is the right call for *starting* an
        episode, versus `replace_state=False` (one AR step, correlated with
        the existing state) for *continuing* one, which is what every
        `advance()` step's `_advance_one_step` call uses instead.

        Called from both `prepare_initial_state` (a fresh fetch) and
        `prime_from_state` (the restart path, an already-packed state
        loaded from disk) -- for FCN3, unlike AIFS, the restart path needs
        no OTHER bookkeeping rebuilt (there is no anemoi-`Runner`-style
        `_input_tensor_by_name` cache; every non-prognostic input channel is
        recomputed fresh from the packed physical state + date on every
        `advance()` call), so `prime_from_state` is just this.
        """
        self.wrapper.set_rng(reset=True, seed=self._noise_seed)
        self.wrapper.update_state(replace_state=True, batch_size=1)

    def prime_from_state(self, state: FCN3State) -> FCN3State:
        self._prime_noise()
        return state

    # ------------------------------------------------------------------
    # Differentiable rollout
    # ------------------------------------------------------------------

    def _advance_one_step(
        self, x_phys: torch.Tensor, date: datetime.datetime, use_checkpoint: bool, latent_increment: torch.Tensor = None
    ) -> torch.Tensor:
        """One FCN3 step, split into a non-checkpointed stateful prelude and
        a checkpointed pure-tensor compute -- NOT simply
        `torch.utils.checkpoint.checkpoint(self.wrapper.encode_process, ...)`,
        and this split is load-bearing, not stylistic.

        `ModelWrapper.encode_process` -> `SingleStepWrapper.encode_process` ->
        `SingleStepWrapper._preprocess` advances FCN3's own stochastic noise
        state (`Preprocessor2D.update_internal_state` -> `DiffusionNoiseS2.update`,
        an AR(1)-in-spectral-space step that mutates a persistent buffer using
        a PRIVATE `torch.Generator`, not the global RNG) and caches the
        solar-zenith "unpredicted feature" for this step. `torch.utils.checkpoint`
        with `use_reentrant=False` RE-RUNS its wrapped function during backward;
        `preserve_rng_state=True` (the default) only saves/restores the GLOBAL
        torch RNG, not a module's own private `torch.Generator` -- exactly the
        failure mode `makani.models.stepper._assert_checkpoint_safe` exists to
        catch for `MultiStepWrapper`'s own `multistep_checkpoint` option (see
        that function's docstring and `_forward_train`'s comment: "the stateful
        preprocessor calls ... stay outside the checkpoint so they run once and
        are not re-executed during the backward recompute"). Checkpointing the
        stateful call here would silently double-advance the noise state's AR
        sequence every time backward() ran, and would recompute a DIFFERENT
        noise draw than the one actually used in the forward pass (the state
        buffer keeps advancing on every recompute) -- corrupting both the
        gradient (computed w.r.t. inputs the recompute silently changed) and
        every later step's noise trajectory.

        The fix, mirroring `_forward_train`'s own pattern exactly: run
        `wrapper._prepare_input` (normalize + cache zenith) and
        `step_wrapper._preprocess` (noise AR step + append + static features)
        OUTSIDE checkpoint, producing a plain tensor `inpans` with no more
        pending side effects. Only encode -> process -> [+ latent_increment]
        -> decode -> `net.clamp_water_channels(...)` -- a pure,
        side-effect-free function of `(inpans, latent_increment)` -- is
        checkpointed; recomputing it during backward is fully safe.

        Uses `_chunked_encode`/`_chunked_decode` (module-level functions
        above) instead of `net.encode_process(inpans)`/`net.decode(...)`
        directly -- functionally identical (chunk_size >= 13 reproduces the
        unmodified behavior exactly), but processes the 13 atmospheric
        pressure-level groups `self._atmo_chunk_size` at a time instead of
        all at once, which is what makes a multi-step differentiable
        rollout fit in memory at all -- see this repo's CLAUDE.md "decode()
        memory investigation" section for the measurements that motivated
        this and __init__'s `atmo_chunk_size` docstring for the mechanism.
        """
        step_wrapper = self._step_wrapper
        net = self._net
        chunk_size = self._atmo_chunk_size

        # --- stateful prelude: must run exactly once, never inside checkpoint ---
        xn = self.wrapper._prepare_input(x_phys, date, normalized_data=False)
        inpans = step_wrapper._preprocess(xn, update_state=True, replace_state=False)

        # --- pure tensor compute: safe to checkpoint/recompute ---
        def _pure_forward(inpans_, latent_increment_):
            x_aux = net.encode_auxiliary_channels(inpans_)
            latent = _chunked_encode(net, inpans_, chunk_size)
            latent = net.process(latent, x_aux)
            if latent_increment_ is not None:
                latent = latent + latent_increment_
            yn = _chunked_decode(net, latent, chunk_size)
            yn = net.clamp_water_channels(yn)
            return yn

        if use_checkpoint:
            yn = torch.utils.checkpoint.checkpoint(
                _pure_forward, inpans, latent_increment, use_reentrant=False
            )
        else:
            yn = _pure_forward(inpans, latent_increment)

        yn = self._preprocessor.correct_bias(yn)  # no-op: this checkpoint has no bias_correction configured
        y = self._preprocessor.history_denormalize(yn, target=True)  # no-op: history_normalization_mode == "none"
        return y * self.wrapper.out_scale + self.wrapper.out_bias

    def advance(
        self,
        state: FCN3State,
        steps: int = 1,
        use_checkpoint: bool = True,
        latent_increment: torch.Tensor = None,
    ) -> FCN3State:
        """Differentiable multi-step rollout. `latent_increment` (shape ==
        `latent_shape`) is added to the encoder/processor output ONLY on
        the first of `steps` internal single-step advances -- FCN3 re-embeds
        from the physical state on every step (there is no persistent latent
        state carried across steps, same as AIFS -- see
        `AtmoSphericNeuralOperatorNet.forward`: every call runs `encode()`
        fresh), so applying it more than once would double-count it.

        Unlike `AIFSModel.advance`, `date` is passed to `_advance_one_step`
        BEFORE incrementing it, not after: FCN3's zenith-angle forcing
        (`ModelWrapper._prepare_input` -> `_zenith_features`) represents
        insolation at the CURRENT input's valid time, used to predict the
        NEXT step -- not (as for AIFS's dynamic forcings, which describe the
        state being CONSTRUCTED) the target time.
        """
        x = state.state
        date = state.date
        for i in range(steps):
            step_latent_increment = latent_increment if i == 0 else None
            x = self._advance_one_step(x, date, use_checkpoint, step_latent_increment)
            date = date + self._timestep
        return FCN3State(x, date)
