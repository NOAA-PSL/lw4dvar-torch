"""
Differentiable SFNO-73ch-small (NVIDIA NGC nvidia/modulus/sfno_73ch_small)
model wrapper for the 4D-Var solver.

Loaded with makani 0.2.0 (the fcstnet3 env) from the converted package in
backends/sfno/ngc_cache/ -- see sfno_prefetch_checkpoint.py for the four
compatibility adjustments to the April-2024 package, and the 6h-forecast
check (z500 2.3 m / sp 0.35 hPa RMS vs ERA5, persistence 24 m / 2.6 hPa)
confirming the converted weights compute correctly.

Architecture (config.json + makani source): SphericalFourierNeuralOperatorNet,
embed_dim 384, 8 blocks, instance norm, dhconv filters -- the same network
body as ACE2 -- but scale_factor 3: block 0 transforms the full 721x1440 grid
down to 240x480, blocks 1-6 run at 240x480, block 7 transforms back up. Big
skip on. Inputs: the 73 channels + cos zenith angle, orography, land and sea
masks (77); outputs the 73 channels; 6h step; self-starting (n_history 0);
deterministic (no noise). No SST/skin temperature and no forcing -- the ocean
lower boundary is free-running (unlike ACE2's prescribed SST).

Channels, names, grid and packed-state layout `(1, 73, 721, 1440)` are
FCN3's plus 'sp', so SFNOModel subclasses FCN3Model and reuses its
decode_state / resolve_columns / pressure_levels / prepare_initial_state /
state_layout / wrap_state; it overrides loading, the (absent) noise, and the
single step.

Latent control (option B): the output of block 6 -- the last 240x480 latent,
shape (384, 240, 480) ~ 44M, comparable to ACE2's (384, 180, 360) -- via a
forward hook on blocks[-2], registered/removed inside the checkpointed
function so the backward recompute re-applies it. The increment then passes
through block 7's upsampling and the decoder.
"""

import datetime
import os
import sys

import numpy as np
import torch

from makani.models.model_package import LocalPackage, load_model_package
from makani.models.stepper import _assert_checkpoint_safe

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fcn3"))
from fcn3_model import FCN3Model, FCN3State, _FAMILY_RE  # noqa: E402

DEFAULT_PACKAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngc_cache", "sfno_73ch_small")


class SFNOModel(FCN3Model):
    def __init__(self, package_root: str = DEFAULT_PACKAGE, device: str = "cuda", latent_block: int = -2):
        """
        package_root : converted NGC package (sfno_prefetch_checkpoint.py).
        latent_block : index of the block whose OUTPUT receives the latent
            increment; -2 (block 6, the last 240x480 latent) is option B.
            -1 would be block 7's full-resolution output (384x721x1440).
        """
        self._device = torch.device(device)
        if self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.wrapper = load_model_package(LocalPackage(package_root), pretrained=True, device=device, multistep=False)
        self.wrapper.to(self._device)
        self.wrapper.eval()
        for p in self.wrapper.parameters():
            p.requires_grad_(False)
        self._step_wrapper = self.wrapper.model  # SingleStepWrapper
        self._net = self._step_wrapper.model  # SphericalFourierNeuralOperatorNet
        self._preprocessor = self._step_wrapper.preprocessor
        _assert_checkpoint_safe(self._net)

        params = self.wrapper.params
        self._channel_names = list(params.channel_names)
        self._timestep = datetime.timedelta(hours=int(params.dt) * int(params.dhours))
        lat1d = np.asarray(params.lat, dtype=np.float64)
        lon1d = np.asarray(params.lon, dtype=np.float64)
        self._nlat, self._nlon = lat1d.size, lon1d.size
        self._lats = np.repeat(lat1d, self._nlon)
        self._lons = np.tile(lon1d, self._nlat)

        self._levels_by_base = {}
        self._single_by_base = {}
        for i, name in enumerate(self._channel_names):
            m = _FAMILY_RE.fullmatch(name)
            if m:
                self._levels_by_base.setdefault(m.group(1), []).append((int(m.group(2)), i))
            else:
                self._single_by_base[name] = i
        for base in self._levels_by_base:
            self._levels_by_base[base].sort(key=lambda t: t[0], reverse=True)  # surface first, as FCN3

        self._latent_block = self._net.blocks[latent_block]
        # latent shape = that block's output: 240x480 for blocks 0..6, full grid for block 7
        n_blocks = len(self._net.blocks)
        idx = latent_block % n_blocks
        h, w = (self._nlat, self._nlon) if idx == n_blocks - 1 else (int(self._net.h), int(self._net.w))
        self._latent_shape = (int(self._net.embed_dim), h, w)

    # SFNO is deterministic -- no noise to prime (FCN3Model's noise calls would fail)
    def _prime_noise(self) -> None:
        pass

    def prime_from_state(self, state: FCN3State) -> FCN3State:
        return state

    def decode_state(self, state, time_index: int = -1, only=None):
        """FCN3Model.decode_state plus the native surface pressure under the
        canonical 'surface_pressure' alias (SFNO has 'sp'; FCN3 does not)."""
        out = super().decode_state(state, time_index=time_index, only=only)
        if "sp" in out:
            out["surface_pressure"] = out["sp"]
        return out

    def _advance_one_step(self, x_phys, date, use_checkpoint, latent_increment=None):
        # stateful-free prelude outside the checkpoint: normalize + zenith, then append
        # orography/land-sea masks (no noise for SFNO)
        xn = self.wrapper._prepare_input(x_phys, date, normalized_data=False)
        inp = self._step_wrapper._preprocess(xn, update_state=False)

        def _pure(inp_, inc_):
            handle = None
            if inc_ is not None:
                handle = self._latent_block.register_forward_hook(lambda _m, _i, out: out + inc_)
            try:
                return self._net(inp_)
            finally:
                if handle is not None:
                    handle.remove()

        if use_checkpoint:
            yn = torch.utils.checkpoint.checkpoint(_pure, inp, latent_increment, use_reentrant=False)
        else:
            yn = _pure(inp, latent_increment)
        return yn * self.wrapper.out_scale + self.wrapper.out_bias
