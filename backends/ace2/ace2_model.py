"""
Differentiable ACE2-ERA5 (Ai2 Climate Emulator v2, `allenai/ACE2-ERA5`)
model wrapper for the 4D-Var solver.

Same forecast_model.LatentForecastModel contract as AIFSModel/FCN3Model/
AuroraModel. Facts this implementation depends on, established by loading
the real checkpoint and reading the installed `fme` package source (not
documentation):

- Architecture: `fme.ace.models.modulus.sfnonet.SphericalFourierNeuralOperatorNet`
  (SFNO, embed_dim=384, 8 blocks, scale_factor=1) on a 180x360
  Legendre-Gauss grid, latitudes ordered SOUTH-TO-NORTH (unlike ERA5/FCN3/
  Aurora's north-to-south), 6h timestep. Deterministic -- no noise inputs.
- Self-starting (one time level, like FCN3): state is `(1, n_vars, 180, 360)`,
  channels-first.
- Vertical coordinate: 8 hybrid sigma-pressure LAYERS (not pressure levels),
  interfaces `p_i = ak_i + bk_i * PRESsfc`, index 0 = top. Prognostic layer
  families: air_temperature_k, specific_total_water_k, eastward_wind_k,
  northward_wind_k (k=0..7), plus single-level PRESsfc, surface_temperature,
  TMP2m, Q2m, UGRD10m, VGRD10m. 12 output-only diagnostics (fluxes,
  PRATEsfc, TMP850, h500, ...) are also carried in the packed state so
  decode_state can expose them, but never feed the network.
- Forcings: land_fraction, ocean_fraction, sea_ice_fraction, DSWRFtoa,
  HGTsfc, global_mean_co2 are network inputs read from the yearly
  `forcing_YYYY.nc` files at every step (DSWRFtoa at the TARGET time,
  `next_step_forcing_names`, the rest at the input time). The checkpoint's
  `ocean` config then overwrites surface_temperature over ocean with the
  forcing file's SST at the target time, and a conservation corrector
  (dry-air mass, moisture budget, positivity) runs after every step. All of
  this is `fme`'s own `Stepper.step` -- called directly here, not
  reimplemented, so it stays numerically identical to `fme` inference.
- Latent: the output of the SFNO's last block (before the big-skip concat
  and decoder), shape `(embed_dim, h, w)` = `(384, 180, 360)`. Injected via
  a forward hook on `blocks[-1]`, registered/removed INSIDE the checkpointed
  function so the backward recompute re-applies it.
- Runs in fp32 (fme's NullOptimization.autocast is a no-op) -- none of
  Aurora's/AIFS's fp16-autocast gradient overflow concerns apply.

Pressure-level fields: the solver's forward operators and z500 diagnostics
need geopotential (and t/q) on fixed PRESSURE levels, which ACE2 doesn't
carry. decode_state derives 'z', 't', 'q', 'u', 'v' on `plevs` from the
native layers (hydrostatic integration up from HGTsfc, isothermal within
each layer -- see `_to_pressure_levels`), differentiably. These derived
names are "known variables" for loss/diagnostic purposes but have no
packed-state columns, so `resolve_columns` (control_variables masking)
only accepts native ACE2 names.
"""

import datetime
import glob
import os
import re

import numpy as np
import torch
import xarray as xr

from fme.ace.models.modulus.sfnonet import SphericalFourierNeuralOperatorNet
from fme.ace.stepper.single_module import Stepper
from fme.core.step.args import StepArgs

import forecast_model

GRAV = 9.80665
RD = 287.05
RV = 461.5
LAPSE = 0.0065

# ERA5/WeatherBench 13 pressure levels (hPa), surface first -- the same set
# FCN3 carries natively, so derived ACE2 fields line up with ERA5 verification.
DEFAULT_PLEVS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

_LAYER_RE = re.compile(r"^(.*)_(\d+)$")

# Lowest-native-level fields for ps_operator='ps_native' (see decode_state).
_LOWEST_LEVEL_KEYS = ("t_lowest", "q_lowest", "p_lowest")

# Derived pressure-level family -> native ACE2 layer family it's built from.
_DERIVED_FROM = {"t": "air_temperature", "q": "specific_total_water", "u": "eastward_wind", "v": "northward_wind"}


class ACE2State:
    """`.state` is the packed `(1, n_vars, 180, 360)` tensor (channel order
    == ACE2Model.channel_names), `.date` its valid time. Same shape/idiom as
    FCN3State."""

    __slots__ = ("state", "date")

    def __init__(self, state: torch.Tensor, date: datetime.datetime):
        self.state = state
        self.date = date

    def __copy__(self) -> "ACE2State":
        return ACE2State(self.state, self.date)


class _ForcingSource:
    """Lazily opens yearly `forcing_YYYY.nc` files and returns the forcing
    fields at an exact 6-hourly date as `(1, nlat, nlon)` device tensors,
    cached per date (forcings are constants w.r.t. the control, and a
    100-epoch run re-reads the same ~20 dates every epoch)."""

    def __init__(self, forcing_dir: str, names: list[str], device: torch.device, max_cache: int = 512):
        self._dir = forcing_dir
        self._names = names
        self._device = device
        self._max_cache = max_cache
        self._datasets: dict[int, xr.Dataset] = {}
        self._cache: dict[np.datetime64, dict[str, torch.Tensor]] = {}

    def _dataset(self, year: int) -> xr.Dataset:
        if year not in self._datasets:
            path = os.path.join(self._dir, f"forcing_{year}.nc")
            if not os.path.exists(path) or _is_lfs_pointer(path):
                raise FileNotFoundError(
                    f"{path} not found -- fetch it with "
                    f"`python backends/ace2/ace2_prefetch_checkpoint.py {year}` from a login node"
                )
            self._datasets[year] = xr.open_dataset(path)
        return self._datasets[year]

    def __call__(self, date: datetime.datetime) -> dict[str, torch.Tensor]:
        key = np.datetime64(date.replace(tzinfo=None), "ns")
        if key not in self._cache:
            ds = self._dataset(date.year).sel(time=key)
            fields = {}
            for n in self._names:
                arr = np.asarray(ds[n].values, dtype=np.float32)
                if arr.ndim == 0:  # global_mean_co2 is a per-time scalar; fme broadcasts it to the grid
                    arr = np.full(ds["land_fraction"].shape, arr, dtype=np.float32)
                fields[n] = torch.from_numpy(arr)[None].to(self._device)
            if len(self._cache) >= self._max_cache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = fields
        return self._cache[key]


def _is_lfs_pointer(path: str) -> bool:
    """True for a git-lfs pointer stub (backends/ace2/ACE2-ERA5 is a
    pointer-only submodule -- only the years fetched with
    ace2_prefetch_checkpoint.py have real content)."""
    if os.path.getsize(path) > 1024:
        return False
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def _find_sfno(stepper: Stepper) -> SphericalFourierNeuralOperatorNet:
    for m in stepper.modules:
        for sub in m.modules():
            if isinstance(sub, SphericalFourierNeuralOperatorNet):
                return sub
    raise RuntimeError("no SphericalFourierNeuralOperatorNet found in the ACE2 stepper")


class ACE2Model(forecast_model.LatentForecastModel):
    """Wraps a loaded ACE2-ERA5 checkpoint (`fme` Stepper) for differentiable
    rollouts. `model.stepper` stays public, same escape-hatch rationale as
    AIFSModel.runner / FCN3Model.wrapper."""

    def __init__(
        self,
        checkpoint_path: str,
        forcing_dir: str,
        device: str = "cuda",
        plevs: tuple = DEFAULT_PLEVS,
    ):
        """
        Parameters
        ----------
        checkpoint_path : str
            `ace2_era5_ckpt.tar` (see ace2_prefetch_checkpoint.py).
        forcing_dir : str
            Directory holding `forcing_YYYY.nc` for every year a rollout
            touches.
        device : str
            Must agree with `fme.core.device.get_device()` -- fme places its
            own module/normalizer/corrector buffers there on its own (CUDA if
            available), so a mismatch is an error rather than silently
            splitting tensors across devices.
        plevs : tuple
            Pressure levels (hPa) decode_state derives 'z'/'t'/'q'/'u'/'v' on.
        """
        from fme.core.device import get_device

        self._device = torch.device(device)
        if get_device().type != self._device.type:
            raise ValueError(f"device={device!r} but fme will place the model on {get_device()}")
        if self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stepper_state = ckpt["stepper"]
        sigma = stepper_state["sigma_coordinates"]
        self._ak = torch.as_tensor(sigma["ak"], dtype=torch.float32, device=self._device)
        self._bk = torch.as_tensor(sigma["bk"], dtype=torch.float32, device=self._device)
        self._timestep = datetime.timedelta(seconds=int(stepper_state["encoded_timestep"]) / 1e6)
        self._nlat, self._nlon = (int(s) for s in stepper_state["img_shape"])

        self.stepper = Stepper.from_state(stepper_state)
        self.stepper.set_eval()
        # Gradients only w.r.t. our latent increment, never the weights --
        # same lesson as every other backend (torch.load doesn't freeze).
        for p in self.stepper.modules.parameters():
            p.requires_grad_(False)

        cfg = stepper_state["config"]
        in_names, out_names = list(cfg["in_names"]), list(cfg["out_names"])
        self._prognostic = [n for n in in_names if n in out_names]
        self._input_only = [n for n in in_names if n not in out_names]
        self._next_step_forcing = set(cfg.get("next_step_forcing_names", []))
        self._next_step_input = list(self.stepper._step_obj.next_step_input_names)
        # Packed state: prognostics first (the only channels fed back as
        # network input), then output-only diagnostics.
        self._channel_names = self._prognostic + [n for n in out_names if n not in in_names]
        self._chan_idx = {n: i for i, n in enumerate(self._channel_names)}

        self._layer_cols: dict[str, list[int]] = {}
        self._single_cols: dict[str, int] = {}
        for i, name in enumerate(self._channel_names):
            m = _LAYER_RE.fullmatch(name)
            if m:
                self._layer_cols.setdefault(m.group(1), []).append((int(m.group(2)), i))
            else:
                self._single_cols[name] = i
        # top (k=0) to bottom, matching ak/bk interface order
        self._layer_cols = {b: [i for _, i in sorted(v)] for b, v in self._layer_cols.items()}

        self._forcing = _ForcingSource(
            forcing_dir, sorted(set(self._input_only) | set(self._next_step_input)), self._device
        )
        first = [f for f in sorted(glob.glob(os.path.join(forcing_dir, "forcing_*.nc"))) if not _is_lfs_pointer(f)]
        if not first:
            raise FileNotFoundError(
                f"no fetched forcing_*.nc in {forcing_dir} (only LFS pointers?) -- run "
                "`python backends/ace2/ace2_prefetch_checkpoint.py YEAR ...` from a login node"
            )
        with xr.open_dataset(first[0]) as ds:
            lat1d = np.asarray(ds["latitude"].values, dtype=np.float64)
            lon1d = np.asarray(ds["longitude"].values, dtype=np.float64)
            hgt = np.asarray(ds["HGTsfc"].values, dtype=np.float32)  # static (lat, lon)
        if (lat1d.size, lon1d.size) != (self._nlat, self._nlon):
            raise ValueError(f"forcing grid {(lat1d.size, lon1d.size)} != checkpoint img_shape {(self._nlat, self._nlon)}")
        # static surface height (m), for the hydrostatic pressure-level
        # derivation. Negative values (spectral ringing over ocean, ~21k
        # cells down to -40 m) are clipped to 0, exactly as fme's own
        # atmosphere_data._height_at_interface does.
        self._hgtsfc = torch.from_numpy(np.maximum(hgt, 0.0).reshape(-1)).to(self._device)
        self._lats = np.repeat(lat1d, self._nlon)
        self._lons = np.tile(lon1d, self._nlat)

        self._plevs = np.asarray(plevs, dtype=np.float32)
        # Views the shared, backend-agnostic output writers in
        # long_window_4dvar_utils.py (save_xr_trajectory/_save_xr_state)
        # iterate: {family: [(level, column-or-None), ...]} in decode_state's
        # row order, and {single name: column}. Native layer families use
        # level = layer index 0..7 (top->bottom); the derived pressure-level
        # families (no packed columns) use level = hPa, surface-first.
        self._levels_by_base = {b: list(enumerate(cols)) for b, cols in self._layer_cols.items()}
        for b in ("z",) + tuple(_DERIVED_FROM):
            self._levels_by_base[b] = [(int(p), None) for p in self._plevs]
        self._single_by_base = dict(self._single_cols)
        self._sfno = _find_sfno(self.stepper)
        self._latent_shape = (int(self._sfno.embed_dim), *(int(s) for s in self._sfno.img_shape_eff))

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
        """(embed_dim, h, w) of the SFNO's last-block output."""
        return self._latent_shape

    @property
    def surface_geopotential(self) -> torch.Tensor:
        """(n_points,) g * HGTsfc (m^2/s^2) -- ACE2's own static orography
        (from the HF forcing files, negatives clipped to 0 like fme does),
        consistent with its PRESsfc. get_verif supplies this as the
        'geopotential_at_surface' the ps-obs QC/reduction uses."""
        return GRAV * self._hgtsfc

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def state_layout(self) -> str:
        return "channels_first"

    def resolve_columns(self, name: str) -> list[int]:
        """Native packed-state columns only (layer families before singles).
        Derived pressure-level names ('z', 't', ...) have no columns -- they
        can't be control_variables -- but ARE known variables, see
        is_known_variable."""
        if name in self._layer_cols:
            return list(self._layer_cols[name])
        if name in self._single_cols:
            return [self._single_cols[name]]
        raise KeyError(f"{name!r} is not a known ACE2 variable or layer family")

    def is_known_variable(self, name: str) -> bool:
        if name in ("z", "sp", "z500") or name in _DERIVED_FROM or name in _LOWEST_LEVEL_KEYS:
            return True
        return super().is_known_variable(name)

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        if base != "z" and base not in _DERIVED_FROM:
            raise KeyError(f"{base!r} is not a derived pressure-level family (ACE2 layers are hybrid sigma)")
        return self._plevs.copy()

    def wrap_state(self, state: torch.Tensor, date: datetime.datetime) -> ACE2State:
        return ACE2State(state, date)

    def _prime_noise(self) -> None:
        """No-op: ACE2 is deterministic. Present because compute_optimal
        calls it unconditionally (see FCN3Model._prime_noise)."""

    # ------------------------------------------------------------------
    # Pressure-level derivation
    # ------------------------------------------------------------------

    def _to_pressure_levels(self, flat: torch.Tensor, bases: set) -> dict[str, torch.Tensor]:
        """Derive {base: (n_plevs, n_points)} on self._plevs from the native
        hybrid-sigma layers of one packed time level `flat` (n_vars, n_points).

        Geopotential: hydrostatic integration up from g*HGTsfc through layer
        interfaces, with each layer isothermal at its virtual temperature
        (so phi is linear in ln p within a layer). Above interface 1 (top
        layer, whose top interface is p=0) the top layer's Tv is extended
        isothermally. Below ground (p > ps) uses the standard-atmosphere
        extrapolation from a surface temperature derived from the lowest
        layer with a 6.5 K/km lapse rate (the ERA5 convention).

        t/q/u/v: linear in ln p between layer midpoints; held at the top
        layer's value above it; below the lowest midpoint, t follows the
        6.5 K/km lapse rate and q/u/v are held constant.
        """
        ps = flat[self._single_cols["PRESsfc"]]  # (P,) Pa
        T = flat[self._layer_cols["air_temperature"]]  # (8, P)
        q = flat[self._layer_cols["specific_total_water"]]
        nlay = T.shape[0]
        p_int = self._ak[:, None] + self._bk[:, None] * ps[None, :]  # (9, P), top -> bottom
        lnp_int = torch.log(p_int[1:].clamp_min(1.0))  # interfaces 1..8, ascending pressure
        lnp_t = torch.log(torch.as_tensor(self._plevs * 100.0, device=flat.device))[:, None].expand(-1, ps.shape[0])  # (L, P)
        kappa = RD * LAPSE / GRAV
        out = {}

        tv = T * (1.0 + (RV / RD - 1.0) * q)
        p_mid_bot = 0.5 * (p_int[-2] + p_int[-1])
        t_sfc = T[-1] * (ps / p_mid_bot) ** kappa
        if "z" in bases:
            # phi at interfaces 1..8 (bottom interface 8 == surface)
            dphi = RD * tv[1:] * (lnp_int[1:] - lnp_int[:-1])  # layers 1..7
            phi_sfc = GRAV * self._hgtsfc
            phi_int = torch.cat([phi_sfc[None] + torch.flip(torch.cumsum(torch.flip(dphi, [0]), 0), [0]), phi_sfc[None]], 0)  # (8, P)
            idx = torch.searchsorted(lnp_int.T.contiguous(), lnp_t.T.contiguous()).T  # (L, P), 0..8
            j = idx.clamp(max=nlay - 1)  # layer containing the target level; base interface is j+1 == phi_int[j]
            phi_above = torch.gather(phi_int, 0, j) + RD * torch.gather(tv, 0, j) * (torch.gather(lnp_int, 0, j) - lnp_t)
            tv_sfc = t_sfc * (1.0 + (RV / RD - 1.0) * q[-1])
            phi_below = phi_sfc[None] + GRAV * (tv_sfc[None] / LAPSE) * (1.0 - torch.exp(kappa * (lnp_t - torch.log(ps)[None])))
            out["z"] = torch.where(idx >= nlay, phi_below, phi_above)

        need = [b for b in _DERIVED_FROM if b in bases]
        if need:
            lnp_mid = torch.log((0.5 * (p_int[:-1] + p_int[1:])).clamp_min(1.0))  # (8, P)
            k = torch.searchsorted(lnp_mid.T.contiguous(), lnp_t.T.contiguous()).T  # (L, P), 0..8
            lo = (k - 1).clamp(0, nlay - 1)
            hi = k.clamp(0, nlay - 1)
            x_lo, x_hi = torch.gather(lnp_mid, 0, lo), torch.gather(lnp_mid, 0, hi)
            w = torch.where(hi > lo, (lnp_t - x_lo) / (x_hi - x_lo), torch.zeros_like(lnp_t))
            for b in need:
                f = flat[self._layer_cols[_DERIVED_FROM[b]]]
                v = (1 - w) * torch.gather(f, 0, lo) + w * torch.gather(f, 0, hi)
                if b == "t":
                    below = T[-1][None] * torch.exp(kappa * (lnp_t - lnp_mid[-1][None]))
                    v = torch.where(k >= nlay, below, v)
                out[b] = v
        return out

    # ------------------------------------------------------------------
    # State construction / decoding
    # ------------------------------------------------------------------

    def decode_state(
        self, state: ACE2State, time_index: int = -1, only: "list[str] | None" = None
    ) -> dict[str, torch.Tensor]:
        """{name: tensor}: native layer families as (8, n_points) top->bottom,
        native singles as (n_points,), plus derived pressure-level families
        'z'/'t'/'q'/'u'/'v' as (n_plevs, n_points) surface-first, 'sp' (Pa,
        == PRESsfc), the lowest-native-level 't_lowest'/'q_lowest'/
        'p_lowest' (K, kg/kg, Pa; for ps_operator='ps_native'), and the
        canonical aliases geopotential/temperature/specific_humidity/
        surface_pressure. `time_index` is ignored (one
        time level)."""
        flat = state.state[0].reshape(len(self._channel_names), -1)
        want = None if only is None else set(only)
        out: dict[str, torch.Tensor] = {}
        for base, cols in self._layer_cols.items():
            if want is None or base in want:
                out[base] = flat[cols]
        for name, i in self._single_cols.items():
            if want is None or name in want:
                out[name] = flat[i]
        derived = set(_DERIVED_FROM) | {"z"} if want is None else want & (set(_DERIVED_FROM) | {"z"})
        if derived:
            out.update(self._to_pressure_levels(flat, derived))
        if want is None or "sp" in want:
            out["sp"] = flat[self._single_cols["PRESsfc"]]
        # 500 hPa geopotential (m^2/s^2) from ACE2's OWN decoder output h500,
        # not the hydrostatically derived 'z' -- h500 verifies at 2.8 m rms vs
        # ERA5 after 6h; the 8-layer hydrostatic z500 has a -19 m bias. The
        # driver's z500 diagnostic prefers 'z500' when a backend supplies it.
        # NaN on an initial state that hasn't been stepped (diagnostic channel).
        if want is None or "z500" in want:
            out["z500"] = GRAV * flat[self._single_cols["h500"]]
        # Lowest native layer (k=7, bottom), for ps_operator='ps_native':
        # its temperature, total water, and midpoint pressure (Pa) between
        # its top interface (ak[7] + bk[7]*ps) and the surface (bk[8] == 1).
        if want is None or "t_lowest" in want:
            out["t_lowest"] = flat[self._layer_cols["air_temperature"][-1]]
        if want is None or "q_lowest" in want:
            out["q_lowest"] = flat[self._layer_cols["specific_total_water"][-1]]
        if want is None or "p_lowest" in want:
            ps = flat[self._single_cols["PRESsfc"]]
            out["p_lowest"] = 0.5 * ((self._ak[-2] + self._bk[-2] * ps) + (self._ak[-1] + self._bk[-1] * ps))
        for base, alias in (("z", "geopotential"), ("t", "temperature"), ("q", "specific_humidity"), ("sp", "surface_pressure")):
            if base in out:
                out[alias] = out[base]
        return out

    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> ACE2State:
        """Pack `{"fields": {name: array}}` (arrays `(180, 360)` on this
        model's south-to-north grid, or flat) into a fresh state. Only the
        prognostic channels are required; output-only diagnostic channels
        are NaN at the initial time (they don't exist until one step has
        run, and never feed the network) unless supplied."""
        fields = input_state["fields"]
        arrs = []
        for name in self._channel_names:
            if name in fields:
                a = np.asarray(fields[name], dtype=np.float32).reshape(self._nlat, self._nlon)
            elif name in self._prognostic:
                raise KeyError(f"initial state is missing prognostic field {name!r}")
            else:
                a = np.full((self._nlat, self._nlon), np.nan, dtype=np.float32)
            arrs.append(a)
        tensor = torch.from_numpy(np.stack(arrs)[None]).to(self._device)
        return ACE2State(tensor, date)

    def prime_from_state(self, state: ACE2State) -> ACE2State:
        return state

    # ------------------------------------------------------------------
    # Differentiable rollout
    # ------------------------------------------------------------------

    def _advance_one_step(
        self, x: torch.Tensor, date: datetime.datetime, use_checkpoint: bool, latent_increment: torch.Tensor = None
    ) -> torch.Tensor:
        """One 6h step from `x` (valid at `date`) via fme's own Stepper.step.
        Forcings are read here, outside the checkpoint (constants w.r.t. the
        control); everything inside `_pure` is a side-effect-free function
        of (x, latent_increment), safe to recompute during backward."""
        f_now = self._forcing(date)
        f_next = self._forcing(date + self._timestep)
        forcing_in = {n: (f_next[n] if n in self._next_step_forcing else f_now[n]) for n in self._input_only}
        next_step = {n: f_next[n] for n in self._next_step_input}
        n_prog = len(self._prognostic)

        def _pure(x_, inc_):
            inputs = {n: x_[:, i] for i, n in enumerate(self._prognostic)}
            inputs.update(forcing_in)
            handle = None
            if inc_ is not None:
                handle = self._sfno.blocks[-1].register_forward_hook(lambda _m, _i, out: out + inc_)
            try:
                y = self.stepper.step(StepArgs(input=inputs, next_step_input_data=next_step))
            finally:
                if handle is not None:
                    handle.remove()
            return torch.stack([y[n] for n in self._channel_names], dim=1)

        if use_checkpoint:
            # use_reentrant=TRUE, deliberately (unlike FCN3/Aurora). With the
            # non-reentrant variant, the FIRST gradient-enabled checkpointed
            # call after any torch.no_grad() rollout (e.g. the background
            # forecast) returns a wrong latent gradient on CUDA -- cos 0.12
            # vs finite differences at 1 step, 110x wrong norm at 4 steps --
            # while every later call, a plain (non-checkpointed) call, and the
            # same sequence on CPU are all exact (probe_ace2_ckpt_grad.py,
            # probe_ace2_ckpt_first.py). Root cause not identified (fme/
            # torch_harmonics have no grad-mode-dependent forward state; the
            # CPU/CUDA split points at a lazily initialized CUDA resource).
            # Reentrant checkpointing backpropagates only through the graph
            # it rebuilds during backward, so a first-forward/recompute
            # disagreement can't reach the gradient; verified exact on GPU.
            # Needs .backward() (not torch.autograd.grad) -- what the solver uses.
            y = torch.utils.checkpoint.checkpoint(_pure, x[:, :n_prog], latent_increment, use_reentrant=True)
        else:
            y = _pure(x[:, :n_prog], latent_increment)
        return y

    def advance(
        self,
        state: ACE2State,
        steps: int = 1,
        use_checkpoint: bool = True,
        latent_increment: torch.Tensor = None,
    ) -> ACE2State:
        """Differentiable multi-step rollout; `latent_increment` (shape ==
        latent_shape) is applied on the first step only (the SFNO re-encodes
        from physical space every step, same as FCN3/AIFS)."""
        x = state.state
        date = state.date
        for i in range(steps):
            x = self._advance_one_step(x, date, use_checkpoint, latent_increment if i == 0 else None)
            date = date + self._timestep
        return ACE2State(x, date)
