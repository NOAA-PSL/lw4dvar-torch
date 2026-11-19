"""
Differentiable wrapper for AIFS checkpoints on hybrid model LAYERS (aifs-r2-l2: ERA5 32 layers from
the ACE2 data, eagle/long-window/PLAN_levels_norm_capacity.md, arm L2). Select with `exp.aifsml: True`.

Same as AIFS37Model (anemoi-models 0.19 fork, multi-dataset checkpoint, ICs and truth from the
training zarrs) except the vertical coordinate:

- Prognostic layer families t/q/u/v/cw_mlKK (KK = 00 top .. 31 bottom), interfaces
  p = ak + bk * sp (ak/bk from the D1 zarr's attrs). They are packed-state columns, so they (and
  sp etc.) are what `control_variables` names: `t_ml`, `q_ml`, `u_ml`, `v_ml`, `cw_ml`.
- `decode_state` also derives t/q/u/v/z on the 37 ERA5 pressure levels (1-1000 hPa, surface first),
  differentiably, with ACE2's method (backends/ace2/ace2_model.py `_to_pressure_levels`: hydrostatic
  z up from g * orog, layers isothermal at their Tv; t/q/u/v linear in ln p between layer midpoints;
  ERA5-style 6.5 K/km extrapolation below ground). So every pressure-level operator (raobs, ps, the
  A1-level `a1q70` ATMS decoder) and the ERA5 37-level verification work unchanged. Derived names have
  no columns: they can't be control variables.
- Derived singles: `tcw` = sum_k (q + cw)_k dp_k / g (the ATMS decoder's input) and
  `geopotential_at_surface` = g * orog (the model's own, smoother, orography, consistent with its sp).
- The checkpoint's 14-level pressure diagnostics (gh_/t_/q_/u_/v_ at 10-1000 hPa) are output-only and not
  in the packed state; they are not used here.
"""

import datetime

import numpy as np
import torch

import aifs37_model
from aifs37_model import ZARR_37, ZARR_STL1, ZARR_V8

GRAV = 9.80665
RD = 287.05
RV = 461.5
LAPSE = 0.0065
ZARR_ML32 = ("/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_ml32/chunks/"
             "era5-o96-ml32-1979-01-01T00-2023-12-31T18.zarr")
PLEVS37 = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
           250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1)
LAYER_FAMILIES = ("t", "q", "u", "v", "cw")
_DERIVED = ("t", "q", "u", "v")


class ERA5ZarrML(aifs37_model.ERA5Zarr):
    """The L2 training join, one date at a time: v8 (surface, forcings, tcw for verification), the
    37-level zarr (pressure-level t/q/u/v/z: verification is ERA5's own 37 levels, as for A1/T1), stl1,
    then D1's layers, sp and orog. `z` (surface geopotential) is replaced by g * orog."""

    def __init__(self, cache_size: int = 64):
        import zarr
        self._src = []
        # later sources override earlier ones: D1 last, but only its layers, sp and orog (its 14-level
        # diagnostics must not override the 37-level ERA5 truth, and L2 took 2t/msl/... from v8)
        d1_keep = lambda n: n in ("sp", "orog") or any(n.startswith(f + "_ml") for f in LAYER_FAMILIES)  # noqa: E731
        for path, keep in ((ZARR_V8, None), (ZARR_37, None), (ZARR_STL1, None), (ZARR_ML32, d1_keep)):
            z = zarr.open(path, "r")
            names = list(z.attrs["variables"])
            idx = np.array([i for i, n in enumerate(names) if keep is None or keep(n)])
            start = datetime.datetime.fromisoformat(str(z["dates"][0]).replace("T", " ")[:19])
            self._src.append((z["data"], [names[i] for i in idx], idx, start))
        d1 = zarr.open(ZARR_ML32, "r")
        self.ak = np.asarray(d1.attrs["ak_interfaces"], dtype=np.float64)
        self.bk = np.asarray(d1.attrs["bk_interfaces"], dtype=np.float64)
        self.latitudes = np.asarray(zarr.open(ZARR_V8, "r")["latitudes"][:])
        self.longitudes = np.asarray(zarr.open(ZARR_V8, "r")["longitudes"][:])
        if not (np.allclose(np.asarray(d1["latitudes"][:]), self.latitudes)
                and np.allclose(np.asarray(d1["longitudes"][:]), self.longitudes)):
            raise ValueError("D1 grid differs from v8's")
        self._cache: dict = {}
        self._cache_size = cache_size

    def fields(self, date: datetime.datetime) -> dict:
        new = date.replace(tzinfo=None) not in self._cache
        out = super().fields(date)
        if new:
            out["z"] = (GRAV * out["orog"]).astype(np.float32)
        return out


def to_pressure_levels(ps, T, q, u, v, ak, bk, phi_sfc, plevs_hpa, bases):
    """{base: (n_plevs, P)} on plevs_hpa (surface first) from layers (L, P) top -> bottom; ACE2's method
    (see the module docstring). ak, bk: (L+1,) tensors; ps, phi_sfc: (P,)."""
    nlay = T.shape[0]
    p_int = ak[:, None] + bk[:, None] * ps[None, :]                    # (L+1, P)
    lnp_int = torch.log(p_int[1:].clamp_min(1.0))                       # interfaces 1..L
    lnp_t = torch.log(torch.as_tensor(np.asarray(plevs_hpa, float) * 100.0, device=ps.device,
                                      dtype=ps.dtype))[:, None].expand(-1, ps.shape[0])
    kappa = RD * LAPSE / GRAV
    out = {}
    tv = T * (1.0 + (RV / RD - 1.0) * q)
    p_mid_bot = 0.5 * (p_int[-2] + p_int[-1])
    t_sfc = T[-1] * (ps / p_mid_bot) ** kappa
    if "z" in bases:
        dphi = RD * tv[1:] * (lnp_int[1:] - lnp_int[:-1])
        phi_int = torch.cat([phi_sfc[None] + torch.flip(torch.cumsum(torch.flip(dphi, [0]), 0), [0]), phi_sfc[None]], 0)
        idx = torch.searchsorted(lnp_int.T.contiguous(), lnp_t.T.contiguous()).T
        j = idx.clamp(max=nlay - 1)
        phi_above = torch.gather(phi_int, 0, j) + RD * torch.gather(tv, 0, j) * (torch.gather(lnp_int, 0, j) - lnp_t)
        tv_sfc = t_sfc * (1.0 + (RV / RD - 1.0) * q[-1])
        phi_below = phi_sfc[None] + GRAV * (tv_sfc[None] / LAPSE) * (1.0 - torch.exp(kappa * (lnp_t - torch.log(ps)[None])))
        out["z"] = torch.where(idx >= nlay, phi_below, phi_above)
    need = [b for b in _DERIVED if b in bases]
    if need:
        lnp_mid = torch.log((0.5 * (p_int[:-1] + p_int[1:])).clamp_min(1.0))
        k = torch.searchsorted(lnp_mid.T.contiguous(), lnp_t.T.contiguous()).T
        lo, hi = (k - 1).clamp(0, nlay - 1), k.clamp(0, nlay - 1)
        x_lo, x_hi = torch.gather(lnp_mid, 0, lo), torch.gather(lnp_mid, 0, hi)
        # safe denominator: where hi == lo the discarded branch would be 0/0, whose NaN gradient
        # (w.r.t. sp, through lnp_mid) torch.where does not mask
        same = hi <= lo
        den = torch.where(same, torch.ones_like(x_hi), x_hi - x_lo)
        w = torch.where(same, torch.zeros_like(lnp_t), (lnp_t - x_lo) / den)
        for b, f in (("t", T), ("q", q), ("u", u), ("v", v)):
            if b not in need:
                continue
            val = (1 - w) * torch.gather(f, 0, lo) + w * torch.gather(f, 0, hi)
            if b == "t":
                val = torch.where(k >= nlay, T[-1][None] * torch.exp(kappa * (lnp_t - lnp_mid[-1][None])), val)
            out[b] = val
    return out


class AIFSMLModel(aifs37_model.AIFS37Model):
    """AIFS37Model for hybrid-layer checkpoints. Same constructor arguments."""

    plevs = PLEVS37

    def __init__(self, *args, **kwargs):
        orig = aifs37_model.ERA5Zarr
        aifs37_model.ERA5Zarr = ERA5ZarrML      # AIFS37Model.__init__ builds self.era5 = ERA5Zarr()
        try:
            super().__init__(*args, **kwargs)
        finally:
            aifs37_model.ERA5Zarr = orig
        # layer families from the input variables; AIFS37Model grouped them as base 't_ml' etc.
        self._layer_cols: dict[str, list[int]] = {}
        for fam in LAYER_FAMILIES:
            cols = sorted((int(n.rsplit("_ml", 1)[1]), i) for n, i in self.var_to_idx.items()
                          if n.startswith(fam + "_ml") and n.rsplit("_ml", 1)[1].isdigit())
            if cols:
                if [k for k, _ in cols] != list(range(len(cols))):
                    raise ValueError(f"{fam}: layers {[k for k, _ in cols]} are not 0..n-1")
                self._layer_cols[fam + "_ml"] = [i for _, i in cols]
        nl = len(self._layer_cols["t_ml"])
        if len(self.era5.ak) != nl + 1:
            raise ValueError(f"{nl} layers vs {len(self.era5.ak)} ak interfaces")
        self._levels_by_base = {b: [(p, None) for p in self.plevs] for b in ("z",) + _DERIVED}
        # virtual singles (derived in decode_state; no columns) + the real ones, minus the layer bases
        singles = {n: i for n, i in self._single_by_base.items() if not n.endswith("_ml")}
        singles.pop("geopotential_at_surface", None)
        self._single_by_base = {**singles, "tcw": None, "geopotential_at_surface": None}
        self._ak = torch.as_tensor(self.era5.ak, dtype=torch.float32, device=self.device)
        self._bk = torch.as_tensor(self.era5.bk, dtype=torch.float32, device=self.device)
        self._orog_col = self.var_to_idx["orog"]
        self._sp_col = self.var_to_idx["sp"]

    def resolve_columns(self, name: str) -> list[int]:
        if name in self._layer_cols:
            return list(self._layer_cols[name])
        if name in self._levels_by_base or self._single_by_base.get(name, 0) is None:
            raise KeyError(f"{name!r} is derived from the layers (no packed-state column); "
                           f"control the layer families {sorted(self._layer_cols)} instead")
        if name in self._single_by_base:
            return [self._single_by_base[name]]
        if name in self.var_to_idx:
            return [self.var_to_idx[name]]
        raise KeyError(f"{name!r} is not a known variable or family")

    def is_known_variable(self, name: str) -> bool:
        return name in self._layer_cols or name in self._levels_by_base or name in self._single_by_base \
            or name in self.var_to_idx

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        if base not in self._levels_by_base:
            raise KeyError(f"{base!r} is not a derived pressure-level family")
        return np.array(self.plevs, dtype=np.float32)

    def decode_state(self, aifs_state, time_index: int = -1, only=None) -> dict:
        """Layer families (L, P) top -> bottom, real singles (P,), derived pressure-level t/q/u/v/z
        (37, P) surface first, derived tcw and geopotential_at_surface, and the canonical aliases."""
        x = aifs_state.state[0, time_index]            # (P, n_vars)
        want = None if only is None else set(only)
        if want is not None and {"geopotential", "temperature", "specific_humidity", "surface_pressure"} & want:
            want |= {"z", "t", "q", "sp"}
        out = {}
        for fam, cols in self._layer_cols.items():
            if want is None or fam in want:
                out[fam] = x[:, cols].T
        for name, i in self._single_by_base.items():
            if i is not None and (want is None or name in want):
                out[name] = x[:, i]
        ps = x[:, self._sp_col]
        derived = set(_DERIVED) | {"z"} if want is None else want & (set(_DERIVED) | {"z"})
        if derived:
            L = self._layer_cols
            out.update(to_pressure_levels(ps, x[:, L["t_ml"]].T, x[:, L["q_ml"]].T, x[:, L["u_ml"]].T,
                                          x[:, L["v_ml"]].T, self._ak.to(x.dtype), self._bk.to(x.dtype),
                                          GRAV * x[:, self._orog_col], self.plevs, derived))
        if want is None or "tcw" in want:
            w = x[:, self._layer_cols["q_ml"]].T
            if "cw_ml" in self._layer_cols:
                w = w + x[:, self._layer_cols["cw_ml"]].T
            dp = self._ak.to(x.dtype).diff()[:, None] + self._bk.to(x.dtype).diff()[:, None] * ps[None]
            out["tcw"] = (w * dp).sum(0) / GRAV
        if want is None or "geopotential_at_surface" in want:
            out["geopotential_at_surface"] = GRAV * x[:, self._orog_col]
        for base, alias in (("z", "geopotential"), ("t", "temperature"), ("q", "specific_humidity"),
                            ("sp", "surface_pressure")):
            if base in out:
                out[alias] = out[base]
        return out
