"""Pressure-level fields from hybrid sigma-pressure layers, in torch.

Backend-neutral (numpy/torch only) version of the derivation in
`backends/ace2/ace2_model.ACE2Model._to_pressure_levels` -- same scheme,
parameterized by the layer interfaces instead of ACE2's packed columns, so
any hybrid-layer backend (the JAX NeuralGCM backend; ACE2 could be moved
onto it) can produce the solver's pressure-level families differentiably.

Layers are indexed top -> bottom, interfaces p_i = ak_i + bk_i * ps
(i = 0..nlay, bk_nlay = 1 so the bottom interface is the surface).
"""

import numpy as np
import torch

GRAV = 9.80665
RD = 287.05
RV = 461.5
LAPSE = 0.0065

# ERA5/WeatherBench 13 pressure levels (hPa), surface first.
DEFAULT_PLEVS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)


def to_pressure_levels(
    ps: torch.Tensor,
    temperature: torch.Tensor,
    humidity: torch.Tensor,
    phi_sfc: torch.Tensor,
    ak: torch.Tensor,
    bk: torch.Tensor,
    plevs_hpa: np.ndarray,
    others: "dict[str, torch.Tensor] | None" = None,
    want_z: bool = True,
) -> dict[str, torch.Tensor]:
    """{'z', 't', 'q', other names: (n_plevs, n_points)} on `plevs_hpa`.

    ps (P,) Pa; temperature/humidity (nlay, P) K, kg/kg; phi_sfc (P,)
    m^2/s^2; ak (nlay+1,) Pa, bk (nlay+1,). `others` maps extra names to
    (nlay, P) layer fields (u, v, ...) interpolated like q. 't' and 'q' are
    always returned; 'z' when `want_z`.

    Geopotential: hydrostatic integration up from phi_sfc through layer
    interfaces, each layer isothermal at its virtual temperature (phi linear
    in ln p within a layer); the top layer's Tv is extended above its top
    interface. Below ground (p > ps): standard-atmosphere extrapolation from
    a surface temperature derived from the lowest layer with a 6.5 K/km
    lapse rate (the ERA5 convention).

    t/q/others: linear in ln p between layer midpoints; held at the top
    layer's value above it; below the lowest midpoint, t follows the 6.5 K/km
    lapse rate and the rest are held constant.
    """
    T, q = temperature, humidity
    nlay = T.shape[0]
    n_points = ps.shape[0]
    p_int = ak[:, None] + bk[:, None] * ps[None, :]  # (nlay+1, P), top -> bottom
    lnp_int = torch.log(p_int[1:].clamp_min(1.0))  # interfaces 1..nlay
    lnp_t = torch.log(torch.as_tensor(np.asarray(plevs_hpa) * 100.0, dtype=ps.dtype, device=ps.device))
    lnp_t = lnp_t[:, None].expand(-1, n_points)  # (L, P)
    kappa = RD * LAPSE / GRAV
    out = {}

    tv = T * (1.0 + (RV / RD - 1.0) * q)
    p_mid_bot = 0.5 * (p_int[-2] + p_int[-1])
    t_sfc = T[-1] * (ps / p_mid_bot) ** kappa
    if want_z:
        dphi = RD * tv[1:] * (lnp_int[1:] - lnp_int[:-1])  # layers 1..nlay-1
        phi_int = torch.cat(
            [phi_sfc[None] + torch.flip(torch.cumsum(torch.flip(dphi, [0]), 0), [0]), phi_sfc[None]], 0
        )  # (nlay, P): interfaces 1..nlay
        idx = torch.searchsorted(lnp_int.T.contiguous(), lnp_t.T.contiguous()).T  # (L, P), 0..nlay
        j = idx.clamp(max=nlay - 1)  # layer containing the level; its base interface is phi_int[j]
        phi_above = torch.gather(phi_int, 0, j) + RD * torch.gather(tv, 0, j) * (torch.gather(lnp_int, 0, j) - lnp_t)
        tv_sfc = t_sfc * (1.0 + (RV / RD - 1.0) * q[-1])
        phi_below = phi_sfc[None] + GRAV * (tv_sfc[None] / LAPSE) * (
            1.0 - torch.exp(kappa * (lnp_t - torch.log(ps)[None]))
        )
        out["z"] = torch.where(idx >= nlay, phi_below, phi_above)

    lnp_mid = torch.log((0.5 * (p_int[:-1] + p_int[1:])).clamp_min(1.0))  # (nlay, P)
    k = torch.searchsorted(lnp_mid.T.contiguous(), lnp_t.T.contiguous()).T  # (L, P), 0..nlay
    lo = (k - 1).clamp(0, nlay - 1)
    hi = k.clamp(0, nlay - 1)
    x_lo, x_hi = torch.gather(lnp_mid, 0, lo), torch.gather(lnp_mid, 0, hi)
    w = torch.where(hi > lo, (lnp_t - x_lo) / (x_hi - x_lo), torch.zeros_like(lnp_t))

    def interp(f):
        return (1 - w) * torch.gather(f, 0, lo) + w * torch.gather(f, 0, hi)

    below = T[-1][None] * torch.exp(kappa * (lnp_t - lnp_mid[-1][None]))
    out["t"] = torch.where(k >= nlay, below, interp(T))
    out["q"] = interp(q)
    for name, f in (others or {}).items():
        out[name] = interp(f)
    return out
