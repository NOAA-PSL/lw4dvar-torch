"""The surface-pressure observation operators: get_surface_pressure
(`logpinterp`) and preduce (`ps`, `ps_native`)."""

import math

import pytest
import torch

import long_window_4dvar_utils as utils

T_ISO = 250.0  # K
P_REF = 1013.25  # hPa: pressure at zero geopotential
# surface-first, as every backend's pressure_levels()/decode_state() orders them
LEVELS = torch.tensor([1000.0, 850.0, 700.0, 500.0, 300.0, 100.0], dtype=torch.float64)


def isothermal_geopotential(p):
    """Geopotential (m^2/s^2) of pressure p in an isothermal atmosphere."""
    return utils.RD * T_ISO * torch.log(P_REF / p)


def isothermal_ps(phi_s):
    return P_REF * torch.exp(-phi_s / (utils.RD * T_ISO))


@pytest.mark.parametrize("height_m", [0.0, 300.0, 1500.0, 4000.0])
def test_logpinterp_exact_for_isothermal_atmosphere(height_m):
    # log p is linear in geopotential here, so interpolation and the linear
    # extrapolation below the 1000 hPa level (height 0: ps > 1000) are exact.
    phi_s = torch.tensor([height_m * utils.GRAV], dtype=torch.float64)
    z = isothermal_geopotential(LEVELS)[:, None]
    ps = utils.get_surface_pressure(LEVELS, z, phi_s)
    torch.testing.assert_close(ps, isothermal_ps(phi_s), rtol=1e-12, atol=0.0)


def test_logpinterp_vectorized_over_obs():
    heights = torch.tensor([0.0, 300.0, 1500.0, 4000.0], dtype=torch.float64)
    phi_s = heights * utils.GRAV
    z = isothermal_geopotential(LEVELS)[:, None].expand(-1, heights.numel())
    ps = utils.get_surface_pressure(LEVELS, z, phi_s)
    torch.testing.assert_close(ps, isothermal_ps(phi_s), rtol=1e-12, atol=0.0)


T0, LAPSE = 288.15, 0.0065  # K, K/m: standard-atmosphere troposphere
EXPONENT = utils.RD * LAPSE / utils.GRAV


def lapse_geopotential(p):
    """Geopotential of pressure p for a constant lapse rate from P_REF at z = 0."""
    return utils.GRAV * T0 / LAPSE * (1.0 - (p / P_REF) ** EXPONENT)


def lapse_ps(phi_s):
    return P_REF * (1.0 - LAPSE * phi_s / (utils.GRAV * T0)) ** (1.0 / EXPONENT)


@pytest.mark.parametrize("height_m", [300.0, 1500.0, 2500.0, 4000.0])
def test_logpinterp_uses_bracketing_levels(height_m):
    # With a lapse rate log p is curved in geopotential, so only the two
    # levels around the station interpolate well (within 1.6 hPa on these
    # levels). The level-order bug the AIFS port once shipped (searching the
    # surface-first levels as if ascending) picks the wrong pair: 89-324 hPa
    # off for these heights. (An isothermal atmosphere can't catch it: there
    # every pair of levels extrapolates exactly.)
    phi_s = torch.tensor([height_m * utils.GRAV], dtype=torch.float64)
    ps = utils.get_surface_pressure(LEVELS, lapse_geopotential(LEVELS)[:, None], phi_s)
    assert abs(float(ps - lapse_ps(phi_s))) < 3.0


def preduce(ps, dz, t=280.0, q=0.0):
    """preduce with the model surface dz metres above the station."""
    f = lambda v: torch.tensor([v], dtype=torch.float64)  # noqa: E731
    return utils.preduce(f(ps), f(700.0), f(t), f(q), f(dz), f(0.0))


def test_preduce_identity_at_same_height():
    assert float(preduce(1000.0, 0.0)) == pytest.approx(1000.0, rel=1e-14)


def test_preduce_hydrostatic_for_small_offset():
    # 10 m lower station: dp/p ~ g dz / (Rd Tv) with Tv the surface value
    ps, dz, t = 1000.0, 10.0, 280.0
    alpha = utils.RD * 0.0065 / utils.GRAV
    t_surf = t * (ps / 700.0) ** alpha
    expected = utils.GRAV * dz / (utils.RD * t_surf)
    got = float(preduce(ps, dz, t)) / ps - 1.0
    assert got == pytest.approx(expected, rel=1e-3)


def test_preduce_moisture_reduces_correction():
    # higher virtual temperature -> smaller pressure change for the same dz
    dry = float(preduce(1000.0, 500.0, q=0.0)) - 1000.0
    moist = float(preduce(1000.0, 500.0, q=0.015)) - 1000.0
    assert 0.0 < moist < dry


def test_preduce_sign():
    assert float(preduce(900.0, 200.0)) > 900.0  # station below the model surface
    assert float(preduce(900.0, -200.0)) < 900.0


def test_constants_consistent():
    assert utils.FV == pytest.approx(utils.RV / utils.RD - 1.0)
    assert math.isclose(utils.GRAV, 9.80665)
