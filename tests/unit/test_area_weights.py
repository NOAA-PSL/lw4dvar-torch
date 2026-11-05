"""forecast_model.area_weights_from_lats on regular, Gaussian and reduced
grids, checked against exact spherical areas rather than the code's own
formula."""

import numpy as np
import pytest

from forecast_model import area_weights_from_lats

# Area-weighted mean of sin^2(lat) over the sphere.
SIN2_MEAN = 1.0 / 3.0


def flat_grid(row_lats, counts):
    return np.concatenate([np.full(n, lat) for lat, n in zip(row_lats, counts)])


def weighted_mean(f, w):
    return float(np.sum(f * w) / np.sum(w))


def test_mean_is_one():
    lats = flat_grid(np.linspace(-89.5, 89.5, 180), [360] * 180)
    assert area_weights_from_lats(lats).mean() == pytest.approx(1.0, rel=1e-12)


def test_regular_grid_matches_exact_cell_areas():
    rows = np.arange(-89.5, 90.0, 1.0)
    w = area_weights_from_lats(flat_grid(rows, [8] * rows.size))
    exact = np.sin(np.radians(rows + 0.5)) - np.sin(np.radians(rows - 0.5))
    exact = np.repeat(exact / exact.mean(), 8)
    np.testing.assert_allclose(w, exact, rtol=1e-12)


def test_regular_grid_with_pole_rows():
    # ERA5-style grid with pole rows: they own their polar caps (not zero area)
    rows = np.linspace(-90.0, 90.0, 37)
    lats = flat_grid(rows, [12] * rows.size)
    w = area_weights_from_lats(lats)
    cap = 1.0 - np.sin(np.radians(87.5))  # area fraction x2 of a 2.5-deg cap
    pole_share = w[lats == 90.0].sum() / w.sum()
    assert pole_share == pytest.approx(cap / 2.0, rel=1e-12)


def test_gaussian_grid_integrates_exactly():
    x, _ = np.polynomial.legendre.leggauss(32)
    rows = np.degrees(np.arcsin(x))
    lats = flat_grid(rows, [64] * rows.size)
    w = area_weights_from_lats(lats)
    assert weighted_mean(np.sin(np.radians(lats)) ** 2, w) == pytest.approx(SIN2_MEAN, rel=1e-12)


def test_reduced_grid_area_shares():
    # Octahedral-style rows: fewer points toward the poles (AIFS's N320).
    x, _ = np.polynomial.legendre.leggauss(40)
    rows = np.degrees(np.arcsin(x))
    counts = 20 + 4 * np.minimum(np.arange(40), np.arange(40)[::-1])
    lats = flat_grid(rows, counts)
    w = area_weights_from_lats(lats)
    assert weighted_mean(np.sin(np.radians(lats)) ** 2, w) == pytest.approx(SIN2_MEAN, rel=1e-12)
    # area poleward of 60 deg: 1 - sin(60) of the sphere, about 13.4 %
    poleward = np.abs(lats) > 60.0
    share = w[poleward].sum() / w.sum()
    band = np.abs(rows) > 60.0
    exact = (np.polynomial.legendre.leggauss(40)[1][band]).sum() / 2.0
    assert share == pytest.approx(exact, rel=1e-12)


def test_cos_lat_is_wrong_on_reduced_grid():
    # The bug the function replaced: cos(lat) per point double-counts the
    # latitude factor where rows already thin out. This test documents that
    # the checks above would catch it.
    x, _ = np.polynomial.legendre.leggauss(40)
    rows = np.degrees(np.arcsin(x))
    counts = 20 + 4 * np.minimum(np.arange(40), np.arange(40)[::-1])
    lats = flat_grid(rows, counts)
    coslat = np.cos(np.radians(lats))
    assert abs(weighted_mean(np.sin(np.radians(lats)) ** 2, coslat) - SIN2_MEAN) > 0.02
