"""Horizontal interpolation of model fields to observation locations: the
three bilinear schemes and the k-d-tree inverse-distance scheme
(backends/{fcn3,ace2,aifs}/*_grid.py), and get_grid_interpolator's choice
among them. Weights are float32, so values agree to ~1e-6 relative."""

import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import long_window_4dvar_utils as utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_grid_module(backend, name):
    """Load backends/<backend>/<name>.py by path (aifs_grid and fcn3_grid
    both define GridInterpolator, so keep them separate)."""
    spec = importlib.util.spec_from_file_location(
        f'_test_{name}', os.path.join(REPO_ROOT, 'backends', backend, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fcn3_grid = load_grid_module('fcn3', 'fcn3_grid')
ace2_grid = load_grid_module('ace2', 'ace2_grid')
aifs_grid = load_grid_module('aifs', 'aifs_grid')


def regular_grid(dlat=5.0, dlon=5.0):
    """FCN3-style: north-to-south rows including both poles, lon from 0."""
    lat1d = np.arange(90.0, -90.0 - 1e-9, -dlat)
    lon1d = np.arange(0.0, 360.0, dlon)
    return np.tile(lon1d, lat1d.size), np.repeat(lat1d, lon1d.size)


def gaussian_grid(nlat=36, nlon=72):
    """ACE2-style: south-to-north Gaussian rows, no pole rows."""
    lat1d = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))
    lon1d = np.arange(nlon) * 360.0 / nlon
    return np.tile(lon1d, nlat), np.repeat(lat1d, nlon)


def reduced_grid(nrow=36):
    """AIFS-style: north-to-south Gaussian rows, fewer points toward the
    poles (about the latitude spacing everywhere, so doubling nrow halves
    both spacings), each row starting at 0 deg."""
    rows = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(nrow)[0]))[::-1]
    counts = np.maximum(8, np.round(2 * nrow * np.cos(np.radians(rows)))).astype(int)
    lons = np.concatenate([np.arange(n) * 360.0 / n for n in counts])
    lats = np.concatenate([np.full(n, lat) for lat, n in zip(rows, counts)])
    return lons, lats


BILINEAR = {
    'regular': lambda: fcn3_grid.BilinearGridInterpolator(*regular_grid()),
    'gaussian': lambda: ace2_grid.RectilinearBilinearInterpolator(*gaussian_grid()),
    'reduced': lambda: aifs_grid.ReducedGaussianBilinearInterpolator(*reduced_grid()),
}
ALL = {**BILINEAR, 'kdtree': lambda: fcn3_grid.GridInterpolator(*gaussian_grid())}

RNG = np.random.default_rng(0)
# obs inside the grids' latitude range (Gaussian edge rows are at ~87.5 deg)
OB_LON = RNG.uniform(-180.0, 360.0, 200)
OB_LAT = RNG.uniform(-85.0, 85.0, 200)


def interpolate(interp, field, ob_lon=OB_LON, ob_lat=OB_LAT):
    idx, wts = interp.weights(ob_lon, ob_lat)
    return interp.interp(idx, wts, torch.as_tensor(field, dtype=torch.float64)).numpy()


@pytest.mark.parametrize('kind', ALL)
def test_weights_are_a_partition_of_unity(kind):
    interp = ALL[kind]()
    idx, wts = interp.weights(OB_LON, OB_LAT)
    assert idx.shape == wts.shape == (OB_LON.size, 4)
    assert int(idx.min()) >= 0 and int(idx.max()) < interp.model_lons.size
    assert float(wts.min()) >= 0.0
    torch.testing.assert_close(wts.sum(-1), torch.ones(OB_LON.size), atol=1e-6, rtol=0)


@pytest.mark.parametrize('kind', ALL)
def test_grid_points_reproduce_grid_values(kind):
    # (the k-d tree adds 1e-6 to every distance, so a point exactly on the
    # grid keeps ~0.9996 of the weight, not all of it)
    interp = ALL[kind]()
    pick = RNG.choice(interp.model_lons.size, 50, replace=False)
    field = RNG.normal(size=interp.model_lons.size)
    got = interpolate(interp, field, interp.model_lons[pick], interp.model_lats[pick])
    np.testing.assert_allclose(got, field[pick], rtol=0, atol=5e-3 if kind == 'kdtree' else 1e-5)


@pytest.mark.parametrize('kind', BILINEAR)
def test_bilinear_exact_for_linear_fields(kind):
    # a + b*lat + c*lon is reproduced exactly away from the 0/360 seam (the
    # field itself jumps there; the reduced grid's polar rows have seam cells
    # up to 45 deg wide, hence |lat| < 70); a + b*lat everywhere.
    interp = BILINEAR[kind]()
    lon, lat = interp.model_lons, interp.model_lats
    inner = (OB_LON % 360.0 > 30.0) & (OB_LON % 360.0 < 330.0) & (np.abs(OB_LAT) < 70.0)
    got = interpolate(interp, 3.0 + 0.5 * lat + 0.01 * lon, OB_LON[inner] % 360.0, OB_LAT[inner])
    np.testing.assert_allclose(got, 3.0 + 0.5 * OB_LAT[inner] + 0.01 * (OB_LON[inner] % 360.0), rtol=1e-5)
    got = interpolate(interp, 3.0 + 0.5 * lat)
    np.testing.assert_allclose(got, 3.0 + 0.5 * OB_LAT, rtol=1e-5)


@pytest.mark.parametrize('kind', ALL)
def test_longitude_wraps(kind):
    # the same place written as -0.4, 359.6 and 719.6 deg, on a periodic
    # field that changes fastest across the seam: using only the 355 deg
    # column (no wrap to 0 deg) would be ~0.08 off
    interp = ALL[kind]()
    lat = np.array([10.0, 10.0, 10.0])
    field = np.sin(np.radians(interp.model_lons)) * np.cos(np.radians(interp.model_lats))
    got = interpolate(interp, field, np.array([-0.4, 359.6, 719.6]), lat)
    np.testing.assert_allclose(got, got[0], rtol=1e-6, atol=1e-9)
    truth = np.sin(np.radians(-0.4)) * np.cos(np.radians(10.0))
    # bilinear is off by ~1e-5 here; inverse-distance weighting is only
    # first-order accurate (0.021 on this 5-deg grid)
    assert abs(got[0] - truth) < (0.05 if kind == 'kdtree' else 2e-3)


@pytest.mark.parametrize('kind', BILINEAR)
def test_bilinear_second_order_accurate(kind):
    # halving the grid spacing cuts the error on a smooth field ~4x (in the
    # asymptotic range: the 18-row reduced grid is still too coarse, 3x)
    grids = {'regular': (fcn3_grid.BilinearGridInterpolator, regular_grid, [(10.0, 10.0), (5.0, 5.0)]),
             'gaussian': (ace2_grid.RectilinearBilinearInterpolator, gaussian_grid, [(18, 36), (36, 72)]),
             'reduced': (aifs_grid.ReducedGaussianBilinearInterpolator, reduced_grid, [(36,), (72,)])}
    cls, make, sizes = grids[kind]
    f = lambda lon, lat: np.sin(np.radians(2 * lat)) * np.cos(np.radians(3 * lon))  # noqa: E731
    errors = []
    for size in sizes:
        lons, lats = make(*size)
        got = interpolate(cls(lons, lats), f(lons, lats), OB_LON, np.clip(OB_LAT, -75, 75))
        errors.append(np.abs(got - f(OB_LON, np.clip(OB_LAT, -75, 75))).max())
    assert 3.0 < errors[0] / errors[1] < 5.0


@pytest.mark.parametrize('kind', ['gaussian', 'reduced'])
def test_poleward_of_edge_row_clamps(kind):
    # no pole row: an ob at 89.9 deg takes the edge row's value
    interp = BILINEAR[kind]()
    edge = interp.model_lats.max()
    field = np.where(interp.model_lats == edge, 7.0, 0.0)
    got = interpolate(interp, field, np.array([0.0, 123.0]), np.array([89.9, 89.9]))
    np.testing.assert_allclose(got, 7.0, rtol=1e-6)


def test_kdtree_inverse_distance():
    # an ob midway between two equal-distance points weights them equally
    interp = fcn3_grid.GridInterpolator(np.array([0.0, 10.0, 180.0, 90.0]), np.array([0.0, 0.0, 0.0, 60.0]))
    idx, wts = interp.weights(np.array([5.0]), np.array([0.0]), k=2)
    assert sorted(idx[0].tolist()) == [0, 1]
    torch.testing.assert_close(wts[0], torch.tensor([0.5, 0.5]))


def test_constructors_reject_wrong_grids():
    lons, lats = gaussian_grid()
    with pytest.raises(ValueError, match='uniformly spaced'):
        fcn3_grid.BilinearGridInterpolator(lons, lats)
    with pytest.raises(ValueError, match='south-to-north'):
        ace2_grid.RectilinearBilinearInterpolator(*regular_grid())
    with pytest.raises(ValueError, match='north-to-south'):
        aifs_grid.ReducedGaussianBilinearInterpolator(lons, lats)


@pytest.mark.parametrize('backend, grid_interp, expected', [
    ('aifs', None, 'ReducedGaussianBilinearInterpolator'),
    ('aifs', 'kdtree', 'GridInterpolator'),
    ('fcn3', None, 'BilinearGridInterpolator'),
    ('aurora', None, 'BilinearGridInterpolator'),
    ('sfno', 'kdtree', 'GridInterpolator'),
    ('ace2', None, 'RectilinearBilinearInterpolator'),
    ('ace2', 'kdtree', 'GridInterpolator'),
])
def test_get_grid_interpolator_choice(backend, grid_interp, expected):
    grid = {'aifs': reduced_grid, 'ace2': gaussian_grid}.get(backend, regular_grid)()
    model = SimpleNamespace(lons=grid[0], lats=grid[1])
    exp = {'model_backend': backend, **({'grid_interp': grid_interp} if grid_interp else {})}
    assert type(utils.get_grid_interpolator(model, exp)).__name__ == expected


def test_get_grid_interpolator_rejects_unknown_kind():
    model = SimpleNamespace(lons=regular_grid()[0], lats=regular_grid()[1])
    with pytest.raises(ValueError, match='grid_interp'):
        utils.get_grid_interpolator(model, {'model_backend': 'fcn3', 'grid_interp': 'cubic'})
