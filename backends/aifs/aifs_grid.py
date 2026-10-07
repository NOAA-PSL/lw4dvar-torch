"""
Irregular-grid geometry helpers for the AIFS 4D-Var port.

AIFS's N320 reduced Gaussian grid is a flat list of 542080 (lat, lon)
points (640 latitude rows, north-to-south, 18..1280 equally spaced
longitudes per row), not a regular lat/lon grid -- `interp2d`'s bilinear
regular-grid assumption (long_window_4dvar_utils.py) doesn't apply. Two
interchangeable interpolators (selected by `grid_interp`):
ReducedGaussianBilinearInterpolator (row-wise bilinear, see its docstring)
and GridInterpolator, a k-nearest-neighbor inverse-distance-weighted
interpolator. Both are built once (per set of observation locations) from a k-d tree
over the model grid, so per-epoch cost inside the loss function is just a
fixed-index gather + fixed-weight dot product (trivially differentiable,
since the indices/weights don't depend on the model state).
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Unit-sphere Cartesian coordinates -- avoids the antimeridian/pole
    discontinuities of raw (lon, lat) nearest-neighbor search."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


class GridInterpolator:
    """k-d tree over the AIFS model grid, for interpolating model fields to
    arbitrary (lon, lat) observation locations."""

    def __init__(self, model_lons: np.ndarray, model_lats: np.ndarray):
        self.model_lons = np.asarray(model_lons, dtype=np.float64)
        self.model_lats = np.asarray(model_lats, dtype=np.float64)
        self.tree = cKDTree(_lonlat_to_xyz(self.model_lons, self.model_lats))
        self.coslat = np.cos(np.radians(self.model_lats)).astype(np.float32)

    def weights(self, ob_lon: np.ndarray, ob_lat: np.ndarray, k: int = 4, device=None):
        """Precompute k-NN indices + inverse-distance weights for a batch of
        observation locations. Returns (idx, wts) torch tensors of shape
        (n_obs, k); rows for out-of-range/padding obs are harmless (weights
        still sum to 1, values simply unused downstream since `used`/QC masks
        those rows out of the loss).

        Call once per window (like get_psobs), not per epoch.
        """
        pts = _lonlat_to_xyz(np.asarray(ob_lon, dtype=np.float64), np.asarray(ob_lat, dtype=np.float64))
        dist, idx = self.tree.query(pts, k=k)
        if k == 1:
            dist = dist[:, np.newaxis]
            idx = idx[:, np.newaxis]
        # chordal distance on a unit sphere -> inverse-distance weights
        # (small epsilon guards an observation that lands exactly on a grid point)
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum(axis=-1, keepdims=True)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        wts_t = torch.as_tensor(w, dtype=torch.float32, device=device)
        return idx_t, wts_t

    def interp(self, idx: torch.Tensor, wts: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Apply precomputed (idx, wts) to a field.

        field : (..., n_points) tensor (e.g. (n_levels, n_points) or (n_points,))
        idx, wts : (n_obs, k)
        returns : (..., n_obs)
        """
        gathered = field[..., idx]  # (..., n_obs, k)
        return (gathered * wts).sum(dim=-1)


class ReducedGaussianBilinearInterpolator:
    """Closed-form bilinear interpolation on a reduced Gaussian grid
    (AIFS's N320: 640 Gaussian latitude rows, north-to-south, row i holding
    nlon_i equally spaced longitudes starting at 0 deg -- 18 points at the
    polar rows up to 1280 at the equator; this is the classic reduced
    N320, not the octahedral O320, so row lengths are taken from the data
    rather than from the 20+4i octahedral formula).

    Per observation: linear-in-longitude interpolation within each of the
    two bracketing latitude rows (each with its own spacing), then linear
    in latitude between the two row values -- 4 points, the same (n_obs, 4)
    idx/wts as GridInterpolator.weights, so the two are interchangeable.
    Unlike k-NN inverse-distance weighting, this reproduces linear fields
    exactly. Poleward of the outermost rows (|lat| > 89.78 deg) the latitude
    index is clamped to the edge row, as in ace2_grid.
    """

    def __init__(self, model_lons: np.ndarray, model_lats: np.ndarray):
        model_lons = np.asarray(model_lons, dtype=np.float64)
        model_lats = np.asarray(model_lats, dtype=np.float64)
        starts = np.r_[0, np.flatnonzero(np.diff(model_lats) != 0) + 1]
        counts = np.diff(np.r_[starts, model_lats.size])
        row_lats = model_lats[starts]
        if not np.all(np.diff(row_lats) < 0):
            raise ValueError("model_lats must be grouped into rows ordered north-to-south")
        if np.any(counts < 2):
            raise ValueError("every latitude row needs at least 2 longitudes")
        lon0 = model_lons[starts]
        for s, n, l0 in zip(starts, counts, lon0):
            if not np.allclose(model_lons[s:s + n], l0 + np.arange(n) * (360.0 / n), atol=1e-6):
                raise ValueError(f"row starting at point {s} is not {n} equally spaced longitudes")

        self.nrow = starts.size
        self.row_starts = starts
        self.row_counts = counts
        self.row_lon0 = lon0
        # ascending copy for np.interp (fractional row index in -lat)
        self._neg_row_lats = -row_lats
        self.model_lons = model_lons
        self.model_lats = model_lats
        self.coslat = np.cos(np.radians(model_lats)).astype(np.float32)

    def _row_weights(self, rows: np.ndarray, ob_lon: np.ndarray):
        """Bracketing (flat index, flat index, weight on the second) along
        each observation's latitude row `rows`."""
        n = self.row_counts[rows]
        fj = ((ob_lon - self.row_lon0[rows]) % 360.0) * n / 360.0
        j0f = np.floor(fj)
        wj = fj - j0f
        j0 = j0f.astype(np.int64) % n
        j1 = (j0 + 1) % n
        s = self.row_starts[rows]
        return s + j0, s + j1, wj

    def weights(self, ob_lon: np.ndarray, ob_lat: np.ndarray, k: int = 4, device=None):
        """Precompute bilinear (idx, wts) for a batch of observation
        locations; same contract as GridInterpolator.weights."""
        if k != 4:
            raise ValueError("ReducedGaussianBilinearInterpolator only supports k=4 (the 4 bilinear corners)")
        ob_lon = np.asarray(ob_lon, dtype=np.float64)
        ob_lat = np.asarray(ob_lat, dtype=np.float64)

        # fractional row index (row 0 = northernmost); np.interp clamps to
        # the edge rows poleward of them
        fi = np.interp(-ob_lat, self._neg_row_lats, np.arange(self.nrow, dtype=np.float64))
        fi = np.minimum(fi, self.nrow - 1 - 1e-9)
        i0 = np.floor(fi).astype(np.int64)
        wi = fi - i0

        a0, a1, wa = self._row_weights(i0, ob_lon)
        b0, b1, wb = self._row_weights(i0 + 1, ob_lon)
        idx = np.stack([a0, a1, b0, b1], axis=-1)
        wts = np.stack([(1 - wi) * (1 - wa), (1 - wi) * wa, wi * (1 - wb), wi * wb], axis=-1)
        return (torch.as_tensor(idx, dtype=torch.long, device=device),
                torch.as_tensor(wts, dtype=torch.float32, device=device))

    def interp(self, idx: torch.Tensor, wts: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Apply precomputed (idx, wts) to a field. See GridInterpolator.interp."""
        return (field[..., idx] * wts).sum(dim=-1)
