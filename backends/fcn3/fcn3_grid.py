"""
Grid geometry helpers for the FourCastNet3 4D-Var port.

`GridInterpolator` (k-d-tree/k-NN) was copied verbatim (2026-09-16, renamed
only) from the AIFS v2 port's aifs_grid.py, and takes only flat (lons, lats)
arrays with no assumption about grid regularity -- appropriate for AIFS's
irregular N320 octahedral grid. FCN3's grid, unlike AIFS's, is a *regular*
0.25 deg equiangular 721x1440 lat/lon grid (confirmed by direct inspection
of fourcastnet3/orography.nc's coordinates: `latitude` 90.0 -> -90.0,
`longitude` 0.0 -> 359.75, both uniformly spaced at exactly 0.25 deg, the
721-point latitude axis literally including both poles as real grid rows).
`BilinearGridInterpolator` (2026-09-17, ported from the NeuralGCM
long-window-4dvar-psobs repo's `interp2d` in long_window_4dvar_utils.py --
same bilinear math, rewritten from a single jax.jit whole-array call into
this repo's precompute-once-per-window (idx, wts) torch design) exploits
that regularity for an O(1) closed-form index lookup per observation
instead of a k-d-tree query, and is a drop-in replacement: same
`weights()`/`interp()` call signature as `GridInterpolator`, same (n_obs,
k) tensor shapes. Unlike `interp2d`, no pole-padding/pole-mean handling is
needed here -- FCN3's 721-point latitude axis already includes lat=+-90 as
real grid rows (see above), so ordinary bilinear clamping at the two ends
of the axis is already exact, not an approximation.
FCN3Model.lats/.lons flatten the (721, 1440) grid row-major (lat slowest,
lon fastest) to feed this module a consistent flat-point convention.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def _gather_weighted(field: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """Shared by GridInterpolator.interp/BilinearGridInterpolator.interp:
    apply precomputed (idx, wts) of shape (n_obs, k) to a field of shape
    (..., n_points), returning (..., n_obs)."""
    gathered = field[..., idx]  # (..., n_obs, k)
    return (gathered * wts).sum(dim=-1)


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
    """k-d tree over the model grid, for interpolating model fields to
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
        return _gather_weighted(field, idx, wts)


class BilinearGridInterpolator:
    """Bilinear interpolation on a regular, uniformly-spaced equiangular
    lat/lon grid (FCN3's native 721x1440 grid) -- an O(1) closed-form index
    lookup per observation instead of GridInterpolator's k-d-tree query.
    Ported from the NeuralGCM long-window-4dvar-psobs repo's `interp2d`;
    see this module's docstring for why no pole-padding is needed here.

    Same public interface as GridInterpolator (weights()/interp(), same
    tensor shapes) so it's a drop-in replacement, not a parallel API.
    """

    def __init__(self, model_lons: np.ndarray, model_lats: np.ndarray):
        model_lons = np.asarray(model_lons, dtype=np.float64)
        model_lats = np.asarray(model_lats, dtype=np.float64)
        if model_lons.shape != model_lats.shape:
            raise ValueError("model_lons and model_lats must have the same shape")

        # Recover the two 1D axes from the flattened (lat slowest, lon
        # fastest) grid -- nlon/nlat from the distinct-value counts, axis
        # values from the first repeat/tile cycle (in original, not sorted,
        # order -- order matters for the flat-index arithmetic below).
        nlat = np.unique(model_lats).size
        nlon = np.unique(model_lons).size
        if nlat * nlon != model_lons.size:
            raise ValueError(
                "BilinearGridInterpolator requires a full regular lat/lon "
                f"grid: got {model_lons.size} points but {nlat} x {nlon} "
                "distinct lat/lon values"
            )
        lat1d = model_lats[::nlon]
        lon1d = model_lons[:nlon]
        # Defensive check, not just assumed: confirms the input really is
        # this row-major repeat(lat1d, nlon) / tile(lon1d, nlat) flattening
        # (the same convention FCN3Model.lats/.lons and GridInterpolator's
        # docstring both document) before trusting closed-form arithmetic
        # on it -- raises clearly instead of silently interpolating wrong
        # values if a future caller ever hands this an irregular grid.
        if not np.array_equal(np.repeat(lat1d, nlon), model_lats):
            raise ValueError("model_lats is not a row-major repeat(lat1d, nlon) flattening")
        if not np.array_equal(np.tile(lon1d, nlat), model_lons):
            raise ValueError("model_lons is not a row-major tile(lon1d, nlat) flattening")

        dlat = lat1d[1] - lat1d[0]
        dlon = lon1d[1] - lon1d[0]
        if not np.allclose(np.diff(lat1d), dlat, atol=1e-6):
            raise ValueError("BilinearGridInterpolator requires a uniformly spaced latitude axis")
        if not np.allclose(np.diff(lon1d), dlon, atol=1e-6):
            raise ValueError("BilinearGridInterpolator requires a uniformly spaced longitude axis")

        self.nlat = nlat
        self.nlon = nlon
        self.lat0 = float(lat1d[0])
        self.lon0 = float(lon1d[0])
        self.dlat = float(dlat)
        self.dlon = float(dlon)
        self.lon_period = self.nlon * self.dlon  # 360 for a full global longitude axis
        self.model_lons = model_lons
        self.model_lats = model_lats
        self.coslat = np.cos(np.radians(model_lats)).astype(np.float32)

    def weights(self, ob_lon: np.ndarray, ob_lat: np.ndarray, k: int = 4, device=None):
        """Precompute the 4 surrounding grid corners + bilinear weights for
        a batch of observation locations. Returns (idx, wts) torch tensors
        of shape (n_obs, 4) -- same shape/dtype convention as
        GridInterpolator.weights, so the two are interchangeable at any
        call site.

        Call once per window (like get_psobs), not per epoch.
        """
        if k != 4:
            raise ValueError("BilinearGridInterpolator only supports k=4 (the 4 bilinear corners)")
        ob_lon = np.asarray(ob_lon, dtype=np.float64)
        ob_lat = np.asarray(ob_lat, dtype=np.float64)

        # Fractional latitude index, clamped to the grid's inclusive range.
        # FCN3's latitude axis already spans pole-to-pole as real grid rows
        # (see module docstring), so clamping at the two ends is exact, not
        # an approximation the way it would be on a Gaussian grid missing
        # the poles.
        fi = (ob_lat - self.lat0) / self.dlat
        fi = np.clip(fi, 0.0, self.nlat - 1 - 1e-9)
        i0 = np.floor(fi).astype(np.int64)
        i1 = i0 + 1
        wi = fi - i0

        # Fractional longitude index, wrapped modulo the full 360 deg
        # period -- handles the antimeridian the same way interp2d's
        # explicit wrap-around column did, but without needing to
        # materialize one.
        lon_rel = (ob_lon - self.lon0) % self.lon_period
        fj = lon_rel / self.dlon
        j0 = np.floor(fj).astype(np.int64) % self.nlon
        j1 = (j0 + 1) % self.nlon
        wj = fj - np.floor(fj)

        idx00 = i0 * self.nlon + j0
        idx01 = i0 * self.nlon + j1
        idx10 = i1 * self.nlon + j0
        idx11 = i1 * self.nlon + j1
        w00 = (1 - wi) * (1 - wj)
        w01 = (1 - wi) * wj
        w10 = wi * (1 - wj)
        w11 = wi * wj

        idx = np.stack([idx00, idx01, idx10, idx11], axis=-1)
        wts = np.stack([w00, w01, w10, w11], axis=-1)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        wts_t = torch.as_tensor(wts, dtype=torch.float32, device=device)
        return idx_t, wts_t

    def interp(self, idx: torch.Tensor, wts: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Apply precomputed (idx, wts) to a field. See GridInterpolator.interp."""
        return _gather_weighted(field, idx, wts)
