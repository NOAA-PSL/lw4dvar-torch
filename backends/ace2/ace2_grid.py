"""
Bilinear observation-location interpolation on ACE2's F90 Gaussian grid.

fcn3_grid.BilinearGridInterpolator (FCN3/Aurora) requires a UNIFORMLY spaced
latitude axis and raises on ACE2's Legendre-Gauss latitudes (spacing varies
slightly, 180 rows, south-to-north, no pole rows). This is the same
closed-form bilinear scheme on a RECTILINEAR grid: the fractional latitude
index comes from the actual latitudes (np.interp over the index), longitude
is uniform and wraps exactly as in fcn3_grid. Same public interface
(weights()/interp(), (n_obs, 4) idx/wts, .coslat), so it drops in wherever
the solver uses a grid interpolator.

Poleward of the outermost Gaussian rows (|lat| > 89.24 deg) the latitude
index is clamped to the edge row -- the value there is the edge row's
(longitude-interpolated) value, since the grid has no pole row to
interpolate toward.
"""

import numpy as np
import torch


class RectilinearBilinearInterpolator:
    def __init__(self, model_lons: np.ndarray, model_lats: np.ndarray):
        model_lons = np.asarray(model_lons, dtype=np.float64)
        model_lats = np.asarray(model_lats, dtype=np.float64)
        nlat = np.unique(model_lats).size
        nlon = np.unique(model_lons).size
        if nlat * nlon != model_lons.size:
            raise ValueError(f"not a full rectilinear grid: {model_lons.size} points, {nlat} x {nlon} distinct lat/lon")
        lat1d = model_lats[::nlon]
        lon1d = model_lons[:nlon]
        if not np.array_equal(np.repeat(lat1d, nlon), model_lats):
            raise ValueError("model_lats is not a row-major repeat(lat1d, nlon) flattening")
        if not np.array_equal(np.tile(lon1d, nlat), model_lons):
            raise ValueError("model_lons is not a row-major tile(lon1d, nlat) flattening")
        if not np.all(np.diff(lat1d) > 0):
            raise ValueError("latitude axis must be strictly increasing (south-to-north)")
        dlon = lon1d[1] - lon1d[0]
        if not np.allclose(np.diff(lon1d), dlon, atol=1e-6):
            raise ValueError("longitude axis must be uniformly spaced")

        self.nlat = nlat
        self.nlon = nlon
        self.lat1d = lat1d
        self.lon0 = float(lon1d[0])
        self.dlon = float(dlon)
        self.lon_period = nlon * self.dlon
        self.model_lons = model_lons
        self.model_lats = model_lats
        self.coslat = np.cos(np.radians(model_lats)).astype(np.float32)

    def weights(self, ob_lon: np.ndarray, ob_lat: np.ndarray, k: int = 4, device=None):
        if k != 4:
            raise ValueError("RectilinearBilinearInterpolator only supports k=4 (the 4 bilinear corners)")
        ob_lon = np.asarray(ob_lon, dtype=np.float64)
        ob_lat = np.asarray(ob_lat, dtype=np.float64)

        # fractional row index, linear in latitude between the two bracketing
        # (unevenly spaced) rows; np.interp clamps to the edge rows
        fi = np.interp(ob_lat, self.lat1d, np.arange(self.nlat, dtype=np.float64))
        fi = np.minimum(fi, self.nlat - 1 - 1e-9)
        i0 = np.floor(fi).astype(np.int64)
        i1 = i0 + 1
        wi = fi - i0

        lon_rel = (ob_lon - self.lon0) % self.lon_period
        fj = lon_rel / self.dlon
        j0 = np.floor(fj).astype(np.int64) % self.nlon
        j1 = (j0 + 1) % self.nlon
        wj = fj - np.floor(fj)

        idx = np.stack([i0 * self.nlon + j0, i0 * self.nlon + j1, i1 * self.nlon + j0, i1 * self.nlon + j1], axis=-1)
        wts = np.stack([(1 - wi) * (1 - wj), (1 - wi) * wj, wi * (1 - wj), wi * wj], axis=-1)
        return (torch.as_tensor(idx, dtype=torch.long, device=device),
                torch.as_tensor(wts, dtype=torch.float32, device=device))

    def interp(self, idx: torch.Tensor, wts: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        return (field[..., idx] * wts).sum(dim=-1)
