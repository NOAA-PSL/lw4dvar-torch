"""Hidden-mesh node classes for class-dependent latent priors (latent_eof.py, control_prior q_cov/b_cov
with per-class covariances): latitude band (SH < -20, tropics, NH >= 20) x surface (ocean, land, ice/snow).

Surface class of a node on a date, from the O96 points nearest it: ice/snow if the sea-ice (siconc >= 0.5)
plus snow-covered (sd > 0.01 m w.e.) share is >= 0.5, else land if the mean lsm >= 0.5, else ocean.
lsm from the v8 zarr, siconc and sd from the 37-level zarr (both O96, same grid order).
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

Z37 = '/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr'
Z8 = ('/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/data/training/'
      'era5-o96-1979-2023-6h-v8.zarr')
SURFACES = ('ocean', 'land', 'ice/snow')
BANDS = ('SH', 'tropics', 'NH')


def _xyz(la, lo):
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class NodeClasses:
    """surface(date) -> (n_nodes,) 0 ocean / 1 land / 2 ice-snow; band -> (n_nodes,) 0 SH / 1 tropics / 2 NH;
    band_surface(date) -> band * 3 + surface (9 classes)."""

    def __init__(self, lat, lon):
        import zarr
        self.z37, z8 = zarr.open(Z37, 'r'), zarr.open(Z8, 'r')
        self.v37, v8 = list(self.z37.attrs['variables']), list(z8.attrs['variables'])
        glat, glon = np.radians(np.asarray(self.z37['latitudes'][:])), np.radians(np.asarray(self.z37['longitudes'][:]))
        self.n = len(lat)
        _, self.near = cKDTree(_xyz(np.radians(lat), np.radians(lon))).query(_xyz(glat, glon))  # grid point -> node
        self.cnt = np.bincount(self.near, minlength=self.n).clip(1)
        self.t0 = pd.Timestamp(self.z37.attrs['start_date'])
        self.lsm = np.bincount(self.near, np.asarray(z8['data'][0, v8.index('lsm'), 0]), self.n) / self.cnt
        self.band = np.digitize(np.asarray(lat), [-20, 20])

    def surface(self, date):
        i = int((pd.Timestamp(date).tz_localize(None) - self.t0) / pd.Timedelta('6h'))
        a = np.asarray(self.z37['data'][i, :, 0, :])
        cold = (np.nan_to_num(a[self.v37.index('siconc')]) >= 0.5) | (a[self.v37.index('sd')] > 0.01)
        icy = np.bincount(self.near, cold.astype(float), self.n) / self.cnt
        return np.where(icy >= 0.5, 2, np.where(self.lsm >= 0.5, 1, 0))

    def band_surface(self, date):
        return self.band * 3 + self.surface(date)
