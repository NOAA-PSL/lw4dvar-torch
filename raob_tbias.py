"""Static radiosonde temperature bias correction, a RICH/RISE-like adjustment anchored to ERA5 (2026-11-24).

ERA5 assimilated radiosonde temperatures adjusted with RICH (break-adjusted station biases) plus a solar-elevation
dependent term (RISE). Our PREPBUFR raobs carry no such adjustment (their TOB event stacks hold only the original value
and the VIRTMP step), while every satellite bias correction in lwatms is fitted to ERA5. This module removes a static
estimate of each sounding's bias relative to ERA5 so the radiosondes agree with the rest of the observing system.

The table (written by fit_raob_tbias.py, fitted on a period that is NOT the one being cycled, e.g. 2016 for 2015):
    {"meta": {...},
     "levels": [1000, 925, ...],                      mandatory levels, hPa
     "classes": ["night", "twilight", "day"],         solar elevation < -5 deg, -5..15 deg, > 15 deg
     "bands": [-90, -60, -30, -10, 10, 30, 60, 90],   latitude bands for the fallback
     "station": {"<station_id>": [[c_night, c_twilight, c_day] per level]},   K, NaN = none
     "band":    [[[c per class] per level] per band]}                         K
A sounding's correction is the station's value for the nearest table level (log-p, within 15%) and its class; where the station has none, the value of
its latitude band; above/below the table's levels, zero. The corrected observation is  T - correction.

Usage:
    tb = RaobTBias('figures/raob_tbias/raob_tbias_2016.json')
    t_corrected = t - tb(station_id, lat, lon, time_datetime64, pressure_hpa)
"""
import json

import numpy as np

CLASS_EDGES = (-5.0, 15.0)   # deg: night < -5 <= twilight < 15 <= day


def solar_elevation(lat, lon, time):
    """Solar elevation (deg), simple NOAA-style formula (good to ~0.5 deg). time: datetime64 (any precision)."""
    t = np.datetime64(time, 's')
    day0 = t.astype('datetime64[Y]')
    doy = (t - day0).astype('timedelta64[s]').astype(np.int64) / 86400.0   # fractional day of year, from 0
    gamma = 2 * np.pi / 365.0 * doy
    dec = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
           + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                    - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))          # minutes
    utc_min = (doy - np.floor(doy)) * 1440.0
    lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    ha = np.deg2rad((utc_min + eqt + 4.0 * lon) / 4.0 - 180.0)
    la = np.deg2rad(np.asarray(lat, dtype=np.float64))
    return np.rad2deg(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))


def solar_class(elevation):
    """0 night, 1 twilight, 2 day."""
    return np.digitize(elevation, CLASS_EDGES)


class RaobTBias:
    def __init__(self, path):
        tab = json.load(open(path))
        self.meta = tab.get('meta', {})
        self.levels = np.asarray(tab['levels'], dtype=np.float64)
        self.bands = np.asarray(tab['bands'], dtype=np.float64)
        self.station = {k: np.asarray(v, dtype=np.float64) for k, v in tab['station'].items()}
        self.band = np.asarray(tab['band'], dtype=np.float64)          # (n_band, n_level, 3)
        self.n_station_hits = 0
        self.n_band_hits = 0
        self.n_none = 0

    def __call__(self, station_id, lat, lon, time, pressure_hpa):
        # nearest table level in log-pressure; levels between mandatory ones (975, 950, 225 hPa ...) take their
        # neighbour's value, nothing further than 15% in pressure (no extrapolation above 10 hPa)
        d = np.abs(np.log(self.levels / max(float(pressure_hpa), 1e-3)))
        li = int(np.argmin(d))
        if d[li] > 0.15:
            self.n_none += 1
            return 0.0
        cls = int(solar_class(solar_elevation(lat, lon, time)))
        row = self.station.get(str(station_id).strip())
        if row is not None and np.isfinite(row[li, cls]):
            self.n_station_hits += 1
            return float(row[li, cls])
        bi = int(np.clip(np.searchsorted(self.bands, lat, side='right') - 1, 0, len(self.bands) - 2))
        c = self.band[bi, li, cls]
        if np.isfinite(c):
            self.n_band_hits += 1
            return float(c)
        self.n_none += 1
        return 0.0

    def summary(self):
        return (f'raob T bias table {self.meta.get("fit_period", "?")}: {self.n_station_hits} station, '
                f'{self.n_band_hits} band, {self.n_none} uncorrected values')
