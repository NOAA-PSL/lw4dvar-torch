"""Static per-scene bias correction for the radianced MW / IR decoders (2026-11-21).

The decoders are fitted to ERA5 pairs, but their residual d = y - H(x_ERA5) still has a mean that depends on the scene
(cloud, surface, latitude, season), largest at high latitudes over cloudy ocean and over snow, where it pulls the
analysis (lwatms/figures/hl_ocean). This module removes it with a linear model per sensor, scene and channel,

    bias(ob) = sum_k beta[scene, channel, k] * p_k(ob),

whose predictors come only from the observation file (no model state), so the correction is fixed per ob and applied once
when the obs are read, like `extra_bc`:
    1, clw, clw^2                 GSI ret_amsua cloud liquid water from the obs (clw_obs; MW only)
    si                            scattering index Tb(89) - Tb(165) of the obs (ATMS only)
    s, s^2                        scan angle / 60 deg
    P1, P2, P3 (sin lat)          Legendre in sin(latitude)
    cos d, sin d, P1 cos d, P1 sin d   season (d = 2 pi day-of-year / 365.25), and its hemispheric flip
    sat_k (k = 1..nsat-1)         per-satellite offsets
Scenes as in long_window_4dvar_utils._atms_scene_codes (ocean_clear, land, snow, seaice, ocean_cloudy).

Coefficients are fitted on the 2015-16 ERA5 pair files by a37tmp/scene_bc/fit_scene_bc.py and stored as JSON
({sensor, target, predictors, scenes, channels, beta[scene][channel][k], ...}). `apply(json, frame, date)` returns the
(nobs, nch) bias to subtract from y (NaN-free; 0 where a scene has no fit).
"""
import json
import math

import numpy as np

SCENES = ('ocean_clear', 'land', 'snow', 'seaice', 'ocean_cloudy')
NSAT = {'atms': 2, 'amsua': 6, 'iasi': 3}


def scene_codes(wf, ice, snow, clw, water_frac_min=0.99, clw_max=0.05):
    """As long_window_4dvar_utils._atms_scene_codes (index into SCENES, -1 = mixed)."""
    land = wf <= 0.01
    code = np.full(len(wf), -1, dtype=np.int8)
    code[(wf >= water_frac_min) & (ice <= 0) & (clw <= clw_max)] = 0
    code[(wf >= water_frac_min) & (ice <= 0) & (clw > clw_max)] = 4
    code[land & (snow <= 0) & (ice <= 0)] = 1
    code[land & (snow > 0)] = 2
    code[ice >= 0.5] = 3
    return code


def predictor_names(sensor, opts=None):
    opts = opts or {}
    n = ['1']
    if sensor in ('atms', 'amsua'):
        n += ['clw', 'clw2'] + (['P1clw'] if opts.get('clw_lat') else [])
    if sensor == 'atms':
        n += ['si']
    n += ['scan', 'scan2', 'P1', 'P2', 'P3', 'cosd', 'sind', 'P1cosd', 'P1sind']
    n += [f'sat{k}' for k in range(1, NSAT[sensor])]
    return n


def predictors(sensor, lat, scan, sat, doy, clw=None, tb89=None, tb165=None, opts=None):
    """(nobs, npred) float64 in predictor_names order. clw (kg/m2) for MW; tb89/tb165 (K, bias-corrected obs) for ATMS.
    opts (the table's 'options', 2026-11-23): clw_cap (default 3.0; lower = no extrapolation of clw^2 into the sparse
    heavy-cloud tail), clw_lat (add P1(sin lat) x clw: cloud slope varying with latitude)."""
    opts = opts or {}
    x = np.sin(np.deg2rad(np.asarray(lat, np.float64)))
    s = np.asarray(scan, np.float64) / 60.0
    d = 2 * math.pi * np.asarray(doy, np.float64) / 365.25
    cols = [np.ones_like(x)]
    if sensor in ('atms', 'amsua'):
        c = np.clip(np.nan_to_num(np.asarray(clw, np.float64), nan=0.0), 0.0, float(opts.get('clw_cap', 3.0)))
        cols += [c, c ** 2] + ([x * c] if opts.get('clw_lat') else [])
    if sensor == 'atms':
        si = np.nan_to_num(np.asarray(tb89, np.float64) - np.asarray(tb165, np.float64), nan=0.0)
        cols += [np.clip(si, -20.0, 60.0) / 10.0]
    p1, p2, p3 = x, 0.5 * (3 * x ** 2 - 1), 0.5 * (5 * x ** 3 - 3 * x)
    cols += [s, s ** 2, p1, p2, p3, np.cos(d) * np.ones_like(x), np.sin(d) * np.ones_like(x), p1 * np.cos(d), p1 * np.sin(d)]
    sat = np.asarray(sat).astype(int)
    cols += [(sat == k).astype(np.float64) for k in range(1, NSAT[sensor])]
    return np.stack(cols, 1)


def load(path):
    t = json.load(open(path))
    t['beta'] = np.asarray(t['beta'], np.float64)   # (scene, channel, npred)
    t.setdefault('options', {})
    return t


def apply(table, scene, P):
    """Bias (nobs, nch) for scene codes (nobs,) and predictors P (nobs, npred); 0 for scene -1 or unfitted."""
    beta = table['beta']
    out = np.zeros((len(scene), beta.shape[1]))
    for k in range(beta.shape[0]):
        m = scene == k
        if m.any():
            out[m] = P[m] @ beta[k].T
    return out
