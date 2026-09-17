"""
Historical ERA5 initial-condition/verification fetching for the FourCastNet3
4D-Var port.

Unlike AIFS (aifs_ic.py in the sibling long-window-4dvar-aifsv2 repo), there
is no anemoi `Runner` to derive MARS/CDS requests from a checkpoint's own
variable metadata -- FCN3's 72 channel names (config.json's `channel_names`)
are mapped to ERA5 MARS parameters directly here. This turns out to be
considerably simpler than the AIFS case in two ways:

  - FCN3's grid IS ERA5's own native grid (regular 0.25 deg, 721x1440,
    global) -- fetching at `grid: 0.25/0.25`, `area: 90/0/-90/359.75`
    requires no regridding at all, unlike AIFS's irregular N320 octahedral
    mesh (which anemoi's own input pipeline handles for that port).
  - FCN3 has no lagged time levels (n_history == 0) and no wave/soil-moisture
    variables with real ERA5 gaps to patch (see the AIFS port's
    test_forecast/patch_missing_wave_fields.py) -- every one of FCN3's 72
    channels is a completely standard ERA5 pressure-level or single-level
    field, confirmed by direct test fetches (see CLAUDE.md's "IC fetching"
    section) before writing this module, not assumed from variable names
    alone.

Fetches from `reanalysis-era5-complete` (`class: ea`, MARS-style request)
via `cdsapi`, the same dataset the AIFS port uses -- confirmed working with
this account's ~/.cdsapirc (shared across both repos: it is a per-user
dotfile, not per-conda-env). Requesting `format: netcdf` server-side avoids
needing eccodes/cfgrib installed in the `fcstnet3` env at all (just
`netCDF4`, via `xarray`, both of which are lighter and lower-risk to add
than AIFS's eccodes/earthkit.data stack). Needs internet access -- call
from a login node, never from the (offline) H100 compute nodes; see
fcn3_prefetch_ic.py.
"""

import datetime
import logging
import os

import cdsapi
import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

CDS_DATASET = "reanalysis-era5-complete"

# FCN3's 13 pressure levels (config.json's channel_names, e.g. "z500").
# Order here only affects the MARS request string, not the fields returned
# (read_single_date_fields reads the actual level values back out of the
# fetched dataset's own coordinate).
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# base name -> ERA5/MARS table-128 (or 228, for the 100m winds) paramId, for
# FCN3's 5 pressure-level families. Units/conventions match FCN3's own
# channels directly -- confirmed (not assumed) by comparing FCN3Model's
# in_bias for z500/z1000/z50 (55506 / 936 / 200918 m^2/s^2) against known
# geopotential magnitudes: FCN3's 'z' family is geopotential, the same
# quantity and units ERA5's own 'z' pressure-level field already is, so no
# unit conversion is needed anywhere in this module.
_PL_PARAMIDS = {
    "z": "129.128",
    "t": "130.128",
    "u": "131.128",
    "v": "132.128",
    "q": "133.128",
}

# ERA5 shortname -> MARS paramId, for FCN3's 7 single-level channels. ERA5's
# own shortnames for the wind-height fields (u10/v10/u100/v100) drop the
# trailing "m" FCN3's channel_names use (u10m/v10m/u100m/v100m) -- confirmed
# by a direct test fetch, not assumed; _SFC_RENAME below fixes this up.
_SFC_PARAMIDS = {
    "u10": "165.128",
    "v10": "166.128",
    "u100": "246.228",
    "v100": "247.228",
    "t2m": "167.128",
    "msl": "151.128",
    "tcwv": "137.128",
}
_SFC_RENAME = {"u10": "u10m", "v10": "v10m", "u100": "u100m", "v100": "v100m"}

# ERA5's own surface geopotential ("orography"), fetched alongside the sfc
# request purely for the ps-obs forward operator's station-elevation QC
# (get_surface_pressure/preduce need a "true" reference orography to compare
# station elevation against) -- exactly the role AIFS's own
# verif_ic['geopotential_at_surface'] plays (see long_window_4dvar_utils.py's
# get_verif). This is NOT one of FCN3's 72 model input channels -- FCN3 has
# no orography input channel at all; its own static orography comes from the
# checkpoint's bundled orography.nc, an entirely separate, not-necessarily-
# ERA5-identical field baked in at training time and used internally by
# Preprocessor2D, not something this fetch needs to reproduce. Confirmed by
# a direct test fetch that this comes back under the SAME shortname "z" as
# the pressure-level geopotential when combined into one sfc request
# (standard ECMWF convention: param 129 "Geopotential" keeps shortName "z"
# regardless of typeOfLevel) -- read_single_date_fields renames it to
# "geopotential_at_surface" specifically to avoid colliding with the
# pressure-level 'z' family's own flat keys (e.g. "z500").
_OROGRAPHY_PARAMID = "129.128"


def _cache_paths(cache_dir, date):
    tag = date.strftime("%Y%m%dT%H")
    return (
        os.path.join(cache_dir, f"era5_fcn3_{tag}_pl.nc"),
        os.path.join(cache_dir, f"era5_fcn3_{tag}_sfc.nc"),
    )


def _retrieve(client, request, target):
    tmp = target + ".tmp"
    client.retrieve(CDS_DATASET, request, tmp)
    os.replace(tmp, target)


def fetch_era5(date, cache_dir):
    """Fetch (or reuse cached) ERA5 pressure-level + single-level fields for
    `date`, on FCN3's own native grid -- no regridding needed. Needs
    internet access.

    Parameters
    ----------
    date : datetime.datetime

    Returns
    -------
    (pl_path, sfc_path) : the two cached netCDF file paths.
    """
    # MARS only understands whole-hour dates; drop tz/minutes/seconds
    # (FCN3Model requires tz-aware UTC dates for the zenith-angle forcing --
    # see fcn3_model.py -- but MARS requests are always implicitly UTC).
    date = datetime.datetime(date.year, date.month, date.day, date.hour)
    os.makedirs(cache_dir, exist_ok=True)
    pl_path, sfc_path = _cache_paths(cache_dir, date)

    need_pl = not os.path.exists(pl_path)
    need_sfc = not os.path.exists(sfc_path)
    if need_pl or need_sfc:
        client = cdsapi.Client()
        base = {
            "class": "ea",
            "date": date.strftime("%Y-%m-%d"),
            "expver": "1",
            "stream": "oper",
            "time": date.strftime("%H:00:00"),
            "type": "an",
            "grid": "0.25/0.25",
            "area": "90/0/-90/359.75",
            "format": "netcdf",
        }
        if need_pl:
            LOG.info("Fetching ERA5 pressure-level fields for %s", date)
            request = dict(
                base,
                levtype="pl",
                levelist="/".join(str(lev) for lev in PRESSURE_LEVELS),
                param="/".join(_PL_PARAMIDS.values()),
            )
            _retrieve(client, request, pl_path)
        if need_sfc:
            LOG.info("Fetching ERA5 single-level fields for %s", date)
            request = dict(
                base,
                levtype="sfc",
                param="/".join(list(_SFC_PARAMIDS.values()) + [_OROGRAPHY_PARAMID]),
            )
            _retrieve(client, request, sfc_path)
    else:
        LOG.info("Reusing cached ERA5 fields for %s", date)

    return pl_path, sfc_path


def read_single_date_fields(date, cache_dir):
    """Fetch (or reuse cached) ERA5 fields for `date` and return a flat
    `{field_name: (721, 1440) array}` dict, keyed by FCN3's own channel-name
    convention (e.g. "z500", "t2m") for the 72 model channels, plus
    "geopotential_at_surface" for the ps-obs QC field (see module docstring).

    Both `build_input_state` (a full 72-channel model state) and
    `long_window_4dvar_utils.get_verif` (which needs exactly these same
    flat fields, stacked into decode_state-shaped families) are built on top
    of this single flat fetch -- unlike AIFS, FCN3 has no lagged-vs-single-
    date distinction to maintain (n_history == 0), so there is only ever one
    fetch shape needed here, not aifs_ic.py's separate
    build_input_state/read_single_date_fields fetch paths.
    """
    pl_path, sfc_path = fetch_era5(date, cache_dir)
    fields = {}
    with xr.open_dataset(pl_path) as ds_pl:
        levels = ds_pl["pressure_level"].values.astype(int)
        for base in _PL_PARAMIDS:
            arr = ds_pl[base].isel(valid_time=0).values  # (n_levels, 721, 1440)
            for i, lev in enumerate(levels):
                fields[f"{base}{int(lev)}"] = np.asarray(arr[i], dtype=np.float32)
    with xr.open_dataset(sfc_path) as ds_sfc:
        for name in ds_sfc.data_vars:
            out_name = "geopotential_at_surface" if name == "z" else _SFC_RENAME.get(name, name)
            fields[out_name] = np.asarray(ds_sfc[name].isel(valid_time=0).values, dtype=np.float32)
    return fields


def build_input_state(date, cache_dir):
    """Fetch (or reuse cached) ERA5 fields for `date` and return
    `{"fields": {channel_name: (721, 1440) array}}`, ready for
    `FCN3Model.prepare_initial_state`. Drops "geopotential_at_surface" (not
    one of FCN3's 72 model channels -- QC-only, see module docstring).
    """
    fields = read_single_date_fields(date, cache_dir)
    model_fields = {k: v for k, v in fields.items() if k != "geopotential_at_surface"}
    return {"fields": model_fields}
