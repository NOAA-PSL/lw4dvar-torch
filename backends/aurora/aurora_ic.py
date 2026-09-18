"""
Historical ERA5 initial-condition/verification fetching for the Aurora
4D-Var port.

Follows fcn3_ic.py's approach closely (CDS `reanalysis-era5-complete` via
`cdsapi`, `format: netcdf` server-side so no eccodes/cfgrib is needed in
the `aurora` env, no anemoi `Runner` to lean on) -- adapted for two real
differences from FCN3:

  - Aurora needs TWO lagged time levels as input (`max_history_size=2`,
    like AIFS, not self-starting like FCN3) -- `build_input_state` fetches
    `date - timestep` and `date` and stacks them along a new leading axis,
    unlike FCN3's single-date fetch.
  - Aurora has no separate "orography" fetch requirement the way FCN3
    does: `AuroraModel.decode_state`'s `geopotential_at_surface` is
    sourced from the checkpoint's own bundled static `z` field (like
    AIFS's own checkpoint-native orography column), not a fresh ERA5
    fetch -- see aurora_model.py's decode_state docstring for the
    not-yet-independently-verified assumption this rests on (that
    Aurora's static z matches real ERA5-consistent orography).

Of Aurora's 26 surface variables, only 18 are real ERA5 fetch targets:
7 are output-only (predicted by the model, never present in real input --
`_OUTPUT_ONLY_SURF_VARS` in aurora_model.py; Aurora's own
`_pre_encoder_hook` zero-pads them unconditionally regardless of what's
fed in, so this module never fetches them at all) and `insolation` is
computed analytically from lat/lon/time (`aurora.insolation.insolation`),
not fetched.

**Verified against a real test fetch (2015-01-01T00), not just assumed**,
matching fcn3_ic.py's own discipline: all 18 param IDs resolved to the
expected variable names with physically sane values (e.g. `msl` ~100974
Pa, `2t` ~276.85 K, `sp` ~96743 Pa). One real bug found this way: `ci`
(`siconc`) comes back NaN over land (ERA5's own convention -- sea-ice
concentration is undefined there) -- `np.nan_to_num` fixes this, matching
what the official aurora repo's own example notebook does defensively for
every surf field. No other field showed NaNs.
"""

import datetime
import logging
import os

import cdsapi
import numpy as np
import xarray as xr
from aurora.insolation import insolation as _compute_insolation

LOG = logging.getLogger(__name__)

CDS_DATASET = "reanalysis-era5-complete"

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# ERA5/MARS table-128 paramId, for Aurora's 5 pressure-level families --
# same quantities/units/paramIds as fcn3_ic.py's own _PL_PARAMIDS (both
# backends' atmos_vars are the same z/u/v/t/q families).
_PL_PARAMIDS = {
    "z": "129.128",
    "t": "130.128",
    "u": "131.128",
    "v": "132.128",
    "q": "133.128",
}

# ERA5 shortname -> MARS paramId, for the 18 real-fetchable surface
# variables (of Aurora's 26 surf_vars -- see module docstring for the 8
# that are never fetched).
_SFC_PARAMIDS = {
    "t2m": "167.128",
    "u10": "165.128",
    "v10": "166.128",
    "msl": "151.128",
    "d2m": "168.128",
    "tcwv": "137.128",
    "tcc": "164.128",
    "u100": "246.228",
    "v100": "247.228",
    "sp": "134.128",
    "lcc": "186.128",
    "mcc": "187.128",
    "hcc": "188.128",
    "skt": "235.128",
    "stl1": "139.128",
    "swvl1": "39.128",
    "siconc": "31.128",
    "sd": "141.128",
}
# ERA5 shortname -> Aurora surf_var name. Most of these mirror the
# name-mapping table in the official aurora repo's docs/example_v1p5.ipynb
# exactly (confirmed by reading that notebook directly, not re-derived).
_SFC_RENAME = {
    "t2m": "2t",
    "u10": "10u",
    "v10": "10v",
    "d2m": "2d",
    "u100": "100u",
    "v100": "100v",
    "siconc": "ci",
    "sd": "scaled_sd",
}
# Names not in this dict keep their ERA5 shortname unchanged (msl, tcwv,
# tcc, sp, lcc, mcc, hcc, skt, stl1, swvl1 -- Aurora's own names already
# match ERA5's shortnames for these).

# ERA5's own surface geopotential ("orography"), fetched as an EXTRA field
# purely for the ps-obs forward operator's station-elevation QC -- the same
# role FCN3's own _OROGRAPHY_PARAMID plays (see fcn3_ic.py's docstring).
# Unlike AIFS (whose checkpoint has a real per-date-fetched orography INPUT
# column, so its own get_verif reads real ERA5 'z' for free as part of the
# normal fetch) or FCN3 (whose checkpoint-bundled orography.nc is entirely
# separate from ERA5 and never exposed to decode_state at all), Aurora's
# static 'z' field IS exposed via decode_state (aurora_model.py) but is
# NOT independently verified to equal real ERA5 orography -- rather than
# rely on that unverified assumption for a QC computation that specifically
# needs ERA5 TRUTH, get_verif's aurora branch uses this fresh fetch instead,
# matching FCN3's safer pattern.
_OROGRAPHY_PARAMID = "129.128"


def _cache_paths(cache_dir, date):
    tag = date.strftime("%Y%m%dT%H")
    return (
        os.path.join(cache_dir, f"era5_aurora_{tag}_pl.nc"),
        os.path.join(cache_dir, f"era5_aurora_{tag}_sfc.nc"),
    )


def _retrieve(client, request, target):
    tmp = target + ".tmp"
    client.retrieve(CDS_DATASET, request, tmp)
    os.replace(tmp, target)


def fetch_era5(date, cache_dir):
    """Fetch (or reuse cached) ERA5 pressure-level + single-level fields
    for `date`, on Aurora's native 0.25deg grid -- no regridding needed
    (same grid as FCN3/ERA5 itself). Needs internet access; call from a
    login node, never from the (offline) H100 compute nodes.

    Returns
    -------
    (pl_path, sfc_path) : the two cached netCDF file paths.
    """
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
    `{name: (720, 1440) array}` dict (surf vars, plus 'insolation',
    computed) and `{base: (13, 720, 1440) array}` dict (atmos families),
    keyed by Aurora's own variable-name convention, cropped to 720 rows
    (dropping the South Pole row -- see aurora_model.py's `_NLAT` comment
    for why Aurora's own native grid is 720, not 721, rows).

    Used directly by `long_window_4dvar_utils.get_verif` (needs a flat,
    single-date snapshot); `build_input_state` calls this twice (for the
    two lagged time levels) and stacks the results.
    """
    pl_path, sfc_path = fetch_era5(date, cache_dir)
    fields = {}
    with xr.open_dataset(pl_path) as ds_pl:
        for base in _PL_PARAMIDS:
            arr = ds_pl[base].isel(valid_time=0).values  # (n_levels, 721, 1440)
            fields[base] = np.asarray(arr[:, :-1, :], dtype=np.float32)  # crop -> (13, 720, 1440)
    with xr.open_dataset(sfc_path) as ds_sfc:
        for name in ds_sfc.data_vars:
            # ERA5's surface geopotential (orography) shares the shortname
            # "z" with the pressure-level geopotential family (standard
            # ECMWF convention: shortName tracks the parameter, not the
            # level type) -- rename to avoid colliding with fields["z"]
            # (the atmos family) above, same fix fcn3_ic.py uses.
            out_name = "geopotential_at_surface" if name == "z" else _SFC_RENAME.get(name, name)
            arr = ds_sfc[name].isel(valid_time=0).values  # (721, 1440)
            # 'siconc' (-> 'ci') comes back NaN over land (ERA5's own
            # convention -- sea-ice concentration is undefined there) --
            # confirmed by a real test fetch, not assumed. np.nan_to_num
            # matches the official aurora repo's own example notebook,
            # which applies this to every surf field defensively.
            arr = np.nan_to_num(arr, nan=0.0)
            fields[out_name] = np.asarray(arr[:-1, :], dtype=np.float32)  # crop -> (720, 1440)

    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)[:-1]
    lon = np.linspace(0.0, 360.0, 1440, endpoint=False, dtype=np.float32)
    fields["insolation"] = _compute_insolation([date], lat, lon, enforce_2d=True)[0].astype(np.float32)

    return fields


def build_input_state(date, cache_dir):
    """Fetch (or reuse cached) ERA5 fields at `date - timestep` and `date`
    (Aurora's two lagged time levels) and return
    `{"fields": {name_or_base: (2, ...) array}}`, ready for
    `AuroraModel.prepare_initial_state`. `date` is the LATEST of the two
    levels, matching `AuroraModel.prepare_initial_state`'s own convention.
    """
    timestep = datetime.timedelta(hours=6)
    fields_lo = read_single_date_fields(date - timestep, cache_dir)
    fields_hi = read_single_date_fields(date, cache_dir)
    # 'geopotential_at_surface' is QC-only (get_verif), not one of Aurora's
    # real input channels -- drop it, matching fcn3_ic.build_input_state's
    # own convention (AuroraModel.prepare_initial_state would just ignore
    # it either way, since it's not in _single_by_base, but dropping it
    # here avoids a pointless stack of a field never used).
    return {
        "fields": {
            k: np.stack([fields_lo[k], fields_hi[k]], axis=0)
            for k in fields_lo
            if k != "geopotential_at_surface"
        }
    }
