"""
ERA5 initial conditions / verification for the SFNO-73ch backend.

SFNO's 73 channels are FCN3's 72 (same names, same 0.25 deg 721x1440 grid,
same units) plus native surface pressure 'sp'. So this reuses fcn3_ic's ERA5
fetch and cache files unchanged (backends/fcn3/fcn3_ic.py: pressure-level
z/t/u/v/q on 13 levels, u10m/v10m/u100m/v100m/t2m/msl/tcwv, and ERA5
orography as 'geopotential_at_surface' for the ps-obs QC) and fetches only
the extra 'sp' field (ERA5 param 134) into its own small per-date file,
era5_sfno_<YYYYMMDDTHH>_sp.nc, in the same cache directory. An existing FCN3
cache therefore serves SFNO without re-fetching anything but 'sp'.

Fetching needs internet (CDS) -- run from a login node before a GPU job;
reads are cache-only once warm.
"""

import datetime
import logging
import os
import sys

import cdsapi
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fcn3"))
import fcn3_ic  # noqa: E402

LOG = logging.getLogger(__name__)
_SP_PARAMID = "134.128"


def _sp_path(cache_dir, date):
    return os.path.join(cache_dir, f"era5_sfno_{date.strftime('%Y%m%dT%H')}_sp.nc")


def fetch_sp(date, cache_dir):
    """Fetch (or reuse cached) ERA5 surface pressure for `date` on the 0.25 deg grid."""
    date = datetime.datetime(date.year, date.month, date.day, date.hour)
    path = _sp_path(cache_dir, date)
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        LOG.info("Fetching ERA5 surface pressure for %s", date)
        request = {
            "class": "ea", "date": date.strftime("%Y-%m-%d"), "expver": "1", "stream": "oper",
            "time": date.strftime("%H:00:00"), "type": "an", "grid": "0.25/0.25",
            "area": "90/0/-90/359.75", "format": "netcdf", "levtype": "sfc", "param": _SP_PARAMID,
        }
        fcn3_ic._retrieve(cdsapi.Client(), request, path)
    return path


def read_single_date_fields(date, cache_dir):
    """`{field_name: (721, 1440) array}`: FCN3's 72 channels, 'sp' (Pa), and
    'geopotential_at_surface' (ERA5 orography, m^2/s^2, QC only)."""
    fields = fcn3_ic.read_single_date_fields(date, cache_dir)
    with xr.open_dataset(fetch_sp(date, cache_dir)) as ds:
        fields["sp"] = np.asarray(ds["sp"].isel(valid_time=0).values, dtype=np.float32)
    return fields


def build_input_state(date, cache_dir):
    """`{"fields": {channel_name: (721, 1440) array}}` for SFNOModel.prepare_initial_state."""
    fields = read_single_date_fields(date, cache_dir)
    return {"fields": {k: v for k, v in fields.items() if k != "geopotential_at_surface"}}


if __name__ == "__main__":
    # prefetch: python backends/sfno/sfno_ic.py CACHE_DIR YYYY-MM-DDTHH [...]
    logging.basicConfig(level=logging.INFO)
    cache = sys.argv[1]
    for d in sys.argv[2:]:
        dt = datetime.datetime.strptime(d, "%Y-%m-%dT%H")
        fcn3_ic.fetch_era5(dt, cache)
        fetch_sp(dt, cache)
