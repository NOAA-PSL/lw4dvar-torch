"""
ERA5 initial-condition / verification reader for AIFS3Model (multi-dataset
anemoi API, `aifs3` env). Same entry points as aifs_ic.py, so the driver only
swaps the module (`AIFS3Model.ic_module`).

Read-only: it reads the per-date GRIBs aifs_ic.fetch_era5_grib() writes
(`era5_<YYYYmmddTHH>.grib`, or `..._lagged.grib` holding every lagged input
date) and never fetches. Warm the cache from a login node with
backends/aifs/aifs_prefetch_ic.py in an env/checkpoint on the same grid and
variable set (for the O96 1-degree checkpoints: Bo Huang's `lwaifs2` env with
the aifs2-1.0deg-v1.0 checkpoint), or point `ic_cache` at an existing cache.

The anemoi State is built directly from the GRIB fields (no anemoi `Input`
classes, whose API changed again in anemoi-inference 0.12): fields the
checkpoint doesn't take as input are dropped, computed forcings are left to
the TensorHandler.
"""
import os

import numpy as np

# time-constant fields aifs_ic.fetch_era5_grib retrieves at one date only
_CONSTANTS = {"lsm", "sdor", "slor", "z"}


def _cache_path(cache_dir, date):
    tag = date.strftime("%Y%m%dT%H")
    for name in (f"era5_{tag}.grib", f"era5_{tag}_lagged.grib"):
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"no cached ERA5 GRIB for {date:%Y-%m-%dT%H} in {cache_dir} -- aifs3_ic only reads the cache; "
        "prefetch it from a login node (see this module's docstring)"
    )


def _read_fields(path, dates):
    """{variable: (len(dates), n_points) float32} for every field at `dates`,
    keyed like the checkpoint ('q_850' for pressure levels, 'z' for surface
    geopotential). Fields present only at one date (the constants
    aifs_ic.fetch_era5_grib fetches once) are repeated across `dates`."""
    import earthkit.data as ekd

    by_name = {}
    for f in ekd.from_source("file", path):
        key = (int(f.metadata("dataDate")), int(f.metadata("dataTime")))
        name = f.metadata("shortName")
        if f.metadata("typeOfLevel") == "isobaricInhPa":
            name = f"{name}_{int(f.metadata('level'))}"
        by_name.setdefault(name, {})[key] = f.to_numpy(flatten=True).astype(np.float32)
    fields = {}
    order = [(int(d.strftime("%Y%m%d")), d.hour * 100) for d in dates]
    for name, at in by_name.items():
        if all(k in at for k in order):
            fields[name] = np.stack([at[k] for k in order])
        elif name in _CONSTANTS and len(at) == 1:
            fields[name] = np.repeat(next(iter(at.values()))[np.newaxis], len(order), axis=0)
    return fields


def build_input_state(runner, date, cache_dir, lagged=True):
    """anemoi State for the checkpoint's lagged input dates ending at `date`,
    restricted to the checkpoint's non-computed inputs. `lagged=False` reads
    `date` only."""
    md = next(iter(runner.tensor_handlers.values())).metadata
    dates = [date + h for h in md.lagged] if lagged else [date]
    fields = _read_fields(_cache_path(cache_dir, date), dates)
    inputs = set(md.variable_to_input_tensor_index)
    state_fields = {k: v for k, v in fields.items() if k in inputs}
    return {
        "date": date,
        "latitudes": np.asarray(md.latitudes),
        "longitudes": np.asarray(md.longitudes),
        "fields": state_fields,
    }


def read_single_date_fields(runner, date, cache_dir):
    """{variable: (n_points,) array} for every cached field at `date`
    (verification state; same contract as aifs_ic.read_single_date_fields)."""
    return {k: v[0] for k, v in _read_fields(_cache_path(cache_dir, date), [date]).items()}
