"""
Shared ERA5 initial-condition / verification reader for AIFS checkpoints
trained against the same anemoi-datasets zarr archives (currently
aifs2-1.0deg-v1.0 and aifs-r2-t1), reading straight from that training data
instead of fetching ERA5 GRIB via CDS (aifs_ic.py/aifs3_ic.py). Same entry
points (`build_input_state`/`read_single_date_fields`) and call signature as
those two modules, so the driver only swaps `model.ic_module` (see
long_window_4dvar_utils.get_model()'s `exp['aifs_ic_module']` override).

One shared join for both checkpoints: `_JOIN` below is exactly aifs-r2-t1's
own training join (its `.hydra/config.yaml`, `dataloader.dataset.join`) --
confirmed directly (2026-11-23, see docs/integrate_multiple_backends.md) to
be a strict superset of every variable aifs2-1.0deg-v1.0's checkpoint needs
too, on the identical O96 grid/point ordering. Each checkpoint just selects
its own variable subset out of the one opened dataset at read time
(`_needed_variables`), rather than maintaining two separate join configs.

No cache directory, no login-node prefetch step: this reads directly off
/scratch4 zarr stores, which `anemoi.datasets.open_dataset` opens lazily and
reads chunk-at-a-time -- fast enough to read on demand from a compute node
(confirmed: a single-date read completes in well under a second, no
internet required). `cache_dir` is accepted, to match aifs_ic.py's call
signature, but unused.
"""
import numpy as np
from anemoi.datasets import open_dataset

_LEVELS37 = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
             300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
             875, 900, 925, 950, 975, 1000]

# Bo Huang's own O96 surface/single-level zarr (also used, alone + a small
# 10hPa supplement, to train aifs2-1.0deg-v1.0 itself) + S. Frolov's 37-level
# pressure zarr (adds the full vertical + the land constants aifs-r2-t1
# needs) + his single-field soil-temperature zarr. 1979-01-01..2023-12-31,
# 6-hourly, 40320-point O96 grid, identical point ordering to every
# checkpoint probed against it so far.
_ZARR1 = "/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/data/training/era5-o96-1979-2023-6h-v8.zarr"
_ZARR2 = "/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/era5-o96-37lev-1979-01-01T00-2023-12-31T18-s4.zarr"
_ZARR3 = "/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_37lev/chunks/era5-o96-stl1-1979-01-01T00-2023-12-31T18.zarr"

_JOIN = [
    {
        "dataset": _ZARR1,
        "select": [
            "10u", "10v", "2d", "2t", "cos_julian_day", "cos_latitude", "cos_local_time",
            "cos_longitude", "cp", "insolation", "lsm", "msl", "sdor", "sin_julian_day",
            "sin_latitude", "sin_local_time", "sin_longitude", "skt", "slor", "sp", "tcw",
            "tp", "z",
        ],
    },
    {
        "dataset": _ZARR2,
        "select": (
            [f"q_{l}" for l in _LEVELS37] + [f"t_{l}" for l in _LEVELS37]
            + [f"u_{l}" for l in _LEVELS37] + [f"v_{l}" for l in _LEVELS37]
            + [f"w_{l}" for l in _LEVELS37] + [f"z_{l}" for l in _LEVELS37]
            + ["sd", "siconc", "swvl1", "cvh", "cvl", "lai_hv", "lai_lv", "slt", "fsr"]
        ),
    },
    _ZARR3,
]

# Analytically-computed inputs (lat/lon trig, insolation, local/julian-day
# time trig): every AIFS checkpoint in this repo computes these on the fly
# from date/lat/lon rather than taking them as retrieved data (see
# aifs_ic.py's _retrieved_constant_forcings_variables / aifs_model.py's
# _select_variables_and_masks docstrings). They're real stored columns in
# the zarr (unlike ERA5 GRIB, which never has them at all) -- excluded here
# to match aifs_ic.py/aifs3_ic.py's convention of leaving them to the
# model's own constant/dynamic-forcing providers rather than supplying them
# a second time.
_COMPUTED = {
    "cos_latitude", "cos_longitude", "sin_latitude", "sin_longitude",
    "cos_julian_day", "cos_local_time", "sin_julian_day", "sin_local_time",
    "insolation",
}

_dataset = None


def _shared_dataset():
    global _dataset
    if _dataset is None:
        _dataset = open_dataset(join=_JOIN)
    return _dataset


def _metadata(runner):
    """AIFSModel's runner.checkpoint, or AIFS3Model's single TensorHandler's
    .metadata -- both expose .variable_to_input_tensor_index/.lagged/
    .latitudes/.longitudes the same way (see aifs_model.py's
    _init_variable_index, aifs3_ic.py's build_input_state)."""
    if hasattr(runner, "tensor_handlers"):
        return next(iter(runner.tensor_handlers.values())).metadata
    return runner.checkpoint


def _needed_variables(md):
    return [v for v in md.variable_to_input_tensor_index if v not in _COMPUTED]


def _read_fields(dates, names):
    """{name: (len(dates), n_points) float32} read from the shared dataset."""
    ds = _shared_dataset()
    idx = []
    for d in dates:
        d64 = np.datetime64(d)
        i = int(np.searchsorted(ds.dates, d64))
        if i >= len(ds.dates) or ds.dates[i] != d64:
            raise ValueError(
                f"{d:%Y-%m-%dT%H} not in the shared training zarr "
                f"(range {ds.dates[0]}..{ds.dates[-1]}, 6-hourly)"
            )
        idx.append(i)
    name_to_index = ds.name_to_index
    out = {}
    for name in names:
        col = name_to_index[name]
        out[name] = np.stack([ds[i][col, 0, :] for i in idx]).astype(np.float32)
    return out


def build_input_state(runner, date, cache_dir=None, lagged=True):
    """anemoi State for the checkpoint's lagged input dates ending at
    `date`, restricted to the checkpoint's non-computed inputs. `lagged=False`
    reads `date` only. Same contract as aifs_ic.py/aifs3_ic.py's
    build_input_state; `cache_dir` is accepted but unused."""
    md = _metadata(runner)
    dates = [date + h for h in md.lagged] if lagged else [date]
    fields = _read_fields(dates, _needed_variables(md))
    return {
        "date": date,
        "latitudes": np.asarray(md.latitudes),
        "longitudes": np.asarray(md.longitudes),
        "fields": fields,
    }


def read_single_date_fields(runner, date, cache_dir=None):
    """{variable: (n_points,) array} at a single date (verification state);
    same contract as aifs_ic.py/aifs3_ic.py's read_single_date_fields."""
    md = _metadata(runner)
    fields = _read_fields([date], _needed_variables(md))
    return {k: v[0] for k, v in fields.items()}
