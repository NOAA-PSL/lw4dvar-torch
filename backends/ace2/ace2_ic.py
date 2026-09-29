"""
ACE2-ERA5 initial conditions built directly from Google's PUBLIC ARCO-ERA5
zarr stores (gs://gcp-public-data-arco-era5, anonymous read -- no billing
project, unlike Ai2's requester-pays processed dataset).

The processing reproduces the model-level stream of Ai2's own ERA5 dataset
pipeline (github.com/ai2cm/ace, scripts/era5/pipeline/xr-beam-pipeline.py,
Apache-2.0): 137 ERA5 model levels are pressure-weighted into ACE2's 8 hybrid
layers at native 0.25 deg, then conservatively regridded (xESMF) to the F90
Gaussian grid (180x360, south-to-north, lon 0.5..359.5). The functions marked
"vendored" below are copied from that script with only cosmetic changes, so
Beam/xarray-beam/obstore aren't needed -- we process a handful of dates, not
a multi-decade dataset. Only the 38 prognostic fields are produced; forcings
come from the HF forcing_YYYY.nc files (see ace2_prefetch_checkpoint.py).

The default layer indices [0,48,67,79,90,100,109,119,137] applied to ARCO's
L137 ak/bk reproduce the ACE2-ERA5 checkpoint's own sigma_coordinates
exactly (checked). One deliberate difference from the current upstream
script: it moves the model-top interface from 0 Pa to ~1 Pa
(`ak[0] = (ak[0]+ak[1])/2`), but the checkpoint records ak_0 = 0, so the
default here (`top_interface_midpoint=False`) keeps 0 Pa. Either way it only
changes the pressure weight of the single topmost ERA5 level inside a
~51 hPa-deep layer.

KNOWN, ACCEPTED difference from ACE2-ERA5's training data: that dataset
(2024-11-13) was NOT made by the current upstream script but by its
2024-11-12 predecessor (ai2cm/ace commit aa08fe0312), which read ERA5 on its
NATIVE grids (ARCO `co/` stores: spectral vo/d/t/lnsp, reduced-Gaussian
moisture), regridded with MetView/MIR (`truncation="none"`, i.e. point
sampling), and vertically coarsened AFTER regridding. Validated against HF
initial_conditions/ic_2020.nc (compare_ic_ace2.py): mean differences ~0 in
every field, rms/std 0.1-0.7% in upper layers, 2-6% near the surface, and
PRESsfc 431 Pa rms / 91 hPa max, concentrated over steep terrain. Accepted
because these ICs only START an experiment; the static field the ps
observation operator uses every cycle (HGTsfc) is NOT produced here -- it
comes unchanged from Ai2's own forcing files (identical across years).

Two environments, deliberately:
- fetch_ic / main (ARCO read + xESMF regrid): the `ace2ic` conda env, from a
  LOGIN node (needs internet; xESMF/ESMF come from conda-forge and are kept
  out of the GPU `ace2` env's pip-installed torch stack).
- build_input_state (read the cached netCDF): any env with xarray, incl. `ace2`
  on a compute node. Raises if the cache file is missing.

CLI (repo root, ace2ic env):
    python backends/ace2/ace2_ic.py YYYY-MM-DDTHH [...] [--cache_dir ic_cache_ace2]
    python backends/ace2/ace2_ic.py --verif YYYY-MM-DDTHH [...]   # h500/TMP850 truth
"""

import argparse
import datetime
import logging
import os
from typing import Sequence

import numpy as np
import xarray as xr

URL_FULL_37 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
URL_MODEL_LEVEL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
STORAGE_OPTIONS = {"token": "anon"}

N_INPUT_LAYERS = 137
DEFAULT_OUTPUT_LAYER_INDICES = [0, 48, 67, 79, 90, 100, 109, 119, 137]
OUTPUT_GRID_N = 90  # F90: 180 x 360

SURFACE_VARS = [
    "surface_pressure",
    "skin_temperature",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
WATER_VARS = [
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
]
LAYER_VARS = {  # ACE2 family name -> ERA5 model-level variable(s)
    "air_temperature": ["temperature"],
    "specific_total_water": WATER_VARS,
    "eastward_wind": ["u_component_of_wind"],
    "northward_wind": ["v_component_of_wind"],
}
SURFACE_RENAME = {
    "surface_pressure": "PRESsfc",
    "skin_temperature": "surface_temperature",
    "2m_temperature": "TMP2m",
    "2m_dewpoint_temperature": "DPT2m",
    "10m_u_component_of_wind": "UGRD10m",
    "10m_v_component_of_wind": "VGRD10m",
}


GRAVITY = 9.80665
RDGAS = 287.05
LAPSE_RATE = 0.0065
DEFAULT_FORCING_DIR = "backends/ace2/ACE2-ERA5/forcing_data"  # git submodule (see ace2_prefetch_checkpoint.py)


def load_hgtsfc(forcing_dir: str) -> np.ndarray:
    """ACE2's own static surface height (m), (180, 360), from any HF
    forcing_YYYY.nc (bitwise identical across years)."""
    import glob

    # skip git-lfs pointer stubs (ACE2-ERA5 is a pointer-only submodule)
    files = [f for f in sorted(glob.glob(os.path.join(forcing_dir, "forcing_*.nc"))) if os.path.getsize(f) > 1024]
    if not files:
        raise FileNotFoundError(f"no fetched forcing_*.nc in {forcing_dir} -- see ace2_prefetch_checkpoint.py")
    with xr.open_dataset(files[0]) as ds:
        return ds["HGTsfc"].values.astype(np.float64)


def _reduce_surface_pressure(ps, tv, h_from, h_to):
    """Standard-atmosphere reduction of surface pressure from surface height
    h_from to h_to (m), with surface virtual temperature tv at h_from and a
    6.5 K/km lapse rate."""
    return ps * ((tv - LAPSE_RATE * (h_to - h_from)) / tv) ** (GRAVITY / (RDGAS * LAPSE_RATE))


def cache_path(cache_dir: str, date: datetime.datetime) -> str:
    return os.path.join(cache_dir, f"ace2_ic_{date:%Y%m%d%H}.nc")


# ---------------------------------------------------------------------------
# Vendored from ai2cm/ace scripts/era5/pipeline/xr-beam-pipeline.py
# ---------------------------------------------------------------------------


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lo], midpoints, [hi]])


def _gaussian_latitudes(n) -> np.ndarray:
    from numpy.polynomial.legendre import leggauss

    x, _ = leggauss(round(2 * n))
    return np.sort(np.degrees(np.arcsin(x)))


def _make_target_grid(n=OUTPUT_GRID_N) -> xr.Dataset:
    lat = _gaussian_latitudes(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], _cell_bounds(lat, -90, 90)),
            "lon_b": (["lon_b"], _cell_bounds(lon, 0, 360)),
        }
    )


def _make_source_grid() -> xr.Dataset:
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 359.75, 1440)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], _cell_bounds(lat, -90, 90)),
            "lon_b": (["lon_b"], _cell_bounds(lon, -0.125, 360 - 0.125)),
        }
    )


_REGRIDDER = None


def _regrid(ds):
    """Conservative 0.25 deg -> F90 regrid (vendored `_regrid`, grid fixed)."""
    global _REGRIDDER
    import xesmf as xe

    if _REGRIDDER is None:
        _REGRIDDER = xe.Regridder(_make_source_grid(), _make_target_grid(), "conservative", periodic=True)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.sortby("lat")
    out = _REGRIDDER(ds, keep_attrs=True)
    return out.rename({"lat": "latitude", "lon": "longitude"})


def _saturation_vapor_pressure(t):
    a1, a2, a3, a4 = 611.21, 273.16, 17.502, 32.19
    return a1 * np.exp(a3 * (t - a2) / (t - a4))


def _specific_humidity_from_dewpoint(dewpoint, pressure):
    ewsat = _saturation_vapor_pressure(dewpoint)
    eps = 0.621981
    return eps * ewsat / (pressure - (1 - eps) * ewsat)


def _get_ak_bk(ds_model_level: xr.Dataset, top_interface_midpoint: bool) -> tuple:
    for name in ds_model_level.data_vars:
        if "GRIB_pv" in ds_model_level[name].attrs:
            pv = ds_model_level[name].attrs["GRIB_pv"]
            break
    else:
        raise ValueError("No variable with GRIB_pv attribute found in model-level data")
    ak = np.array(pv[: N_INPUT_LAYERS + 1])
    bk = np.array(pv[N_INPUT_LAYERS + 1 :])
    if top_interface_midpoint:  # upstream's current behavior; see module docstring
        ak[0] = (ak[0] + ak[1]) / 2.0
    return ak, bk


def _compute_layer_thicknesses(ak, bk, surface_pressure: xr.DataArray) -> xr.DataArray:
    dak = ak[1:] - ak[:-1]
    dbk = bk[1:] - bk[:-1]
    dp = dak[:, None, None] + dbk[:, None, None] * surface_pressure.values[None, :, :]
    return xr.DataArray(
        dp,
        dims=["hybrid", "latitude", "longitude"],
        coords={"latitude": surface_pressure.latitude, "longitude": surface_pressure.longitude},
    )


def _vertical_coarsen(var: xr.DataArray, dp: xr.DataArray, output_layer_indices: Sequence[int]) -> dict:
    results = {}
    for i in range(len(output_layer_indices) - 1):
        sl = slice(output_layer_indices[i], output_layer_indices[i + 1])
        dp_fine = dp.isel(hybrid=sl)
        weighted = (var.isel(hybrid=sl) * dp_fine).sum("hybrid")
        results[i] = (weighted / dp_fine.sum("hybrid")).astype(np.float32)
    return results


# ---------------------------------------------------------------------------
# Fetch (login node, ace2ic env)
# ---------------------------------------------------------------------------


def _load_3d(ds_ml: xr.Dataset, name: str, date) -> xr.DataArray:
    logging.info(f"  reading {name}")
    da = ds_ml[name].sel(time=np.datetime64(date, "ns")).load()
    return da.astype(np.float64)


def fetch_ic(
    date: datetime.datetime,
    cache_dir: str,
    output_layer_indices: Sequence[int] = DEFAULT_OUTPUT_LAYER_INDICES,
    top_interface_midpoint: bool = False,
    overwrite: bool = False,
    forcing_dir: str | None = DEFAULT_FORCING_DIR,
) -> str:
    """Build the ACE2 prognostic state at `date` from ARCO-ERA5 and write it
    to `cache_path(cache_dir, date)` (netCDF, (latitude, longitude) fields,
    same variable names/grid as the HF initial_conditions/ic_YYYY.nc files).
    Returns the path. Reads ~2.5-3 GB of compressed model-level data.

    `forcing_dir` (default: the HF snapshot's forcing_data): PRESsfc is
    hydrostatically reduced from the orography it's consistent with here
    (ERA5 surface geopotential, conservatively regridded exactly like
    PRESsfc) to ACE2's own static HGTsfc -- the orography the ps observation
    operator uses. Validated on 2020-01-01T00 vs HF ic_2020.nc: PRESsfc rms
    difference 431 -> 280 Pa (ACE2's own PRESsfc implies an orography within
    26.7 m rms of HGTsfc, closer than any regridding of ERA5's). Q2m is
    computed from the UNadjusted PRESsfc; the hybrid-layer fields are left
    as is. `forcing_dir=None` disables the adjustment."""
    date = date.replace(tzinfo=None)
    path = cache_path(cache_dir, date)
    if os.path.exists(path) and not overwrite:
        logging.info(f"{path} exists, skipping")
        return path
    os.makedirs(cache_dir, exist_ok=True)

    ds_ml = xr.open_zarr(URL_MODEL_LEVEL, chunks=None, storage_options=STORAGE_OPTIONS)
    ds_sfc = xr.open_zarr(URL_FULL_37, chunks=None, storage_options=STORAGE_OPTIONS)
    ak, bk = _get_ak_bk(ds_ml, top_interface_midpoint)

    logging.info(f"{date:%Y-%m-%dT%H}: reading surface fields")
    sfc = ds_sfc[SURFACE_VARS].sel(time=np.datetime64(date, "ns")).load()
    for name in SURFACE_VARS:
        if bool(sfc[name].isnull().any()):
            raise ValueError(f"missing values in ARCO {name} at {date}")
    dp = _compute_layer_thicknesses(ak, bk, sfc["surface_pressure"].astype(np.float64))

    out_2d = xr.Dataset()
    for family, sources in LAYER_VARS.items():
        total = None
        for src in sources:
            da = _load_3d(ds_ml, src, date)
            total = da if total is None else total + da
            del da
        if bool(total.isnull().any()):
            raise ValueError(f"missing values in ARCO {sources} at {date}")
        for i, layer in _vertical_coarsen(total, dp, output_layer_indices).items():
            out_2d[f"{family}_{i}"] = layer
        del total
    for src, dst in SURFACE_RENAME.items():
        out_2d[dst] = sfc[src].astype(np.float32)
    out_2d = out_2d.drop_vars([c for c in out_2d.coords if c not in ("latitude", "longitude")])

    logging.info(f"regridding {len(out_2d.data_vars)} fields to F90")
    regridded = _regrid(out_2d)
    # Q2m from the UNadjusted PRESsfc: consistent with where DPT2m is valid
    # (the ERA5 surface); measured better vs ic_2020.nc (rms/std 2.8e-2 vs
    # 3.3e-2 with the adjusted PRESsfc).
    regridded["Q2m"] = _specific_humidity_from_dewpoint(regridded["DPT2m"], regridded["PRESsfc"])
    if forcing_dir is not None:
        logging.info("reducing PRESsfc to ACE2's HGTsfc")
        orog = ds_sfc["geopotential_at_surface"].sel(time=np.datetime64(date, "ns")).load() / GRAVITY
        h_cons = _regrid(orog.rename("h").to_dataset())["h"]
        tv = regridded["air_temperature_7"] * (1.0 + (461.5 / RDGAS - 1.0) * regridded["specific_total_water_7"])
        hgt = xr.DataArray(load_hgtsfc(forcing_dir), dims=h_cons.dims, coords=h_cons.coords)
        regridded["PRESsfc"] = _reduce_surface_pressure(regridded["PRESsfc"], tv, h_cons, hgt)

    regridded = regridded.astype(np.float32)
    regridded.attrs = {
        "source": "ARCO-ERA5 (gs://gcp-public-data-arco-era5), processed by lw4dvar-torch backends/ace2/ace2_ic.py",
        "valid_time": f"{date:%Y-%m-%dT%H:%M:%S}",
        "output_layer_indices": list(output_layer_indices),
        "top_interface_midpoint": int(top_interface_midpoint),
        "presfc_reduced_to_hgtsfc": int(forcing_dir is not None),
    }
    tmp = path + ".tmp"
    regridded.to_netcdf(tmp)
    os.replace(tmp, path)
    logging.info(f"wrote {path}")
    return path


def verif_cache_path(cache_dir: str, date: datetime.datetime) -> str:
    return os.path.join(cache_dir, f"ace2_verif_{date:%Y%m%d%H}.nc")


def fetch_verif(date: datetime.datetime, cache_dir: str, overwrite: bool = False) -> str:
    """ERA5 verification fields on ACE2's F90 grid, named like ACE2's own
    output-only diagnostics: h500 (500 hPa geopotential height, m) and
    TMP850 (850 hPa temperature, K). Built the way the current upstream
    pipeline builds them (`_process_pressure_level_data`: level select,
    geopotential / g, conservative regrid) from ARCO full_37 -- one ~80 MB
    all-37-level chunk per variable per hour, so ~160 MB per date."""
    date = date.replace(tzinfo=None)
    path = verif_cache_path(cache_dir, date)
    if os.path.exists(path) and not overwrite:
        logging.info(f"{path} exists, skipping")
        return path
    os.makedirs(cache_dir, exist_ok=True)
    ds = xr.open_zarr(URL_FULL_37, chunks=None, storage_options=STORAGE_OPTIONS)
    t = np.datetime64(date, "ns")
    logging.info(f"{date:%Y-%m-%dT%H}: reading z500, t850")
    out = xr.Dataset(
        {
            "h500": ds["geopotential"].sel(time=t, level=500).drop_vars("level").load() / GRAVITY,
            "TMP850": ds["temperature"].sel(time=t, level=850).drop_vars("level").load(),
        }
    )
    for name in out.data_vars:
        if bool(out[name].isnull().any()):
            raise ValueError(f"missing values in ARCO {name} at {date}")
    out = out.drop_vars([c for c in out.coords if c not in ("latitude", "longitude")])
    regridded = _regrid(out).astype(np.float32)
    regridded["h500"].attrs = {"long_name": "Geopotential height at 500 hPa", "units": "m"}
    regridded["TMP850"].attrs = {"long_name": "Temperature at 850 hPa", "units": "K"}
    regridded.attrs = {
        "source": "ARCO-ERA5 full_37 (gs://gcp-public-data-arco-era5), conservative regrid to F90 by lw4dvar-torch backends/ace2/ace2_ic.py",
        "valid_time": f"{date:%Y-%m-%dT%H:%M:%S}",
    }
    tmp = path + ".tmp"
    regridded.to_netcdf(tmp)
    os.replace(tmp, path)
    logging.info(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Read (any env, incl. compute nodes)
# ---------------------------------------------------------------------------


def build_input_state(date: datetime.datetime, cache_dir: str) -> dict:
    """`{"fields": {name: (180, 360) array}}` for ACE2Model.prepare_initial_state,
    from the cache only (compute nodes have no internet)."""
    path = cache_path(cache_dir, date.replace(tzinfo=None))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python backends/ace2/ace2_ic.py {date:%Y-%m-%dT%H}` "
            f"from a login node in the ace2ic env first"
        )
    with xr.open_dataset(path) as ds:
        return {"fields": {n: ds[n].values for n in ds.data_vars}}


def load_verif(date: datetime.datetime, cache_dir: str) -> dict:
    """{'h500': (180, 360), 'TMP850': (180, 360)} numpy arrays from the cache
    (see fetch_verif). Raises if not prefetched."""
    path = verif_cache_path(cache_dir, date.replace(tzinfo=None))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python backends/ace2/ace2_ic.py --verif {date:%Y-%m-%dT%H}` "
            f"from a login node in the ace2ic env first"
        )
    with xr.open_dataset(path) as ds:
        return {n: ds[n].values for n in ds.data_vars}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("dates", nargs="+", help="YYYY-MM-DDTHH")
    p.add_argument("--cache_dir", default="ic_cache_ace2")
    p.add_argument("--top_interface_midpoint", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verif", action="store_true", help="fetch h500/TMP850 verification fields instead of ICs")
    p.add_argument("--forcing_dir", default=DEFAULT_FORCING_DIR,
                   help="HF forcing_data dir (for HGTsfc); 'none' disables the PRESsfc reduction")
    args = p.parse_args()
    for d in args.dates:
        if args.verif:
            fetch_verif(datetime.datetime.strptime(d, "%Y-%m-%dT%H"), args.cache_dir, overwrite=args.overwrite)
            continue
        fetch_ic(
            datetime.datetime.strptime(d, "%Y-%m-%dT%H"),
            args.cache_dir,
            top_interface_midpoint=args.top_interface_midpoint,
            overwrite=args.overwrite,
            forcing_dir=None if args.forcing_dir.lower() == "none" else args.forcing_dir,
        )
