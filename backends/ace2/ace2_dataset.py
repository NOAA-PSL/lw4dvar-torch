"""
ACE2 training/validation dataset built from Google's PUBLIC ARCO-ERA5 zarr
stores (gs://gcp-public-data-arco-era5, anonymous read), written directly in
the layout `fme` trains on: one netCDF per month, `YYYYMM0100.nc`, 6-hourly
`time` x F90 (latitude, longitude), with the static fields and the coarse
`ak_N`/`bk_N` interface coefficients in every file (what `fme`'s
`XarrayDataset` reads the vertical coordinate from).

This is the per-time-step logic of Ai2's own ERA5 pipeline
(github.com/ai2cm/ace scripts/era5/pipeline/xr-beam-pipeline.py, Apache-2.0,
vendored as of ai2cm/ace e9d7fc227, 2026-08-14) without Apache Beam: all five
of its streams --
  model level   8 ERA5 L137 fields pressure-weighted into hybrid layers at
                0.25 deg, then regridded; PRESsfc, PRMSL, surface_temperature,
                TMP2m, DPT2m, Q2m, UGRD10m, VGRD10m at the output time
  mean flux     radiative/turbulent fluxes, PRATEsfc, ... averaged over the 6
                hourly ERA5 means ending at the output time (6-h window)
  surface mean  6-h means of the surface fields (`*_mean`)
  surface anal. sea ice / ocean fraction, SST, soil, snow at the output time
  pressure lvl  h<p>, TMP<p>, Q<p>, UGRD<p>, VGRD<p> (incl. h500, TMP850)
-- plus the static fields (HGTsfc, land_fraction, soil type fractions) and
global_mean_co2. The vertical-coarsening, regridding and physics helpers are
the ones already vendored in ace2_ic.py.

Differences from Ai2's datasets, all deliberate:
- The model-top interface: default ~1 Pa, upstream's current choice and the
  one for models trained from scratch here; `--model_top_zero` gives 0 Pa as
  in the ACE2-ERA5 checkpoint (and ace2_ic.py's ICs). See ace2_ic.py.
- CO2: Ai2's CO2 series is in a private bucket. `co2` mode pulls ACE2-ERA5's
  own `global_mean_co2` (1940-2022) out of the public HF forcing files over
  HTTP range reads (~3 s/year, no full-file download).
- ACE2-ERA5 itself was trained on the 2024-11-13 dataset made by the
  PREVIOUS upstream pipeline (native-grid ERA5, MIR point regridding, vertical
  coarsening after regridding); this follows the current one (conservative
  xESMF regridding, coarsening at 0.25 deg). A model trained here and
  ACE2-ERA5 see slightly different data (ace2_ic.py quantifies the IC
  differences); recompute normalization statistics from this dataset.
- PRESsfc and PRESsfc_mean are made valid at HGTsfc: each 0.25-deg ps is
  reduced hydrostatically to its F90 cell's HGTsfc BEFORE the conservative
  average (`_presfc_at_hgtsfc`), instead of averaging pressures valid at
  different heights (biased high over steep terrain, up to ~1.3 hPa). Q2m and
  Q2m_mean still use the plain average. Files carry the global attribute
  `presfc_reduced_to_hgtsfc=1`; `--no_reduce_presfc` gives the plain regrid,
  and `fix_presfc` mode patches months built without it.

Vertical resolution: `--layer_indices` takes the L137 interface indices of the
coarse layers (default ACE2's 8: 0 48 67 79 90 100 109 119 137); any
increasing list from 0 to 137 works, and `fme` reads the layer count from the
ak_N/bk_N it finds. `--extra_layer_vars` adds layer-averaged ERA5 model-level
fields beyond ACE2's four families (e.g. specific_humidity separate from total
water, ozone_mass_mixing_ratio, fraction_of_cloud_cover -- what a radiance
operator needs), named `<era5_name>_<i>`.

Cost (Polaris login node, 2026-10): the 8 model-level fields are ~10 s of
reads per output time per process, the rest a few s more; a month is ~124
times. Memory ~2-3 GB per worker (Polaris login nodes cap a user at 8 GB
RAM and 8 CPUs: test there, run production on compute nodes). Output ~28 MB/time uncompressed at 8
layers (~3.5 GB/month).

Env: `ace2ic` (xESMF/ESMF, zarr 3, gcsfs; ace2ic-spec.txt). Needs internet.

CLI (repo root):
    python backends/ace2/ace2_dataset.py co2 OUTDIR/co2.nc 1940 2022
    python backends/ace2/ace2_dataset.py build OUTDIR 1979-01 2022-12 \\
        --co2 OUTDIR/co2.nc --workers 8 [--layer_indices ...] \\
        [--extra_layer_vars specific_humidity ozone_mass_mixing_ratio]
    python backends/ace2/ace2_dataset.py fix_presfc OUTDIR 1979-01 1988-12 --workers 16
`build` skips months whose file exists (resumable; files are written to .tmp
and renamed), so disjoint month ranges can run as separate jobs into one
OUTDIR.
"""

import argparse
import datetime
import logging
import multiprocessing as mp
import os
import time as _time
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

import ace2_ic as ic
from ace2_ic import (
    DEFAULT_OUTPUT_LAYER_INDICES,
    GRAVITY,
    LAYER_VARS,
    N_INPUT_LAYERS,
    URL_FULL_37,
    URL_MODEL_LEVEL,
    _get_ak_bk,
    _regrid,
    _specific_humidity_from_dewpoint,
)

DENSITY_OF_LIQUID_WATER = 1000.0  # kg/m**3
TIME_STEP = pd.Timedelta(hours=6)
# Upstream's production START_TIME: the first output time whose 6-h mean-flux
# window lies inside ERA5. Also the epoch of Ai2's time units.
FIRST_TIME = pd.Timestamp("1940-01-01T12:00:00")
TIME_UNITS = "hours since 1940-01-01T12:00:00"
# trust_env: aiohttp ignores http(s)_proxy unless asked -- needed on Polaris
# compute nodes, which reach the internet only through proxy.alcf.anl.gov.
STORAGE_OPTIONS = {**ic.STORAGE_OPTIONS, "session_kwargs": {"trust_env": True}}
HF_FORCING_URL = "https://huggingface.co/allenai/ACE2-ERA5/resolve/main/forcing_data/forcing_{year}.nc"
UPSTREAM = "ai2cm/ace scripts/era5/pipeline/xr-beam-pipeline.py @ e9d7fc227"

# ---------------------------------------------------------------------------
# Vendored variable lists (upstream names; mean_snowfall_rate de-duplicated)
# ---------------------------------------------------------------------------

FULL_37_MEAN_FLUX_VARS = [
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_surface_latent_heat_flux",
    "mean_total_precipitation_rate",
    "mean_vertically_integrated_moisture_divergence",
    "mean_snowfall_rate",
    "mean_top_net_short_wave_radiation_flux_clear_sky",
    "mean_top_net_long_wave_radiation_flux_clear_sky",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "mean_surface_net_short_wave_radiation_flux_clear_sky",
    "mean_surface_downward_long_wave_radiation_flux_clear_sky",
    "mean_surface_net_long_wave_radiation_flux_clear_sky",
    "mean_runoff_rate",
    "mean_eastward_gravity_wave_surface_stress",
    "mean_eastward_turbulent_surface_stress",
    "mean_northward_gravity_wave_surface_stress",
    "mean_northward_turbulent_surface_stress",
]
FULL_37_SURFACE_ANALYSIS_VARS = [
    "sea_ice_cover",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "snow_depth",
    "snow_density",
    "sea_surface_temperature",
    "skin_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
]
FULL_37_INVARIANT_VARS = ["land_sea_mask", "geopotential_at_surface", "soil_type"]
FULL_37_PRESSURE_LEVEL_VARS = [
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
FULL_37_MODEL_LEVEL_SURFACE_VARS = [
    "surface_pressure",
    "mean_sea_level_pressure",
    "skin_temperature",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
FULL_37_SURFACE_MEAN_VARS = [
    "skin_temperature",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
MODEL_LEVEL_SURFACE_RENAME = {
    "surface_pressure": "PRESsfc",
    "mean_sea_level_pressure": "PRMSL",
    "skin_temperature": "surface_temperature",
    "2m_temperature": "TMP2m",
    "2m_dewpoint_temperature": "DPT2m",
    "10m_u_component_of_wind": "UGRD10m",
    "10m_v_component_of_wind": "VGRD10m",
}
# Model-level fields that --extra_layer_vars may add (output `<name>_<i>`).
EXTRA_LAYER_VARS = [
    "specific_humidity",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "ozone_mass_mixing_ratio",
    "fraction_of_cloud_cover",
    "vertical_velocity",
]

# Upstream's levels plus those ACE2.1-ERA5's secondary pressure-level decoder
# is trained on (h/TMP/Q/UGRD/VGRD at 50-1000 hPa, 13 levels; cf.
# ../ACE2.1-ERA5-AIMIP/configs/ace-fine-tune-pressure-level-separate-decoder-config.yaml).
# The ARCO store chunks all 37 levels together, so extra levels cost no reads.
OUTPUT_PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
RENAME_PRESSURE_LEVEL = {
    **{f"specific_humidity_{p}": f"Q{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"temperature_{p}": f"TMP{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"u_component_of_wind_{p}": f"UGRD{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"v_component_of_wind_{p}": f"VGRD{p}" for p in OUTPUT_PRESSURE_LEVELS},
    **{f"geopotential_{p}": f"h{p}" for p in OUTPUT_PRESSURE_LEVELS},
}
# ECMWF soil types (codes.ecmwf.int/grib/param-db/43); 0 is the fill value.
SOIL_TYPES = {
    "undefined": 0,
    "coarse": 1,
    "medium": 2,
    "medium_fine": 3,
    "fine": 4,
    "very_fine": 5,
    "organic": 6,
    "tropical_organic": 7,
}
VARIABLES_WITH_SOME_MISSING_VALUES = [
    "sea_ice_cover",
    "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
]

_W2 = "W/m**2"
_FLUX = "kg/m**2/s"
DESIRED_ATTRS = {
    "DSWRFtoa": ("Downward SW radiative flux at TOA", _W2),
    "USWRFtoa": ("Upward SW radiative flux at TOA", _W2),
    "ULWRFtoa": ("Upward LW radiative flux at TOA", _W2),
    "DSWRFsfc": ("Downward SW radiative flux at surface", _W2),
    "USWRFsfc": ("Upward SW radiative flux at surface", _W2),
    "DLWRFsfc": ("Downward LW radiative flux at surface", _W2),
    "ULWRFsfc": ("Upward LW radiative flux at surface", _W2),
    "UCSWRFtoa": ("Upward SW radiative flux at TOA assuming clear sky", _W2),
    "UCLWRFtoa": ("Upward LW radiative flux at TOA assuming clear sky", _W2),
    "DCSWRFsfc": ("Downward SW radiative flux at surface assuming clear sky", _W2),
    "UCSWRFsfc": ("Upward SW radiative flux at surface assuming clear sky", _W2),
    "DCLWRFsfc": ("Downward LW radiative flux at surface assuming clear sky", _W2),
    "UCLWRFsfc": ("Upward LW radiative flux at surface assuming clear sky", _W2),
    "LHTFLsfc": ("Latent heat flux", _W2),
    "SHTFLsfc": ("Sensible heat flux", _W2),
    "PRATEsfc": ("Surface precipitation rate", _FLUX),
    "tendency_of_total_water_path_due_to_advection": ("Tendency of total water path due to advection", _FLUX),
    "runoff_flux": ("Runoff flux", _FLUX),
    "total_frozen_precipitation_rate": ("Total frozen precipitation rate", _FLUX),
    "eastward_surface_stress": ("Eastward surface stress", "N/m**2"),
    "northward_surface_stress": ("Northward surface stress", "N/m**2"),
    "HGTsfc": ("Topography height", "m"),
    "land_fraction": ("land fraction", ""),
    "sea_ice_fraction": ("sea ice fraction", ""),
    "ocean_fraction": ("ocean fraction", ""),
    "PRESsfc": ("Surface pressure", "Pa"),
    "PRMSL": ("Mean sea level pressure", "Pa"),
    "surface_temperature": ("Skin temperature", "K"),
    "TMP2m": ("2m air temperature", "K"),
    "Q2m": ("2m specific humidity", "kg/kg"),
    "DPT2m": ("2m dewpoint temperature", "K"),
    "UGRD10m": ("10m U component of wind", "m/s"),
    "VGRD10m": ("10m V component of wind", "m/s"),
    "merged_sea_surface_and_skin_temperature": ("Merged sea surface and skin temperature", "K"),
    "surface_snow_amount": ("Surface snow amount", "kg/m**2"),
    "surface_snow_area_fraction": ("Surface snow area fraction", "fraction"),
    "surface_snow_thickness": ("Surface snow thickness", "m"),
    "surface_temperature_mean": ("Mean skin temperature", "K"),
    "TMP2m_mean": ("Mean 2m air temperature", "K"),
    "DPT2m_mean": ("Mean 2m dewpoint temperature", "K"),
    "PRESsfc_mean": ("Mean surface pressure", "Pa"),
    "Q2m_mean": ("Mean 2m specific humidity", "kg/kg"),
    "UGRD10m_mean": ("Mean 10m U component of wind", "m/s"),
    "VGRD10m_mean": ("Mean 10m V component of wind", "m/s"),
    "WIND10m_mean": ("Mean 10m wind speed", "m/s"),
    "global_mean_co2": ("Global mean CO2 mole fraction", "mol/mol"),
    **{f"{s}_soil_type_fraction": (f"Fraction of {s} soil type", "fraction") for s in SOIL_TYPES},
}
LAYER_FAMILY_ATTRS = {
    "air_temperature": ("Temperature", "K"),
    "specific_total_water": ("Specific total water", "kg/kg"),
    "eastward_wind": ("U component of wind", "m/s"),
    "northward_wind": ("V component of wind", "m/s"),
}


def _set_attrs(ds: xr.Dataset) -> xr.Dataset:
    for name in ds.data_vars:
        if name in DESIRED_ATTRS:
            long_name, units = DESIRED_ATTRS[name]
            ds[name].attrs = {"long_name": long_name, "units": units}
    return ds


def _only_horizontal_coords(ds: xr.Dataset) -> xr.Dataset:
    return ds.drop_vars([c for c in ds.coords if c not in ("latitude", "longitude")])


def _check_data_validity(ds: xr.Dataset, when) -> None:
    """Vendored `_check_data_validity` for one time step: raise on any NaN,
    except masked fields (SST, sea ice, waves), which raise only if all-NaN.
    ARCO's valid_stop_time can run ahead of some variables
    (google-research/arco-era5#128), so this is worth its cost."""
    for name, da in ds.data_vars.items():
        bad = da.isnull().all() if name in VARIABLES_WITH_SOME_MISSING_VALUES else da.isnull().any()
        if bool(bad):
            raise ValueError(f"missing values in ARCO {name!r} at {when}")


# ---------------------------------------------------------------------------
# Streams (vendored processing; each returns F90 fields without a time dim)
# ---------------------------------------------------------------------------


def _isclose(a, b, **kwargs):
    return xr.apply_ufunc(np.isclose, a, b, kwargs=kwargs, output_dtypes=[bool])


def process_invariant(ds: xr.Dataset) -> xr.Dataset:
    """HGTsfc, land_fraction and one-hot soil type fractions (vendored
    `_process_invariant`)."""
    out = xr.Dataset()
    out["HGTsfc"] = ds["geopotential_at_surface"] / GRAVITY
    out["land_fraction"] = ds["land_sea_mask"]
    for soil_type, soil_id in SOIL_TYPES.items():
        out[f"{soil_type}_soil_type_fraction"] = _isclose(ds["soil_type"], soil_id, atol=1.0e-3, rtol=0.0).astype(np.float32)
    return _set_attrs(_regrid(_only_horizontal_coords(out)))


def _coarsen_layers(field: np.ndarray, ak, bk, ps: np.ndarray, layer_indices) -> list:
    """Pressure-weighted layer means of one (137, lat, lon) float32 field --
    `_vertical_coarsen` with dp built one coarse layer at a time, so memory
    stays ~2 GB per time step instead of ~10 GB (Polaris login nodes cap a
    user at 8 GB). Returns float64 (lat, lon) arrays."""
    layers = []
    for k0, k1 in zip(layer_indices[:-1], layer_indices[1:]):
        dp = (ak[k0 + 1 : k1 + 1] - ak[k0:k1])[:, None, None] + (bk[k0 + 1 : k1 + 1] - bk[k0:k1])[:, None, None] * ps
        layers.append(np.einsum("kij,kij->ij", field[k0:k1], dp, dtype=np.float64) / dp.sum(0))
    return layers


def _model_level(ml_store: xr.Dataset, t64, sfc: xr.Dataset, ak, bk, layer_indices, families: dict, check: bool) -> xr.Dataset:
    """Vendored `_process_model_level_data`: pressure-weighted coarsening of
    each family at 0.25 deg, then regrid; Q2m from the regridded DPT2m and
    PRESsfc. A family that sums several ERA5 fields (total water) is the sum
    of their coarsened layers -- the layer mean is linear, so this equals
    coarsening the sum -- which lets one 3D field be in memory at a time."""
    ps = sfc["surface_pressure"].values.astype(np.float64)
    out = xr.Dataset()
    for family, sources in families.items():
        total, attrs = None, {}
        for src in sources:
            da = ml_store[src].sel(time=t64).load()
            if check and bool(da.isnull().any()):
                raise ValueError(f"missing values in ARCO {src!r} at {t64}")
            attrs = attrs or da.attrs
            layers = _coarsen_layers(da.values, ak, bk, ps, layer_indices)
            del da
            total = layers if total is None else [a + b for a, b in zip(total, layers)]
        long_name, units = LAYER_FAMILY_ATTRS.get(family, (attrs.get("long_name", family), attrs.get("units", "")))
        for i, layer in enumerate(total):
            out[f"{family}_{i}"] = xr.DataArray(
                layer.astype(np.float32),
                dims=("latitude", "longitude"),
                coords={"latitude": sfc.latitude, "longitude": sfc.longitude},
                attrs={"long_name": f"{long_name} level-{i}", "units": units},
            )
    for src, dst in MODEL_LEVEL_SURFACE_RENAME.items():
        out[dst] = sfc[src]
    out = _regrid(_only_horizontal_coords(out))
    out["Q2m"] = _specific_humidity_from_dewpoint(out["DPT2m"], out["PRESsfc"])
    return out


def _mean_flux(ds: xr.Dataset) -> xr.Dataset:
    """Vendored `_process_mean_flux` on the 6-h mean of hourly ERA5 means."""
    out = xr.Dataset()
    out["DSWRFtoa"] = ds["mean_top_downward_short_wave_radiation_flux"]
    out["USWRFtoa"] = ds["mean_top_downward_short_wave_radiation_flux"] - ds["mean_top_net_short_wave_radiation_flux"]
    out["ULWRFtoa"] = -ds["mean_top_net_long_wave_radiation_flux"]
    out["DSWRFsfc"] = ds["mean_surface_downward_short_wave_radiation_flux"]
    out["USWRFsfc"] = ds["mean_surface_downward_short_wave_radiation_flux"] - ds["mean_surface_net_short_wave_radiation_flux"]
    out["DLWRFsfc"] = ds["mean_surface_downward_long_wave_radiation_flux"]
    out["ULWRFsfc"] = ds["mean_surface_downward_long_wave_radiation_flux"] - ds["mean_surface_net_long_wave_radiation_flux"]
    out["UCSWRFtoa"] = ds["mean_top_downward_short_wave_radiation_flux"] - ds["mean_top_net_short_wave_radiation_flux_clear_sky"]
    out["UCLWRFtoa"] = -ds["mean_top_net_long_wave_radiation_flux_clear_sky"]
    out["DCSWRFsfc"] = ds["mean_surface_downward_short_wave_radiation_flux_clear_sky"]
    out["UCSWRFsfc"] = (
        ds["mean_surface_downward_short_wave_radiation_flux_clear_sky"] - ds["mean_surface_net_short_wave_radiation_flux_clear_sky"]
    )
    out["DCLWRFsfc"] = ds["mean_surface_downward_long_wave_radiation_flux_clear_sky"]
    out["UCLWRFsfc"] = (
        ds["mean_surface_downward_long_wave_radiation_flux_clear_sky"] - ds["mean_surface_net_long_wave_radiation_flux_clear_sky"]
    )
    out["SHTFLsfc"] = -ds["mean_surface_sensible_heat_flux"]
    out["LHTFLsfc"] = -ds["mean_surface_latent_heat_flux"]
    out["PRATEsfc"] = ds["mean_total_precipitation_rate"]
    out["total_frozen_precipitation_rate"] = ds["mean_snowfall_rate"]
    out["runoff_flux"] = ds["mean_runoff_rate"]
    out["tendency_of_total_water_path_due_to_advection"] = -ds["mean_vertically_integrated_moisture_divergence"]
    out["eastward_surface_stress"] = ds["mean_eastward_gravity_wave_surface_stress"] + ds["mean_eastward_turbulent_surface_stress"]
    out["northward_surface_stress"] = ds["mean_northward_gravity_wave_surface_stress"] + ds["mean_northward_turbulent_surface_stress"]
    return _set_attrs(_regrid(_only_horizontal_coords(out)))


def _surface_mean(ds: xr.Dataset) -> xr.Dataset:
    """Vendored `_process_surface_mean`: regrid each hour, derive Q2m and
    wind speed, then average the 6 hours."""
    rg = _regrid(ds.drop_vars([c for c in ds.coords if c not in ("latitude", "longitude", "time")]))
    out = xr.Dataset()
    out["surface_temperature_mean"] = rg["skin_temperature"]
    out["PRESsfc_mean"] = rg["surface_pressure"]
    out["TMP2m_mean"] = rg["2m_temperature"]
    out["DPT2m_mean"] = rg["2m_dewpoint_temperature"]
    out["Q2m_mean"] = _specific_humidity_from_dewpoint(rg["2m_dewpoint_temperature"], rg["surface_pressure"])
    out["UGRD10m_mean"] = rg["10m_u_component_of_wind"]
    out["VGRD10m_mean"] = rg["10m_v_component_of_wind"]
    out["WIND10m_mean"] = np.sqrt(rg["10m_u_component_of_wind"] ** 2 + rg["10m_v_component_of_wind"] ** 2)
    return _set_attrs(_only_horizontal_coords(out.mean("time")))


def _surface_analysis(ds: xr.Dataset, invariant: xr.Dataset) -> xr.Dataset:
    """Vendored `_process_surface_analysis`."""
    ds = _only_horizontal_coords(ds)
    out = xr.Dataset()
    out["sea_ice_fraction"] = ds["sea_ice_cover"].fillna(0.0)
    for k in range(4):
        out[f"soil_moisture_{k}"] = ds[f"volumetric_soil_water_layer_{k + 1}"]
        out[f"soil_temperature_{k}"] = ds[f"soil_temperature_level_{k + 1}"]
    out["surface_snow_amount"] = DENSITY_OF_LIQUID_WATER * ds["snow_depth"]
    frac = (DENSITY_OF_LIQUID_WATER * ds["snow_depth"] / ds["snow_density"]) / 0.1
    out["surface_snow_area_fraction"] = xr.where(frac > 1, 1, frac)
    out["surface_snow_thickness"] = (
        out["surface_snow_amount"] / (ds["snow_density"] * out["surface_snow_area_fraction"])
    ).fillna(0.0)
    rg = _regrid(out)
    # adaptive masking so coastal points get a value
    rg["sea_surface_temperature"] = _regrid(ds["sea_surface_temperature"], skipna=True, na_thres=1.0)
    rg["significant_height_of_combined_wind_waves_and_swell"] = _regrid(
        ds["significant_height_of_combined_wind_waves_and_swell"], skipna=True, na_thres=1.0
    ).fillna(0.0)
    ocean = 1 - invariant["land_fraction"] - rg["sea_ice_fraction"]
    negative = xr.where(ocean < 0, ocean, 0)
    rg["ocean_fraction"] = ocean - negative
    rg["sea_ice_fraction"] = rg["sea_ice_fraction"] + negative
    skin = _regrid(ds["skin_temperature"])
    land_or_ice = (rg["ocean_fraction"] < 0.5) | rg["sea_surface_temperature"].isnull()
    rg["merged_sea_surface_and_skin_temperature"] = xr.where(land_or_ice, skin, rg["sea_surface_temperature"])
    return _set_attrs(rg)


def _pressure_levels(ds: xr.Dataset) -> xr.Dataset:
    """Vendored `_process_pressure_level_data`: level select (geopotential ->
    height), regrid, rename to h500/TMP850/..."""
    out = xr.Dataset()
    for name in ds.data_vars:
        for p in OUTPUT_PRESSURE_LEVELS:
            da = ds[name].sel(level=p)
            if name == "geopotential":
                da = (da / GRAVITY).assign_attrs(long_name=f"Geopotential height at {p} hPa", units="m")
            else:
                da = da.assign_attrs(long_name=f"{ds[name].attrs.get('long_name', name)} at {p} hPa")
            out[f"{name}_{p}"] = da
    return _regrid(_only_horizontal_coords(out)).rename(RENAME_PRESSURE_LEVEL)


# ---------------------------------------------------------------------------
# Surface pressure at HGTsfc
# ---------------------------------------------------------------------------

PRESSURE_REDUCTION_VARS = ["surface_pressure", "2m_temperature", "2m_dewpoint_temperature"]
PRESFC_ATTR = "presfc_reduced_to_hgtsfc"  # global attribute, as in ace2_ic.py's ICs


def _load_regrid_pairs(weights: str) -> tuple:
    """(row, col, S), 0-based, of the 0.25 deg -> F90 conservative regridder's
    sparse weights: F90 cell row[k] gets S[k] x 0.25-deg cell col[k], both
    grids flattened (latitude ascending, longitude) as `_regrid` sees them."""
    with xr.open_dataset(weights) as w:
        return (w["row"].values.astype(np.int64) - 1, w["col"].values.astype(np.int64) - 1, w["S"].values.astype(np.float64))


def _apply_pairs(field: np.ndarray, pairs: tuple, n_out: int) -> np.ndarray:
    """Conservative regrid of one flattened 0.25-deg field (= `_regrid`)."""
    row, col, S = pairs
    return np.bincount(row, S * field[col], minlength=n_out)


def _surface_height(f37: xr.Dataset, t64, pairs: tuple, hgtsfc: xr.DataArray) -> np.ndarray:
    """ERA5 0.25-deg surface height (m), flattened like `_load_regrid_pairs`;
    raises unless it regrids to HGTsfc (guards the flattening order)."""
    z = f37["geopotential_at_surface"].sel(time=t64).load().sortby("latitude")
    h = z.values.astype(np.float64).ravel() / GRAVITY
    err = np.abs(_apply_pairs(h, pairs, hgtsfc.size) - hgtsfc.values.ravel()).max()
    if not err < 1.0:
        raise ValueError(f"0.25-deg surface height regrids to HGTsfc only within {err:.1f} m")
    return h


def _presfc_at_hgtsfc(ds: xr.Dataset, h: np.ndarray, hgtsfc: xr.DataArray, pairs: tuple) -> xr.DataArray:
    """F90 surface pressure valid at HGTsfc, from the 0.25-deg
    PRESSURE_REDUCTION_VARS in `ds` (optionally with a leading time dim) and
    the 0.25-deg surface height `h` (from `_surface_height`).

    The plain conservative regrid averages pressures valid at different
    heights, and since ps falls off ~exponentially with height that average
    exceeds the pressure at the cell's mean height (HGTsfc, the same regrid of
    the ERA5 orography) over steep terrain: measured (1985) ~20 Pa where the
    sub-grid height std is 100-300 m, ~90 Pa at 300-600 m, ~350 Pa above
    600 m, 1.3 hPa at most; ~1 Pa over the 89% of the globe below 100 m.
    Here each 0.25-deg ps is first reduced hydrostatically from its own
    height to the HGTsfc of each F90 cell it overlaps
    (`ace2_ic._reduce_surface_pressure`, 6.5 K/km from the 2-m virtual
    temperature), then averaged with the same weights."""
    row, col, S = pairs
    ds = ds[PRESSURE_REDUCTION_VARS].sortby("latitude").transpose(..., "latitude", "longitude")
    lead_dims = ds["surface_pressure"].dims[:-2]
    n_fine = ds.sizes["latitude"] * ds.sizes["longitude"]
    ps, t, dpt = (ds[v].values.astype(np.float64).reshape(-1, n_fine) for v in PRESSURE_REDUCTION_VARS)
    tv = t * (1.0 + (461.5 / ic.RDGAS - 1.0) * _specific_humidity_from_dewpoint(dpt, ps))
    h_from, h_to = h[col], hgtsfc.values.astype(np.float64).ravel()[row]
    out = np.stack(
        [np.bincount(row, S * ic._reduce_surface_pressure(ps[i, col], tv[i, col], h_from, h_to), minlength=hgtsfc.size) for i in range(ps.shape[0])]
    )
    shape = tuple(ds.sizes[d] for d in lead_dims) + hgtsfc.shape
    coords = {**{d: ds[d] for d in lead_dims if d in ds.coords}, **{d: hgtsfc[d] for d in hgtsfc.dims}}
    return xr.DataArray(out.reshape(shape), dims=lead_dims + hgtsfc.dims, coords=coords)


# ---------------------------------------------------------------------------
# Per-time-step driver (worker processes)
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(weights: str, invariant_path: str, layer_indices, extra_layer_vars, top_interface_midpoint, check, reduce_presfc):
    xr.set_options(keep_attrs=True)
    ic.make_regridder(weights)
    ml = xr.open_zarr(URL_MODEL_LEVEL, chunks=None, storage_options=STORAGE_OPTIONS)
    ak, bk = _get_ak_bk(ml, top_interface_midpoint)
    families = dict(LAYER_VARS)
    families.update({v: [v] for v in extra_layer_vars})
    with xr.open_dataset(invariant_path) as inv:
        invariant = inv.load()
    _W.update(
        ml=ml,
        f37=xr.open_zarr(URL_FULL_37, chunks=None, storage_options=STORAGE_OPTIONS),
        ak=ak,
        bk=bk,
        layer_indices=list(layer_indices),
        families=families,
        invariant=invariant,
        check=check,
        pairs=_load_regrid_pairs(weights) if reduce_presfc else None,
        h=None,  # 0.25-deg surface height, read on first use (`_presfc`)
    )


def _presfc(ds: xr.Dataset, t64) -> xr.DataArray:
    """`_presfc_at_hgtsfc` with this worker's weights, HGTsfc and (cached)
    0.25-deg surface height."""
    w = _W
    hgtsfc = w["invariant"]["HGTsfc"]
    if w["h"] is None:
        w["h"] = _surface_height(w["f37"], t64, w["pairs"], hgtsfc)
    return _presfc_at_hgtsfc(ds, w["h"], hgtsfc, w["pairs"])


def process_time(t: pd.Timestamp) -> xr.Dataset:
    """All time-varying F90 fields at output time `t` (expanded to time=[t])."""
    w = _W
    f37, when = w["f37"], f"{t:%Y-%m-%dT%H}"
    t64 = np.datetime64(t, "ns")
    window = slice(np.datetime64(t - pd.Timedelta(hours=5), "ns"), t64)

    def read(names, when_):
        ds = f37[names].sel(time=when_).load()
        if w["check"]:
            _check_data_validity(ds, when)
        return ds

    sfc = read(FULL_37_MODEL_LEVEL_SURFACE_VARS, t64)
    model_level = _model_level(w["ml"], t64, sfc, w["ak"], w["bk"], w["layer_indices"], w["families"], w["check"])
    # Q2m / Q2m_mean stay computed from the plain regridded ps, consistent
    # with where DPT2m is valid (as in ace2_ic.py)
    if w["pairs"] is not None:
        model_level["PRESsfc"] = _presfc(sfc, t64)
    parts = [model_level]
    flux = read(FULL_37_MEAN_FLUX_VARS, window)
    if flux.sizes["time"] != 6:
        raise ValueError(f"{when}: expected 6 hourly means, got {flux.sizes['time']}")
    parts.append(_mean_flux(flux.mean("time")))
    del flux
    sfc_hourly = read(FULL_37_SURFACE_MEAN_VARS, window)
    surface_mean = _surface_mean(sfc_hourly)
    if w["pairs"] is not None:
        surface_mean["PRESsfc_mean"] = _presfc(sfc_hourly, t64).mean("time")
    parts.append(surface_mean)
    del sfc_hourly
    parts.append(_surface_analysis(read(FULL_37_SURFACE_ANALYSIS_VARS, t64), w["invariant"]))
    parts.append(_pressure_levels(read(FULL_37_PRESSURE_LEVEL_VARS, t64)))
    out = xr.merge([_only_horizontal_coords(p) for p in parts], compat="override")
    return out.astype(np.float32).expand_dims(time=[t64])


# ---------------------------------------------------------------------------
# Build (parent process)
# ---------------------------------------------------------------------------


def _vertical_coordinate(ak, bk, layer_indices) -> xr.Dataset:
    """Coarse-interface ak_N (Pa) / bk_N scalars (vendored
    `_get_vertical_coordinate`)."""
    ds = xr.Dataset()
    for i, k in enumerate(layer_indices):
        ds[f"ak_{i}"] = xr.DataArray(float(ak[k]), attrs={"long_name": "ak", "units": "Pa"})
    for i, k in enumerate(layer_indices):
        ds[f"bk_{i}"] = xr.DataArray(float(bk[k]), attrs={"long_name": "bk", "units": ""})
    return ds


def month_times(month: pd.Timestamp) -> pd.DatetimeIndex:
    times = pd.date_range(month, month + pd.offsets.MonthBegin(1) - TIME_STEP, freq=TIME_STEP)
    return times[times >= FIRST_TIME]


def build(args) -> None:
    layer_indices = list(args.layer_indices)
    if layer_indices[0] != 0 or layer_indices[-1] != N_INPUT_LAYERS or np.any(np.diff(layer_indices) <= 0):
        raise ValueError(f"--layer_indices must increase from 0 to {N_INPUT_LAYERS}: {layer_indices}")
    unknown = set(args.extra_layer_vars) - set(EXTRA_LAYER_VARS)
    if unknown:
        raise ValueError(f"unknown --extra_layer_vars {sorted(unknown)}; choose from {EXTRA_LAYER_VARS}")
    os.makedirs(args.outdir, exist_ok=True)
    xr.set_options(keep_attrs=True)

    with xr.open_dataset(args.co2) as ds:
        co2 = ds["global_mean_co2"].load()
    months = pd.date_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq="MS")
    todo = [m for m in months if args.overwrite or not os.path.exists(os.path.join(args.outdir, f"{m:%Y%m%d%H}.nc"))]
    logging.info(f"{len(todo)} of {len(months)} months to build into {args.outdir}")
    if not todo:
        return
    for m in todo:  # fail now, not after hours of processing
        missing = month_times(m).difference(pd.DatetimeIndex(co2.time.values))
        if len(missing):
            raise ValueError(f"{args.co2} has no global_mean_co2 at {missing[0]} (and {len(missing) - 1} more)")

    # Shared by every month and worker: regrid weights and static fields.
    weights = os.path.join(args.outdir, "regrid_weights_0p25deg_to_F90.nc")
    if not os.path.exists(weights):
        ic.make_regridder().to_netcdf(weights)
    ic.make_regridder(weights)
    invariant_path = os.path.join(args.outdir, "invariant.nc")
    if not os.path.exists(invariant_path):
        # values are constant in time, but early times in the store may hold
        # NaN fill values, so read them at the first time being built
        t0 = np.datetime64(month_times(todo[0])[0], "ns")
        f37 = xr.open_zarr(URL_FULL_37, chunks=None, storage_options=STORAGE_OPTIONS)
        inv_raw = f37[FULL_37_INVARIANT_VARS].sel(time=t0).load()
        _check_data_validity(inv_raw, "invariant")
        process_invariant(inv_raw).astype(np.float32).to_netcdf(invariant_path)
    with xr.open_dataset(invariant_path) as ds:
        invariant = ds.load()
    ml = xr.open_zarr(URL_MODEL_LEVEL, chunks=None, storage_options=STORAGE_OPTIONS)
    static = xr.merge([invariant, _vertical_coordinate(*_get_ak_bk(ml, not args.model_top_zero), layer_indices)])

    attrs = {
        "source": "ARCO-ERA5 (gs://gcp-public-data-arco-era5), processed by lw4dvar-torch backends/ace2/ace2_dataset.py",
        "processing": UPSTREAM,
        "output_layer_indices": layer_indices,
        "extra_layer_vars": " ".join(args.extra_layer_vars),
        "top_interface_midpoint": int(not args.model_top_zero),
        "global_mean_co2_source": os.path.abspath(args.co2),
        PRESFC_ATTR: int(not args.no_reduce_presfc),
    }
    ctx = mp.get_context("spawn")  # gcsfs/zarr event loops are not fork-safe
    initargs = (weights, invariant_path, layer_indices, args.extra_layer_vars, not args.model_top_zero, not args.no_check, not args.no_reduce_presfc)
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=initargs) as pool:
        for m in todo:
            times = month_times(m)
            path = os.path.join(args.outdir, f"{m:%Y%m%d%H}.nc")
            t0 = _time.time()
            steps = []
            for i, step in enumerate(pool.imap(process_time, times), 1):
                steps.append(step)
                if i % 20 == 0:
                    logging.info(f"{m:%Y-%m}: {i}/{len(times)} times, {(_time.time() - t0) / i:.1f} s/time")
            ds = xr.concat(steps, "time")
            del steps
            ds["global_mean_co2"] = co2.sel(time=ds.time).astype(np.float64)
            ds = _set_attrs(xr.merge([ds, static]))
            ds.attrs = attrs
            enc = {"time": {"units": TIME_UNITS, "calendar": "proleptic_gregorian", "dtype": "int64"}}
            if args.complevel:
                enc.update({v: {"zlib": True, "complevel": args.complevel} for v in ds.data_vars if "time" in ds[v].dims})
            tmp = path + ".tmp"
            ds.to_netcdf(tmp, unlimited_dims=["time"], encoding=enc)
            os.replace(tmp, path)
            logging.info(f"wrote {path}: {len(times)} times in {(_time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# Patch PRESsfc in files built without the reduction
# ---------------------------------------------------------------------------


def _init_fix_worker(weights: str, invariant_path: str, check: bool):
    xr.set_options(keep_attrs=True)
    with xr.open_dataset(invariant_path) as inv:
        invariant = inv.load()
    _W.update(
        f37=xr.open_zarr(URL_FULL_37, chunks=None, storage_options=STORAGE_OPTIONS),
        invariant=invariant,
        check=check,
        pairs=_load_regrid_pairs(weights),
        h=None,
    )


def _fix_presfc_time(t: pd.Timestamp) -> tuple:
    """(plain PRESsfc, plain PRESsfc_mean, PRESsfc, PRESsfc_mean) at output
    time `t` as F90 float64 arrays; "plain" = no reduction, i.e. what `build`
    wrote before it (used to check the file being patched)."""
    w = _W
    when, t64 = f"{t:%Y-%m-%dT%H}", np.datetime64(t, "ns")
    window = slice(np.datetime64(t - pd.Timedelta(hours=5), "ns"), t64)
    sfc = w["f37"][PRESSURE_REDUCTION_VARS].sel(time=t64).load()
    hourly = w["f37"][PRESSURE_REDUCTION_VARS].sel(time=window).load()
    if w["check"]:
        _check_data_validity(sfc, when)
        _check_data_validity(hourly, when)
    if hourly.sizes["time"] != 6:
        raise ValueError(f"{when}: expected 6 hourly values, got {hourly.sizes['time']}")
    hgtsfc = w["invariant"]["HGTsfc"]

    def plain(ps):
        ps = ps.sortby("latitude").transpose(..., "latitude", "longitude").values.astype(np.float64)
        flat = ps.reshape(-1, ps.shape[-2] * ps.shape[-1])
        return np.stack([_apply_pairs(f, w["pairs"], hgtsfc.size) for f in flat]).reshape(-1, *hgtsfc.shape)

    return (
        plain(sfc["surface_pressure"])[0],
        plain(hourly["surface_pressure"]).mean(0),
        _presfc(sfc, t64).values,
        _presfc(hourly, t64).mean("time").values,
    )


def fix_presfc(args) -> None:
    """Rewrite PRESsfc and PRESsfc_mean in place, reduced to HGTsfc
    (`_presfc_at_hgtsfc`), in the monthly files of OUTDIR built without the
    reduction (global attribute PRESFC_ATTR absent or 0); all other fields are
    untouched. Each month is first checked against a recomputation of the
    plain regrid (< 1 Pa) so that a file whose times or grid don't match is
    never patched. Reads only the 3 surface fields per hour (~1/30 of
    `build`'s reads). Files are found by name only once complete (`build`
    writes .tmp and renames), so this can run beside a `build` job."""
    import netCDF4

    weights = os.path.join(args.outdir, "regrid_weights_0p25deg_to_F90.nc")
    invariant_path = os.path.join(args.outdir, "invariant.nc")
    todo = []
    for m in pd.date_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq="MS"):
        path = os.path.join(args.outdir, f"{m:%Y%m%d%H}.nc")
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            if int(ds.attrs.get(PRESFC_ATTR, 0)):
                continue
            todo.append((path, pd.DatetimeIndex(ds.time.values)))
    logging.info(f"{len(todo)} months to patch in {args.outdir}")
    if not todo:
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_fix_worker, initargs=(weights, invariant_path, not args.no_check)) as pool:
        for path, times in todo:
            t0 = _time.time()
            plain, plain_mean, new, new_mean = (np.stack(a) for a in zip(*pool.imap(_fix_presfc_time, times)))
            with xr.open_dataset(path) as ds:
                err = max(np.abs(ds["PRESsfc"].values - plain).max(), np.abs(ds["PRESsfc_mean"].values - plain_mean).max())
            if not err < 1.0:
                raise ValueError(f"{path}: plain PRESsfc recomputed differs from the file by {err:.2f} Pa; not patching")
            with netCDF4.Dataset(path, "r+") as nc:
                nc["PRESsfc"][:] = new.astype(np.float32)
                nc["PRESsfc_mean"][:] = new_mean.astype(np.float32)
                nc.setncattr(PRESFC_ATTR, 1)
            d = new - plain
            logging.info(
                f"patched {path}: {len(times)} times in {(_time.time() - t0) / 60:.1f} min "
                f"(check {err:.3f} Pa; change mean {d.mean():.1f}, min {d.min():.0f}, max {d.max():.0f} Pa)"
            )


# ---------------------------------------------------------------------------
# CO2
# ---------------------------------------------------------------------------


def fetch_co2(path: str, first_year: int, last_year: int) -> None:
    """Write ACE2-ERA5's own 6-hourly `global_mean_co2` (mol/mol) for the
    given years to `path`, read out of the public HF forcing_YYYY.nc files
    over HTTP range requests (1940-2022 exist)."""
    import fsspec

    series = []
    for year in range(first_year, last_year + 1):
        url = HF_FORCING_URL.format(year=year)
        with fsspec.open(url, block_size=2**20) as f, xr.open_dataset(f, engine="h5netcdf") as ds:
            series.append(ds["global_mean_co2"].load())
        logging.info(f"{year}: {series[-1].sizes['time']} times")
    co2 = _set_attrs(xr.concat(series, "time").to_dataset())
    co2.attrs = {"source": "global_mean_co2 from huggingface.co/allenai/ACE2-ERA5 forcing_data/forcing_YYYY.nc"}
    co2.to_netcdf(path, encoding={"time": {"units": TIME_UNITS, "dtype": "int64"}})
    logging.info(f"wrote {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="build monthly training files")
    b.add_argument("outdir")
    b.add_argument("start", help="first month, YYYY-MM")
    b.add_argument("end", help="last month, YYYY-MM (inclusive)")
    b.add_argument("--co2", required=True, help="netCDF with 6-hourly global_mean_co2 (see the co2 mode)")
    b.add_argument("--workers", type=int, default=4, help="time steps processed in parallel (~2-3 GB each)")
    b.add_argument("--layer_indices", type=int, nargs="+", default=DEFAULT_OUTPUT_LAYER_INDICES)
    b.add_argument("--extra_layer_vars", nargs="*", default=[], help=f"from {EXTRA_LAYER_VARS}")
    b.add_argument("--model_top_zero", action="store_true", help="model top 0 Pa (ACE2-ERA5) instead of ~1 Pa (upstream, default)")
    b.add_argument("--complevel", type=int, default=0, help="zlib level for time-varying fields (0: uncompressed, like Ai2's)")
    b.add_argument("--no_check", action="store_true", help="skip the NaN checks on the ARCO inputs")
    b.add_argument("--no_reduce_presfc", action="store_true", help="plain regrid of PRESsfc(_mean), not reduced to HGTsfc (pre-2026-10-10 files)")
    b.add_argument("--overwrite", action="store_true")
    f = sub.add_parser("fix_presfc", help="reduce PRESsfc(_mean) to HGTsfc in place in months built without it")
    f.add_argument("outdir")
    f.add_argument("start", help="first month, YYYY-MM")
    f.add_argument("end", help="last month, YYYY-MM (inclusive)")
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--no_check", action="store_true", help="skip the NaN checks on the ARCO inputs")
    c = sub.add_parser("co2", help="extract ACE2-ERA5's global_mean_co2 from the HF forcing files")
    c.add_argument("path")
    c.add_argument("first_year", type=int)
    c.add_argument("last_year", type=int)
    args = p.parse_args()
    if args.mode == "co2":
        fetch_co2(args.path, args.first_year, args.last_year)
    elif args.mode == "fix_presfc":
        fix_presfc(args)
    else:
        build(args)
