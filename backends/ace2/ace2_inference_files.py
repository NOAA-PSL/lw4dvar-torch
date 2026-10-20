"""
Forcing files and initial conditions for running OUR trained ACE2 checkpoint
(ace2-training.md) in lw4dvar, built from our training dataset
(backends/ace2/ace2_dataset.py output: monthly YYYYMM0100.nc, 6-hourly).

Why not the HF files: our dataset has its own HGTsfc (41 m rms from Ai2's),
model top ~1 Pa instead of 0, and PRESsfc consistent with that HGTsfc. The
network takes HGTsfc as an input and the ps observation operator uses the
model's HGTsfc (ACE2Model.surface_geopotential, read from the forcing files),
so a checkpoint trained on our data must be run with forcings and ICs from
the same data.

Writes, under OUT:
  forcing_data/forcing_YYYY.nc -- the HF forcing_YYYY.nc layout (same
      variables, static fields without a time axis, same time encoding), so
      ACE2Model reads it unchanged; set `exp.forcing_dir` to this directory.
  ic_cache/ace2_ic_YYYYMMDDHH.nc -- the ace2_ic.py IC layout (the variables
      of an existing IC, `--ic_template`), set `exp.ic_cache` to this dir.
  --link_verif DIR symlinks DIR's ace2_verif_*.nc (ERA5 h500/TMP850 truth,
      ace2_ic.py --verif) into ic_cache, so z500 scores match runs of the
      official checkpoint exactly.

Usage (ace2 env; no internet needed):
    python backends/ace2/ace2_inference_files.py OUT --years 2014 2015 \
        --ic 2014-12-30T00 --ic_template ic_cache_ace2/ace2_ic_2014123000.nc \
        --link_verif ic_cache_ace2 [--dataset DIR] [--hf_forcing FILE]
"""

import argparse
import datetime
import glob
import os

import xarray as xr

DATASET = "/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer"
HF_FORCING = "backends/ace2/ACE2-ERA5/forcing_data/forcing_2015.nc"


def write_forcing(dataset: str, hf_forcing: str, out_dir: str, year: int) -> str:
    """forcing_YYYY.nc from the year's monthly dataset files, with exactly the
    HF file's variables (static ones without a time axis)."""
    hf = xr.open_dataset(hf_forcing)
    names = list(hf.data_vars)
    files = sorted(glob.glob(os.path.join(dataset, f"{year}[01][0-9]0100.nc")))
    if len(files) != 12:
        raise FileNotFoundError(f"expected 12 monthly files for {year} in {dataset}, found {len(files)}")
    out = xr.concat([xr.open_dataset(f)[names].load() for f in files], dim="time",
                    data_vars="minimal", coords="minimal", compat="override")
    for v in names:
        if "time" not in hf[v].dims and "time" in out[v].dims:
            out[v] = out[v].isel(time=0, drop=True)
    out.time.encoding = {"units": "hours since 1940-01-01T12:00:00",
                         "calendar": "proleptic_gregorian", "dtype": "int64"}
    path = os.path.join(out_dir, f"forcing_{year}.nc")
    out.to_netcdf(path)
    return path


def write_ic(dataset: str, template: str, out_dir: str, date: datetime.datetime) -> str:
    """ace2_ic_YYYYMMDDHH.nc: the template IC's variables at `date`."""
    ref = xr.open_dataset(template)
    ds = xr.open_dataset(os.path.join(dataset, f"{date:%Y%m}0100.nc"))
    ic = ds[list(ref.data_vars)].sel(time=date).load()
    ic.attrs = dict(ref.attrs, source=f"ACE2 training dataset {dataset}", valid_time=f"{date:%Y-%m-%dT%H:%M:%S}",
                    presfc_reduced_to_hgtsfc=0)  # already consistent with the dataset's own HGTsfc
    path = os.path.join(out_dir, f"ace2_ic_{date:%Y%m%d%H}.nc")
    ic.to_netcdf(path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--hf_forcing", default=HF_FORCING, help="an HF forcing_YYYY.nc, for the variable list/layout")
    p.add_argument("--years", type=int, nargs="*", default=[])
    p.add_argument("--ic", nargs="*", default=[], help="IC dates YYYY-MM-DDTHH")
    p.add_argument("--ic_template", help="an existing ace2_ic_*.nc, for the variable list")
    p.add_argument("--link_verif", help="dir whose ace2_verif_*.nc are symlinked into OUT/ic_cache")
    a = p.parse_args()
    fdir, idir = os.path.join(a.out, "forcing_data"), os.path.join(a.out, "ic_cache")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(idir, exist_ok=True)
    for y in a.years:
        print("wrote", write_forcing(a.dataset, a.hf_forcing, fdir, y), flush=True)
    for d in a.ic:
        if not a.ic_template:
            p.error("--ic needs --ic_template")
        print("wrote", write_ic(a.dataset, a.ic_template, idir, datetime.datetime.fromisoformat(d)), flush=True)
    if a.link_verif:
        n = 0
        for f in sorted(glob.glob(os.path.join(os.path.abspath(a.link_verif), "ace2_verif_*.nc"))):
            dst = os.path.join(idir, os.path.basename(f))
            if not os.path.lexists(dst):
                os.symlink(f, dst)
                n += 1
        print(f"linked {n} verification files", flush=True)


if __name__ == "__main__":
    main()
