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

--checkpoint CKPT takes the variable lists from the checkpoint instead of an
HF forcing file and an IC template -- needed when its variables differ from
the released ACE2-ERA5's (the CRTM+land checkpoints: specific_humidity +
condensate, ozone, soil, snow): forcing = its input-only names plus the ocean
SST name and ak_k/bk_k; IC = its prognostic names. --verif_from_dataset
START END writes ace2_verif_*.nc (h500, TMP850) every 6 h from the dataset
itself, for a checkpoint on a grid the ERA5 verification isn't on (the
2.5 deg mini ACE2).

Usage (ace2 env; no internet needed):
    python backends/ace2/ace2_inference_files.py OUT --years 2014 2015 \
        --ic 2014-12-30T00 --ic_template ic_cache_ace2/ace2_ic_2014123000.nc \
        --link_verif ic_cache_ace2 [--dataset DIR] [--hf_forcing FILE]
    python backends/ace2/ace2_inference_files.py OUT --checkpoint CKPT \
        --dataset /scratch4/BMC/gsienkf/whitaker/ace2_era5_2p5deg_16layer \
        --years 2014 2015 --ic 2014-12-30T00 \
        --verif_from_dataset 2014-12-30T00 2015-01-31T18
"""

import argparse
import datetime
import glob
import os

import xarray as xr

DATASET = "/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer"
HF_FORCING = "backends/ace2/ACE2-ERA5/forcing_data/forcing_2015.nc"
# Fields the HF forcing files store without a time axis (ACE2Model reads
# HGTsfc as a static (lat, lon) array).
STATIC = ("HGTsfc", "land_fraction")


def checkpoint_names(checkpoint: str) -> tuple[list[str], list[str]]:
    """(forcing names, prognostic names) from an fme checkpoint (either
    layout): forcing = input-only names + the ocean config's SST name."""
    import torch

    st = torch.load(checkpoint, map_location="cpu", weights_only=False)["stepper"]
    cfg = st["config"]["step"]["config"] if "dataset_info" in st else st["config"]
    ins, outs = list(cfg["in_names"]), list(cfg["out_names"])
    forcing = [n for n in ins if n not in outs]
    sst = (cfg.get("ocean") or {}).get("surface_temperature_name")
    if sst and sst not in forcing:
        forcing.append(sst)
    return forcing, [n for n in ins if n in outs]


def write_forcing(dataset: str, names: list[str], static: set, out_dir: str, year: int) -> str:
    """forcing_YYYY.nc from the year's monthly dataset files with variables
    `names` (those in `static` without a time axis)."""
    files = sorted(glob.glob(os.path.join(dataset, f"{year}[01][0-9]0100.nc")))
    if len(files) != 12:
        raise FileNotFoundError(f"expected 12 monthly files for {year} in {dataset}, found {len(files)}")
    out = xr.concat([xr.open_dataset(f)[names].load() for f in files], dim="time",
                    data_vars="minimal", coords="minimal", compat="override")
    for v in names:
        if v in static and "time" in out[v].dims:
            out[v] = out[v].isel(time=0, drop=True)
    out.time.encoding = {"units": "hours since 1940-01-01T12:00:00",
                         "calendar": "proleptic_gregorian", "dtype": "int64"}
    path = os.path.join(out_dir, f"forcing_{year}.nc")
    out.to_netcdf(path)
    return path


def write_ic(dataset: str, names: list[str], attrs: dict, out_dir: str, date: datetime.datetime) -> str:
    """ace2_ic_YYYYMMDDHH.nc: variables `names` at `date`."""
    ds = xr.open_dataset(os.path.join(dataset, f"{date:%Y%m}0100.nc"))
    ic = ds[names].sel(time=date).load()
    ic.attrs = dict(attrs, source=f"ACE2 training dataset {dataset}", valid_time=f"{date:%Y-%m-%dT%H:%M:%S}",
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
    p.add_argument("--checkpoint", help="take the forcing/IC variable lists from this fme checkpoint")
    p.add_argument("--verif_from_dataset", nargs=2, metavar=("START", "END"),
                   help="write ace2_verif_*.nc (h500, TMP850) every 6 h from the dataset")
    a = p.parse_args()
    if a.checkpoint and a.ic_template:
        p.error("--checkpoint and --ic_template are alternatives")
    if a.verif_from_dataset and a.link_verif:
        p.error("--verif_from_dataset and --link_verif are alternatives")
    fdir, idir = os.path.join(a.out, "forcing_data"), os.path.join(a.out, "ic_cache")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(idir, exist_ok=True)
    if a.checkpoint:
        forcing, prognostic = checkpoint_names(a.checkpoint)
        with xr.open_dataset(sorted(glob.glob(os.path.join(a.dataset, "[12][0-9][0-9][0-9][01][0-9]0100.nc")))[0]) as ds:
            forcing += sorted((n for n in ds.data_vars if n.startswith(("ak_", "bk_"))), key=lambda n: (n[:2], int(n[3:])))
        static = set(STATIC) | {n for n in forcing if n.startswith(("ak_", "bk_"))}
        ic_attrs = {}
    else:
        with xr.open_dataset(a.hf_forcing) as hf:
            forcing = list(hf.data_vars)
            static = {v for v in forcing if "time" not in hf[v].dims}
        prognostic, ic_attrs = None, {}
        if a.ic_template:
            with xr.open_dataset(a.ic_template) as ref:
                prognostic, ic_attrs = list(ref.data_vars), dict(ref.attrs)
    for y in a.years:
        print("wrote", write_forcing(a.dataset, forcing, static, fdir, y), flush=True)
    for d in a.ic:
        if prognostic is None:
            p.error("--ic needs --ic_template or --checkpoint")
        print("wrote", write_ic(a.dataset, prognostic, ic_attrs, idir, datetime.datetime.fromisoformat(d)), flush=True)
    if a.verif_from_dataset:
        t, t1 = (datetime.datetime.fromisoformat(x) for x in a.verif_from_dataset)
        n = 0
        while t <= t1:
            with xr.open_dataset(os.path.join(a.dataset, f"{t:%Y%m}0100.nc")) as ds:
                v = ds[["h500", "TMP850"]].sel(time=t).drop_vars("time").load()
            v.attrs = {"source": f"ACE2 training dataset {a.dataset}", "valid_time": f"{t:%Y-%m-%dT%H:%M:%S}"}
            v.to_netcdf(os.path.join(idir, f"ace2_verif_{t:%Y%m%d%H}.nc"))
            t += datetime.timedelta(hours=6)
            n += 1
        print(f"wrote {n} verification files from the dataset", flush=True)
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
