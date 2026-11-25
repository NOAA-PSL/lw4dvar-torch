"""Alias sidecar for training a secondary decoder on fields the main network
already outputs (fme forbids duplicate names, and has no variable renaming):
writes SIDECAR/YYYYMM0100.nc holding copies of h500 and TMP850 named
h500_dec / TMP850_dec, with the dataset's time/latitude/longitude, for fme to
`merge:` with the main files; and STATS_OUT, a copy of STATS_IN with the
aliases added (same values as the originals) to every stats file.

usage: alias_sidecar.py DATASET SIDECAR STATS_IN STATS_OUT START END [--workers N]
(months whose sidecar exists are skipped, so an interrupted run can be resumed)"""
import argparse, os, shutil
from multiprocessing import Pool

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

ALIASES = {"h500": "h500_dec", "TMP850": "TMP850_dec"}


def one_month(job):
    src, dst = job
    if not os.path.exists(src) or os.path.exists(dst):
        return None
    tmp = dst + ".tmp"
    with netCDF4.Dataset(src) as d, netCDF4.Dataset(tmp, "w") as o:
        d.set_auto_mask(False)
        o.setncatts({k: d.getncattr(k) for k in d.ncattrs()})
        o.setncattr("history", f"{ALIASES} copies from {src} (alias_sidecar.py)")
        for dim in ("time", "latitude", "longitude"):
            o.createDimension(dim, None if d.dimensions[dim].isunlimited() else len(d.dimensions[dim]))
            v = d[dim]
            w = o.createVariable(dim, v.dtype, v.dimensions)
            w.setncatts(v.__dict__)
            w[:] = v[:]
        for a, b in ALIASES.items():
            s = d[a]
            v = o.createVariable(b, "f4", s.dimensions, fill_value=np.float32(np.nan))
            v.setncatts({k: s.getncattr(k) for k in s.ncattrs() if k != "_FillValue"})
            v.long_name = f"{getattr(s, 'long_name', a)} (copy of {a}, secondary-decoder target)"
            v[:] = s[:]
    os.replace(tmp, dst)
    return dst


def main():
    p = argparse.ArgumentParser()
    for a in ("dataset", "sidecar", "stats_in", "stats_out", "start", "end"):
        p.add_argument(a)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.sidecar, exist_ok=True)
    os.makedirs(a.stats_out, exist_ok=True)
    for f in sorted(os.listdir(a.stats_in)):
        if not f.endswith(".nc"):
            continue
        ds = xr.open_dataset(os.path.join(a.stats_in, f)).load()
        for s, t in ALIASES.items():
            ds[t] = ds[s].copy()
        ds.to_netcdf(os.path.join(a.stats_out, f))
        print("wrote", os.path.join(a.stats_out, f), flush=True)
    jobs = [(os.path.join(a.dataset, f"{m:%Y%m%d%H}.nc"), os.path.join(a.sidecar, f"{m:%Y%m%d%H}.nc"))
            for m in pd.date_range(pd.Timestamp(a.start), pd.Timestamp(a.end), freq="MS")]
    n = 0
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one_month, jobs):
            if r:
                n += 1
                if n % 50 == 0:
                    print(n, r, flush=True)
    print("done", n, "months", flush=True)


if __name__ == "__main__":
    main()
