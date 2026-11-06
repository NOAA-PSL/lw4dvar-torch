"""
Sub-6-hourly climatology of the ACE2 state: the part of the hourly ERA5 state
that linear interpolation between 6-hourly states misses (diurnal and
semidiurnal tides, the diurnal cycle of surface and boundary-layer fields),
composited by calendar month and UTC hour, for use in the 4D-Var loss when
observations fall between model steps (`dt_obs` < 6):

    C(t) = x(t) - [(1 - a) x(t_lo) + a x(t_hi)],   a = (t - t_lo) / 6h

with t_lo, t_hi the bracketing 00/06/12/18Z times (C = 0 there). Off-synoptic
states x(t) are processed by `ace2_dataset`'s own model-level pipeline (same
layer indices, model top and PRESsfc reduction as the dataset, read from its
global attributes), and x(t_lo), x(t_hi) are read from the 6-hourly dataset
itself -- so the two sides of the difference are processed identically and
static pipeline differences cancel. Only the model-level stream is processed:
every ACE2 prognostic field (the four layer families, PRESsfc, surface_temperature,
TMP2m, Q2m, UGRD10m, VGRD10m) plus PRMSL and DPT2m. Compositing the deviation
from linear interpolation, rather than the raw hourly state, removes most of
the synoptic variability, so a few years suffice.

Modes:
  accum   DATASET_DIR SUMDIR START END  -- per month, sum / sum of squares /
          count of C by UTC hour into SUMDIR/diurnal_sum_YYYYMM.nc (months
          already there are skipped, so jobs over disjoint ranges can share
          SUMDIR). Needs internet (u1-service) and the next month's dataset
          file (for the last day's 18-24Z bracket).
  combine SUMDIR OUT -- per calendar month over all years in SUMDIR: mean and
          std of C, dims (month 12, hour 24, latitude, longitude).

Cost: ~2/3 of `ace2_dataset build` per processed time (model level only), at
18 off-synoptic times per day. Env: ace2ic.
"""

import argparse
import glob
import logging
import multiprocessing as mp
import os
import time as _time

import numpy as np
import pandas as pd
import xarray as xr

import ace2_dataset as dsmod

HOURS = 24


def _state(t: pd.Timestamp) -> xr.Dataset:
    """Model-level ACE2 state at hourly time `t` (worker)."""
    w = dsmod._W
    t64 = np.datetime64(t, "ns")
    sfc = w["f37"][dsmod.FULL_37_MODEL_LEVEL_SURFACE_VARS].sel(time=t64).load()
    if w["check"]:
        dsmod._check_data_validity(sfc, f"{t:%Y-%m-%dT%H}")
    out = dsmod._model_level(w["ml"], t64, sfc, w["ak"], w["bk"], w["layer_indices"], w["families"], w["check"])
    if w["pairs"] is not None:
        out["PRESsfc"] = dsmod._presfc(sfc, t64)
    return dsmod._only_horizontal_coords(out).astype(np.float32)


def _off_times(month: pd.Timestamp) -> pd.DatetimeIndex:
    t = pd.date_range(month, month + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1), freq="h")
    return t[t.hour % 6 != 0]


def accum(args) -> None:
    os.makedirs(args.sumdir, exist_ok=True)
    months = pd.date_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq="MS")
    todo = [m for m in months if not os.path.exists(os.path.join(args.sumdir, f"diurnal_sum_{m:%Y%m}.nc"))]
    logging.info(f"{len(todo)} of {len(months)} months to accumulate into {args.sumdir}")
    if not todo:
        return
    path = lambda m: os.path.join(args.dataset_dir, f"{m:%Y%m}0100.nc")  # noqa: E731
    for m in todo:  # fail now, not after hours of processing
        for p in (path(m), path(m + pd.offsets.MonthBegin(1))):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
    with xr.open_dataset(path(todo[0])) as ds:
        attrs = dict(ds.attrs)
    if str(attrs.get("extra_layer_vars", "")).strip():
        raise ValueError("extra_layer_vars datasets not supported yet")
    layer_indices = [int(i) for i in np.atleast_1d(attrs["output_layer_indices"])]
    top_mid = bool(int(attrs["top_interface_midpoint"]))
    reduce_ps = bool(int(attrs.get(dsmod.PRESFC_ATTR, 0)))
    weights = os.path.join(args.dataset_dir, "regrid_weights_0p25deg_to_F90.nc")
    invariant = os.path.join(args.dataset_dir, "invariant.nc")
    logging.info(f"pipeline from {path(todo[0])}: layers {layer_indices}, top_interface_midpoint {top_mid}, "
                 f"PRESsfc reduced {reduce_ps}")

    ctx = mp.get_context("spawn")  # gcsfs/zarr event loops are not fork-safe
    initargs = (weights, invariant, layer_indices, [], top_mid, not args.no_check, reduce_ps)
    with ctx.Pool(args.workers, initializer=dsmod._init_worker, initargs=initargs) as pool:
        for m in todo:
            t0 = _time.time()
            times = _off_times(m)[: args.test_times or None]
            nxt = m + pd.offsets.MonthBegin(1)
            s = acc = acc2 = None
            count = np.zeros(HOURS, dtype=np.int64)
            for i, x in enumerate(pool.imap(_state, times), 1):
                if s is None:
                    names = list(x.data_vars)
                    with xr.open_dataset(path(m)) as a, xr.open_dataset(path(nxt)) as b:
                        s = xr.concat([a[names], b[names].isel(time=[0])], "time").load()
                    shape = (HOURS, len(names)) + x[names[0]].shape
                    acc = np.zeros(shape)
                    acc2 = np.zeros(shape)
                t = times[i - 1]
                lo = t.floor("6h")
                a = (t - lo) / pd.Timedelta(hours=6)
                xlo = s.sel(time=np.datetime64(lo, "ns"))
                xhi = s.sel(time=np.datetime64(lo + pd.Timedelta(hours=6), "ns"))
                c = np.stack([x[n].values - ((1 - a) * xlo[n].values + a * xhi[n].values) for n in names]).astype(np.float64)
                acc[t.hour] += c
                acc2[t.hour] += c * c
                count[t.hour] += 1
                if i % 72 == 0:
                    logging.info(f"{m:%Y-%m}: {i}/{len(times)} times, {(_time.time() - t0) / i:.1f} s/time")
            lat, lon = s.latitude, s.longitude
            dims = ("hour", "latitude", "longitude")
            out = xr.Dataset(
                {f"{n}_sum": (dims, acc[:, k]) for k, n in enumerate(names)}
                | {f"{n}_sumsq": (dims, acc2[:, k]) for k, n in enumerate(names)}
                | {"count": (("hour",), count)},
                coords={"hour": np.arange(HOURS), "latitude": lat, "longitude": lon},
                attrs={"month": f"{m:%Y-%m}", "variables": " ".join(names), "dataset_dir": args.dataset_dir,
                       "output_layer_indices": layer_indices, "top_interface_midpoint": int(top_mid),
                       dsmod.PRESFC_ATTR: int(reduce_ps)},
            )
            p = os.path.join(args.sumdir, f"diurnal_sum_{m:%Y%m}.nc")
            out.to_netcdf(p + ".tmp")
            os.replace(p + ".tmp", p)
            logging.info(f"wrote {p}: {len(times)} times in {(_time.time() - t0) / 60:.1f} min")


def combine(args) -> None:
    files = sorted(glob.glob(os.path.join(args.sumdir, "diurnal_sum_??????.nc")))
    by_month = {}
    for f in files:
        by_month.setdefault(int(os.path.basename(f)[-5:-3]), []).append(f)
    missing = sorted(set(range(1, 13)) - set(by_month))
    if missing:
        raise ValueError(f"no sums for calendar months {missing} in {args.sumdir}")
    means, stds, counts, years = [], [], [], {}
    for mo in range(1, 13):
        tot = None
        for f in by_month[mo]:
            with xr.open_dataset(f) as ds:
                tot = ds.load() if tot is None else tot + ds.load()
                names = ds.attrs["variables"].split()
                attrs = dict(ds.attrs)
        n = tot["count"].where(tot["count"] > 0)
        means.append(xr.Dataset({v: tot[f"{v}_sum"] / n for v in names}).fillna(0.0))
        stds.append(xr.Dataset({f"{v}_std": np.sqrt(np.maximum(tot[f"{v}_sumsq"] / n - (tot[f"{v}_sum"] / n) ** 2, 0))
                                for v in names}).fillna(0.0))
        counts.append(tot["count"])
        years[mo] = [os.path.basename(f)[12:16] for f in by_month[mo]]
    out = xr.merge([xr.concat(means, "month"), xr.concat(stds, "month")]).astype(np.float32)
    out["count"] = xr.concat(counts, "month")
    out = out.assign_coords(month=np.arange(1, 13))
    attrs.pop("month")
    out.attrs = attrs | {
        "definition": "C = x(t) - linear interpolation between the bracketing 00/06/12/18Z states; "
                      "mean (and std) over all days of each calendar month, by UTC hour; zero at hour % 6 == 0",
        "years": " ".join(sorted({y for v in years.values() for y in v})),
    }
    out.to_netcdf(args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)
    logging.info(f"wrote {args.out}: years {out.attrs['years']}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("accum", help="accumulate monthly sums of C by UTC hour")
    a.add_argument("dataset_dir")
    a.add_argument("sumdir")
    a.add_argument("start", help="YYYY-MM")
    a.add_argument("end", help="YYYY-MM")
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--no_check", action="store_true", help="skip the ARCO missing-value checks")
    a.add_argument("--test_times", type=int, default=0, help="only the first N off-synoptic times per month (testing)")
    c = sub.add_parser("combine", help="climatology (mean/std by calendar month and UTC hour)")
    c.add_argument("sumdir")
    c.add_argument("out")
    args = p.parse_args()
    accum(args) if args.mode == "accum" else combine(args)


if __name__ == "__main__":
    main()
