"""Summarize innovation statistics (O-B, O-A) and loss history, pooled over ALL
DA cycles (all *_observation_diagnostics_*.nc files) in an output directory.

Per lead time, statistics are pooled over every used observation from every
cycle (sums of O-B, (O-B)^2, ... over all files, then divided by the total
count) -- not an average of each file's RMS. O-A only exists from the
analysis time on (+dt_verif), so it is counted only where finite.

Usage: python diagnostics/analyze_obstats.py OUT_DIR [SUFFIX]
  SUFFIX (optional, e.g. 120h_50it) keeps only files from one run when several
  runs wrote into the same directory.
"""
import collections
import glob
import json
import re
import sys

import numpy as np
import xarray as xr

OUT_DIR = sys.argv[1]
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else None
#OUT_DIR = "output/test_aifs_latent_reset_skt_12h_100it_5day/"

pattern = f"/*_observation_diagnostics_{SUFFIX}.nc" if SUFFIX else "/*_observation_diagnostics_*.nc"
diag_files = sorted(glob.glob(OUT_DIR + pattern))
assert diag_files, "no observation_diagnostics.nc found"
dates = [re.match(r"(\d{4}-\d{2}-\d{2}T\d{2})", f.split("/")[-1]).group(1) for f in diag_files]
dups = sorted({d for d in dates if dates.count(d) > 1})
print(f"Loaded {len(diag_files)} files: {dates[0]} .. {dates[-1]}")
if dups:
    print(f"** WARNING: {len(dups)} date(s) have files from more than one run (e.g. {dups[0]}); "
          f"pass a SUFFIX argument to pick one run **")
print()

# pooled sums per lead hour
keys = ["n_avail", "n_used", "sb", "sbb", "na", "sa", "saa"]
acc = collections.defaultdict(lambda: dict.fromkeys(keys, 0.0))
for f in diag_files:
    with xr.open_dataset(f) as ds:
        used = ds["surface_pressure_used"].values.astype(bool)
        avail = ds["surface_pressure_available"].values.astype(bool)
        omb = ds["surface_pressure_omb"].values
        oma = ds["surface_pressure_oma"].values
        lead = ds["lead_time_hours"].values
    for i, h in enumerate(lead):
        a = acc[int(h)]
        m = used[i] & np.isfinite(omb[i])
        ma = used[i] & np.isfinite(oma[i])
        a["n_avail"] += avail[i].sum(); a["n_used"] += m.sum()
        a["sb"] += omb[i, m].sum(); a["sbb"] += (omb[i, m] ** 2).sum()
        a["na"] += ma.sum(); a["sa"] += oma[i, ma].sum(); a["saa"] += (oma[i, ma] ** 2).sum()

n_files = len(diag_files)
print(f"Pooled over {n_files} cycles (n_avail/n_used are per-cycle averages):")
print(f"{'lead(h)':>8} {'n_avail':>8} {'n_used':>8} {'mean(O-B)':>10} {'rms(O-B)':>10} {'mean(O-A)':>10} {'rms(O-A)':>10}")
tot = dict.fromkeys(keys, 0.0)
for h in sorted(acc):
    a = acc[h]
    for k in keys:
        tot[k] += a[k]
    nb, na = a["n_used"], a["na"]
    ob = f"{a['sb']/nb:10.3f} {np.sqrt(a['sbb']/nb):10.3f}" if nb else f"{'--':>10} {'--':>10}"
    oa = f"{a['sa']/na:10.3f} {np.sqrt(a['saa']/na):10.3f}" if na else f"{'--':>10} {'--':>10}"
    print(f"{h:8d} {a['n_avail']/n_files:8.0f} {nb/n_files:8.0f} {ob} {oa}")

print()
nb, na = tot["n_used"], tot["na"]
rb, ra = np.sqrt(tot["sbb"] / nb), np.sqrt(tot["saa"] / na)
print(f"ALL TIMES: n_used={int(nb)}  mean(O-B)={tot['sb']/nb:.3f}  rms(O-B)={rb:.3f}  "
      f"mean(O-A)={tot['sa']/na:.3f}  rms(O-A)={ra:.3f}  (O-A over {int(na)} obs)")
print(f"RMS reduction: {100*(1 - ra/rb):.1f}%")
# same comparison restricted to lead times where an analysis exists
lb = [h for h in acc if acc[h]["na"] > 0]
sbb = sum(acc[h]["sbb"] for h in lb); nbb = sum(acc[h]["n_used"] for h in lb)
print(f"RMS reduction over lead times with an analysis (>= +{min(lb)}h): "
      f"{100*(1 - ra/np.sqrt(sbb/nbb)):.1f}%")

print()
lpat = f"/*_loss_latent_{SUFFIX}.json" if SUFFIX else "/*_loss_*.json"
# only cycles that also have a diagnostics file (a cycle still running writes
# its loss file first; stale files from other runs are skipped too)
loss_files = [lf for lf in sorted(glob.glob(OUT_DIR + lpat))
              if re.match(r"(\d{4}-\d{2}-\d{2}T\d{2})", lf.split("/")[-1]).group(1) in set(dates)]
first, last, best, red = [], [], [], []
for lf in loss_files:
    with open(lf) as f:
        losses = [h["loss"] for h in json.load(f)]
    first.append(losses[0]); last.append(losses[-1]); best.append(min(losses))
    red.append(100 * (1 - min(losses) / losses[0]))
if loss_files:
    print(f"Loss history over {len(loss_files)} cycles (means): epoch 1 = {np.mean(first):.2f}, "
          f"final = {np.mean(last):.2f}, min = {np.mean(best):.2f}; "
          f"reduction (min vs epoch 1) mean {np.mean(red):.1f}% (range {min(red):.1f}-{max(red):.1f}%)")
