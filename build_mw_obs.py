#!/usr/bin/env python
"""Obs files for a microwave sensor other than ATMS (AMSU-A; MHS later), one parquet per 6-h cycle,
in the layout `get_atmsobs` reads (radianced `scripts/build_lw4dvar_obs.py` does the same for ATMS).

Reads radianced's ERA5 pairs `<pre>/<YYYYMMDDHH>.parquet` (build_era5_pre.py; every satellite of the
sensor together, `sat_id` = index in atmsdec.diag.SENSORS[sensor]["sats"]) and keeps the obs side:

    lat, lon (deg), dt_hours (obs time minus cycle time, -3..3), sat_id
    Sat_Zenith_Angle, Scan_Angle, Sat_Azimuth_Angle (deg)
    land_frac, water_frac, ice_frac, snow_frac (FOV fractions, GSI diag), clw_obs (kg/m2)
    yobc_01..NN   y_o - BC (K), NaN where the channel is off in the replay for that satellite and date
    qc_01..NN     GSI qc flag (information only)
    withheld      True for a random 20% of obs (seed = cycle YYYYMMDDHH)

Example (radianced data, 2015-01 .. 2015-07):
    python build_mw_obs.py --sensor amsua --nch 15 --pre /scratch4/BMC/gsienkf/Sergey.Frolov/atms/data/amsua_era5_pre \
        --out /scratch4/BMC/gsienkf/Sergey.Frolov/obs/amsua_2015 --start 2015010100 --end 2015073118
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensor", required=True)
    p.add_argument("--nch", type=int, required=True)
    p.add_argument("--pre", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", default="0000000000")
    p.add_argument("--end", default="9999999999")
    p.add_argument("--withheld_fraction", type=float, default=0.2)
    a = p.parse_args()
    keep = {"lat": "lat", "lon": "lon", "Obs_Time": "dt_hours", "sat_id": "sat_id",
            "Sat_Zenith_Angle": "Sat_Zenith_Angle", "Scan_Angle": "Scan_Angle",
            "Sat_Azimuth_Angle": "Sat_Azimuth_Angle", "land_frac": "land_frac", "water_frac": "water_frac",
            "ice_frac": "ice_frac", "snow_frac": "snow_frac", "clw_obs": "clw_obs"}
    keep.update({f"ObsBC-{i}": f"yobc_{i + 1:02d}" for i in range(a.nch)})
    keep.update({f"qc-{i}": f"qc_{i + 1:02d}" for i in range(a.nch)})
    os.makedirs(a.out, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(a.pre, "*.parquet")))
             if a.start <= os.path.basename(f)[:10] <= a.end]
    n_tot = n_wh = 0
    for f in files:
        cyc = os.path.basename(f)[:10]
        d = pd.read_parquet(f, columns=list(keep)).rename(columns=keep).astype(np.float32)
        d["lon"] = d["lon"] % 360.0
        d["withheld"] = np.random.default_rng(int(cyc)).random(len(d)) < a.withheld_fraction
        d.to_parquet(os.path.join(a.out, f"{cyc}.parquet"), index=False)
        n_tot += len(d)
        n_wh += int(d["withheld"].sum())
    print(f"{a.sensor}: wrote {len(files)} cycles to {a.out}: {n_tot:,} obs, {n_wh:,} withheld")


if __name__ == "__main__":
    main()
