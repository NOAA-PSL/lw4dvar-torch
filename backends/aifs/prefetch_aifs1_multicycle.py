"""
Prefetch a few more ERA5 dates into ic_cache_aifs1/ for aifs-single-1.1,
in preparation for a multi-cycle test (n_init > 1). Run from the repo
root on a login node (needs internet). Uses the same (now-fixed)
aifs_ic.read_single_date_fields path get_verif() calls per cycle -- each
date needs its own single-time, non-lagged prognostic+constant-forcing
fetch.

2015-01-01T00 is already cached (from the single-cycle IC-reading fix
validation). This fetches a handful of additional 12h-spaced dates
following it.
"""
import datetime
import sys

sys.path.insert(0, "backends/aifs")

from anemoi.inference.runners.simple import SimpleRunner
import aifs_ic

CHECKPOINT = "backends/aifs/aifs-single-1.1/aifs-single-mse-1.1.ckpt"
CACHE_DIR = "ic_cache_aifs1"

DATES = [
    datetime.datetime(2015, 1, 1, 12, tzinfo=datetime.timezone.utc),
    datetime.datetime(2015, 1, 2, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2015, 1, 2, 12, tzinfo=datetime.timezone.utc),
]

print("loading runner...", flush=True)
runner = SimpleRunner(CHECKPOINT, device="cpu")

for date in DATES:
    print(f"=== fetching {date.isoformat()} ===", flush=True)
    fields = aifs_ic.read_single_date_fields(runner, date, CACHE_DIR)
    print(f"  got {len(fields)} fields", flush=True)

print("Done.", flush=True)
