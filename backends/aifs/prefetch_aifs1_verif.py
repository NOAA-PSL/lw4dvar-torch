"""
Prefetch the specific verification dates config_test_aifs1.yml's get_verif()
call needs (window start + the z500-diagnostic-truth date at
window_start + dt_verif) -- missed earlier since the CPU-based debugging
run never got far enough to reach get_verif() before hitting the expected
CPU/flash_attn incompatibility.
"""
import datetime
import sys

sys.path.insert(0, "backends/aifs")

from anemoi.inference.runners.simple import SimpleRunner
import aifs_ic

CHECKPOINT = "backends/aifs/aifs-single-1.1/aifs-single-mse-1.1.ckpt"
CACHE_DIR = "ic_cache_aifs1"

DATES = [
    datetime.datetime(2015, 1, 1, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2015, 1, 1, 6, tzinfo=datetime.timezone.utc),
]

print("loading runner...", flush=True)
runner = SimpleRunner(CHECKPOINT, device="cpu")

for date in DATES:
    print(f"=== fetching {date.isoformat()} ===", flush=True)
    fields = aifs_ic.read_single_date_fields(runner, date, CACHE_DIR)
    print(f"  got {len(fields)} fields", flush=True)

print("Done.", flush=True)
