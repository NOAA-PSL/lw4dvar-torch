"""
Selectively fetch LFS content into the backends/ace2/ACE2-ERA5 git submodule
(https://huggingface.co/allenai/ACE2-ERA5, pinned like the AIFS/FCN3
submodules).

Unlike those model packages, allenai/ACE2-ERA5 also carries ~45 GB of yearly
forcing files (1940-2022) and a sample of training data, so the submodule is
used POINTER-ONLY by default and only the files a run needs are pulled:
the checkpoint, forcing_YYYY.nc for every year a rollout touches (forcings
-- SST, sea ice, insolation, CO2, ... -- are read at every step), and the
sample ic_2020.nc used by smoke_test_ace2.py / compare_ic_ace2.py.

Fresh clone of lw4dvar-torch: initialize the submodule WITHOUT smudging (or
git-lfs downloads everything, ~75 GB):
    GIT_LFS_SKIP_SMUDGE=1 git submodule update --init backends/ace2/ACE2-ERA5
then run this script. It also records the pulled paths in the submodule's
local lfs.fetchinclude so a later plain `git lfs pull` stays selective.

Must run from a LOGIN node (compute nodes have no internet). Run from the
repo root (any env with git-lfs):
    python backends/ace2/ace2_prefetch_checkpoint.py [YEAR ...]
(default years: 2014 2015 2020 -- the repo's test dates + the ic_2020 sample).
"""

import os
import subprocess
import sys

SUBMODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ACE2-ERA5")
BASE_FILES = ["ace2_era5_ckpt.tar", "initial_conditions/ic_2020.nc"]


def _git(*args):
    return subprocess.run(["git", "-C", SUBMODULE, *args], check=True, capture_output=True, text=True).stdout


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or [2014, 2015, 2020]
    wanted = BASE_FILES + [f"forcing_data/forcing_{y}.nc" for y in years]
    try:
        current = _git("config", "--get", "lfs.fetchinclude").strip()
    except subprocess.CalledProcessError:
        current = ""
    include = sorted(set(filter(None, current.split(","))) | set(wanted))
    _git("config", "lfs.fetchinclude", ",".join(include))
    _git("config", "lfs.fetchexclude", "training_validation_data/**")
    subprocess.run(["git", "-C", SUBMODULE, "lfs", "pull", "--include", ",".join(wanted)], check=True)
    for f in wanted:
        size = os.path.getsize(os.path.join(SUBMODULE, f))
        print(f"{f}: {size / 2**20:.0f} MiB" + ("  (STILL A POINTER?)" if size < 1024 else ""))
