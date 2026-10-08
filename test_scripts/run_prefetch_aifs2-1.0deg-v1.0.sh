#!/bin/bash
#SBATCH -J fetch_ic_aifs2_1p0deg_v1p0
#SBATCH -o fetch_ic_aifs2_1p0deg_v1p0.%j.out
#SBATCH -e fetch_ic_aifs2_1p0deg_v1p0.%j.out
#SBATCH --account=gsienkf
#SBATCH --partition=u1-service
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH -t 24:00:00

# Warms ic_cache_aifs2-1.0deg-v1.0/ (backends/aifs/aifs_ic.py) for
# config_test_aifs2-1.0deg-v1.0.yml, BEFORE submitting
# run_aifs2-1.0deg-v1.0.sh -- the H100 partition that job runs on has no
# internet (see CLAUDE.md), so the CDS/ERA5 fetch has to happen here first.
#
# u1-service (not the GPU partition), because this is network-bound (CDS
# retrieval + MARS queueing), not compute-bound -- same reasoning as the
# aifs-single-mse-2.0-1deg project's own submit_fetch_ics_*.slurm scripts
# (confirmed: u1-service has internet, u1-h100 does not). 24h is
# u1-service's hard MaxTime (confirmed via `scontrol show partition
# u1-service`), not a choice here.
#
# Resumable: aifs_ic.fetch_era5_grib skips any date already cached in
# ic_cache_aifs2-1.0deg-v1.0/, so if this hits the walltime limit, just
# resubmit.
#
# Must run in the `lwaifs2` env, not the shared aifs2 env -- see
# integrate_aifs2-1deg-v1.0_backend.md for why.
#
# Submit with: sbatch run_prefetch_aifs2-1.0deg-v1.0.sh

set -x

source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwaifs2
projdir="$(pwd)/../"
tmpdir="$(pwd)/tmp"
mkdir -p ${tmpdir}
cd ${tmpdir}

# aifs_prefetch_ic.py reads config.yml (a fixed filename, see
# long_window_4dvar_utils.load_config()'s default) -- point it at this
# checkpoint's real config, matching every other run_*.sh's convention.
cp -r  ${projdir}/test_scripts/config_test_aifs2-1.0deg-v1.0_ic.yml config.yml

python -u ${projdir}/backends/aifs/aifs_prefetch_ic.py
status=$?

echo "aifs_prefetch_ic.py exited with status ${status}"
exit "${status}"
