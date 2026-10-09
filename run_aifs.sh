#!/bin/bash
#SBATCH -J lw4dvar_aifs
#SBATCH -o logs/lw4dvar_aifs.%j.out
#SBATCH -e logs/lw4dvar_aifs.%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH -t 01:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (not the cluster default, which has drifted before
# -- see long-window-4dvar-fcstnetv3's CLAUDE.md) -- matches the
# torch==2.7.1+cu128 both aifs2 and lwaifs2 (and fcstnet3) pin, so pinning
# this rather than relying on the job tolerating whatever the cluster
# default happens to be.
module load cuda/12.8.1
module load rdhpcs-conda
# lwaifs2 (the user's personal clone of the shared aifs2 env), NOT aifs2
# itself -- this repo's get_psobs unconditionally reads the PREPBUFR
# parquet archive now (see psobs_parquet.py / integrate_prepbufr_obs.md), which
# needs pyarrow; the shared aifs2 env does not have it installed (confirmed
# 2026-10-02 -- see integrate_prepbufr_obs.md's job-readiness-check log entry).
conda activate /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/envs/lwaifs2

# ERA5 initial-condition/verification fetches (backends/aifs/aifs_ic.py)
# need internet access, which this H100 node does not have -- the
# ic_cache/ directory must already be warmed from a login node before this
# job runs.
python -u long_window_4dvar.py config_test_aifs.yml
