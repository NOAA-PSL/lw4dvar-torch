#!/bin/bash
#SBATCH -J lw4dvar_aifs_r2_t1
#SBATCH -o lw4dvar_aifs_r2_t1_%j.out
#SBATCH -e lw4dvar_aifs_r2_t1_%j.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 01:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# Resolve the repo root regardless of where `sbatch` was run from: SLURM
# copies this script to a spool dir before executing it, so $0 points there,
# not here. scontrol's Command is what was passed to sbatch -- relative to
# the job's initial cwd (always the submission dir) when sbatch was given a
# bare relative filename -- so resolve it with readlink -f before any cd.
SCRIPT=$(readlink -f "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\S+' | cut -d= -f2)")
cd "$(dirname "$(dirname "$SCRIPT")")"

# S. Frolov's aifs-r2-t1 checkpoint: multi-dataset anemoi API, same
# AIFS3Model dispatch as aifs2-1deg-ic1 (run_aifs3.sh) -- but the repo's
# shared aifs3 env (anemoi-models==0.19.0.post35) can't unpickle it
# (AttributeError on anemoi.models.layers.attention.FlexAttentionWrapper,
# added upstream after post35; this checkpoint recorded post36). aifs3
# isn't group-writable by this account, so this run uses S. Frolov's own
# already-working `anemoi-dev` env instead (CUDA torch 2.10/12.9, same
# anemoi-inference commit as aifs3, anemoi-models 0.19.0.post38) -- see
# aifs_inference_aifs-r2-t1.yaml and docs/integrate_multiple_backends.md.
# Its torch is a conda-forge build that ships its own CUDA runtime, so no
# conda activation is needed: call the env's python directly.
#
# Triton (used by anemoi-models' graph-attention kernel) compiles a small
# CUDA driver-API helper with the system gcc on first use and needs cuda.h,
# which the conda env doesn't ship -- the cuda module provides it via CPATH.
module load cuda/12.8.1

# ICs/verification come from the shared training-data zarr join
# (backends/aifs/aifs_zarr_ic.py, exp.aifs_ic_module: 'aifs_zarr_ic') read
# straight off /scratch4 -- no login-node GRIB prefetch, no ic_cache needed.
CONFIG=${1:-configs/config_test_aifs-r2-t1.yml}
/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/envs/anemoi-dev/bin/python -u src/long_window_4dvar.py $CONFIG
