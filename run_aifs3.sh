#!/bin/bash
#SBATCH -J lw4dvar_aifs3
#SBATCH -o lw4dvar_aifs3.out
#SBATCH -e lw4dvar_aifs3.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 01:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# AIFS checkpoints trained with the multi-dataset anemoi API (anemoi-models
# 0.19, anemoi-inference 0.12) -- e.g. the 1-degree aifs2-1deg-ic1 -- run in
# the aifs3 env, where get_model() picks aifs3_model.AIFS3Model. Its torch
# (2.10, CUDA 12.9) is a conda-forge build that ships its own CUDA runtime, so
# no conda activation is needed: call the env's python directly. ERA5
# ICs/verification must already be in exp.ic_cache (aifs3_ic.py only reads
# the cache).
#
# Triton (used by anemoi-models' graph-attention kernel) compiles a small
# CUDA driver-API helper with the system gcc on first use and needs cuda.h,
# which the conda env doesn't ship -- the cuda module provides it via CPATH.
module load cuda/12.8.1
CONFIG=${1:-config_test_aifs3.yml}
/scratch4/BMC/gsienkf/whitaker/conda/envs/aifs3/bin/python -u long_window_4dvar.py $CONFIG
