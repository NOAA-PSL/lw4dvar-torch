#!/bin/bash
#SBATCH -J lw4dvar_graphcast
#SBATCH -o lw4dvar_graphcast.out
#SBATCH -e lw4dvar_graphcast.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 04:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# GraphCast backend (model_backend: graphcast, JAX via backends/jax_bridge.py):
# the graphcast-torch env is a venv on the ngcm-sl conda env (JAX 0.11.1)
# whose .pth file adds ngcm-torch's site-packages (torch 2.7.1+cu128), plus
# graphcast 0.2 (from /scratch4/BMC/gsienkf/Jeffrey.Whitaker/python/weathernext),
# jraph, trimesh, rtree and dm-tree (see integrate_jax_backend.md).
#
# XLA_PYTHON_CLIENT_PREALLOCATE=false: JAX and torch share the GPU; without
# it JAX reserves 75 % of the device at startup.
#
# Default: run the solver on a config; any other .py is run instead (e.g.
#   sbatch run_graphcast.sh backends/graphcast/smoke_test_graphcast.py).
ENV=/scratch4/BMC/gsienkf/whitaker/conda/envs/graphcast-torch
export PATH=${ENV}/bin:${PATH}
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ "${1:-}" == *.py ]]; then
  python -u -W ignore::DeprecationWarning "$@"
else
  python -u -W ignore::DeprecationWarning src/long_window_4dvar.py ${1:-configs/config_test_graphcast.yml}
fi
