#!/bin/bash
#SBATCH -J lw4dvar_ngcm
#SBATCH -o lw4dvar_ngcm.out
#SBATCH -e lw4dvar_ngcm.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 04:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# JAX backend (model_backend: ngcm): the ngcm-torch env is a venv layered on
# the ngcm-sl conda env (JAX 0.11.1, ngcm-train's dependencies) that adds
# torch 2.7.1+cu128 against the base env's CUDA 12.9 libraries (see
# integrate_multiple_backends.md). ngcm-train itself (model build/restore)
# is put on the path by the backend from exp.ngcm_train_path.
#
# XLA_PYTHON_CLIENT_PREALLOCATE=false: JAX and torch share the GPU; without
# it JAX reserves 75 % of the device at startup.
#
# Default: run the solver on a config; any other .py is run instead (e.g.
#   sbatch run_ngcm.sh backends/ngcm/smoke_test_ngcm.py EXP STEP DATA).
ENV=/scratch4/BMC/gsienkf/whitaker/conda/envs/ngcm-torch
export PATH=${ENV}/bin:${PATH}
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ "${1:-}" == *.py ]]; then
  python -u -W ignore::DeprecationWarning "$@"
else
  python -u -W ignore::DeprecationWarning src/long_window_4dvar.py ${1:-configs/config_test_ngcm.yml}
fi
