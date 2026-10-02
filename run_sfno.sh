#!/bin/bash
#SBATCH -J lw4dvar_sfno
#SBATCH -o lw4dvar_sfno.out
#SBATCH -e lw4dvar_sfno.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 04:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (matches the torch==2.7.1+cu128 every env pins).
module load cuda/12.8.1
module load rdhpcs-conda
# SFNO runs with makani, in the FCN3 env
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/fcstnet3

# Compute nodes have no internet: the converted checkpoint
# (backends/sfno/sfno_prefetch_checkpoint.py) and ERA5 ICs/verification incl.
# 'sp' (backends/sfno/sfno_ic.py CACHE_DIR DATE ...) must be fetched from a
# login node first.
python -u long_window_4dvar.py ${1:-config_test_sfno.yml}
