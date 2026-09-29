#!/bin/bash
#SBATCH -J lw4dvar_ace2
#SBATCH -o lw4dvar_ace2.out
#SBATCH -e lw4dvar_ace2.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 04:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (matches the torch==2.7.1+cu128 every env pins).
module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/ace2

# Compute nodes have no internet: the ACE2 checkpoint + yearly forcing files
# (backends/ace2/ace2_prefetch_checkpoint.py, ace2 env) and the ICs /
# verification (backends/ace2/ace2_ic.py [--verif], ace2ic env) must already
# be fetched from a login node.
python -u long_window_4dvar.py ${1:-config_test_ace2.yml}
