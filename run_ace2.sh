#!/bin/bash
#SBATCH -J lw4dvar_ace2
#SBATCH -o lw4dvar_ace2.out
#SBATCH -e lw4dvar_ace2.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 08:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (matches the torch==2.7.1+cu128 every env pins).
module load cuda/12.8.1
# Own user-owned 'lwace2' env (/scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/
# envs/lwace2), not the shared /scratch4/BMC/gsienkf/whitaker/conda/envs/ace2
# -- built 2026-10-21 from this repo's own ace2-spec.txt/ace2-requirements.txt
# (pip check clean), same rationale as the aifs2-1.0deg-v1.0 -> lwaifs2
# precedent: independence from a shared, not-account-writable env another
# user actively develops in. Named 'ace2' at first, renamed to 'lwace2' to
# match the lwaifs2 naming convention. See integrate_multiple_backends.md.
source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwace2

# Compute nodes have no internet: the ACE2 checkpoint + yearly forcing files
# (backends/ace2/ace2_prefetch_checkpoint.py, ace2 env) and the ICs /
# verification (backends/ace2/ace2_ic.py [--verif], ace2ic env) must already
# be fetched from a login node.
python -u long_window_4dvar.py ${1:-config_test_ace2.yml}
