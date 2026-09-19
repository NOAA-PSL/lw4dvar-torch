#!/bin/bash
#SBATCH -J lw4dvar_aurora
#SBATCH -o lw4dvar_aurora.out
#SBATCH -e lw4dvar_aurora.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 08:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (not the cluster default, which has drifted before
# -- see long-window-4dvar-fcstnetv3's CLAUDE.md) -- matches the
# torch==2.7.1+cu128 all three backends' envs pin.
module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aurora

# surface pressure observation files, organized by date-stamped text files --
# grid/model-independent, shared between backends.
#ln -sfnT /scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs psobs
#ln -sfnT config.yml.template config.yml

# ERA5 initial-condition/verification fetches (backends/aurora/aurora_ic.py)
# need internet access, which this H100 node does not have -- the
# ic_cache/ directory must already be warmed from a login node before this
# job runs (see backends/aurora/aurora_prefetch_checkpoint.py for the
# separate checkpoint-fetch step, also login-node-only).
python -u backends/aurora/profile_aurora_rollout.py
