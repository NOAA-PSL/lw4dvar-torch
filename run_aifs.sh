#!/bin/bash
#SBATCH -J lw4dvar_aifs
#SBATCH -o lw4dvar_aifs.out
#SBATCH -e lw4dvar_aifs.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 08:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly (not the cluster default, which has drifted before
# -- see long-window-4dvar-fcstnetv3's CLAUDE.md) -- matches the
# torch==2.7.1+cu128 both the aifs2 and fcstnet3 envs pin, so pinning this
# for BOTH backends' run scripts rather than relying on the aifs2 job
# tolerating whatever the cluster default happens to be.
module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2

# surface pressure observation files, organized by date-stamped text files --
# grid/model-independent, shared between backends.
#ln -sfnT /scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs psobs
#ln -sfnT config.yml.template config.yml

# ERA5 initial-condition/verification fetches (backends/aifs/aifs_ic.py)
# need internet access, which this H100 node does not have -- the
# ic_cache/ directory must already be warmed from a login node before this
# job runs.
python -u long_window_4dvar.py config_test_aifs.yml
