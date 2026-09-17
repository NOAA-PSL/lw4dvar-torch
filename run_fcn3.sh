#!/bin/bash
#SBATCH -J lw4dvar_fcn3
#SBATCH -o lw4dvar_fcn3.out
#SBATCH -e lw4dvar_fcn3.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 08:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 (not the login/compute-node default, which has drifted
# before) matches the CUDA toolkit torch==2.7.1+cu128 and the from-source
# torch_harmonics CUDA extension were built against -- load-bearing for
# this backend specifically (see CLAUDE.md's "fcstnet3 conda environment"
# section in the original long-window-4dvar-fcstnetv3 repo).
module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/fcstnet3

# surface pressure observation files, organized by date-stamped text files --
# grid/model-independent, shared between backends.
#ln -sfnT /scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs psobs
#ln -sfnT config.yml.template config.yml

# ERA5 initial-condition/verification fetches (backends/fcn3/fcn3_ic.py)
# need internet access, which this H100 node does not have -- the
# ic_cache/ directory must already be warmed from a login node before this
# job runs (see fcn3_prefetch_ic.py).
python -u long_window_4dvar.py config_test_fcn3.yml
