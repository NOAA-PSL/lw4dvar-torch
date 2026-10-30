#!/bin/bash
#SBATCH -J lw4dvar_aifs2_1p0deg_v1p0_ps_raobs_amv_gpsro
#SBATCH -o logs/lw4dvar_aifs2_1p0deg_v1p0_ps_raobs_amv_gpsro_%j.out
#SBATCH -e logs/lw4dvar_aifs2_1p0deg_v1p0_ps_raobs_amv_gpsro_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH -t 08:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# cuda/12.8.1 explicitly, matching the torch==2.7.1+cu128 this env pins (see
# run_aifs.sh's own note on this).
module load cuda/12.8.1

# This checkpoint (a user-trained 1-degree/O96 AIFS-single-2.0) needs
# hydra-core>=1.3.6 to unpickle (its saved model object references
# hydra._internal.target_policy, absent from hydra-core==1.3.5, the repo's
# shared aifs2 env's pin) -- see aifs_inference_aifs2-1.0deg-v1.0.yaml's
# docstring. The shared aifs2 env (/scratch4/BMC/gsienkf/whitaker/conda/envs/
# aifs2) is not writable by this user, so this run uses a separate,
# user-owned env (lwaifs2) instead -- otherwise byte-identical to
# aifs2-requirements.txt except hydra-core==1.3.7 (see
# aifs2-1.0deg-v1.0-requirements.txt/aifs2-1.0deg-v1.0-spec.txt).
source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwaifs2

# Observation-ablation series (4/4): ps + raobs + amv + gpsro. Same
# sdate/n_init/window as the ps-only / ps+raobs / ps+raobs+amv siblings,
# varying ONLY the observations block -- see integrate_gpsro.md. This is
# the first real (non-offline) exercise of the gpsro forward operator.

# ERA5 initial-condition/verification fetches (backends/aifs/aifs_ic.py)
# need internet access, which this H100 node does not have -- the
# ic_cache_aifs2-1.0deg-v1.0/ directory must already be warmed from a login
# node before this job runs (see backends/aifs/aifs_prefetch_ic.py). Already
# warmed through 2015-07 for this checkpoint, well past this run's range.
python -u long_window_4dvar.py config_test_aifs2-1.0deg-v1.0_ps_raobs_amv_gpsro.yml
