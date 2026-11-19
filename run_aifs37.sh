#!/bin/bash
#SBATCH -J lw4dvar_aifs37
#SBATCH -o logs/lw4dvar_aifs37.%j.out
#SBATCH -e logs/lw4dvar_aifs37.%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --mem=240g
#SBATCH -t 02:00:00

# The 37-level AIFS checkpoints A1/A2 (aifs37: True; notes/integrate_aifs37_models.md) need the
# anemoi-dev env (anemoi-models 0.19 fork, anemoi-inference 0.12, torch 2.10, flash_attn), shared
# with their training: don't install into it. Usage: sbatch [-J name] run_aifs37.sh CONFIG.yml
source /scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh
conda activate anemoi-dev
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python -u long_window_4dvar.py "$1"
