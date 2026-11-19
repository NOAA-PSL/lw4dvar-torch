#!/bin/bash
#SBATCH -J latent_err
#SBATCH -o logs/latent_err_%j.out
#SBATCH -e logs/latent_err_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 12
#SBATCH --mem=240g
#SBATCH -t 01:00:00

# Latent-space B / Q estimates from differences with ERA5 (estimate_latent_errors.py), on accept A.
source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate
python -u estimate_latent_errors.py config_lwatms_1mo_accept_a.yml output/lwatms_1mo_accept_a \
    output/latent_errors_accept_a.npz 2015-01-06T00 2015-01-20T00
