#!/bin/bash
#SBATCH -J latent_fit
#SBATCH -o logs/latent_fit_%j.out
#SBATCH -e logs/latent_fit_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 12
#SBATCH --mem=240g
#SBATCH -t 02:00:00

# Latent-space B / Q samples by inverting one step onto ERA5 (fit_latent_to_era5.py), on accept A.
source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate
python -u fit_latent_to_era5.py config_lwatms_1mo_accept_a.yml output/lwatms_1mo_accept_a \
    output/latent_fit_accept_a.npz 2015-01-07T00 2015-01-21T00
