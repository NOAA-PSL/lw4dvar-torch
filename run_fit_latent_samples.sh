#!/bin/bash
#SBATCH -J latent_fit
#SBATCH -o logs/latent_samples_%j.out
#SBATCH -e logs/latent_samples_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 12
#SBATCH --mem=240g
#SBATCH -t 02:00:00

# 25 dates (Jan 7-31: the cached lagged ERA5 starts Jan 6) of latent B / Q24 samples (L-BFGS, 60 it) for the structured control prior (latent_eof.py).
source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate
python -u fit_latent_to_era5.py config_lwatms_1mo_accept_a.yml output/lwatms_1mo_accept_a output/latent_fit_accept_a_lbfgs60.npz 2015-01-07T00 2015-01-31T00 --kinds B,Q24 --optimizer lbfgs --iters 60 --lr 0.05 --save output/latent_samples_accept_a_lbfgs60
