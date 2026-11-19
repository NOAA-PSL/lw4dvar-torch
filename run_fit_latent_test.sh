#!/bin/bash
#SBATCH -J latent_fit
#SBATCH -o logs/latent_fittest_%j.out
#SBATCH -e logs/latent_fittest_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 12
#SBATCH --mem=240g
#SBATCH -t 01:00:00

# Latent-space B / Q samples by inverting one step onto ERA5 (fit_latent_to_era5.py), on accept A.
source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate
for o in "lbfgs 60 0.05" "lbfgs 150 0.05"; do set -- $o; echo "== $1 $2"; python -u fit_latent_to_era5.py config_lwatms_1mo_accept_a.yml output/lwatms_1mo_accept_a output/latent_fit_test_$1$2.npz 2015-01-07T00 2015-01-07T00 --kinds B,Q24 --optimizer $1 --iters $2 --lr $3 --save output/latent_samples_test_$1$2; done
