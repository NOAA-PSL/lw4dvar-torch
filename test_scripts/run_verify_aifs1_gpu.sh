#!/bin/bash
#SBATCH -J verify_aifs1
#SBATCH -o verify_aifs1_gpu.out
#SBATCH -e verify_aifs1_gpu.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 00:15:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32g
#SBATCH --qos=gpu

module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs1

python -u backends/aifs/verify_aifs1_gpu.py
