#!/bin/bash
#SBATCH -J smoke_test_ace2
#SBATCH -o smoke_test_ace2.out
#SBATCH -e smoke_test_ace2.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 00:30:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/ace2

cd /scratch4/BMC/gsienkf/Jeffrey.Whitaker/lw4dvar-torch
python -u backends/ace2/smoke_test_ace2.py
