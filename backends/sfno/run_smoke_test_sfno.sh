#!/bin/bash
#SBATCH -J smoke_test_sfno
#SBATCH -o smoke_test_sfno.out
#SBATCH -e smoke_test_sfno.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 00:45:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/fcstnet3

cd /scratch4/BMC/gsienkf/Jeffrey.Whitaker/lw4dvar-torch
python -u backends/sfno/smoke_test_sfno.py
