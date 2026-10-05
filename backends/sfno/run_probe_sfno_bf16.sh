#!/bin/bash
#SBATCH -J probe_sfno_bf16
#SBATCH -o probe_sfno_bf16.out
#SBATCH -e probe_sfno_bf16.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 01:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/fcstnet3

cd /scratch4/BMC/gsienkf/Jeffrey.Whitaker/lw4dvar-torch
python -u backends/sfno/probe_sfno_bf16.py
