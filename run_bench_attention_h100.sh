#!/bin/bash
#SBATCH -J bench_attn_h100
#SBATCH -o logs/bench_attn_h100_%j.out
#SBATCH -e logs/bench_attn_h100_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 00:30:00

# Sliding-window attention microbenchmark (bench_attention.py) on one H100.
module load cuda/12.8.1
source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwaifs2
python -u bench_attention.py
