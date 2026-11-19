#!/bin/bash
#SBATCH -J bench_attn_amd
#SBATCH -o logs/bench_attn_amd_%j.out
#SBATCH -e logs/bench_attn_amd_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 00:30:00

# Sliding-window attention microbenchmark (bench_attention.py) on one MI300X.
source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate
python -u bench_attention.py
