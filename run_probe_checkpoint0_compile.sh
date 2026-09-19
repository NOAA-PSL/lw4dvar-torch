#!/bin/bash
#SBATCH -J aurora_probe_ckpt0_compile
#SBATCH -o probe_aurora_checkpoint0_compile.out
#SBATCH -e probe_aurora_checkpoint0_compile.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 00:45:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aurora

set -x
# checkpoint_stride=0 (no outer checkpoint) + torch.compile, at the real
# 20-step production window -- checking memory safety (earlier uncompiled
# no-outer-checkpoint sweep hit 81.76GiB at 20 steps, only ~11GiB margin)
# and steady-state timing for this combination, before committing to a
# full 100-epoch job.
python -u backends/aurora/probe_aurora_checkpoint.py --steps 20 --no_outer_checkpoint --compile --n_calls 3
