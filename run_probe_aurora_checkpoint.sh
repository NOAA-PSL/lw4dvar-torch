#!/bin/bash
#SBATCH -J aurora_probe_checkpoint
#SBATCH -o probe_aurora_checkpoint.out
#SBATCH -e probe_aurora_checkpoint.err
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

# Baseline: outer checkpoint on every step (current production default,
# checkpoint_stride=1), at increasing rollout lengths up to the real
# 20-step production window.
for steps in 4 8 16 20; do
    python -u probe_aurora_checkpoint.py --steps $steps
done

# Same sweep with the outer checkpoint disabled (checkpoint_stride=0
# equivalent) -- relies solely on Aurora's own internal per-Swin3D-block
# checkpointing. Answers whether the outer layer is redundant
# double-checkpointing (memory should stay similar, time should drop) or
# actually load-bearing for memory at these window lengths (OOM without
# it at some point).
for steps in 4 8 16 20; do
    python -u probe_aurora_checkpoint.py --steps $steps --no_outer_checkpoint
done
