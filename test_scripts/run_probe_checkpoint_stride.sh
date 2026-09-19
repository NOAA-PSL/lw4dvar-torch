#!/bin/bash
#SBATCH -J aurora_probe_stride
#SBATCH -o probe_aurora_checkpoint_stride.out
#SBATCH -e probe_aurora_checkpoint_stride.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 02:00:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aurora

set -x

# Uncompiled sweep at the real 20-step production window: both extremes
# (stride=1, fully disabled) are already known to fit (66.12 / 81.76 GiB),
# so this fills in the intermediate values to check the interpolation
# holds, not expecting an OOM here.
for stride in 2 4 5 10 20; do
    python -u backends/aurora/probe_aurora_checkpoint_stride.py --steps 20 --checkpoint_stride $stride
done

# Compiled sweep: stride=1 is known to fit (the real 100-epoch run used
# it), and a fully-disabled-equivalent OOM'd -- find where the actual
# boundary is, increasing stride until OOM.
for stride in 2 4 5 10 20; do
    python -u backends/aurora/probe_aurora_checkpoint_stride.py --steps 20 --checkpoint_stride $stride --compile
done
