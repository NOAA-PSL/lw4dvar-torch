#!/bin/bash
#SBATCH -J test_aifs1
#SBATCH -o test_aifs1.out
#SBATCH -e test_aifs1.err
#SBATCH --account=gpu-ai4wp
#SBATCH -t 00:30:00
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96g
#SBATCH --qos=gpu

# Quick smoke test for the aifs1 env / AIFS-single-1.1 checkpoint --
# verifies the get_shard_shapes/get_shape_shards import fix (see
# CLAUDE.md) and the full encode/process/decode forward+backward path
# actually run cleanly against this checkpoint, not just that it loads.

module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs1

python -u long_window_4dvar.py config_test_aifs1.yml
