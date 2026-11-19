#!/bin/bash
#SBATCH -J adjq
#SBATCH -o logs/adjq.%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 12
#SBATCH --mem=240g
#SBATCH -t 02:00:00
# Adjoint-quality battery launcher (notes/adjoint_quality.md). Submit from the repo root; arguments go to the CLI:
#   sbatch diagnostics/adjoint_quality/run_aq.sh all --config CFG --out DIR [--golden PAT] [--smoke]
#   sbatch diagnostics/adjoint_quality/run_aq.sh single_ob --config CFG --out DIR --loc sh nh
# AQ_ENV picks the conda env (default anemoi-rocm: aifs37 A1/A2 on MI300X). For AIFS-single on H100:
#   AQ_ENV=/scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/envs/lwaifs2 sbatch -p u1-h100 -q gpu \
#     --gres=gpu:h100:1 --mem=96g diagnostics/adjoint_quality/run_aq.sh all --config CFG --out DIR
set -o pipefail
AQ_ENV=${AQ_ENV:-anemoi-rocm}
if [[ "$AQ_ENV" == anemoi-* ]]; then
  source /scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh
  export ATTN=flex_sliding_window
else
  module load cuda/12.8.1 rdhpcs-conda
fi
conda activate "$AQ_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
python -u -m diagnostics.adjoint_quality.aq "$@" 2>&1 | grep -v "^INFO\|UserWarning\|warnings.warn"
