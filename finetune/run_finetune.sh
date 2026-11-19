#!/bin/bash
#SBATCH -J a2ft
#SBATCH -o logs/a2ft.%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 12
#SBATCH --mem=240g
#SBATCH -t 2-00:00:00
# A2 fine-tuning launcher (finetune/finetune_a2.py, notes/adjoint_finetune_plan.md). From the repo root:
#   sbatch -J a2ft_C0 finetune/run_finetune.sh --arm C0 --out output/finetune/C0
# H100 overflow: FT_ENV=anemoi-dev sbatch -A gpu-ai4wp -q gpu -p u1-h100 --gres=gpu:h100:1 --mem=96g finetune/run_finetune.sh ...
set -o pipefail
source /scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh
export ATTN=flex_sliding_window
conda activate ${FT_ENV:-anemoi-rocm}
cd "${SLURM_SUBMIT_DIR:-.}"
python -u finetune/finetune_a2.py "$@" 2>&1 | grep -v "^INFO\|UserWarning\|warnings.warn"
