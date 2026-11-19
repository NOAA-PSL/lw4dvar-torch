#!/bin/bash
#SBATCH -J a2ft_eval
#SBATCH -o logs/a2ft_eval.%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpu
#SBATCH --partition=u1-h100
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=96g
#SBATCH -t 03:00:00
# Stage-2 skill / blur / robustness eval of an A2 fine-tune (finetune/eval_skill.py), from the repo root:
#   sbatch finetune/run_eval.sh --label C0 --weights output/finetune/C0/step_4000.pt --out output/finetune/eval/C0.json
set -o pipefail
source /scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh
export ATTN=flex_sliding_window
conda activate ${FT_ENV:-anemoi-dev}
cd "${SLURM_SUBMIT_DIR:-.}"
python -u finetune/eval_skill.py "$@" 2>&1 | grep -v "^INFO\|UserWarning\|warnings.warn"
