#!/bin/bash
#SBATCH -J fcst_skill
#SBATCH -o logs/fcst_skill_%j.out
#SBATCH -e logs/fcst_skill_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 12
#SBATCH --mem=240g
#SBATCH -t 04:00:00
# Daily 5-day forecast skill (forecast_skill.py). Args: one or more "LABEL SRC START END" quadruples.
source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate
while [ $# -ge 4 ]; do python -u forecast_skill.py config_fcst_skill.yml "$1" "$2" "$3" "$4" || exit 1; shift 4; done
