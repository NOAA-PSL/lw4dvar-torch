#!/bin/bash
#SBATCH -J fcst_skill
#SBATCH -o logs/fcst_skill_%j.out
#SBATCH -e logs/fcst_skill_%j.out
#SBATCH --account=gpu-ai4wp
#SBATCH --qos=gpuwf
#SBATCH --partition=u1-mi300x
#SBATCH --gres=gpu:mi300x:1
#SBATCH -c 16
#SBATCH --mem=240g
#SBATCH -t 08:00:00
# Daily 5-day forecast skill (diagnostics/forecast_skill.py, see forecast_skill.md), from the repo root:
#   CONFIG=config.yml sbatch diagnostics/run_forecast_skill.sh LABEL SRC START END [LABEL SRC START END ...]
# SRC is 'era5' or a run's output directory; FCST_SKILL_ARGS passes extra options to every run.
# Set ENV to the backend's python environment (default: the aifs2-1.0deg-v1.0 MI300X venv).
source ${ENV:-/scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm}/bin/activate
while [ $# -ge 4 ]; do
    python -u diagnostics/forecast_skill.py ${CONFIG:?set CONFIG} "$1" "$2" "$3" "$4" $FCST_SKILL_ARGS || exit 1
    shift 4
done
