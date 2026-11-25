#!/bin/bash
# sbatch [-d ...] tools/decoder_tables/run_fit.sh SENSOR BUNDLE PRE TARGET OUT_PREFIX [STRIDE]
#SBATCH -A gsienkf   # override with sbatch -A
#SBATCH -q batch
#SBATCH -p u1-compute
#SBATCH -c 32
#SBATCH --mem=160g
#SBATCH -t 03:00:00
cd "${LWATMS:-$SLURM_SUBMIT_DIR}"
source "${VENV:-/scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm}/bin/activate"
python -u tools/decoder_tables/fit_scene_bc.py "$1" "$2" "$3" "$4" "$5" "${6:-4}" 16
