#!/bin/bash
# One job of an unattended chain (tools/cycling/chain_next.py). Partition, GPU and time come from the sbatch command line.
# Usage: sbatch [...] tools/cycling/chain_job.sh ENV BASE_CFG LAST_DATE CHUNK
#   ENV: bo-h100 | bo-amd | a37-h100 | a37-amd
set -u
cd "${LWATMS:-$SLURM_SUBMIT_DIR}"
ENV=$1; BASE=$2; LAST=$3; CHUNK=$4
EXTRA=()
case $ENV in
  bo-h100) module load cuda/12.8.1; source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/bin/activate lwaifs2 ;;
  bo-amd)  source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate ;;
  a37-h100) source ${CONDA_SH:-/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh}; conda activate anemoi-dev ;;
  a37-amd) source ${CONDA_SH:-/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh}; conda activate anemoi-rocm
           EXTRA=("  aifs_attention: 'flex_sliding_window'") ;;
  *) echo "unknown ENV $ENV"; exit 1 ;;
esac
CFG=$(python tools/cycling/chain_next.py "$BASE" "$LAST" "$CHUNK" "${EXTRA[@]}")
echo "$(date -u +%FT%T) $SLURM_JOB_ID $ENV $BASE -> $CFG"
case $CFG in
  DONE) exit 0 ;;
  STOP) echo "no progress in two attempts: chain stopped"; exit 1 ;;
esac
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python -u long_window_4dvar.py "$CFG"
