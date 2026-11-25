#!/bin/bash
# One fake-cycled free run (Stage 0 E2; eagle/long-window/PLAN_levels_norm_capacity.md).
# Usage: sbatch [flags] eval/layer_models/run_fr.sh ENV CONFIG.yml    ENV: bo-amd | a37-amd
set -u
cd "${LWATMS:-$SLURM_SUBMIT_DIR}"
case $1 in
  bo-amd)  source /scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm/bin/activate ;;
  a37-amd) source ${CONDA_SH:-/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh}
           conda activate anemoi-rocm ;;
  *) echo "unknown ENV $1"; exit 1 ;;
esac
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python -u long_window_4dvar.py "$2"
