#!/bin/bash
# Round-2 fake-cycled free runs (F3 in eagle/long-window/PLAN_levels_norm_capacity.md) for one aifs-r2 run:
# the final checkpoint plus epochs 4, 8, 12, 16, 20, each 180 x 12 h from 2015-01-01 (as the A0/A1/A2 runs).
# Configs are config_fr_a2.yml with the checkpoint swapped. 1 MI300X each, gsienkf/gpuwf, 3 h.
# Usage: eval/layer_models/r2_freeruns.sh <arm>      e.g. c, t1, w1   (run dir EAGLE/run/aifs-r2-<arm>)
set -euo pipefail
ARM=$1
# Paths (2026-11-25, made portable for Jeff from Sergey's untracked scratch originals):
#   LWATMS   your lwatms checkout (writable: logs/, output/, figures/, eval/layer_models/cfg/); default = this repo
#   RUNROOT  dir holding aifs-r2-<arm>/training (checkpoints; read-only is fine)
#   CONDA_SH conda.sh that provides anemoi-rocm / anemoi-dev;  ACCOUNT slurm account;  DRYRUN=1 prints sbatch lines
LWATMS=${LWATMS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNROOT=${RUNROOT:-/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/long-window/EAGLE/run}
CONDA_SH=${CONDA_SH:-/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/EAGLE/conda/etc/profile.d/conda.sh}
ACCOUNT=${ACCOUNT:-gsienkf}
export LWATMS CONDA_SH
SB() { if [[ ${DRYRUN:-0} == 1 ]]; then echo "DRYRUN sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch "$@"; fi; }
cd "$LWATMS"
D=eval/layer_models
mkdir -p ${D}/cfg logs
RUN=${RUNROOT}/aifs-r2-${ARM}/training
CK=$(ls -td ${RUN}/outputs/checkpoint/*/ | head -1)
grep -qs "max_steps=100000\` reached" ${RUN}/runscript.training.out || { echo "aifs-r2-${ARM}: training not finished"; exit 1; }
AMD="-A ${ACCOUNT} -q gpuwf -p u1-mi300x --gres=gpu:mi300x:1 -c 12 --mem=128g"
declare -A CKPTS=([final]=inference-last.ckpt)
for e in 004 008 012 016 020; do
  f=$(cd ${CK} && ls inference-anemoi-by_epoch-epoch_${e}-step_*.ckpt 2>/dev/null | head -1 || true)
  [[ -n ${f} ]] && CKPTS[e${e#0}]=${f}
done
for tag in "${!CKPTS[@]}"; do
  n=r2${ARM}_${tag}
  sed -e "s#^  path_model: .*#  path_model: '${CK}'#" -e "s#^  model_name: .*#  model_name: '${CKPTS[$tag]}'#" \
      -e "s#fr_a2#fr_${n}#g" -e "s#fake-cycled free run, a2#fake-cycled free run, aifs-r2-${ARM} ${tag}#" \
      ${D}/fr_base.yml > ${D}/cfg/fr_${n}.yml
  [[ ${ARM} == l2* ]] && sed -i "s#^  aifs37: True#  aifs37: True\n  aifsml: True#" ${D}/cfg/fr_${n}.yml   # hybrid-layer backend
  [[ ${ARM} == l3* ]] && sed -i "s#^  aifs37: True#  aifs37: True\n  aifsml: True\n  aifsml_zarr: '/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_ml37/chunks/era5-o96-ml37-1979-01-01T00-2023-12-31T18.zarr'#" ${D}/cfg/fr_${n}.yml   # L3: D3 layers
  echo "${n} $(SB --parsable ${AMD} -t 03:00:00 -J fr_${n} -o logs/fr_${n}_%j.out ${D}/run_fr.sh a37-amd ${D}/cfg/fr_${n}.yml)"
done | tee -a ${D}/jobs.txt
