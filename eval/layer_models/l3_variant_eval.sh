#!/bin/bash
# Evaluation of an L3 variant once its training is done (2026-11-21; launched by the L3 watchdog or by hand):
#   free runs  eval/layer_models/r2_freeruns.sh <arm>   (final + epochs 4..20, 90 days from 2015-01-01)
#   fcst bias  eval/layer_models/fcst_bias_maps.py, 71 starts (L3-gold config on the variant's checkpoint)
#   drift      after all free runs: drift.py (vs 37-level ERA5) + l3diag/runaway.py (vs D3, by layer/region) -> summary
# Usage: bash eval/layer_models/l3_variant_eval.sh <arm>   (l3w | l3n ...). Prints "<jobid> other <label>" lines for jobs.list.
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
mkdir -p logs figures/fcst_bias output/freerun_diag eval/layer_models/cfg
RUN=${RUNROOT}/aifs-r2-${ARM}/training
D3=/scratch4/BMC/gsienkf/Sergey.Frolov/era5_o96_ml37/chunks/era5-o96-ml37-1979-01-01T00-2023-12-31T18.zarr
AMD="-A ${ACCOUNT} -q gpuwf -p u1-mi300x --gres=gpu:mi300x:1 -c 12 --mem=240g -x e02g03"
CK=$(ls -td $RUN/outputs/checkpoint/*/ | head -1)
[ -f ${CK}inference-last.ckpt ] || { echo "no inference-last.ckpt in $CK"; exit 1; }
out=$(bash eval/layer_models/r2_freeruns.sh $ARM)
frjobs=$(echo "$out" | awk '{print $2}' | paste -sd:)
echo "$out" | awk '{print $2" other fr_"$1}'
python3 - "$ARM" "$CK" "$D3" <<'PY'
import re, sys
arm, ck, d3 = sys.argv[1:]
s = open('eval/layer_models/cycle_gold_ice.yml').read().replace('a37_r2t1_gold_ice', f'a37_r2{arm}_fcst')
old = "  aifs37: True\n"; assert s.count(old) == 1
s = s.replace(old, old + "  aifsml: True\n  aifsml_zarr: '" + d3 + "'\n")
s = re.sub(r"^  path_model: .*$", f"  path_model: '{ck}'", s, flags=re.M)
s = re.sub(r"^  control_variables: .*$", "  control_variables: [u_ml, v_ml, t_ml, q_ml, sp, msl, 2t, 2d, 10u, 10v]", s, flags=re.M)
open(f'eval/layer_models/cfg/{arm}_fcst.yml', 'w').write(s)
PY
j=$(SB --parsable $AMD -t 03:00:00 -J fb_$ARM -o logs/fcst_bias_${ARM}_%j.out --wrap \
    "source ${CONDA_SH}; conda activate anemoi-rocm
     python -u eval/layer_models/fcst_bias_maps.py ${LWATMS} eval/layer_models/cfg/${ARM}_fcst.yml figures/fcst_bias/${ARM}.npz 2015-01-05T00 2015-05-25T00 --amd")
echo "$j other fcst_bias_$ARM"
runs=$(echo "$out" | awk '{print "output/fr_"$1}' | paste -sd' ')
d=$(SB --parsable -d afterany:$frjobs -J drift_$ARM -A ${ACCOUNT} -q batch -p u1-compute -c 8 --mem=64g -t 02:00:00 -o logs/drift_${ARM}_%j.out --wrap \
    "source ${CONDA_SH}; conda activate anemoi-dev; cd ${LWATMS}
     python eval/layer_models/drift.py output/freerun_diag $runs
     python eval/layer_models/runaway.py output/freerun_diag/runaway_${ARM}_final.npz output/fr_r2${ARM}_final 4 | awk 'NR%5==1'
     python eval/layer_models/gate_summary.py ${ARM} > output/freerun_diag/gates_${ARM}.txt; cat output/freerun_diag/gates_${ARM}.txt")
echo "$d other drift_$ARM"
