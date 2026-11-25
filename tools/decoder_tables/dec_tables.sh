#!/bin/bash
#SBATCH -J dec_tables
#SBATCH -A gsienkf
#SBATCH -q batch
#SBATCH -p u1-compute
#SBATCH -c 96
#SBATCH --mem=300g
#SBATCH -t 03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH --export=ALL
# Observation-error and bias tables for a decoder set, the gold tables' recipe (from l3_v2_tables.sh, 2026-11-22).
# Env: TAG (output dir figures/dec_<TAG>), BA BM BI (ATMS / AMSU-A / IASI bundles), PA PM PI (their ERA5 pair dirs).
# Defaults: T1 p-level pairs. q on all 37 levels for the q70 decoders: QFILL=''.
set -u
cd "${LWATMS:-$SLURM_SUBMIT_DIR}"
source "${VENV:-/scratch4/BMC/gsienkf/Sergey.Frolov/envs/lwaifs2-rocm}/bin/activate"
A=/scratch4/BMC/gsienkf/Sergey.Frolov/atms; D=tools/decoder_tables; T=${TAG:?set TAG}; O=figures/dec_$T; mkdir -p $D/logs $O
PA=${PA:-$A/data/era5_pre37_1516}; PM=${PM:-$A/data/amsua_era5_pre37}; PI=${PI:-$A/data/iasi_era5_pre37}
for b in $BA $BM $BI; do [ -s $b ] || { echo "missing $b"; exit 1; }; done
# ATMS decoder bias on ERA5 (ch 10-13 extra_bc), 8 chunks
for r in "2015010100 2015012118" "2015012200 2015021118" "2015021200 2015030418" "2015030500 2015032518" \
         "2015032600 2015041518" "2015041600 2015050618" "2015050700 2015052718" "2015052800 2015061518"; do
  set -- $r
  PRE=$PA BUNDLE=$BA NTHREADS=2 python -u $D/era5_ref.py $D/ref8${T}_$1.npz $1 $2 > $D/logs/ref8${T}_$1.log 2>&1 &
done
# AMSU-A and IASI decoder bias on ERA5, 3 chunks each (as the gold dbg_* files)
for s in amsua iasi; do B=$BM; [ $s = iasi ] && B=$BI
  for r in "2015010100 2015022818" "2015030100 2015043018" "2015050100 2015061518"; do set -- $r
    PRE=$( [ $s = amsua ] && echo $PM || echo $PI ) BUNDLE=$B NTHREADS=4 python -u $D/mw_ref.py $s $O/dbg_${s}_$1.npz $1 $2 2 > $D/logs/${T}_${s}_$1.log 2>&1 &
  done
done
# R: ATMS sigma / inter-channel corr, scene-cloud table; IASI corr
PRE=$PA BUNDLE=$BA QFILL='' STRAT=1 NTHREADS=8 python -u atms_decoder_error.py 120 $O/decoder_era5_$T > $D/logs/${T}_atms_err.log 2>&1 &
PRE=$PA BUNDLE=$BA QFILL='' WINDOW=1 OUTDIR=$O NTHREADS=8 python -u atms_cloud_error.py 60 oberr_table_w_$T.json > $D/logs/${T}_atms_cloud.log 2>&1 &
PRE=$PI BUNDLE=$BI NTHREADS=8 python -u iasi_decoder_error.py 120 $O/iasi_decoder_era5_$T > $D/logs/${T}_iasi_err.log 2>&1 &
wait
REF="$D/ref8${T}_*.npz" python $D/make_extra_bc.py $O/atms_${T}_era5_2015_ch10-13.json 10,11,12,13
REF="$O/dbg_amsua_*.npz" python $D/make_extra_bc.py $O/amsua_${T}_era5_2015_ch4-13.json 4,5,6,7,8,9,10,11,12,13
REF="$O/dbg_iasi_*.npz" python $D/make_extra_bc.py $O/iasi_${T}_era5_2015_ch1-50.json $(seq -s, 1 50)
ls -la $O/
grep -h "Traceback\|Error" $D/logs/ref8${T}_*.log $D/logs/${T}_*.log | head
