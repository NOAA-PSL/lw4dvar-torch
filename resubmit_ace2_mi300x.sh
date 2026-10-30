#!/bin/bash
# Keep a u1-mi300x ACE2 training run going in 24-h jobs; run from cron on a
# login node, e.g. (crontab -e on ufe01):
#   */15 * * * * cd /scratch4/BMC/gsienkf/Jeffrey.Whitaker/lw4dvar-torch-ace2train; bash -l ./resubmit_ace2_mi300x.sh >> resubmit_ace2_mi300x.log 2>&1
# QOS gpuwf (the only one allowed on u1-mi300x) caps a job at 8 GPUs x 24 h
# (MaxTRESRunMinsPerAccount 11520 GPU-min) and a user at ONE submitted job,
# so jobs can't be chained with --dependency; see ace2-training.md.
# Submits the next job (fme resumes from the latest checkpoint) only when
#   - no ace2_train_mi300x job of ours is queued or running,
#   - out.log has no "Time taken for epoch MAX_EPOCHS", and
#   - the previous job (in $STATE) ended in TIMEOUT, NODE_FAIL or PREEMPTED.
# Any other end state (FAILED, CANCELLED, OUT_OF_MEMORY, ...) writes $STOP and
# stops resubmitting; fix the problem, delete $STOP, and it picks up again.
CONFIG=${CONFIG:-backends/ace2/training/ace2-crtm-16layer-train-config-ursa.yaml}
EXP=${EXP:-/scratch4/BMC/gsienkf/whitaker/ace2_training/ace2-crtm-land-16layer-1979-2025-no2015}
MAX_EPOCHS=${MAX_EPOCHS:-120}
LOGS=${LOGS:-ace2_crtm_land16_mi300x}
# fme --override dotlist (no commas). 2026-10-29: the 16-layer run is limited
# by reading data from /scratch4 (~2.9 GB of float32 per batch, GPUs idle
# unevenly, slower at busy hours), so use 16 loader workers per rank (twice the
# reads in flight) with prefetch 2 (same 32 batches buffered per rank).
OVERRIDE=${OVERRIDE-train_loader.num_data_workers=16 train_loader.prefetch_factor=2}
STATE=${LOGS}.lastjob
STOP=${LOGS}.STOP

cd "$(dirname "$0")"
exec 9> ${LOGS}.lock
flock -n 9 || exit 0
now() { date '+%F %T'; }

[[ -e $STOP ]] && exit 0
if grep -q "Time taken for epoch ${MAX_EPOCHS} " ${EXP}/out.log 2>/dev/null; then
    echo "$(now) epoch ${MAX_EPOCHS} done; stopping (remove the crontab entry)"
    touch $STOP; exit 0
fi
[[ -n "$(squeue -h -u $USER -n ace2_train_mi300x 2>/dev/null)" ]] && exit 0

last=$(cat $STATE 2>/dev/null)
if [[ -n "$last" ]]; then
    st=$(sacct -n -X -P -j $last -o state 2>/dev/null | head -1 | cut -d' ' -f1)
    case "$st" in
        TIMEOUT|NODE_FAIL|PREEMPTED) ;;
        ""|PENDING|RUNNING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)
            exit 0 ;;  # sacct not caught up yet
        *)  echo "$(now) job $last ended $st before epoch ${MAX_EPOCHS}; not resubmitting ($STOP)"
            echo "$last $st" > $STOP; exit 0 ;;
    esac
fi

j=$(sbatch --parsable -t 1-00:00:00 -o ${LOGS}_%j.out -e ${LOGS}_%j.err \
    --export="ALL,CONFIG=${CONFIG},OVERRIDE=${OVERRIDE}" run_ace2_train_mi300x.sh 2>&1)
if [[ "$j" =~ ^[0-9]+$ ]]; then
    echo $j > $STATE
    echo "$(now) previous ${last:-none} ${st:-}; submitted $j [${OVERRIDE}] ($(grep -c 'Time taken for epoch' ${EXP}/out.log) epochs done)"
else
    echo "$(now) sbatch failed, will retry: $j"
fi
