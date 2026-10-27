#!/bin/bash
#SBATCH -J ace2_diurnal
#SBATCH -o ace2_diurnal.out
#SBATCH -e ace2_diurnal.err
#SBATCH --account=gsienkf
#SBATCH --partition=u1-service
#SBATCH -t 24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g

# Sub-6-hourly (diurnal/tidal) climatology of the ACE2 state from ARCO-ERA5
# (backends/ace2/ace2_diurnal_clim.py). u1-service has internet. Submit from
# the repo root:
#   sbatch --export=ALL,START=2014-01,END=2014-12 run_ace2_diurnal_ursa.sh
#   sbatch -t 01:00:00 --export=ALL,MODE=combine run_ace2_diurnal_ursa.sh
# Optional exports: DATASET_DIR (the 6-hourly dataset; needs END+1 month),
# SUMDIR, OUT, WORKERS, EXTRA (e.g. "--test_times 6"). Months already in
# SUMDIR are skipped. u1-service: 1 node, 24 h, 64 CPUs per user.
DATASET_DIR=${DATASET_DIR:-/scratch4/BMC/gsienkf/whitaker/ace2_era5_1deg_8layer}
SUMDIR=${SUMDIR:-${DATASET_DIR}/diurnal_sums}
WORKERS=${WORKERS:-16}
PY=/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2ic/bin/python
if [ "${MODE:-accum}" = "combine" ]; then
    ${PY} -u backends/ace2/ace2_diurnal_clim.py combine ${SUMDIR} ${OUT:-${DATASET_DIR}/diurnal_clim.nc}
else
    ${PY} -u backends/ace2/ace2_diurnal_clim.py accum ${DATASET_DIR} ${SUMDIR} \
        ${START:?set START=YYYY-MM} ${END:?set END=YYYY-MM} --workers ${WORKERS} ${EXTRA}
fi
