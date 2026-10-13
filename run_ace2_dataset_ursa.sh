#!/bin/bash
#SBATCH -J ace2_dataset
#SBATCH -o ace2_dataset.out
#SBATCH -e ace2_dataset.err
#SBATCH --account=gsienkf
#SBATCH --partition=u1-service
#SBATCH -t 24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g

# Build ACE2 training files from public ARCO-ERA5 on Ursa
# (backends/ace2/ace2_dataset.py; see ace2-training.md). Ursa compute nodes
# have no internet; u1-service (ufe05-14) does, directly (no proxy). Submit
# from the repo root:
#   sbatch --export=ALL,START=2020-01,END=2020-01 run_ace2_dataset_ursa.sh
# Optional exports: OUTDIR, CO2, WORKERS, THREADS, EXTRA (extra `build` flags,
# e.g. "--layer_indices 0 48 ... 137"). Months already in OUTDIR are skipped,
# so jobs over disjoint month ranges can share one OUTDIR. u1-service allows
# 1 node and 24 h per job; WORKERS x ~3 GB must fit in --mem.
# MODE=fix_presfc patches PRESsfc(_mean) in place in months of OUTDIR built
# before the reduction to HGTsfc (2026-10-10); use its own log files:
#   sbatch -o ace2_fix_presfc.out -e ace2_fix_presfc.err \
#       --export=ALL,MODE=fix_presfc,START=1979-01,END=1988-12 run_ace2_dataset_ursa.sh
# MODE=stats computes the normalization statistics (default 1990-01..2019-12
# into OUTDIR/stats-START-END, or STATSDIR); reads only OUTDIR. Its workers
# reach ~14 GB each: at the default 128 GB, 16 of them were OOM-killed and the
# pool hung (2026-10-13). u1-service allows 64 CPUs per user and 3.9 GB per
# CPU (~246 GB), so use 240 GB and 12 workers:
#   sbatch -t 02:00:00 -c 64 --mem=240g -o ace2_stats.out -e ace2_stats.err \
#       --export=ALL,MODE=stats,WORKERS=12 run_ace2_dataset_ursa.sh
# The CO2 series must exist first (login node, ace2 env -- needs h5netcdf):
#   python backends/ace2/ace2_dataset.py co2 $OUTDIR/co2.nc 1940 2022

# THREADS=N caps numpy/BLAS/OpenMP/Blosc/zarr threads per worker.
if [ -n "${THREADS}" ]; then
    export OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS}
    export NUMEXPR_NUM_THREADS=${THREADS} BLOSC_NTHREADS=${THREADS} ZARR_THREADING__MAX_WORKERS=${THREADS}
fi

OUTDIR=${OUTDIR:-/scratch4/BMC/gsienkf/Jeffrey.Whitaker/ace2_era5_1deg_8layer}
WORKERS=${WORKERS:-16}

PY=/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2ic/bin/python
if [ "${MODE:-build}" = "stats" ]; then
    ${PY} -u backends/ace2/ace2_dataset.py stats ${OUTDIR} \
        ${STATSDIR:-${OUTDIR}/stats-${START:-1990-01}-${END:-2019-12}} \
        ${START:-1990-01} ${END:-2019-12} --workers ${WORKERS}
elif [ "${MODE:-build}" = "fix_presfc" ]; then
    ${PY} -u backends/ace2/ace2_dataset.py fix_presfc \
        ${OUTDIR} ${START:?set START=YYYY-MM} ${END:?set END=YYYY-MM} --workers ${WORKERS}
else
    ${PY} -u backends/ace2/ace2_dataset.py build \
        ${OUTDIR} ${START:?set START=YYYY-MM} ${END:?set END=YYYY-MM} \
        --co2 ${CO2:-${OUTDIR}/co2.nc} --workers ${WORKERS} ${EXTRA}
fi
