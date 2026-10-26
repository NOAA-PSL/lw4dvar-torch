#!/bin/bash
#SBATCH -J ace2_dataset
#SBATCH -o ace2_dataset.out
#SBATCH -e ace2_dataset.err
#SBATCH --account=gsienkf
#SBATCH --partition=service
#SBATCH -t 08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g

# Build ACE2 training files from public ARCO-ERA5 on Hercules (MSU)
# (backends/ace2/ace2_dataset.py; see ace2-training.md). Hercules compute nodes
# have no internet; the service partition (hercules-dtn-[1-4]) does. Submit
# from the repo root:
#   sbatch --export=ALL,START=2020-01,END=2020-01 run_ace2_dataset_hercules.sh
# Optional exports: OUTDIR, CO2, WORKERS, THREADS, EXTRA (extra `build` flags,
# e.g. "--layer_indices 0 48 ... 137"). Months already in OUTDIR are skipped,
# so jobs over disjoint month ranges can share one OUTDIR. service allows 1 node;
# the batch QOS caps jobs at 8 h. WORKERS x ~3 GB must fit in --mem.
# MODE=fix_presfc patches PRESsfc(_mean) in place in months of OUTDIR built
# before the reduction to HGTsfc (2026-10-10); use its own log files:
#   sbatch -o ace2_fix_presfc.out -e ace2_fix_presfc.err \
#       --export=ALL,MODE=fix_presfc,START=1979-01,END=1988-12 run_ace2_dataset_hercules.sh
# MODE=add_layer_families adds the --layer_families set's missing families
# (default crtm: specific_humidity, specific_total_condensate, ozone) to the
# months of OUTDIR in place; use its own log files:
#   sbatch -o add.out -e add.err --export=ALL,MODE=add_layer_families,START=1979-01,END=1984-12,OUTDIR=... run_ace2_dataset_hercules.sh
# The CO2 series must exist first (login node, ace2ic env here has h5netcdf):
#   python backends/ace2/ace2_dataset.py co2 $OUTDIR/co2.nc 1940 2022

# THREADS=N caps numpy/BLAS/OpenMP/Blosc/zarr threads per worker.
if [ -n "${THREADS}" ]; then
    export OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS}
    export NUMEXPR_NUM_THREADS=${THREADS} BLOSC_NTHREADS=${THREADS} ZARR_THREADING__MAX_WORKERS=${THREADS}
fi

OUTDIR=${OUTDIR:-/work2/noaa/gsienkf/whitaker/ace2_era5_1deg_8layer}
WORKERS=${WORKERS:-16}

ENV=/work2/noaa/gsienkf/whitaker/conda/envs/ace2ic
PY=${ENV}/bin/python
# Without `conda activate`, ESMPy would use whatever ESMFMKFILE the submitting
# shell exported (e.g. base miniconda's ESMF 8.9.0 -> VersionMismatch).
export ESMFMKFILE=${ENV}/lib/esmf.mk
if [ "${MODE:-build}" = "fix_presfc" ]; then
    ${PY} -u backends/ace2/ace2_dataset.py fix_presfc \
        ${OUTDIR} ${START:?set START=YYYY-MM} ${END:?set END=YYYY-MM} --workers ${WORKERS}
elif [ "${MODE}" = "add_layer_families" ]; then
    ${PY} -u backends/ace2/ace2_dataset.py add_layer_families \
        ${OUTDIR} ${START:?set START=YYYY-MM} ${END:?set END=YYYY-MM} --workers ${WORKERS} ${EXTRA}
else
    ${PY} -u backends/ace2/ace2_dataset.py build \
        ${OUTDIR} ${START:?set START=YYYY-MM} ${END:?set END=YYYY-MM} \
        --co2 ${CO2:-${OUTDIR}/co2.nc} --workers ${WORKERS} ${EXTRA}
fi
