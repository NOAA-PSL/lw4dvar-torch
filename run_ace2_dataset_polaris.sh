#!/bin/bash
#PBS -N ace2_dataset
#PBS -A moonshot-reanalysis
#PBS -q capacity
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:eagle
#PBS -o ace2_dataset.out
#PBS -e ace2_dataset.err

# Build ACE2 training files from public ARCO-ERA5 on a Polaris compute node
# (backends/ace2/ace2_dataset.py; see ace2-training.md). Login nodes cap a
# user at 8 GB / 8 CPUs, too small for this. Submit from the repo root:
#   qsub -v START=2020-01,END=2020-01 run_ace2_dataset_polaris.sh
#   qsub -q debug -l walltime=01:00:00 -v START=2020-01,END=2020-01 run_ace2_dataset_polaris.sh
# Optional -v: OUTDIR, CO2, WORKERS, EXTRA (extra `build` flags, e.g.
# "--layer_indices 0 48 ... 137"). Months already in OUTDIR are skipped, so
# jobs over disjoint month ranges can share one OUTDIR.
# The CO2 series must exist first (login node, ace2 env -- needs h5netcdf):
#   python backends/ace2/ace2_dataset.py co2 $OUTDIR/co2.nc 1940 2022

cd ${PBS_O_WORKDIR}

# Compute nodes reach the internet only through the ALCF proxy.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy}

# THREADS=N caps numpy/BLAS/OpenMP/Blosc/zarr threads per worker (default: the
# libraries' own; capping to 1 was not faster in the 2020-01 tests -- see
# ace2-training.md).
if [ -n "${THREADS}" ]; then
    export OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS}
    export NUMEXPR_NUM_THREADS=${THREADS} BLOSC_NTHREADS=${THREADS} ZARR_THREADING__MAX_WORKERS=${THREADS}
fi

OUTDIR=${OUTDIR:-/lus/eagle/projects/moonshot-reanalysis/jwhitaker/ace2_era5_1deg_8layer}
# ~2-3 GB per worker; a Polaris node has 32 cores / 512 GB. Reads, not
# compute, are expected to bound throughput -- tune from the s/time log lines.
WORKERS=${WORKERS:-16}

/home/jwhitaker/.conda/envs/ace2ic/bin/python -u backends/ace2/ace2_dataset.py build \
    ${OUTDIR} ${START:?set START=YYYY-MM} ${END:?set END=YYYY-MM} \
    --co2 ${CO2:-${OUTDIR}/co2.nc} --workers ${WORKERS} ${EXTRA}
