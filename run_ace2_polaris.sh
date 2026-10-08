#!/bin/bash
#PBS -N lw4dvar_ace2
#PBS -A moonshot-reanalysis
#PBS -q capacity
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:eagle
#PBS -o lw4dvar_ace2.out
#PBS -e lw4dvar_ace2.err

# ALCF Polaris (PBS) version of run_ace2.sh.  Submit from the repo root:
#   qsub run_ace2_polaris.sh                          # config_test_ace2_polaris.yml
#   qsub -v CONFIG=my_config.yml run_ace2_polaris.sh
# (PBS does not pass script arguments, hence CONFIG instead of $1.)
# For short tests use the debug queue (<= 1 h): qsub -q debug -l walltime=01:00:00 ...
# A Polaris node has 4 A100s; this run uses one (cuda:0).

cd ${PBS_O_WORKDIR}

# cudatoolkit 12.8.1 explicitly: torch-harmonics' CUDA kernels were built
# against it and resolve libcudart.so.12 from the loaded toolkit (matches
# the torch==2.7.1+cu128 pin).  Load it AFTER conda: the conda module puts
# CUDA 12.9.1's libraries on the path, overriding an earlier 12.8.1 load.
module use /soft/modulefiles
module load conda
module load cudatoolkit-standalone/12.8.1
conda activate /home/jwhitaker/.conda/envs/ace2

# The ACE2 checkpoint + yearly forcing files
# (backends/ace2/ace2_prefetch_checkpoint.py, ace2 env) and the ICs /
# verification (backends/ace2/ace2_ic.py [--verif], ace2ic env) must already
# be fetched from a login node.
python -u long_window_4dvar.py ${CONFIG:-config_test_ace2_polaris.yml}
