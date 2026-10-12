#!/bin/bash
#PBS -N ace2_train
#PBS -A moonshot-reanalysis
#PBS -q capacity
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -l filesystems=home:eagle
#PBS -o ace2_train.out
#PBS -e ace2_train.err

# Train an ACE2 checkpoint with fme on one Polaris node (4 x 40 GB A100, one
# torchrun rank per GPU); see ace2-training.md. Submit from the repo root:
#   qsub run_ace2_train_polaris.sh       # backends/ace2/training/ace2-era5-vars-train-config.yaml
#   qsub -v CONFIG=other.yaml,OVERRIDE="max_epochs=1" run_ace2_train_polaris.sh
# fme resumes from experiment_dir/training_checkpoints when it exists, so a
# 24-h job can simply be resubmitted (or chained with qsub -W depend=afterany:JOB)
# until max_epochs is reached. Training reads only Eagle: no internet needed.

cd ${PBS_O_WORKDIR}

# Same module/env recipe as run_ace2_polaris.sh (cudatoolkit after conda).
module purge
module use /soft/modulefiles
module load conda
module load cudatoolkit-standalone/12.8.1
conda activate /home/jwhitaker/.conda/envs/ace2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPUS=${NGPUS:-4}
# fme_train.py = fme.ace.train with non-reentrant SFNO gradient checkpointing.
torchrun --standalone --nproc_per_node ${NGPUS} backends/ace2/training/fme_train.py \
    ${CONFIG:-backends/ace2/training/ace2-era5-vars-train-config.yaml} \
    ${OVERRIDE:+--override ${OVERRIDE}}
