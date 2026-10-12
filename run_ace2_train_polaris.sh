#!/bin/bash
#PBS -N ace2_train
#PBS -A moonshot-reanalysis
#PBS -q capacity
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=168:00:00
#PBS -l filesystems=home:eagle
#PBS -o ace2_train.out
#PBS -e ace2_train.err

# Train an ACE2 checkpoint with fme on Polaris nodes (4 x 40 GB A100 each, one
# rank per GPU, DDP over NCCL; Slingshot between nodes via the AWS-OFI NCCL
# plugin); see ace2-training.md. Submit from the repo root:
#   qsub run_ace2_train_polaris.sh       # 4 nodes, backends/ace2/training/ace2-era5-vars-train-config.yaml
#   qsub -l select=1:system=polaris run_ace2_train_polaris.sh     # 1 node
#   qsub -v CONFIG=other.yaml,OVERRIDE="max_epochs=1" run_ace2_train_polaris.sh
# capacity: 1-4 nodes, <= 168 h, one running job. fme resumes from
# experiment_dir/training_checkpoints when it exists, so a job can simply be
# resubmitted (or chained with qsub -W depend=afterany:JOB) until max_epochs.
# Global batch = train_loader.batch_size (16) whatever the GPU count, so up to
# 16 GPUs (4 nodes) train identically; inline-inference ICs must be a
# multiple of the GPU count. NNODES=N uses the first N nodes of the job.
# Training reads only Eagle: no internet needed.

cd ${PBS_O_WORKDIR}

# Same module/env recipe as run_ace2_polaris.sh (cudatoolkit after conda).
module purge
module use /soft/modulefiles
module load conda
module load cudatoolkit-standalone/12.8.1
conda activate /home/jwhitaker/.conda/envs/ace2

# NCCL over Slingshot (ALCF's recommended settings for PyTorch DDP on Polaris).
export NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1 NCCL_COLLNET_ENABLE=1 NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:/soft/libraries/hwloc/lib:${LD_LIBRARY_PATH}
export FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DEFAULT_CX_MAX=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NNODES=${NNODES:-$(sort -u ${PBS_NODEFILE} | wc -l)}
HOSTFILE=$(mktemp -p ${PBS_O_WORKDIR} .ace2_train_hosts.XXXX)
sort -u ${PBS_NODEFILE} | head -n ${NNODES} > ${HOSTFILE}
MASTER=$(head -n 1 ${HOSTFILE})
echo "ace2_train: ${NNODES} node(s): $(paste -sd' ' ${HOSTFILE})"

# fme_train.py = fme.ace.train with non-reentrant SFNO gradient checkpointing.
mpiexec -n ${NNODES} --ppn 1 --hostfile ${HOSTFILE} --depth 64 --cpu-bind depth \
    torchrun --nnodes ${NNODES} --nproc_per_node 4 \
        --rdzv_id ${PBS_JOBID%%.*}-${NNODES} --rdzv_backend c10d --rdzv_endpoint ${MASTER}:29400 \
        backends/ace2/training/fme_train.py \
        ${CONFIG:-backends/ace2/training/ace2-era5-vars-train-config.yaml} \
        ${OVERRIDE:+--override ${OVERRIDE}}
status=$?
rm -f ${HOSTFILE}
exit ${status}
