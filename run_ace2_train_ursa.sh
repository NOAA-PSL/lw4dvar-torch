#!/bin/bash
#SBATCH -J ace2_train
#SBATCH -o ace2_train_ursa.out
#SBATCH -e ace2_train_ursa.err
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00

# Train an ACE2 checkpoint with fme on Ursa u1-h100 nodes (2 x 80 GB H100
# each, one rank per GPU, DDP over NCCL/InfiniBand); see ace2-training.md.
# Ursa twin of run_ace2_train_polaris.sh. Submit from the repo root:
#   sbatch run_ace2_train_ursa.sh        # 4 nodes, backends/ace2/training/ace2-era5-vars-train-config-ursa.yaml
#   sbatch -N 2 run_ace2_train_ursa.sh   # 2 nodes (4 samples/GPU)
#   sbatch -N 2 -t 01:00:00 --export=ALL,CONFIG=backends/ace2/training/ace2-smoke-ursa.yaml run_ace2_train_ursa.sh
#   sbatch --export=ALL,OVERRIDE="max_epochs=1" run_ace2_train_ursa.sh
# QOS gpu: <= 20 GPUs (10 nodes) per job, <= 7 days. Global batch =
# train_loader.batch_size (16) whatever the GPU count, so 1-8 nodes train
# identically; inline-inference ICs (16) must be a multiple of the GPU count.
# 1 node (8 samples/GPU) needs
#   OVERRIDE="stepper.step.config.builder.config.checkpointing=3".
# fme resumes from experiment_dir/training_checkpoints when it exists, so a
# job can simply be resubmitted (or chained with --dependency=afterany:JOB)
# until max_epochs. Training reads only /scratch4: no internet needed.
# --cpus-per-task=192 is essential: without it the job (even --exclusive) is
# bound to ONE core, and both ranks plus all their data-loader workers share
# it -- 4.1 s/batch on 2 or 4 nodes alike (2026-10-13).

# cuda/12.8.1 explicitly (matches the env's torch==2.7.1+cu128), as run_ace2.sh.
module load cuda/12.8.1
module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/ace2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

NNODES=${SLURM_JOB_NUM_NODES}
MASTER=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
echo "ace2_train: ${NNODES} node(s): $(scontrol show hostnames ${SLURM_JOB_NODELIST} | paste -sd' ')"

# fme_train.py = fme.ace.train with non-reentrant SFNO gradient checkpointing.
srun --ntasks-per-node=1 --cpus-per-task=192 --kill-on-bad-exit=1 \
    torchrun --nnodes ${NNODES} --nproc_per_node 2 \
        --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER}:29400 \
        backends/ace2/training/fme_train.py \
        ${CONFIG:-backends/ace2/training/ace2-era5-vars-train-config-ursa.yaml} \
        ${OVERRIDE:+--override ${OVERRIDE}}
