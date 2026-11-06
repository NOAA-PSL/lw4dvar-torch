#!/bin/bash
#SBATCH -J ace2_train_gh
#SBATCH -o ace2_train_gh.out
#SBATCH -e ace2_train_gh.err
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-gh
#SBATCH --qos=gpuwf
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 1-00:00:00  # QOS gpuwf: 11520 GH200-min per account = 8 GPUs x 24 h

# Train an ACE2 checkpoint with fme on Ursa u1-gh nodes (one NVIDIA GH200
# per node: 96 GB HBM, 72 Grace aarch64 cores, 213 GB RAM, IB mlx5_0; one
# rank per node, DDP over NCCL); see ace2-training.md. aarch64 twin of
# run_ace2_train_ursa.sh, in the ace2-gh env (the ace2 pip lock with the
# aarch64 torch 2.7.1+cu128 wheel, which bundles its CUDA libraries, so no
# cuda module). Submit from the repo root:
#   sbatch run_ace2_train_gh.sh            # 8 nodes = 8 GPUs, 2 samples/GPU
#   sbatch -N 4 -t 01:00:00 --export=ALL,CONFIG=... run_ace2_train_gh.sh
# Global batch = train_loader.batch_size (16) whatever the GPU count;
# inline-inference ICs (16) must be a multiple of the node count.
# fme resumes from experiment_dir/training_checkpoints, so a job can simply
# be resubmitted until max_epochs.

# No rdhpcs-conda module on the aarch64 nodes: activate the env by PATH.
ENV=/scratch4/BMC/gsienkf/whitaker/conda/envs/ace2-gh
export PATH=${ENV}/bin:${PATH} CONDA_PREFIX=${ENV}

export PYTHONNOUSERSITE=1  # keep ~/.local packages out of the env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

NNODES=${SLURM_JOB_NUM_NODES}
MASTER=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
echo "ace2_train_gh: ${NNODES} node(s): $(scontrol show hostnames ${SLURM_JOB_NODELIST} | paste -sd' ')"

# fme_train.py = fme.ace.train with non-reentrant SFNO gradient checkpointing.
srun --ntasks-per-node=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
    torchrun --nnodes ${NNODES} --nproc_per_node 1 \
        --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER}:29400 \
        backends/ace2/training/fme_train.py \
        ${CONFIG:-backends/ace2/training/ace2-crtm-32layer-e512-train-config-ursa.yaml} \
        ${OVERRIDE:+--override ${OVERRIDE}}
