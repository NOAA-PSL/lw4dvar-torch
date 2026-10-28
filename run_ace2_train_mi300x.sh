#!/bin/bash
#SBATCH -J ace2_train_mi300x
#SBATCH -o ace2_train_mi300x.out
#SBATCH -e ace2_train_mi300x.err
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-mi300x
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00

# Train an ACE2 checkpoint with fme on Ursa u1-mi300x nodes (8 x 192 GB AMD
# MI300X each, one rank per GPU, DDP over RCCL; one mlx5 IB HCA per node);
# see ace2-training.md. ROCm twin of run_ace2_train_ursa.sh, in the
# ace2-rocm env (the ace2 env's pip lock with torch 2.7.1+rocm6.3; the wheel
# bundles its own ROCm/RCCL, so no rocm module is loaded). Submit from the
# repo root:
#   sbatch run_ace2_train_mi300x.sh       # 1 node = 8 GPUs, 2 samples/GPU
#   sbatch --export=ALL,CONFIG=backends/ace2/training/ace2-crtm-16layer-train-config-ursa.yaml run_ace2_train_mi300x.sh
#   sbatch --export=ALL,MIOPEN_FIND_MODE=FAST run_ace2_train_mi300x.sh  # no conv tuning; same speed
# Global batch = train_loader.batch_size (16): 1 node = 2 samples/GPU, 0.78
# s/batch steady vs 0.97 on 4 u1-h100 nodes (2026-10-28), plus ~5-6 min of
# first-epoch warm-up per job. Inline-inference ICs must be a multiple of 8 x
# nodes.
# fme resumes from experiment_dir/training_checkpoints, so resubmit/chain.

module load rdhpcs-conda
conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/ace2-rocm

export PYTHONNOUSERSITE=1  # keep ~/.local packages out of the env
# (no expandable_segments: unsupported by torch 2.7.1+rocm6.3)
export OMP_NUM_THREADS=8
# MIOpen/rocFFT caches: per rank on node-local /tmp, set by
# rocm_rank_env.sh. MIOPEN_FIND_MODE (e.g. FAST) passes through --export.
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

NNODES=${SLURM_JOB_NUM_NODES}
MASTER=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
echo "ace2_train_mi300x: ${NNODES} node(s): $(scontrol show hostnames ${SLURM_JOB_NODELIST} | paste -sd' ')"

# fme_train.py = fme.ace.train with non-reentrant SFNO gradient checkpointing.
# rocm_rank_env.sh = per-rank MIOpen caches, then python.
srun --ntasks-per-node=1 --cpus-per-task=96 --kill-on-bad-exit=1 \
    torchrun --nnodes ${NNODES} --nproc_per_node 8 \
        --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER}:29400 \
        --no-python backends/ace2/training/rocm_rank_env.sh \
        backends/ace2/training/fme_train.py \
        ${CONFIG:-backends/ace2/training/ace2-era5-vars-train-config-ursa.yaml} \
        ${OVERRIDE:+--override ${OVERRIDE}}
