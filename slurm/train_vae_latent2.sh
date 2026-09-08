#!/bin/bash
#SBATCH --job-name=d2-vae2
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/vae2_%j.out
#SBATCH --error=logs/vae2_%j.err

# Note: --account=comp3710 is required. The partition sets
# AllowAccounts=comp3710, so a job under the default personal account sits in
# PENDING with Reason=PartitionConfig and never starts.
#
# Note: no --mem directive. The a100 nodes report mem=1M, so this cluster does
# not schedule on memory; any --mem request matches no node.

# Part 4, Task 1: train the VAE on OASIS.
#
# 9,664 training slices at 256x256, batch 32, so about 302 steps per epoch.
# The two hour limit is headroom - the run itself should be well under one.
#
# This is the two-dimensional-latent companion to slurm/train_vae.sh.
#
# With only two latent dimensions the manifold needs no dimensionality
# reduction at all: sweep a grid across the plane, decode every point, and the
# result is the manifold itself rather than a projection of it. The cost is
# blurrier reconstructions, because two numbers cannot carry much detail.
#
# Both runs are worth having, and they answer different questions. The
# 32-dimensional model shows the VAE reconstructs faithfully; this one shows
# what its latent space actually looks like.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train_vae.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --latent-dim 2 \
    --beta 1.0 \
    --out-dir runs/vae_latent2

echo "finished $(date)"
