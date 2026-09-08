#!/bin/bash
#SBATCH --job-name=d2-vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/vae_%j.out
#SBATCH --error=logs/vae_%j.err

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
# --latent-dim 32 keeps reconstructions sharp; the manifold is then visualised
# by reducing those 32 dimensions locally. For the classic decoded-grid picture,
# see slurm/train_vae_latent2.sh, which trains a second model with a 2D latent.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train_vae.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --latent-dim 32 \
    --beta 1.0 \
    --out-dir runs/vae

echo "finished $(date)"
