#!/bin/bash
#SBATCH --job-name=d2-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err

# Note: --account=comp3710 is required. The partition sets
# AllowAccounts=comp3710, so a job under the default personal account sits in
# PENDING with Reason=PartitionConfig and never starts.
#
# Note: no --mem directive. The a100 nodes report mem=1M, so this cluster does
# not schedule on memory; any --mem request matches no node.

# Part 4, Task 2: UNet segmentation of OASIS.
#
# 9,664 training slices at 256x256, batch 16, so about 604 steps per epoch.
# The three hour limit is headroom; the run should take well under one.
#
# The bar is DSC > 0.9 for EVERY label, so the checkpoint is selected on the
# worst class rather than the mean - see train_unet.py.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train_unet.py \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-3 \
    --base-channels 32 \
    --out-dir runs/unet

echo "finished $(date)"
