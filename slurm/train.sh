#!/bin/bash
#SBATCH --job-name=d2-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Note: no --mem directive. The a100 nodes report CfgTRES mem=1M, i.e. this
# cluster does not schedule on memory at all; asking for --mem=16G leaves the
# job pending forever with "Requested node configuration is not available".
# --cpus-per-task is 4 of the node's 8 cores, so the job can share a node that
# is already partly allocated rather than waiting for a whole one.

# The full baseline run for part 3.2 requirement 1: >90% test accuracy.
# 30 epochs of ResNet-18 on CIFAR-10 should take well under the 30 minute
# DAWNBench guideline on an A100; the 40 minute limit is headroom, not a target.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch

python train.py \
    --epochs 30 \
    --batch-size 128 \
    --lr 0.1 \
    --data-dir $HOME/data \
    --out-dir runs/baseline

echo "finished $(date)"
