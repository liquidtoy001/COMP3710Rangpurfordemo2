#!/bin/bash
#SBATCH --job-name=d2-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

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
