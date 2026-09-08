#!/bin/bash
#SBATCH --job-name=d2-smoke
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# Five batches on a GPU node. Proves the environment, the data directory, the
# model and the checkpoint writing all work before a long job is queued.
# Run this first, every time something changes.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch

python train.py \
    --epochs 1 \
    --limit-batches 5 \
    --data-dir $HOME/data \
    --out-dir runs/smoke

echo "finished $(date)"
