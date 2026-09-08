#!/bin/bash
#SBATCH --job-name=d2-smoke
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# Note: no --mem directive. The a100 nodes report CfgTRES mem=1M, i.e. this
# cluster does not schedule on memory at all; asking for --mem=16G leaves the
# job pending forever with "Requested node configuration is not available".
# --cpus-per-task is 4 of the node's 8 cores, so the job can share a node that
# is already partly allocated rather than waiting for a whole one.

# Five batches on a GPU node. Proves the environment, the data directory, the
# model and the checkpoint writing all work before a long job is queued.
# Run this first, every time something changes.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# Unbuffered Python output. When stdout is a file rather than a terminal,
# Python block-buffers it, so a long job's progress does not appear in the log
# until the buffer fills or the process exits. That makes a running job look
# hung. This costs nothing and makes `tail -f logs/...out` work as expected.
export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train.py \
    --epochs 1 \
    --limit-batches 5 \
    --data-dir $HOME/data \
    --out-dir runs/smoke

echo "finished $(date)"
