#!/bin/bash
#SBATCH --job-name=d2-smoke-t
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoket_%j.out
#SBATCH --error=logs/smoket_%j.err

# The same five-batch smoke test as smoke.sh, but on the a100-test partition.
#
# comp3710 has ten nodes with exactly one A100 each, and those GPUs are usually
# all held - often by jobs from other partitions (cosc3500, a100-grind) that run
# on the same physical machines. A smoke test that takes seconds of compute can
# then wait hours behind a queue of half-hour training jobs.
#
# a100-test (nodes a100-a, a100-b) sets AllowAccounts=ALL and exists for exactly
# this: short checks that something works before it is queued for real. No
# --account directive is needed here.
#
# Use this to validate code changes; use slurm/train.sh on comp3710 for real
# training runs. Do not run long jobs here - it is a shared test partition, and
# its default time limit is ten minutes.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train.py \
    --epochs 1 \
    --limit-batches 5 \
    --data-dir $HOME/data \
    --out-dir runs/smoke

echo "finished $(date)"
