#!/bin/bash
#SBATCH --job-name=d2-amp
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=logs/amp_%j.out
#SBATCH --error=logs/amp_%j.err

# Note: --account=comp3710 is required. The partition sets
# AllowAccounts=comp3710, so a job submitted under the default personal
# account sits in PENDING with Reason=PartitionConfig - which never clears
# on its own, because it is a permissions mismatch and not a queue.
#
# Note: no --mem directive. The a100 nodes report CfgTRES=cpu=8,mem=1M, i.e.
# this cluster does not schedule on memory at all; asking for --mem=16G leaves
# the job pending forever with "Requested node configuration is not available".

# First measurement for part 3.2c: what does mixed precision alone buy?
#
# Every hyperparameter is identical to slurm/train.sh - same 30 epochs, same
# batch size, same peak learning rate, same seed. The ONLY difference is --amp.
# That is deliberate: with one variable changed, the difference in wall clock
# and in final accuracy is attributable to mixed precision and nothing else.
#
# The DAWNBench tricks that come after this (larger batch, channels-last memory
# format, label smoothing, a tuned one-cycle) each need their own controlled
# run, or the ablation table cannot say which change bought what.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train.py \
    --epochs 30 \
    --batch-size 128 \
    --lr 0.1 \
    --amp \
    --data-dir $HOME/data \
    --out-dir runs/amp

echo "finished $(date)"
