#!/bin/bash
#SBATCH --job-name=d2-demo
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/demo_%j.out
#SBATCH --error=logs/demo_%j.err

# Note: --account=comp3710 is required. The partition sets
# AllowAccounts=comp3710, so a job submitted under the default personal
# account sits in PENDING with Reason=PartitionConfig - which never clears
# on its own, because it is a permissions mismatch and not a queue.
#
# Note: no --mem directive. The a100 nodes report CfgTRES mem=1M, i.e. this
# cluster does not schedule on memory at all; asking for --mem=16G leaves the
# job pending forever with "Requested node configuration is not available".
# --cpus-per-task is 4 of the node's 8 cores, so the job can share a node that
# is already partly allocated rather than waiting for a whole one.

# The live demonstration run: inference plus one epoch of training.
#
# On the day, prefer running demo_run.py directly inside an interactive session
# that is already allocated (see README), so the demonstrator watches the output
# appear rather than waiting in the queue. This batch version exists as the
# fallback if the interactive session is lost.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# Unbuffered Python output. When stdout is a file rather than a terminal,
# Python block-buffers it, so a long job's progress does not appear in the log
# until the buffer fills or the process exits. That makes a running job look
# hung. This costs nothing and makes `tail -f logs/...out` work as expected.
export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python demo_run.py \
    --checkpoint runs/baseline/best.pt \
    --data-dir $HOME/data

echo "finished $(date)"
