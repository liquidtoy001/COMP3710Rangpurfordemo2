#!/bin/bash
#SBATCH --job-name=d2-tta
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/tta_%j.out
#SBATCH --error=logs/tta_%j.err

# Part 3.2c: score the mixed-precision checkpoint with test-time flip
# augmentation (see tta_eval.py for why a flip, and why only a flip).
#
# A batch job rather than an interactive one, so the result is written whether
# or not anyone is watching when the GPU arrives. On a100-test, whose test QoS
# allows 20 minutes; scoring 10,000 images twice takes well under one.
#
# On the GPU rather than the cpu partition, because the plain score has to
# reproduce the recorded 93.86%, which was measured on an A100: CPU arithmetic
# could flip a borderline prediction or two, and the whole question is 14
# images wide.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python tta_eval.py \
    --checkpoint runs/amp/best.pt \
    --data-dir $HOME/data

echo "finished $(date)"
