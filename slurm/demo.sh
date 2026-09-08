#!/bin/bash
#SBATCH --job-name=d2-demo
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/demo_%j.out
#SBATCH --error=logs/demo_%j.err

# The live demonstration run: inference plus one epoch of training.
#
# On the day, prefer running demo_run.py directly inside an interactive session
# that is already allocated (see README), so the demonstrator watches the output
# appear rather than waiting in the queue. This batch version exists as the
# fallback if the interactive session is lost.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch

python demo_run.py \
    --checkpoint runs/baseline/best.pt \
    --data-dir $HOME/data

echo "finished $(date)"
