#!/bin/bash
#SBATCH --job-name=d2-unet-demo
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/unetdemo_%j.out
#SBATCH --error=logs/unetdemo_%j.err

# Batch fallback for the live UNet inference (Part 4, Task 2).
#
# The demonstration itself should run demo_unet.py inside an interactive
# session that is already allocated, so the demonstrator watches the output
# appear; see the README. This script exists for when that session is lost.
#
# a100-test rather than comp3710: it sets AllowAccounts=ALL and is usually free,
# while comp3710's A100s are normally all held. Inference takes well under a
# minute, inside a100-test's default time limit.
#
# Arguments pass straight through to demo_unet.py:
#     sbatch slurm/demo_unet.sh --slices 12 200 431
#     sbatch slurm/demo_unet.sh --random 4

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python demo_unet.py --checkpoint runs/unet/best.pt "$@"

echo "finished $(date)"
