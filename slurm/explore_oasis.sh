#!/bin/bash
#SBATCH --job-name=d2-oasis
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --output=logs/oasis_%j.out
#SBATCH --error=logs/oasis_%j.err

# Note: no --gres and no --account. Reading a directory listing needs no GPU,
# and the cpu partition has no AllowAccounts restriction. The cpu nodes are
# normally idle, so this starts immediately - unlike anything on comp3710.

# Find out what the OASIS dataset actually looks like before writing any Part 4
# code. Read-only.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python explore_oasis.py --root /home/groups/comp3710 --depth 3

echo "finished $(date)"
