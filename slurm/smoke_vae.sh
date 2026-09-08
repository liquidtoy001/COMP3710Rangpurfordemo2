#!/bin/bash
#SBATCH --job-name=d2-vae-smoke
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/vaesmoke_%j.out
#SBATCH --error=logs/vaesmoke_%j.err

# One epoch on 128 images, on the usually-free a100-test partition. Run this
# before queueing a real VAE run.
#
# This exists as a script rather than an sbatch --wrap one-liner for a reason:
# a bare --wrap inherits Slurm's defaults, which are one CPU and *no GPU*. The
# first attempt at this smoke test therefore ran on the CPU and reported
# "device: cpu" while sitting on an A100 node - it validated the data pipeline
# but not the thing it was supposed to validate. Directives in a file do not
# get forgotten.
#
# a100-test sets AllowAccounts=ALL, so no --account is needed here.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train_vae.py \
    --epochs 1 \
    --limit 128 \
    --latent-dim 32 \
    --out-dir runs/vae_smoke

echo "finished $(date)"
