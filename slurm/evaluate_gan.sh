#!/bin/bash
#SBATCH --job-name=d2-gan-eval
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/ganeval_%j.out
#SBATCH --error=logs/ganeval_%j.err

# Evaluate a trained GAN (see evaluate_gan.py). Usage:
#
#     sbatch slurm/evaluate_gan.sh runs/gan128
#
# Needs the Task 1 VAE checkpoints (runs/vae_l2_beta50, runs/vae_l32_beta10)
# and the Task 2 UNet (runs/unet) still on the cluster; any that are missing
# are skipped with a message rather than failing the job.

RUN_DIR=${1:?usage: sbatch slurm/evaluate_gan.sh <run_dir>}

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1
# `source` passes this script's own arguments to the activate script, which
# then tries to activate an environment named after the first one. Hide them.
ARGS=("$@")
set --
source $HOME/miniconda3/bin/activate
set -- "${ARGS[@]}"
conda activate torch

python evaluate_gan.py "$RUN_DIR"

echo "finished $(date)"
