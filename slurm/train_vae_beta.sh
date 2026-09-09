#!/bin/bash
#SBATCH --job-name=d2-vae-beta
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/vaebeta_%j.out
#SBATCH --error=logs/vaebeta_%j.err

# Beta sweep for the VAE. Usage:
#
#     sbatch slurm/train_vae_beta.sh <latent_dim> <beta>
#     sbatch slurm/train_vae_beta.sh 2 150
#
# Why this sweep exists. At beta=1 the KL term came out at 0.43% of the loss
# for a 32-dimensional latent and 0.076% for a 2-dimensional one. That is
# effectively no regularisation: the reconstruction term is summed over 65,536
# pixels while the KL is summed over a handful of latent dimensions, so beta=1
# does not weight them comparably at all.
#
# The consequences were visible in the first runs. The codes spread out to
# |mu| = 24 when the prior is N(0, 1), so sampling z ~ N(0, I) landed in regions
# the encoder never visited and roughly a third of the decoded samples were
# noise rather than brains. It also explains why a 2-dimensional latent scored
# within 3% of a 32-dimensional one: with the KL ignored, the model behaves like
# a plain autoencoder and the extra capacity buys little.
#
# Raising beta to about 26 (latent 32) or 147 (latent 2) would make the KL
# around a tenth of the loss. This sweep brackets those figures.
#
# Each run takes roughly three minutes, so the sweep is cheap - but one variable
# changes per run, or the comparison says nothing.

LATENT_DIM=${1:?usage: sbatch slurm/train_vae_beta.sh <latent_dim> <beta>}
BETA=${2:?usage: sbatch slurm/train_vae_beta.sh <latent_dim> <beta>}
OUT_DIR="runs/vae_l${LATENT_DIM}_beta${BETA}"

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "latent_dim=$LATENT_DIM beta=$BETA -> $OUT_DIR"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train_vae.py \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --latent-dim "$LATENT_DIM" \
    --beta "$BETA" \
    --out-dir "$OUT_DIR"

echo "finished $(date)"
