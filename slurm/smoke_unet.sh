#!/bin/bash
#SBATCH --job-name=d2-unet-smoke
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/unetsmoke_%j.out
#SBATCH --error=logs/unetsmoke_%j.err

# One epoch on 128 slices, on the usually-free a100-test partition. Confirms the
# mask pairing, the label remapping, the model, the Dice metric and the artefact
# writing before a real run is queued.
#
# a100-test sets AllowAccounts=ALL, so no --account is needed. --gres and
# --cpus-per-task are spelled out because Slurm's defaults are one CPU and no
# GPU, which once produced a "successful" smoke test that ran entirely on the
# CPU.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

python train_unet.py \
    --epochs 1 \
    --limit 128 \
    --batch-size 16 \
    --out-dir runs/unet_smoke

echo "finished $(date)"
