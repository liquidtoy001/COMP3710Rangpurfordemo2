#!/bin/bash
#SBATCH --job-name=d2-gan
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/gan_%j.out
#SBATCH --error=logs/gan_%j.err

# Train the Task 3 GAN at one resolution. Usage:
#
#     sbatch slurm/train_gan.sh <resolution> [steps]
#     sbatch slurm/train_gan.sh 128
#
# Output goes to runs/gan<resolution>. The job asks for three hours, not the
# twelve the comp3710 QoS allows, because shorter jobs fit into gaps in the
# queue sooner. train_gan.py stops itself after 170 minutes with last.pt saved,
# leaving ten minutes' margin, so if the run is not finished, submit the same
# command again: it resumes from where it stopped, with its logs appended.
#
# Step counts. The defaults below are starting points for the batch sizes
# given; set them from the milliseconds per step that slurm/smoke_gan.sh
# measured, so a run fits in one or two jobs.
#
# No --mem, deliberately: the nodes advertise mem=1M and a --mem request pends
# forever. --account=comp3710 is required on this partition.

RESOLUTION=${1:?usage: sbatch slurm/train_gan.sh <resolution> [steps]}
case "$RESOLUTION" in
    64)  DEFAULT_STEPS=20000; BATCH=64 ;;
    128) DEFAULT_STEPS=40000; BATCH=64 ;;
    256) DEFAULT_STEPS=60000; BATCH=32 ;;
    *)   echo "resolution must be 64, 128 or 256"; exit 1 ;;
esac
STEPS=${2:-$DEFAULT_STEPS}
OUT_DIR="runs/gan${RESOLUTION}"

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "resolution=$RESOLUTION steps=$STEPS batch=$BATCH -> $OUT_DIR"
nvidia-smi

export PYTHONUNBUFFERED=1
source $HOME/miniconda3/bin/activate
conda activate torch

python train_gan.py \
    --resolution "$RESOLUTION" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --max-minutes 170 \
    --out-dir "$OUT_DIR"

echo "finished $(date)"
