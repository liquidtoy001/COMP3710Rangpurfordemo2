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
#     sbatch slurm/train_gan.sh <resolution> [extra train_gan.py arguments]
#     sbatch slurm/train_gan.sh 128
#
# Output goes to runs/gan<resolution> unless --out-dir is given. The job asks
# for three hours, not the twelve the comp3710 QoS allows, because shorter jobs
# fit into gaps in the queue sooner. train_gan.py stops itself after 170 minutes
# with last.pt saved, so if the run is not finished, submit the same command
# again: it resumes, with its logs appended.
#
# The model configuration is read from slurm/gan<resolution>.args when the job
# *starts*, not when it is submitted. comp3710 can queue for hours, and the
# configuration may still be settled by a quick experiment on a100-test in that
# time: committing a new .args file and running git pull on the cluster before
# the job starts changes it without giving up the place in the queue. The
# configuration actually used is printed at the top of the log and saved in
# metrics.json and every checkpoint. Arguments given on the command line come
# after the file's, so they win.
#
# Step counts, set from what slurm/smoke_gan.sh measured on an A100 (job
# 590977): 42 ms per step at 64x64, 97 ms at 128x128 with batch 64, and 188 ms
# at 256x256 with batch 32, with spectral normalisation and no R1 penalty. So
# 40,000 steps at 128 take about 65 minutes and 50,000 at 256 about 157. An R1
# penalty adds a second backward pass through the discriminator, which costs more.
#
# No --mem, deliberately: the nodes advertise mem=1M and a --mem request pends
# forever. --account=comp3710 is required on this partition.

RESOLUTION=${1:?usage: sbatch slurm/train_gan.sh <resolution> [extra train_gan.py arguments]}
shift
case "$RESOLUTION" in
    64)  DEFAULT_STEPS=20000; BATCH=64 ;;
    128) DEFAULT_STEPS=40000; BATCH=64 ;;
    256) DEFAULT_STEPS=50000; BATCH=32 ;;
    *)   echo "resolution must be 64, 128 or 256"; exit 1 ;;
esac

CONFIG_FILE="slurm/gan${RESOLUTION}.args"
CONFIG=""
if [ -f "$CONFIG_FILE" ]; then
    CONFIG=$(grep -v '^#' "$CONFIG_FILE" | tr -d '\r' | tr '\n' ' ')
fi

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "repository at $(git rev-parse --short HEAD)"
echo "configuration from $CONFIG_FILE: $CONFIG"
echo "command-line arguments: $*"
nvidia-smi

export PYTHONUNBUFFERED=1
source $HOME/miniconda3/bin/activate
conda activate torch

# $CONFIG is deliberately unquoted, so its words become separate arguments.
python train_gan.py \
    --resolution "$RESOLUTION" \
    --steps "$DEFAULT_STEPS" \
    --batch-size "$BATCH" \
    --max-minutes 170 \
    --out-dir "runs/gan${RESOLUTION}" \
    $CONFIG "$@"

echo "finished $(date)"
