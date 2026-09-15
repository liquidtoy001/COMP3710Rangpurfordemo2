#!/bin/bash
#SBATCH --job-name=d2-unet-demo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/unetdemo_%j.out
#SBATCH --error=logs/unetdemo_%j.err

# Batch fallback for the live UNet inference (Part 4, Task 2).
#
# The demonstration itself runs demo_unet.py through srun, so the demonstrator
# watches the output appear; see the README. This script exists for when the
# terminal is lost.
#
# On the cpu partition, not a GPU one. Inference is only forward passes, and in
# rehearsal on 14 Sep 2026 all 544 test slices took 85 s on 4 cores (2 min end
# to end), reproducing the committed Dice to 4.3e-6. The cpu partition allocated
# at once, while a job on a100-test's two GPUs waited about 13 minutes.
#
# Arguments pass straight through to demo_unet.py:
#     sbatch slurm/demo_unet.sh --slices 12 200 431
#     sbatch slurm/demo_unet.sh --random 4

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

export PYTHONUNBUFFERED=1

# `source` passes this script's own arguments to the activate script, which
# then tries to activate an environment named after the first one. Hide them.
ARGS=("$@")
set --
source $HOME/miniconda3/bin/activate
set -- "${ARGS[@]}"
conda activate torch

python demo_unet.py --checkpoint runs/unet/best.pt --device cpu "$@"

echo "finished $(date)"
