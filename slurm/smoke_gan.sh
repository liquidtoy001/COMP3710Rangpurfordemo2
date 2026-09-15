#!/bin/bash
#SBATCH --job-name=d2-gan-smoke
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/gansmoke_%j.out
#SBATCH --error=logs/gansmoke_%j.err

# Smoke test for Task 3, on a100-test, before anything queues on comp3710.
#
# comp3710's GPUs are hard to get, so a full run that dies on its first step
# costs hours of queueing. This job checks, on the real OASIS data:
#
#   1. a 64x64 GAN trains, stops on its time limit, and resumes from last.pt
#   2. evaluate_gan.py runs end to end on the result, with the Task 1 VAEs and
#      the Task 2 UNet
#   3. how many milliseconds a step takes at 128x128 and at 256x256, which is
#      what the step counts in slurm/train_gan.sh have to be set from
#
# The test QoS caps a job at 20 minutes.

set -e
echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

export PYTHONUNBUFFERED=1
source $HOME/miniconda3/bin/activate
conda activate torch

echo; echo "==== 1a. 64x64, stopped by --max-minutes ===="
python train_gan.py --resolution 64 --steps 2000 --log-every 100 --sample-every 500 \
    --checkpoint-every 500 --max-minutes 0.2 --fresh --out-dir runs/gan_smoke

echo; echo "==== 1b. the same command again, which must resume ===="
python train_gan.py --resolution 64 --steps 2000 --log-every 100 --sample-every 500 \
    --checkpoint-every 500 --out-dir runs/gan_smoke

echo; echo "==== 2. evaluation ===="
python evaluate_gan.py runs/gan_smoke --num-generated 256

echo; echo "==== 3. speed at 128x128 and 256x256 ===="
python train_gan.py --resolution 128 --batch-size 64 --steps 300 --log-every 100 \
    --sample-every 100000 --checkpoint-every 100000 --fresh --out-dir runs/gan_speed128
python train_gan.py --resolution 256 --batch-size 32 --steps 300 --log-every 100 \
    --sample-every 100000 --checkpoint-every 100000 --fresh --out-dir runs/gan_speed256

echo "finished $(date)"
