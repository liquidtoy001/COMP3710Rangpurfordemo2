#!/bin/bash
#SBATCH --job-name=d2-gan-sweep
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/gansweep_%j.out
#SBATCH --error=logs/gansweep_%j.err

# Which GAN configuration avoids mode collapse on OASIS? A quick comparison on
# a100-test, before the long runs on comp3710.
#
# The first configuration collapsed in the smoke test (job 590977): the
# discriminator's scores for real and generated slices both sank to zero by step
# 300, and by step 1,000 every progress sample was the same brain. Collapse
# showed within 1,000 steps, so 2,500 steps at 64x64 is enough to tell a
# configuration that collapses from one that does not - not enough to judge
# final quality. Each run changes as little as possible from its neighbour:
#
#   a  hinge + spectral norm, translation only       is cutout the cause?
#   b  logistic + R1 gamma 1, no spectral norm        a gradient penalty instead
#   c  logistic + R1 gamma 10, no spectral norm       ... stronger
#   d  as c, without any DiffAugment                  does augmentation matter here?
#
# Output: runs/sweep_<name>/ with progress grids and diversity.csv, and a summary
# of each run's diversity ratio at the end of this log.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "repository at $(git rev-parse --short HEAD)"
nvidia-smi

export PYTHONUNBUFFERED=1
source $HOME/miniconda3/bin/activate
conda activate torch

COMMON="--resolution 64 --steps 2500 --batch-size 64 --log-every 250 --sample-every 500 --checkpoint-every 100000 --fresh"
R1="--loss logistic --no-spectral-norm --lr-g 2e-4 --lr-d 2e-4 --beta2 0.99"

run() {
    name=$1; shift
    echo; echo "==== $name: $* ===="
    python train_gan.py $COMMON --out-dir "runs/sweep_$name" "$@" || echo "run $name FAILED"
}

run a_hinge_sn_translation --loss hinge --diffaugment translation
run b_r1_1 $R1 --r1-gamma 1 --diffaugment translation
run c_r1_10 $R1 --r1-gamma 10 --diffaugment translation
run d_r1_10_noaug $R1 --r1-gamma 10 --diffaugment ""

echo; echo "==== diversity ratio (generated / real) at each sample, per run ===="
for dir in runs/sweep_*; do
    echo "$(basename "$dir"): $(tail -n +2 "$dir/diversity.csv" | cut -d, -f4 | xargs printf '%.2f ')"
done

echo "finished $(date)"
