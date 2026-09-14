#!/bin/bash
# Runs one command on a node that srun has just allocated, with the torch
# environment active. Used for the live demonstration, so the command starts
# the moment the allocation arrives:
#
#     srun --partition=a100-test --gres=gpu:1 --cpus-per-task=4 --time=00:10:00 \
#         bash slurm/live.sh python demo_run.py --data-dir $HOME/data
#
# Why not `srun --pty bash` and then type the command: in rehearsal on 14 Sep
# 2026 an interactive allocation on a100-test arrived after waiting about 13 minutes
# while nobody was watching that window, and its 10 minutes ran out unused.
# Handing srun the command itself wastes none of the allocation, and the output
# still streams to the terminal as it is produced.
#
# There are no #SBATCH lines: srun does not read them, and the partition, GPU
# and time limit belong on the srun command line where they can be seen.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

# Without this, Python block-buffers output that is not going to a terminal,
# and the demonstrator would see nothing until the run ended.
export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

cd "${SLURM_SUBMIT_DIR:-.}"
"$@"
status=$?

echo "finished $(date)"
exit $status
