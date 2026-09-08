#!/bin/bash
#SBATCH --job-name=d2-download
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# Note: no --mem and no --gres. Downloading needs no GPU, and this cluster does
# not schedule on memory (the nodes report mem=1M).

# Fetch CIFAR-10 once, unattended.
#
# This was originally done by hand inside an `srun --pty bash` session, which is
# a trap: an interactive allocation's wall clock runs in real time whether or
# not you are typing, so time spent reading, waiting on a slow `import torch`,
# or being distracted counts against it. A 30 minute session expired mid
# download and took the SSH connection with it.
#
# A batch job does not care. Submit it and walk away.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

source $HOME/miniconda3/bin/activate
conda activate torch

# Remove any partial archive from an interrupted attempt. torchvision verifies
# the MD5 and would re-download anyway, but deleting it makes that explicit.
rm -f $HOME/data/cifar-10-python.tar.gz

python prepare_data.py --data-dir $HOME/data

echo "finished $(date)"
