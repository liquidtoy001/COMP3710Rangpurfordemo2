#!/bin/bash
#SBATCH --job-name=d2-download
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# Note: no --mem and no --gres. Downloading needs no GPU, and this cluster does
# not schedule on memory (the nodes report mem=1M).

# Fetch CIFAR-10 once, unattended and resumably.
#
# Two things were learnt the hard way here.
#
# First, this must be a batch job, not `srun --pty bash`. An interactive
# allocation's wall clock runs in real time whether or not you are typing, so
# reading, thinking and waiting on a slow first `import torch` all count against
# it. A 30 minute interactive session expired at 62% and took the SSH
# connection with it.
#
# Second, the route from this cluster to cs.toronto.edu runs at roughly
# 70 KB/s, so the 170 MB archive takes around 40 minutes. torchvision's own
# downloader cannot resume - a partial file fails its MD5 check and is fetched
# again from the start - so a job that times out makes no lasting progress.
# curl --continue-at does resume, which makes a timeout merely inconvenient:
# resubmitting picks up where the last attempt stopped.

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

# Unbuffered Python output. When stdout is a file rather than a terminal,
# Python block-buffers it, so a long job's progress does not appear in the log
# until the buffer fills or the process exits. That makes a running job look
# hung. This costs nothing and makes `tail -f logs/...out` work as expected.
export PYTHONUNBUFFERED=1

source $HOME/miniconda3/bin/activate
conda activate torch

DATA_DIR=$HOME/data
ARCHIVE=$DATA_DIR/cifar-10-python.tar.gz
URL=https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

mkdir -p "$DATA_DIR"

if [ -f "$ARCHIVE" ]; then
    echo "existing archive: $(du -h "$ARCHIVE" | cut -f1) - resuming from there"
else
    echo "no existing archive - starting from the beginning"
fi

# --continue-at -  resume from whatever is already on disk
# --retry          survive a transient drop without losing the whole job
#
# A curl failure is not fatal here. prepare_data.py verifies the archive's MD5
# itself, so let it be the judge of whether the file is usable.
curl --location \
     --continue-at - \
     --retry 5 \
     --retry-delay 10 \
     --output "$ARCHIVE" \
     "$URL" || echo "curl exited $? - continuing to verification anyway"

echo "archive now: $(du -h "$ARCHIVE" 2>/dev/null | cut -f1)"

# torchvision checks the MD5, skips its own download if the archive is already
# complete and valid, and extracts it.
python prepare_data.py --data-dir "$DATA_DIR"

echo "finished $(date)"
