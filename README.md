# COMP3710 Demonstration 2 - Rangpur work

Cluster code for COMP3710 Lab Demonstration 2 (Pattern Recognition), UQ
Semester 2 2026. Everything here runs on **Rangpur**, UQ's HPC cluster.

The notebook covering parts 1-3.1 lives in the course repository under
`demo2/`; this repository holds only the work that needs a GPU node.

| Part | Task | Marks | Status |
| --- | --- | ---: | --- |
| 3.2a | ResNet-18 on CIFAR-10, >90% test accuracy | 1 | code written, not yet run |
| 3.2b | Inference + one training epoch live during the demo | 1 | script written, not yet rehearsed |
| 3.2c | Mixed precision, 94% at V100-360s or better | 2 | not started |
| 4.4 | OASIS recognition tasks (VAE / UNet / GAN) | 7 | not started |

## Layout

| File | Purpose |
| --- | --- |
| `resnet.py` | ResNet-18 written from scratch, with the CIFAR stem substitution |
| `data.py` | CIFAR-10 transforms, loaders, and device selection |
| `prepare_data.py` | One-off dataset download, to be run on a **CPU** node |
| `train.py` | Training loop, one-cycle schedule, checkpointing, metrics |
| `plot_run.py` | Turns a run's `metrics.json` into training curves and a summary |
| `demo_run.py` | The live demonstration script for 3.2b |
| `slurm/smoke.sh` | Five batches on a GPU node - run this before any long job |
| `slurm/train.sh` | The full 30-epoch baseline run |
| `slurm/demo.sh` | Batch fallback for the live run |

Checkpoints, datasets and Slurm logs are deliberately not tracked; see
`.gitignore`.

## Working arrangement

Code is edited and committed **locally**, then pulled onto the cluster. The
cluster only ever pulls, never pushes, which keeps the commit history clean and
avoids configuring GitHub credentials on a shared machine.

```
local machine:  edit -> git commit -> git push
Rangpur:        git pull -> sbatch -> results in logs/ and runs/
results back:   scp
```

## First-time setup on Rangpur

Follow `COMP3710-Rangpur.pdf` (week 2) for the Miniconda and PyTorch install.
In summary, and noting that **the login node is a lobby, not a workshop** - do
none of this on `login` itself:

```bash
# 1. Connect (UQ VPN required from off campus)
ssh s49133336@rangpur.compute.eait.uq.edu.au

# 2. Clone this repository
git clone https://github.com/liquidtoy001/COMP3710Rangpurfordemo2.git
cd COMP3710Rangpurfordemo2
mkdir -p logs

# 3. Grab a CPU node for the install and the download - not a GPU node
srun --partition=cpu --time=01:00:00 --pty bash

# ... Miniconda install and `conda create -n torch python=3.11 pip -y`
#     then `pip3 install --no-cache-dir torch torchvision`, per the guide ...

conda activate torch
python prepare_data.py --data-dir $HOME/data   # ~170 MB, once only
exit                                            # free the node
```

`prepare_data.py` is separate from `train.py` on purpose: a GPU job that
downloads is a GPU job holding an A100 idle on the network, and one that fails
outright if the download does.

## Running

Always smoke test before queueing a long job:

```bash
sbatch slurm/smoke.sh
squeue --me
cat logs/smoke_<jobid>.out
```

Five batches, a few seconds of compute. If that prints a best accuracy and
writes `runs/smoke/metrics.json`, the environment, data path, model and
checkpoint writing all work.

Then the real run:

```bash
sbatch slurm/train.sh
cat logs/train_<jobid>.out
```

30 epochs of ResNet-18 on CIFAR-10. Expect comfortably under the 30 minute
DAWNBench guideline on an A100; the job's 40 minute limit is headroom, not a
target. The result lands in `runs/baseline/best.pt` and `runs/baseline/metrics.json`.

## Training records

Training runs on the cluster as a script, not as a notebook, because `sbatch`
jobs are non-interactive: a job may sit in the queue for half an hour and then
run while the laptop that submitted it is closed. There is no kernel and nobody
watching. The notebook's job is to *read* these records afterwards and present
them.

Every run leaves four artefacts in its `--out-dir`:

| Artefact | Contents |
| --- | --- |
| `history.csv` | One row per epoch, **flushed as the run goes** |
| `metrics.json` | The same history plus the arguments, device and totals |
| `best.pt` | Weights from the best epoch (~45 MB, never committed) |
| `curves.png` | Accuracy, loss, learning rate and per-epoch time |

The Slurm job's own `logs/train_<jobid>.out` sits alongside them, carrying the
job ID, the node name, `nvidia-smi` output and the per-epoch lines with
timestamps.

`history.csv` is written incrementally and flushed every epoch on purpose: if
the job hits its Slurm time limit or the node fails, the record up to that point
survives. A summary written only at the end would be lost.

Bringing the results back:

```bash
scp -r s49133336@rangpur.compute.eait.uq.edu.au:~/COMP3710Rangpurfordemo2/runs/baseline runs/
scp s49133336@rangpur.compute.eait.uq.edu.au:~/COMP3710Rangpurfordemo2/logs/train_12345.out logs/
python plot_run.py runs/baseline
```

`plot_run.py` also prints the numbers worth quoting: best accuracy, total
training time, mean time per epoch, the epoch at which the run crossed 90%, and
whether the under-30-minute guideline was met.

matplotlib is not needed on the cluster - `train.py` skips the figure with a
note if it is not installed, and `plot_run.py` regenerates it locally from
`metrics.json`. Install it there only if you want the figure produced in the
job itself:

```bash
pip install --no-cache-dir matplotlib
```

**What the demonstrator sees:** a real training log with a job ID and a node
name, timestamped per-epoch progress, and curves generated from it - evidence of
a run that actually happened on Rangpur, rather than cells re-executed on the
spot.

## Demonstration day

Requirement 2 of part 3.2 says the model must run inference and a single epoch
of training on the cluster **during the demonstration**, so the split is:

* **beforehand** - the full training run, with `metrics.json` and the training
  log committed as evidence
* **live** - `demo_run.py`, which loads the checkpoint, runs inference over the
  test set with a per-class breakdown, then trains exactly one epoch. It does
  not overwrite the checkpoint, so it is safe to run twice.

Hold an interactive GPU session **before the demonstrator arrives**, so no time
is lost in the queue:

```bash
tmux new -s demo
srun --partition=comp3710 --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --pty bash
conda activate torch
cd ~/COMP3710Rangpurfordemo2
# then, when the demonstrator is watching:
python demo_run.py --checkpoint runs/baseline/best.pt --data-dir $HOME/data | tee logs/demo_day.log
```

`tee` keeps a copy of what the demonstrator saw. `slurm/demo.sh` is the batch
fallback if the interactive session is lost.

**The demonstration is given from a MacBook, from a fresh clone.** SSH access,
the UQ VPN and `~/.ssh/config` must all be verified on that machine, not only on
the machine the code was written on.

## Design notes

Points the code makes deliberately, and that should be explainable on the day:

* **The CIFAR stem.** torchvision's ImageNet ResNet-18 opens with a 7x7 stride-2
  convolution and a stride-2 max pool, sized for 224x224 inputs. On a 32x32
  CIFAR image that reduces the feature map to 8x8 before the first residual
  block runs. `resnet.py` uses a 3x3 stride-1 stem with no max pool instead, so
  the first stage still sees the image at full resolution.
* **The model is written out, not imported.** The lab sheet does not allow
  pre-built models without the demonstrator's approval.
* **Convolutions carry no bias**, because the BatchNorm that follows has its own
  learnable shift.
* **The shortcut is an identity wherever shapes allow**, and only projects with
  a 1x1 convolution when a block downsamples or changes channel count.
* **SGD with Nesterov momentum and a one-cycle schedule**, rather than Adam:
  higher final accuracy on CIFAR-10, and what the DAWNBench reference solutions
  use.
* **Timing calls `torch.cuda.synchronize()` first.** CUDA is asynchronous, so
  without it a measurement records how long it took to *queue* the work.
* **`--amp` exists but is off.** The baseline is measured in full precision;
  mixed precision is part 3.2c and should be reported as a delta against this.

## Sources and AI use

* Lab sheet: `COMP3710_Lab_2_2026_v2.01.pdf` (Shekhar Chandra, v2.01)
* Cluster guide: `COMP3710-Rangpur.pdf` (course material, week 2)
* Architecture: He et al., *Deep Residual Learning for Image Recognition*,
  CVPR 2016
* The CIFAR stem substitution and the one-cycle recipe are standard practice,
  widely documented in the DAWNBench reference solutions linked from the lab
  sheet's appendix B
* Claude (Anthropic) was used to draft this code; it was reviewed, smoke tested
  and is explainable by the author. A fuller AI usage record will be added as
  the work progresses.
