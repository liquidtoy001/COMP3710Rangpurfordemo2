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
| `slurm/download.sh` | Fetch CIFAR-10 once, as a batch job on a CPU node |
| `slurm/smoke.sh` | Five batches on a GPU node - run this before any long job |
| `slurm/smoke_test_partition.sh` | The same check on `a100-test`, which is usually free |
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

## What this cluster actually looks like

Confirmed with `sinfo` and `scontrol show node` on 8 Sep 2026. These facts are
what the Slurm scripts are shaped around:

| Fact | Consequence |
| --- | --- |
| `comp3710` holds the A100 nodes `a100-0` .. `a100-9` | one `--gres=gpu:1` per job |
| The partition sets `AllowAccounts=comp3710` | **every GPU job needs `--account=comp3710`** |
| Nodes report `CfgTRES=cpu=8,mem=1M` | **never pass `--mem`** - see below |
| 8 cores per node, usually in `mix` state | `--cpus-per-task=4`, not 8 |
| Home quota 17 GB (`/home/Student/s4913333`) | fine for CIFAR-10 and a few checkpoints |

**The account trap.** Jobs default to the personal account (`s4913333`), which
the `comp3710` partition does not accept. Without `--account=comp3710` the job
sits in `PENDING` with `Reason=PartitionConfig` and never starts - it is a
permissions mismatch, not a queue, so waiting does not help. The `cpu` partition
has no such restriction, which is why the download job ran without it.

**One GPU per node, and they are always taken.** Each `a100-*` node carries
exactly one A100 (`gpu:a100:1`) alongside 8 CPU cores. The cores are mostly idle
- a typical node shows 1 or 2 of 8 allocated - but the single GPU is held, often
by jobs from other partitions (`cosc3500`, `a100-grind`) that share the same
physical machines. So the GPU is the bottleneck and trimming `--cpus-per-task`
does nothing for queue time.

The practical consequence: a smoke test that needs seconds of compute can wait
hours behind half-hour training jobs. Use `slurm/smoke_test_partition.sh`, which
targets `a100-test` (`AllowAccounts=ALL`, nodes `a100-a` and `a100-b`) and
normally starts at once. Keep real training on `comp3710`.

**A queue full of jobs that will never run.** `squeue -p comp3710` shows dozens
of old jobs stuck at `Reason=PartitionConfig` - other students who hit the same
missing-`--account` problem and never diagnosed it. They are not competing for
resources, so the real queue is much shorter than it looks.

**The `--mem` trap.** The nodes advertise 1 MB of memory, meaning this cluster
does not schedule on memory at all. A job asking for `--mem=16G` matches no node
and pends forever with *"Requested node configuration is not available"*. The
job scripts therefore carry no `--mem` directive, which is deliberate and should
not be "fixed".

Requesting all 8 cores would mean waiting for an entirely free node, and the
A100s are normally partly allocated. Four cores lets a job share one.

## Setup on Rangpur

The Miniconda and PyTorch install from `COMP3710-Rangpur.pdf` (week 2) was done
in August and does not need repeating. Verify it still works, then clone:

```bash
ssh s4913333@rangpur.compute.eait.uq.edu.au     # UQ VPN required off campus

source $HOME/miniconda3/bin/activate
conda activate torch
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

git clone https://github.com/liquidtoy001/COMP3710Rangpurfordemo2.git
cd COMP3710Rangpurfordemo2
mkdir -p logs
```

Then download CIFAR-10 once (~170 MB), as a batch job on a CPU node:

```bash
sbatch slurm/download.sh
squeue --me
cat logs/download_*.out
```

Two deliberate choices here.

`prepare_data.py` is separate from `train.py` because a GPU job that downloads
is a GPU job holding an A100 idle on the network, and one that fails outright if
the download does.

The download is a **batch** job rather than an `srun --pty` session because an
interactive allocation's wall clock runs in real time whether or not you are
typing. Time spent reading, waiting on a slow first `import torch`, or simply
being distracted all counts against it. A 30 minute interactive session expired
part-way through this download on the first attempt and took the SSH connection
with it. Batch jobs do not care: submit and walk away.

Interactive sessions are still the right tool for debugging and for holding a
GPU before the demonstration - just not for anything that runs unattended.

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
timestamps. Follow a running job with:

```bash
tail -f logs/train_<jobid>.out
```

The job scripts set `PYTHONUNBUFFERED=1` so that works. Without it Python
block-buffers stdout when it is a file rather than a terminal, and a running job
looks hung because nothing reaches the log until the buffer fills.

A job is finished when it no longer appears in `squeue --me`. To check how it
ended:

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode
```

`COMPLETED` with `0:0` is success; `TIMEOUT` means the `--time` limit was too
short and `FAILED` means the script itself errored.

`history.csv` is written incrementally and flushed every epoch on purpose: if
the job hits its Slurm time limit or the node fails, the record up to that point
survives. A summary written only at the end would be lost.

Bringing the results back:

```bash
scp -r s4913333@rangpur.compute.eait.uq.edu.au:~/COMP3710Rangpurfordemo2/runs/baseline runs/
scp s4913333@rangpur.compute.eait.uq.edu.au:~/COMP3710Rangpurfordemo2/logs/train_12345.out logs/
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
srun --partition=comp3710 --account=comp3710 --gres=gpu:1 --cpus-per-task=4 \n    --time=01:00:00 --pty bash
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
