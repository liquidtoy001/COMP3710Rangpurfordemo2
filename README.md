# COMP3710 Demonstration 2 - Rangpur work

Cluster code for COMP3710 Lab Demonstration 2 (Pattern Recognition), UQ
Semester 2 2026. Everything here runs on **Rangpur**, UQ's HPC cluster.

The notebook covering parts 1-3.1 lives in the course repository under
`demo2/`; this repository holds only the work that needs a GPU node.

| Part | Task | Marks | Status |
| --- | --- | ---: | --- |
| 3.2a | ResNet-18 on CIFAR-10, >90% test accuracy | 1 | **93.87% in 3.8 min - MET** |
| 3.2b | Inference + one training epoch live during the demo | 1 | script written, not yet rehearsed |
| 3.2c | Mixed precision, 94% at V100-360s or better | 2 | AMP 9.1% faster, same accuracy |
| 4.4 Task 1 | OASIS VAE + manifold visualisation | (3/7 tier) | trained, both latent sizes |
| 4.4 Task 2 | OASIS UNet, DSC > 0.9 all labels | (5/7 tier) | **worst class 0.9646 - MET** |
| 4.4 Task 3 | OASIS GAN | (7/7 tier) | not attempting yet |

## Layout

| File | Purpose |
| --- | --- |
| `resnet.py` | ResNet-18 written from scratch, with the CIFAR stem substitution |
| `data.py` | CIFAR-10 transforms, loaders, and device selection |
| `prepare_data.py` | One-off dataset download, to be run on a **CPU** node |
| `train.py` | Training loop, one-cycle schedule, checkpointing, metrics |
| `plot_run.py` | Turns a CIFAR run's `metrics.json` into curves and a summary |
| `plot_vae.py` | VAE figures: curves, reconstructions, samples, the manifold |
| `plot_unet.py` | UNet figures: per-class Dice, curves, segmentation overlays |
| `demo_run.py` | The live demonstration script for 3.2b |
| `slurm/download.sh` | Fetch CIFAR-10 once, as a batch job on a CPU node |
| `slurm/smoke.sh` | Five batches on a GPU node - run this before any long job |
| `slurm/smoke_test_partition.sh` | The same check on `a100-test`, which is usually free |
| `slurm/train.sh` | The full 30-epoch baseline run |
| `slurm/train_amp.sh` | The same run with mixed precision, for 3.2c |
| `slurm/demo.sh` | Batch fallback for the live run |
| `oasis.py` | OASIS dataset: paths, mask pairing, label remapping |
| `vae.py` | The convolutional VAE |
| `train_vae.py` | Trains it and saves the manifold visualisation data |
| `slurm/smoke_vae.sh` | One epoch on 128 images, on the free `a100-test` |
| `slurm/train_vae.sh` | 30-epoch VAE run, 32-dimensional latent |
| `slurm/train_vae_latent2.sh` | The same with a 2D latent, for the decoded grid |
| `slurm/train_vae_beta.sh` | Beta sweep: `sbatch ... <latent_dim> <beta>` |
| `unet.py` | The UNet, the Dice loss and the per-class Dice metric |
| `train_unet.py` | Trains it and reports per-class DSC |
| `slurm/smoke_unet.sh` | One epoch on 128 slices, on the free `a100-test` |
| `slurm/train_unet.sh` | The 30-epoch UNet run |
| `explore_oasis.py` | Read-only probe of the OASIS dataset, before any Part 4 code |
| `slurm/explore_oasis.sh` | Runs that probe on a CPU node |

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

## Results so far

All measured on an A100-PCIE-40GB, 9 September 2026.

| Run | Result | Time |
| --- | --- | --- |
| 3.2a baseline | **93.87%** test accuracy - target met | 230.7 s |
| 3.2c mixed precision | 93.86% - unchanged within noise | 209.7 s (**9.1% faster**) |
| Task 1 VAE, latent 32 | best validation loss 16382 at epoch 14 | 181.7 s |
| Task 1 VAE, latent 2 | best validation loss 16885 at epoch 23 | 179.7 s |
| Task 2 UNet | **worst class DSC 0.9646** - every label above 0.9 | 17.7 min |

Three things in that table are worth being able to explain, because they are
what a demonstrator will ask about.

**Mixed precision bought only 9%.** Accuracy is unchanged, so nothing was lost
numerically - but a 9% speed-up is modest for half precision. ResNet-18 on
32x32 inputs is small enough that an A100 is never saturated: the bottleneck is
kernel launch overhead and the data loader, not arithmetic. That is the argument
for a larger batch size as the next 3.2c experiment, and it is why the DAWNBench
reference solutions raise batch size and use channels-last together with AMP
rather than relying on AMP alone.

**The 2-dimensional VAE scores within 3% of the 32-dimensional one, and the
reason is the opposite of what was expected.** Posterior collapse was the first
hypothesis; it is ruled out, because all 32 dimensions have a `mu` that varies.
The actual problem is too *little* regularisation, not too much.

At beta=1 the KL term is **0.43%** of the loss for the 32-dimensional latent and
**0.076%** for the 2-dimensional one. That is not a weighting, it is an absence:
reconstruction is summed over 65,536 pixels while KL is summed over a handful of
latent dimensions, so beta=1 does not put the two on comparable footing. The
model is trained as a plain autoencoder in all but name, which is exactly why
extra latent capacity buys so little.

Two consequences show up in the figures:

* The codes spread to `|mu| = 24` when the prior is N(0, 1). Drawing
  `z ~ N(0, I)` therefore lands in regions the encoder never visited, and about
  a third of `samples.png` is noise rather than brains.
* The manifold sweep was hard-coded to +/-2.5, which covered a small blob at the
  centre of a cloud spanning z1 in [-7, 25] - so every decoded tile looked
  identical. That was a plotting bug, now fixed: the grid is swept across the
  2nd-98th percentile of the codes the encoder actually produced, and the figure
  is labelled with those coordinates.

Making the KL about a tenth of the loss needs beta around 26 for the
32-dimensional latent and 147 for the 2-dimensional one. `slurm/train_vae_beta.sh`
brackets those:

```bash
sbatch slurm/train_vae_beta.sh 32 10
sbatch slurm/train_vae_beta.sh 32 30
sbatch slurm/train_vae_beta.sh 2 50
sbatch slurm/train_vae_beta.sh 2 150
```

The expected trade is blurrier reconstructions for a latent space that matches
its prior - so `samples.png` should stop producing noise, and the manifold
should become a smooth sweep rather than a cloud with holes in it.

**The 30-epoch CIFAR baseline finished in under four minutes**, against a
DAWNBench guideline of thirty. The time requirement was never going to bind
here; the accuracy one is what mattered.

## Queueing work in parallel

GPU queue time on `comp3710` is measured in hours, so it is worth having every
job that is *ready* in the queue at once. Ready means the code exists and has
passed a smoke test - not that the task is on the list.

`slurm/train.sh` and `slurm/train_amp.sh` can both be queued now. They differ in
exactly one flag, `--amp`, with the same epochs, batch size, learning rate and
seed, so the difference in wall clock and final accuracy is attributable to
mixed precision alone. That is the first row of part 3.2c's ablation table.

What is *not* worth queueing is a speculative pile of variants. Each later
DAWNBench change - larger batch, channels-last, label smoothing, a retuned
one-cycle - needs its own controlled run, or the table cannot say which change
bought what.

Part 4's tasks cannot be queued yet, and the blocker is not the queue: nothing
can be written until the OASIS layout is known. That probe runs on an idle CPU
node in seconds, so it is the next thing to do, not something to wait for.

Parts 1, 2 and 3.1 need no cluster at all. LFW is about 1,300 images and trains
on a laptop.

## OASIS, as it actually is

Confirmed with `explore_oasis.py` on 9 September 2026:

```
/home/groups/comp3710/OASIS/
    keras_png_slices_train/          9,664 PNG   256x256 greyscale uint8
    keras_png_slices_validate/       1,120 PNG
    keras_png_slices_test/             544 PNG
    keras_png_slices_seg_{train,validate,test}/   matching masks, same counts
```

The split is already made, so no train/test division is done in code.

**Masks are stored as 0 / 85 / 170 / 255, not 0 / 1 / 2 / 3.** There are four
classes, spread across the 8-bit range so the masks are viewable as ordinary
greyscale PNGs. Feeding those raw values to cross entropy would treat them as
indices into a 256-class output - an index error at best, nonsense training at
worst. `oasis.encode_mask` maps them back, and raises if it meets a value
outside the known set rather than silently guessing.

**Image and mask filenames differ by their prefix**: `case_001_slice_0.nii.png`
pairs with `seg_001_slice_0.nii.png`. `oasis.mask_path_for` anchors that
substitution to the start of the name.

## Task 1: the VAE

`slurm/train_vae.sh` trains a 32-dimensional latent for 30 epochs;
`slurm/train_vae_latent2.sh` trains a 2-dimensional one. Both are worth having:
the first shows the model reconstructs faithfully, the second shows what the
latent space looks like without any dimensionality reduction in between.

Full marks need the manifold *visualised*, not just the model trained. Since
matplotlib is not on the cluster, `train_vae.py` saves what the figures need:

| File | Contents |
| --- | --- |
| `latents.npy` | `mu` for every test image - the manifold, before reduction |
| `reconstructions.npy` | Test images beside their rebuilds |
| `samples.npy` | Images decoded from `z ~ N(0, I)` - novel brains |
| `manifold_grid.npy` | A decoded sweep of the plane (2D latent runs only) |

Smoke test on `a100-test` before queueing a real run:

```bash
sbatch slurm/smoke_vae.sh
cat logs/vaesmoke_*.out
```

Use the script, not `sbatch --wrap`. A bare `--wrap` inherits Slurm's defaults -
one CPU and **no GPU** - so the first attempt at this ran on the CPU while
sitting on an A100 node, reported `device: cpu`, and validated everything except
the thing it existed to validate. Directives in a file cannot be forgotten.

`samples.npy` is the quickest read on whether training worked: a collapsed VAE
returns the same blurry average for every draw from the prior.

Two numerical details that are easy to get wrong and hard to debug afterwards:

* The reconstruction term is **summed** over pixels, not averaged. Averaging
  would shrink it by a factor of 65,536 against the KL term, and the model would
  collapse to emitting the dataset mean.
* `logvar` is clamped to [-10, 10]. The KL term contains `exp(logvar)`; left
  unbounded, a few bad early steps drove it to 1e12 in testing and took the run
  with it.

## Task 2: the UNet

The requirement is **DSC above 0.9 for every label**, with categorical
(one-hot) output. Four decisions follow from that wording and each should be
explainable on the day.

**Skip connections are the point.** Downsampling four times is what buys a large
receptive field, so a pixel's label can depend on distant context - but by the
bottleneck each position covers a 16x16 patch and precise boundaries are gone.
The decoder can recover *what* is present from the bottleneck but not exactly
*where* its edges are. The skips hand back the high-resolution encoder features
that place those edges to the pixel. For a per-pixel task that is the difference
between a usable mask and a blurry blob.

**Dice loss, not cross entropy alone.** Background dominates a brain slice, so a
model predicting background everywhere would score well on pixel accuracy and on
unweighted cross entropy while being useless. Dice normalises each class by its
own size, so a small structure counts as much as a large one. Cross entropy is
kept alongside it because it gives stronger gradients early, when predictions
are near-uniform and Dice's gradient is weak.

**The reported DSC uses the hard argmax; the loss uses soft probabilities.**
Argmax has zero gradient almost everywhere and cannot be trained through, so the
two necessarily differ - the training loss will always look slightly better than
the true coefficient. That is by design, not a bug.

**Dice is accumulated over the whole split, not averaged per batch.** Many
slices do not contain every class, and a per-slice Dice for an absent class is
0/0. Fudging that to either 0 or 1 distorts the average badly. Summing
intersections and cardinalities first and dividing once sidesteps it.

**The checkpoint is selected on the worst class, not the mean.** A mean is
easily carried over 0.9 by the background class while a tissue class languishes,
and the mean is not what is being marked.

```bash
sbatch slurm/smoke_unet.sh      # one epoch, 128 slices, on a100-test
sbatch slurm/train_unet.sh      # the real run
```

If a class falls short of 0.9, the levers in order are: more epochs; flip
augmentation; a wider network (`--base-channels 64`); and weighting the Dice
term towards the failing class. Change one at a time, or the ablation cannot say
what helped.

## Part 4: before writing any code

The three Part 4 tasks all read the preprocessed OASIS brain MR data from
`/home/groups/comp3710/`, but the lab sheet says nothing about how it is laid
out. The directory structure, file format, image size, image count, whether a
train/test split already exists and how many segmentation labels there are all
decide how the dataset class is written - so find out first rather than guess:

```bash
sbatch slurm/explore_oasis.sh
cat logs/oasis_*.out
```

It needs no GPU and no `--account`, so it runs on the idle `cpu` nodes
immediately instead of queueing behind the A100s. It is read-only.

The output should answer: where OASIS is and whether it is pre-split; the format
and image size (which fixes the VAE's input layer); whether label maps are
present and how many classes they contain (which fixes the width of the UNet's
one-hot output); and the total image count.

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
