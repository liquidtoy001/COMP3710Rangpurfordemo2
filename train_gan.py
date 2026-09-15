"""Train the OASIS GAN (Part 4, Task 3).

Everything the lab sheet asks to see as evidence of training is written as the
run goes, so it survives a job that is cut off:

    history.csv       every --log-every steps: both losses, the discriminator's
                      mean score on real training slices, on real validation
                      slices it never trains on, and on generated slices
    diversity.csv     every --sample-every steps: how different the generated
                      images are from each other, relative to real ones
    progress/         every --sample-every steps: a PNG grid of the same 64
                      noise vectors, so the grids show one set of brains forming
    last.pt           every --checkpoint-every steps: everything needed to resume
    final.pt          at the end: the averaged generator, ready for evaluate_gan.py

No checkpoint is chosen by a quality score. A GAN has no validation loss that
says which step is best, and choosing by eye would be choosing on the evidence.
The run ends at --steps and the final averaged generator is the result - with
one exception, a collapse rule fixed before the run:

    snapshots/        the averaged generator at the most recent sample, from
                      --collapse-after on, whose diversity ratio was at least
                      --healthy-ratio (only the latest is kept, so the rule
                      offers no set of candidates to pick among)
    collapse rule     from --collapse-after on, if the diversity ratio is below
                      --collapse-ratio at --collapse-patience samples in a row,
                      training stops, and final.pt is the last healthy snapshot

The rule exists because the first 128x128 OASIS run (job 591530) was healthy for
10,000 steps, its ratio between 0.79 and 0.92, then collapsed to 0.10 by step
14,000 - and last.pt, overwritten every 2,000 steps, no longer held a healthy
generator. It looks only at diversity, which measures collapse and not quality,
and it is recorded in metrics.json whenever it fires.

Resuming is automatic. A Slurm job has a time limit, so the script watches its
own wall clock (--max-minutes) and Slurm's SIGTERM, writes last.pt and exits;
resubmitting the same command continues from that step.

Smoke test, then a full run:

    python train_gan.py --resolution 64 --steps 200 --limit 512 --out-dir runs/gan_smoke
    sbatch slurm/train_gan.sh 128
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from gan import (
    EMA,
    Discriminator,
    Generator,
    count_parameters,
    diff_augment,
    LOSSES,
    r1_penalty,
    recalibrate_batchnorm,
)
from oasis import DEFAULT_ROOT, OASIS

GRID_SIDE = 8  # progress grids are 8x8
DIVERSITY_BATCH = 64
DIVERSITY_SIZE = 64  # diversity is measured on 64x64 copies, which is cheap and enough

HISTORY_FIELDS = [
    "step", "d_loss", "g_loss", "r1", "d_real", "d_fake", "d_train_clean", "d_validate_clean",
    "seconds_per_step",
]
# The memorisation check scores this many training and validation slices.
# Large enough that the two means are not dominated by which slices happened
# to be drawn.
OVERFIT_CHECK_BATCH = 256
DIVERSITY_FIELDS = ["step", "generated_distance", "real_distance", "ratio"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type


def load_split(root: str, split: str, resolution: int, device: torch.device,
               workers: int, limit: int | None = None) -> torch.Tensor:
    """Every slice of a split, resized, as one uint8 tensor on ``device``.

    Held in memory rather than read from disk each step: 9,664 slices are 633 MB
    at 256x256 and 158 MB at 128x128, which fits easily on an A100, and it takes
    PNG decoding out of the training loop entirely. The PNGs are decoded once, by
    DataLoader workers.

    Downsampling uses area averaging, which is exact for these power-of-two
    factors: each output pixel is the mean of the block it covers.
    """
    dataset = OASIS(root, split=split, with_masks=False, limit=limit)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=workers)
    chunks = []
    for images in loader:
        if resolution != images.shape[-1]:
            images = F.interpolate(images, size=(resolution, resolution), mode="area")
        chunks.append((images * 255.0).round().to(torch.uint8))
    return torch.cat(chunks).to(device)


def to_model_range(images: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1], the generator's tanh range."""
    return images.float().div_(127.5).sub_(1.0)


def save_grid(images: torch.Tensor, path: Path) -> None:
    """Write a GRID_SIDE x GRID_SIDE grid of [-1, 1] images as a PNG with Pillow.

    Pillow, not matplotlib, because matplotlib is not installed on Rangpur.
    """
    images = ((images[: GRID_SIDE * GRID_SIDE].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    images = images.cpu().numpy()[:, 0]
    rows = [np.concatenate(list(images[r * GRID_SIDE:(r + 1) * GRID_SIDE]), axis=1)
            for r in range(GRID_SIDE)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


@torch.no_grad()
def mean_pairwise_distance(images: torch.Tensor) -> float:
    """Mean Euclidean distance between every pair of images, on 64x64 copies in [0, 1].

    A generator in mode collapse draws near-identical images, so this falls far
    below the same figure for real slices. Logged as a ratio, generated over
    real, as an early warning while the run is still going.
    """
    small = F.interpolate((images.float() + 1) / 2, size=(DIVERSITY_SIZE, DIVERSITY_SIZE), mode="area")
    flat = small.flatten(1)
    distances = torch.cdist(flat, flat)
    count = flat.size(0)
    return (distances.sum() / (count * (count - 1))).item()


class StopRequest:
    """Set when Slurm sends SIGTERM ahead of the time limit."""

    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:  # noqa: ARG002 - signal handler signature
        print("\nSIGTERM received: saving a checkpoint and stopping", flush=True)
        self.requested = True


def open_csv(path: Path, fields: list[str], resuming: bool):
    exists = path.exists() and resuming
    handle = path.open("a" if exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default="runs/gan128")
    parser.add_argument("--resolution", type=int, default=128, choices=(64, 128, 256))
    parser.add_argument("--steps", type=int, default=60_000, help="generator updates in total")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=32, help="channels at full resolution")
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=4e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--loss", default="hinge", choices=sorted(LOSSES))
    parser.add_argument("--r1-gamma", type=float, default=0.0,
                        help="weight of the R1 gradient penalty on real images; 0 disables it")
    parser.add_argument("--no-spectral-norm", action="store_true",
                        help="plain discriminator, normally used together with --r1-gamma")
    parser.add_argument("--diffaugment", default="translation,cutout",
                        help="comma-separated DiffAugment policy; empty string to disable")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="checkpoint and stop after this long, to finish inside a Slurm limit")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    parser.add_argument("--healthy-ratio", type=float, default=0.8,
                        help="save a generator snapshot at samples with at least this diversity ratio")
    parser.add_argument("--collapse-ratio", type=float, default=0.6,
                        help="a diversity ratio below this counts towards declaring collapse")
    parser.add_argument("--collapse-patience", type=int, default=2,
                        help="consecutive low samples that declare collapse")
    parser.add_argument("--collapse-after", type=int, default=5000,
                        help="ignore diversity before this step: every run dips early, then recovers")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="use only N slices per split (smoke tests)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "progress").mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    if args.fresh:
        (out_dir / "COMPLETE").unlink(missing_ok=True)
    elif (out_dir / "COMPLETE").exists():
        raise SystemExit(f"{out_dir} is complete ({(out_dir / 'COMPLETE').read_text().strip()}); "
                         "use --fresh or another --out-dir to train again")
    resuming = last_path.exists() and not args.fresh

    if device.type == "cuda":
        # TF32 matrix multiplies on the A100: faster, and precise enough for
        # training. Full float16 is avoided; GAN training is fragile enough.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 78)
    print(f"job id        : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"host          : {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"device        : {describe_device(device)}")
    print(f"torch         : {torch.__version__}")
    print(f"resolution    : {args.resolution}   z dim: {args.z_dim}   width: {args.width}")
    print(f"steps         : {args.steps}   batch size: {args.batch_size}   "
          f"lr G {args.lr_g} / D {args.lr_d}   betas ({args.beta1}, {args.beta2})")
    print(f"loss          : {args.loss}   R1 gamma: {args.r1_gamma}   "
          f"spectral norm: {not args.no_spectral_norm}")
    print(f"DiffAugment   : {args.diffaugment or 'off'}   EMA decay: {args.ema_decay}")
    print(f"resuming      : {resuming}")
    print("=" * 78)

    load_start = time.perf_counter()
    train_images = load_split(args.data_root, "train", args.resolution, device, args.workers, args.limit)
    validate_images = load_split(args.data_root, "validate", args.resolution, device, args.workers, args.limit)
    print(f"loaded {len(train_images):,} training and {len(validate_images):,} validation slices "
          f"at {args.resolution}x{args.resolution} in {time.perf_counter() - load_start:.1f}s")

    generator = Generator(args.resolution, args.z_dim, args.width).to(device)
    discriminator = Discriminator(args.resolution, args.width, spectral=not args.no_spectral_norm).to(device)
    discriminator_loss, generator_loss = LOSSES[args.loss]
    print(f"generator     : {count_parameters(generator):,} parameters")
    print(f"discriminator : {count_parameters(discriminator):,} parameters")

    # beta1 = 0: momentum in Adam lets an update keep pushing after the other
    # network has already moved, which feeds oscillation. Two learning rates,
    # the discriminator's four times the generator's (TTUR, Heusel et al. 2017),
    # keep the discriminator a little ahead so its gradient stays informative.
    optimiser_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimiser_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    ema = EMA(generator, args.ema_decay)

    set_seed(args.seed)
    # The same 64 noise vectors for every progress grid, so successive grids
    # show the same brains taking shape rather than different random draws.
    fixed_z = torch.randn(GRID_SIDE * GRID_SIDE, args.z_dim, generator=torch.Generator().manual_seed(args.seed)).to(device)
    step = 0
    trained_seconds = 0.0
    collapse = {"low_samples": 0, "last_healthy": None, "detected_at": None}

    if resuming:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        for key in ("resolution", "z_dim", "width", "no_spectral_norm"):
            if checkpoint["args"].get(key, False) != getattr(args, key):
                raise SystemExit(f"{last_path} was trained with {key}={checkpoint['args'].get(key, False)}, "
                                 f"not {getattr(args, key)}; use another --out-dir or --fresh")
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        ema.shadow.load_state_dict(checkpoint["ema"])
        optimiser_g.load_state_dict(checkpoint["optimiser_g"])
        optimiser_d.load_state_dict(checkpoint["optimiser_d"])
        fixed_z = checkpoint["fixed_z"].to(device)
        step = checkpoint["step"]
        trained_seconds = checkpoint["trained_seconds"]
        collapse = checkpoint.get("collapse", collapse)
        # map_location moved every tensor to the device, but RNG states must be
        # CPU byte tensors whichever generator they belong to.
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        print(f"resumed from step {step:,} ({trained_seconds / 60:.1f} min trained so far)")

    history_file, history_writer = open_csv(out_dir / "history.csv", HISTORY_FIELDS, resuming)
    diversity_file, diversity_writer = open_csv(out_dir / "diversity.csv", DIVERSITY_FIELDS, resuming)

    def save_checkpoint() -> None:
        state = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "ema": ema.shadow.state_dict(),
            "optimiser_g": optimiser_g.state_dict(),
            "optimiser_d": optimiser_d.state_dict(),
            "fixed_z": fixed_z.cpu(),
            "step": step,
            "trained_seconds": trained_seconds,
            "collapse": collapse,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
            "args": vars(args),
        }
        # Write then rename, so a job killed mid-write cannot leave a corrupt
        # last.pt behind in place of the good one.
        temporary = last_path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, last_path)

    def generator_record(model: Generator) -> dict:
        """The format final.pt and snapshots share, which evaluate_gan.py loads."""
        return {
            "generator": model.state_dict(),
            "resolution": args.resolution,
            "z_dim": args.z_dim,
            "width": args.width,
            "step": step,
            "args": vars(args),
        }

    @torch.no_grad()
    def sample_and_measure() -> None:
        sampler = recalibrate_batchnorm(ema.shadow, device)
        grid = sampler(fixed_z)
        save_grid(grid, out_dir / "progress" / f"step_{step:06d}.png")
        generated = sampler(torch.randn(DIVERSITY_BATCH, args.z_dim, device=device))
        real_index = torch.randperm(len(train_images), device=device)[:DIVERSITY_BATCH]
        generated_distance = mean_pairwise_distance(generated)
        real_distance = mean_pairwise_distance(to_model_range(train_images[real_index]))
        diversity_writer.writerow({
            "step": step,
            "generated_distance": generated_distance,
            "real_distance": real_distance,
            "ratio": generated_distance / real_distance,
        })
        diversity_file.flush()
        ratio = generated_distance / real_distance
        print(f"          diversity at step {step:,}: generated {generated_distance:.2f} vs "
              f"real {real_distance:.2f} (ratio {ratio:.2f})", flush=True)

        if step < args.collapse_after:
            return
        if ratio >= args.healthy_ratio:
            snapshot = out_dir / "snapshots" / f"step_{step:06d}.pt"
            snapshot.parent.mkdir(exist_ok=True)
            record = generator_record(sampler)
            record["diversity_ratio"] = ratio
            torch.save(record, snapshot)
            previous = collapse["last_healthy"]
            if previous is not None and previous != str(snapshot):
                Path(previous).unlink(missing_ok=True)
            collapse["last_healthy"] = str(snapshot)
        collapse["low_samples"] = collapse["low_samples"] + 1 if ratio < args.collapse_ratio else 0
        if collapse["low_samples"] >= args.collapse_patience:
            collapse["detected_at"] = step
            print(f"          COLLAPSE: diversity below {args.collapse_ratio} at "
                  f"{args.collapse_patience} samples in a row", flush=True)

    stop = StopRequest()
    session_start = time.perf_counter()
    window = {"d_loss": 0.0, "g_loss": 0.0, "r1": 0.0, "d_real": 0.0, "d_fake": 0.0, "count": 0}
    synchronise(device)
    window_start = time.perf_counter()

    if step == 0:
        sample_and_measure()  # the untrained generator, for the start of the progress sequence

    while step < args.steps:
        generator.train()
        discriminator.train()

        # ---- discriminator update: real slices up, generated slices down ----
        index = torch.randint(0, len(train_images), (args.batch_size,), device=device)
        real = to_model_range(train_images[index])
        with torch.no_grad():
            fake = generator(torch.randn(args.batch_size, args.z_dim, device=device))
        if args.r1_gamma > 0:
            real.requires_grad_(True)
        real_scores = discriminator(diff_augment(real, args.diffaugment))
        fake_scores = discriminator(diff_augment(fake, args.diffaugment))
        d_loss = discriminator_loss(real_scores, fake_scores)
        r1 = r1_penalty(real_scores, real) if args.r1_gamma > 0 else torch.zeros((), device=device)
        d_loss = d_loss + 0.5 * args.r1_gamma * r1
        optimiser_d.zero_grad(set_to_none=True)
        d_loss.backward()
        optimiser_d.step()

        # ---- generator update: make the discriminator score its images as real ----
        fake = generator(torch.randn(args.batch_size, args.z_dim, device=device))
        g_loss = generator_loss(discriminator(diff_augment(fake, args.diffaugment)))
        optimiser_g.zero_grad(set_to_none=True)
        g_loss.backward()
        optimiser_g.step()

        ema.update(generator)
        step += 1

        window["d_loss"] += d_loss.detach()
        window["g_loss"] += g_loss.detach()
        window["d_real"] += real_scores.detach().mean()
        window["d_fake"] += fake_scores.detach().mean()
        window["r1"] += r1.detach()
        window["count"] += 1

        if step % args.log_every == 0 or step == args.steps:
            # Validation slices are never trained on. If the discriminator scores
            # training slices far higher than validation slices, it is memorising
            # the training set - the overfitting DiffAugment is there to prevent.
            # Both are scored the same way: unaugmented, and on the same number
            # of slices. d_real above is scored on augmented slices, so it is not
            # comparable with a validation score and is not used for this.
            with torch.no_grad():
                discriminator.eval()
                train_index = torch.randint(0, len(train_images), (OVERFIT_CHECK_BATCH,), device=device)
                validate_index = torch.randint(0, len(validate_images), (OVERFIT_CHECK_BATCH,), device=device)
                d_train_clean = discriminator(to_model_range(train_images[train_index])).mean().item()
                d_validate_clean = discriminator(to_model_range(validate_images[validate_index])).mean().item()
            synchronise(device)
            elapsed = time.perf_counter() - window_start
            trained_seconds += elapsed
            count = window["count"]
            record = {
                "step": step,
                "d_loss": (window["d_loss"] / count).item(),
                "g_loss": (window["g_loss"] / count).item(),
                "d_real": (window["d_real"] / count).item(),
                "d_fake": (window["d_fake"] / count).item(),
                "r1": (window["r1"] / count).item(),
                "d_train_clean": d_train_clean,
                "d_validate_clean": d_validate_clean,
                "seconds_per_step": elapsed / count,
            }
            history_writer.writerow(record)
            history_file.flush()
            print(f"step {step:7,}/{args.steps:,}  D {record['d_loss']:.3f}  G {record['g_loss']:+.3f}  "
                  f"scores real {record['d_real']:+.2f} fake {record['d_fake']:+.2f} | "
                  f"unaugmented train {d_train_clean:+.2f} validate {d_validate_clean:+.2f}  "
                  f"{1000 * record['seconds_per_step']:.0f} ms/step",
                  flush=True)
            if not all(np.isfinite(value) for value in record.values()):
                # Deliberately no checkpoint here: last.pt still holds the last
                # healthy state, and overwriting it with diverged weights would
                # destroy the only thing worth resuming from.
                raise SystemExit("non-finite loss or score: training has diverged; "
                                 f"last.pt is from step {step - step % args.checkpoint_every:,} or earlier")
            window = {"d_loss": 0.0, "g_loss": 0.0, "r1": 0.0, "d_real": 0.0, "d_fake": 0.0, "count": 0}

        if step % args.sample_every == 0 or step == args.steps:
            sample_and_measure()
            if collapse["detected_at"] is not None:
                break

        out_of_time = (args.max_minutes is not None
                       and time.perf_counter() - session_start > args.max_minutes * 60)
        if (out_of_time or stop.requested) and window["count"] > 0:
            # Stopping mid-window: bank the unlogged training time before saving.
            synchronise(device)
            trained_seconds += time.perf_counter() - window_start
        if step % args.checkpoint_every == 0 or step == args.steps or out_of_time or stop.requested:
            save_checkpoint()
            if (out_of_time or stop.requested) and step < args.steps:
                print(f"\nstopped at step {step:,} of {args.steps:,}; last.pt is saved. "
                      "Resubmit the same command to continue.")
                history_file.close()
                diversity_file.close()
                return

        if window["count"] == 0:
            # A window has just been logged. Restart its clock only now, after
            # sampling and checkpointing, so seconds_per_step measures training.
            synchronise(device)
            window_start = time.perf_counter()

    history_file.close()
    diversity_file.close()

    if collapse["detected_at"] is None:
        final = recalibrate_batchnorm(ema.shadow, device)
        torch.save(generator_record(final), out_dir / "final.pt")
        save_grid(final(fixed_z), out_dir / "final_grid.png")
        final_source = f"the averaged generator at step {step:,}"
    elif collapse["last_healthy"] is not None:
        save_checkpoint()  # the collapsed state, for the record; training will not resume from it
        healthy = torch.load(collapse["last_healthy"], map_location=device, weights_only=False)
        healthy["stopped_by_collapse_rule"] = True
        healthy["collapse_detected_at"] = collapse["detected_at"]
        torch.save(healthy, out_dir / "final.pt")
        final = Generator(args.resolution, args.z_dim, args.width).to(device)
        final.load_state_dict(healthy["generator"])
        final.eval()
        with torch.no_grad():
            save_grid(final(fixed_z), out_dir / "final_grid.png")
        final_source = (f"the last healthy snapshot, step {healthy['step']:,} "
                        f"(collapse declared at step {collapse['detected_at']:,})")
    else:
        save_checkpoint()
        raise SystemExit(f"collapse declared at step {collapse['detected_at']:,} with no healthy "
                         "snapshot to fall back on; no final.pt written")
    # A finished or collapsed run is complete; resubmitting must not train on.
    (out_dir / "COMPLETE").write_text(final_source + "\n", encoding="utf-8")

    summary = {
        "steps": step,
        "trained_seconds": trained_seconds,
        "final": final_source,
        "collapse_rule": {
            "healthy_ratio": args.healthy_ratio,
            "collapse_ratio": args.collapse_ratio,
            "patience": args.collapse_patience,
            "after_step": args.collapse_after,
            "detected_at": collapse["detected_at"],
            "last_healthy_snapshot": collapse["last_healthy"],
        },
        "device": describe_device(device),
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 78)
    print(f"finished {step:,} steps in {trained_seconds / 60:.1f} min of training")
    print(f"final.pt           : {final_source}")
    print(f"progress grids     : {out_dir / 'progress'}")
    print(f"\nNext:  sbatch slurm/evaluate_gan.sh {out_dir}")


if __name__ == "__main__":
    main()
