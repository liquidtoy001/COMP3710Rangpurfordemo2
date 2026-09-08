"""Train the OASIS UNet (Part 4, Task 2).

The lab sheet's bar is specific: **Dice above 0.9 for every label**, not on
average. So the per-class scores are printed every epoch, the checkpoint is
selected on the *worst* class rather than the mean, and the final summary states
plainly whether the requirement is met.

Segmentation results must also be visualised to justify the scores, and
inference must be run live at the demonstration. This script saves the arrays
``plot_unet.py`` turns into overlays; ``demo_unet.py`` handles the live run.

Smoke test first:

    python train_unet.py --epochs 1 --limit 64 --out-dir runs/unet_smoke

Full run:

    sbatch slurm/train_unet.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from oasis import DEFAULT_ROOT, NUM_CLASSES, build_loaders
from unet import CombinedLoss, DiceScore, UNet, count_parameters

# How many test slices to save as image / ground truth / prediction triples.
NUM_VISUALISED = 8

DICE_TARGET = 0.9


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


def train_one_epoch(
    model: UNet,
    loader,
    device: torch.device,
    criterion: CombinedLoss,
    optimiser: torch.optim.Optimizer,
    limit_batches: int | None = None,
) -> float:
    model.train()
    loss_sum = 0.0
    seen = 0

    for batch_index, (images, masks) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimiser.step()

        loss_sum += loss.item() * images.size(0)
        seen += images.size(0)

    return loss_sum / seen


@torch.no_grad()
def evaluate(
    model: UNet,
    loader,
    device: torch.device,
    criterion: CombinedLoss,
    limit_batches: int | None = None,
) -> tuple[float, list[float]]:
    """Return ``(mean_loss, per_class_dice)``.

    Dice is accumulated across the whole loader and computed once at the end -
    see :class:`unet.DiceScore` for why per-batch averaging would be wrong.
    """
    model.eval()
    score = DiceScore(NUM_CLASSES, device=device)
    loss_sum = 0.0
    seen = 0

    for batch_index, (images, masks) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss_sum += criterion(logits, masks).item() * images.size(0)
        seen += images.size(0)
        score.update(logits.argmax(dim=1), masks)

    return loss_sum / seen, score.compute()


def format_dice(per_class: list[float]) -> str:
    return " ".join(f"c{index}:{value:.4f}" for index, value in enumerate(per_class))


@torch.no_grad()
def save_visualisation_data(model: UNet, loader, device: torch.device, out_dir: Path) -> None:
    """Save image / ground truth / prediction triples for the overlay figures.

    The lab sheet requires the segmentation results to be visualised in order to
    justify the DSC scores - a number on its own is not evidence.
    """
    model.eval()
    images, masks = next(iter(loader))
    images = images[:NUM_VISUALISED].to(device)
    masks = masks[:NUM_VISUALISED]

    predictions = model(images).argmax(dim=1).cpu()

    np.savez(
        out_dir / "segmentations.npz",
        images=images.cpu().numpy(),
        ground_truth=masks.numpy(),
        predictions=predictions.numpy(),
    )
    print(f"  {images.size(0)} image/truth/prediction triples -> segmentations.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default="runs/unet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="N images per split (smoke tests)")
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"job id        : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"host          : {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"device        : {describe_device(device)}")
    print(f"torch         : {torch.__version__}")
    print(f"classes       : {NUM_CLASSES}   base channels: {args.base_channels}")
    print(f"epochs        : {args.epochs}   batch size: {args.batch_size}   lr: {args.lr}")
    print(f"target        : DSC > {DICE_TARGET} for EVERY class")
    print("=" * 78)

    train_loader, validate_loader, test_loader = build_loaders(
        args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        with_masks=True,
        limit=args.limit,
    )
    print(f"train {len(train_loader.dataset):,} | "
          f"validate {len(validate_loader.dataset):,} | "
          f"test {len(test_loader.dataset):,} slices")

    model = UNet(num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    print(f"parameters    : {count_parameters(model):,}")

    criterion = CombinedLoss(num_classes=NUM_CLASSES)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    csv_path = out_dir / "history.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = (
        ["epoch", "train_loss", "val_loss"]
        + [f"val_dice_c{index}" for index in range(NUM_CLASSES)]
        + ["val_dice_min", "val_dice_mean", "epoch_seconds", "lr"]
    )
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    history = []
    best_min_dice = -1.0
    synchronise(device)
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        synchronise(device)
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model, train_loader, device, criterion, optimiser, args.limit_batches
        )
        synchronise(device)
        epoch_seconds = time.perf_counter() - epoch_start

        val_loss, per_class = evaluate(
            model, validate_loader, device, criterion, args.limit_batches
        )
        scheduler.step()

        min_dice = min(per_class)
        mean_dice = sum(per_class) / len(per_class)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_dice_c{index}": value for index, value in enumerate(per_class)},
            "val_dice_min": min_dice,
            "val_dice_mean": mean_dice,
            "epoch_seconds": epoch_seconds,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        writer.writerow(record)
        csv_file.flush()

        # Selected on the worst class, not the mean. The requirement is that
        # *every* label clears 0.9, and a mean is easily carried over the line
        # by the background class while a tissue class languishes.
        marker = ""
        if min_dice > best_min_dice:
            best_min_dice = min_dice
            marker = "  <- best"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": per_class,
                    "val_dice_min": min_dice,
                    "num_classes": NUM_CLASSES,
                    "base_channels": args.base_channels,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train {train_loss:.4f}  val {val_loss:.4f}  "
            f"dice[{format_dice(per_class)}]  "
            f"min {min_dice:.4f}  {epoch_seconds:6.1f}s{marker}",
            flush=True,
        )

    total_seconds = time.perf_counter() - run_start
    csv_file.close()

    # Final numbers come from the best checkpoint on the held-out test split,
    # which was never used for selection.
    print("\nevaluating the best checkpoint on the test split")
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_dice = evaluate(model, test_loader, device, criterion)
    save_visualisation_data(model, test_loader, device, out_dir)

    summary = {
        "best_epoch": checkpoint["epoch"],
        "validation_dice": checkpoint["val_dice"],
        "test_dice": test_dice,
        "test_dice_min": min(test_dice),
        "test_dice_mean": sum(test_dice) / len(test_dice),
        "test_loss": test_loss,
        "target_met": min(test_dice) > DICE_TARGET,
        "total_seconds": total_seconds,
        "device": describe_device(device),
        "args": vars(args),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 78)
    print(f"best epoch          : {checkpoint['epoch']}")
    print("test set Dice per class:")
    for index, value in enumerate(test_dice):
        status = "ok " if value > DICE_TARGET else "LOW"
        bar = "#" * int(value * 40)
        print(f"  class {index}  {value:.4f}  {status}  {bar}")
    print(f"worst class         : {min(test_dice):.4f}")
    print(f"mean                : {sum(test_dice) / len(test_dice):.4f}")
    print(f"requirement (all > {DICE_TARGET}) : "
          f"{'MET' if min(test_dice) > DICE_TARGET else 'NOT MET'}")
    print(f"total wall clock    : {total_seconds:.1f}s ({total_seconds / 60:.1f} min)")
    print(f"checkpoint          : {out_dir / 'best.pt'}")
    print("\nCopy this directory back and run:  python plot_unet.py " + str(out_dir))


if __name__ == "__main__":
    main()
