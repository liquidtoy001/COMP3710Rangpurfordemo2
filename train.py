"""Train ResNet-18 on CIFAR-10 to the DAWNBench accuracy target.

Part 3.2 requirement 1 of the lab sheet: more than 90% test accuracy, trainable
in a fast time (usually under 30 minutes on a cluster).

Smoke test first, on a few batches, to prove the plumbing works:

    python train.py --epochs 1 --limit-batches 5 --data-dir $HOME/data

Full run:

    python train.py --epochs 30 --data-dir $HOME/data --out-dir runs/baseline
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
import torch.nn as nn

from data import build_loaders, describe_device, resolve_device
from resnet import ResNet18, count_parameters


def set_seed(seed: int) -> None:
    """Seed every source of randomness the run touches.

    Not enough for bit-exact reproducibility on a GPU (cuDNN picks algorithms
    non-deterministically), but enough that two runs are comparable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronise(device: torch.device) -> None:
    """Wait for queued GPU work before reading the clock.

    CUDA calls are asynchronous: without this, a timing measurement records how
    long it took to *queue* the work, not to run it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    limit_batches: int | None = None,
) -> tuple[float, float]:
    """Return ``(accuracy, mean_loss)`` on the given loader."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss_sum += criterion(logits, targets).item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return correct / total, loss_sum / total


def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    scaler,
    limit_batches: int | None = None,
) -> tuple[float, float]:
    """One pass over the training set. Returns ``(accuracy, mean_loss)``."""
    model.train()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch_index, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        # autocast runs the eligible operations in half precision. It is a no-op
        # when scaler is disabled, so the same code path serves both the
        # full-precision baseline and the mixed-precision experiments.
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, targets)

        # In fp16 the small gradients underflow to zero, so the loss is scaled
        # up before the backward pass and the gradients scaled back down before
        # the optimiser step.
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # One-cycle steps per batch, not per epoch.
        if scheduler is not None:
            scheduler.step()

        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return correct / total, loss_sum / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data", help="CIFAR-10 location (see prepare_data.py)")
    parser.add_argument("--out-dir", default="runs/baseline", help="where checkpoints and metrics go")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1, help="peak learning rate for the one-cycle schedule")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--amp", action="store_true", help="mixed precision (part 3.2c); off for the baseline")
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="stop each epoch after N batches - for smoke tests only",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Context for the run log. Worth having in the output file, because a job
    # that ran three days ago is otherwise impossible to interpret.
    print("=" * 68)
    print(f"job id        : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"host          : {os.environ.get('SLURMD_NODENAME', os.uname().nodename if hasattr(os, 'uname') else 'unknown')}")
    print(f"device        : {describe_device(device)}")
    print(f"torch         : {torch.__version__}")
    print(f"mixed precision: {args.amp}")
    print(f"epochs        : {args.epochs}   batch size: {args.batch_size}   peak lr: {args.lr}")
    print("=" * 68)

    train_loader, test_loader = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        download=False,
        pin_memory=device.type == "cuda",
    )
    print(f"train batches : {len(train_loader)}   test batches: {len(test_loader)}")

    model = ResNet18(num_classes=10).to(device)
    print(f"parameters    : {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # SGD with Nesterov momentum rather than Adam: on CIFAR-10 with a one-cycle
    # schedule it reaches a higher final accuracy, and it is what the DAWNBench
    # reference solutions use.
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    steps_per_epoch = args.limit_batches if args.limit_batches else len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
    )

    # Disabled unless --amp, in which case autocast and loss scaling turn on
    # together.
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    # Per-epoch history is appended to a CSV *as the run goes*, and flushed
    # every epoch. If the job hits its Slurm time limit or the node dies, the
    # record of everything up to that point survives - which a summary written
    # only at the end would not.
    csv_path = out_dir / "history.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch", "train_accuracy", "train_loss",
            "test_accuracy", "test_loss", "epoch_seconds", "lr",
        ],
    )
    csv_writer.writeheader()
    csv_file.flush()

    history = []
    best_accuracy = 0.0
    synchronise(device)
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        synchronise(device)
        epoch_start = time.perf_counter()

        train_accuracy, train_loss = train_one_epoch(
            model, train_loader, device, criterion, optimiser, scheduler, scaler, args.limit_batches
        )
        synchronise(device)
        epoch_seconds = time.perf_counter() - epoch_start

        test_accuracy, test_loss = evaluate(
            model, test_loader, device, criterion, args.limit_batches
        )

        record = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "epoch_seconds": epoch_seconds,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        csv_writer.writerow(record)
        csv_file.flush()

        marker = ""
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            marker = "  <- best"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "test_accuracy": test_accuracy,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train {train_accuracy:6.2%}  test {test_accuracy:6.2%}  "
            f"loss {train_loss:.4f}  {epoch_seconds:6.1f}s{marker}",
            flush=True,
        )

    total_seconds = time.perf_counter() - run_start
    csv_file.close()

    summary = {
        "best_test_accuracy": best_accuracy,
        "final_test_accuracy": history[-1]["test_accuracy"] if history else None,
        "total_seconds": total_seconds,
        "device": describe_device(device),
        "args": vars(args),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 68)
    print(f"best test accuracy : {best_accuracy:.2%}")
    print(f"total wall clock   : {total_seconds:.1f}s ({total_seconds / 60:.1f} min)")
    print(f"target (>90%)      : {'MET' if best_accuracy > 0.90 else 'NOT MET'}")
    print(f"checkpoint         : {out_dir / 'best.pt'}")
    print(f"metrics            : {out_dir / 'metrics.json'}")
    print(f"per-epoch history  : {csv_path}")

    # Training curves, if matplotlib happens to be installed on the cluster.
    # It is not needed there: plot_run.py regenerates the figure locally from
    # metrics.json after the results are copied back.
    try:
        from plot_run import plot_history

        figure_path = plot_history(summary, out_dir / "curves.png")
        print(f"curves             : {figure_path}")
    except ImportError:
        print("curves             : skipped (matplotlib not installed here);"
              " run plot_run.py locally after scp")


if __name__ == "__main__":
    main()
