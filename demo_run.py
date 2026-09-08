"""Live demonstration script for part 3.2 requirement 2.

The lab sheet requires that the model "run inference and a single epoch of
training on the Ranpur compute cluster **during the demonstration**". Full
training is done beforehand; this script does only the two things that have to
happen in front of the demonstrator, and should finish in a couple of minutes:

    1. load the trained checkpoint and run inference over the whole test set
    2. run exactly one epoch of training, timed

It never overwrites the checkpoint, so it is safe to run twice if the first
attempt is interrupted.

    python demo_run.py --checkpoint runs/baseline/best.pt --data-dir $HOME/data
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from data import CLASS_NAMES, build_loaders, describe_device, resolve_device
from train import evaluate, synchronise, train_one_epoch
from resnet import ResNet18, count_parameters


@torch.no_grad()
def per_class_accuracy(model: nn.Module, loader, device: torch.device) -> list[float]:
    """Accuracy for each of the ten classes, to show the model really works."""
    model.eval()
    correct = torch.zeros(len(CLASS_NAMES))
    total = torch.zeros(len(CLASS_NAMES))

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1).cpu()
        for label, prediction in zip(targets, predictions):
            total[label] += 1
            correct[label] += int(label == prediction)

    return (correct / total).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for the single demo epoch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)

    print("=" * 68)
    print("COMP3710 demonstration 2 - part 3.2, live run on Rangpur")
    print(f"job id : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"host   : {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"device : {describe_device(device)}")
    print("=" * 68)

    train_loader, test_loader = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        download=False,
        pin_memory=device.type == "cuda",
    )

    model = ResNet18(num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nloaded {args.checkpoint}")
    print(f"  trained for   : {checkpoint['epoch']} epochs")
    print(f"  recorded acc  : {checkpoint['test_accuracy']:.2%}")
    print(f"  parameters    : {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()

    # ---- 1. Inference -------------------------------------------------------
    print("\n[1/2] inference over the 10,000 test images")
    synchronise(device)
    start = time.perf_counter()
    accuracy, loss = evaluate(model, test_loader, device, criterion)
    synchronise(device)
    inference_seconds = time.perf_counter() - start

    print(f"  test accuracy : {accuracy:.2%}")
    print(f"  test loss     : {loss:.4f}")
    print(f"  time          : {inference_seconds:.2f}s "
          f"({10_000 / inference_seconds:,.0f} images/second)")

    print("\n  per-class accuracy:")
    for name, class_accuracy in zip(CLASS_NAMES, per_class_accuracy(model, test_loader, device)):
        bar = "#" * int(class_accuracy * 40)
        print(f"    {name:<12} {class_accuracy:6.2%}  {bar}")

    # ---- 2. One epoch of training ------------------------------------------
    print(f"\n[2/2] one epoch of training (lr={args.lr})")
    optimiser = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    synchronise(device)
    start = time.perf_counter()
    train_accuracy, train_loss = train_one_epoch(
        model, train_loader, device, criterion, optimiser, scheduler=None, scaler=scaler
    )
    synchronise(device)
    epoch_seconds = time.perf_counter() - start

    print(f"  train accuracy: {train_accuracy:.2%}")
    print(f"  train loss    : {train_loss:.4f}")
    print(f"  time          : {epoch_seconds:.2f}s for {len(train_loader)} batches")

    after_accuracy, _ = evaluate(model, test_loader, device, criterion)
    print(f"\n  test accuracy after the extra epoch: {after_accuracy:.2%} "
          f"(was {accuracy:.2%})")
    print("\nThe checkpoint on disk is unchanged; this run trained a copy in memory.")


if __name__ == "__main__":
    main()
