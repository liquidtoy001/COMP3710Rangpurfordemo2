"""Test-time augmentation for part 3.2c: the mixed-precision model, scored with flips.

Part 3.2c asks for 94% test accuracy from a model trained with mixed precision in
about 360 s on a V100. The AMP run (``slurm/train_amp.sh``) reached 93.86%, which
is 14 of the 10,000 test images short. This script asks whether the same trained
weights clear 94% when each test image is also classified mirrored left to
right, with the two predictions averaged. No retraining, so the training time on
record does not change.

Why a horizontal flip, and only a flip. The model was trained with
``RandomHorizontalFlip``, so a mirrored image is exactly the kind of input it
learned to classify, and a mirrored CIFAR-10 object is still the same class. Flip
averaging is the test-time augmentation that fast DAWNBench CIFAR-10 entries
use. It is fixed here, before the script has ever been run, together with every
other choice below, because the test set must be scored once and not searched:
trying crops, scales or averaging rules until one crosses 94% would be tuning on
the test set, and the resulting number would not mean what it claims.

The choices, fixed in advance:

* the checkpoint: ``runs/amp/best.pt``. Its epoch is the last one (30), so it is
  the model the run ended with rather than one picked for its test score
* the augmentation: identity plus horizontal flip, nothing else
* the rule: average the two softmax distributions, then take the arg max
* the precision: full precision, as ``train.evaluate`` scores every epoch, so
  the plain accuracy below must reproduce the recorded 93.86% exactly

The plain score is computed in the same pass, from the same forward passes, so
any change is attributable to the flip alone. The script also counts the test
images the flip corrected and the ones it broke: with 10,000 images, one
accuracy figure has a standard error of about 0.24 percentage points, so how
many predictions actually changed is the honest measure of the effect.

    python tta_eval.py --checkpoint runs/amp/best.pt --data-dir $HOME/data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import CLASS_NAMES, build_loaders, describe_device, resolve_device
from resnet import ResNet18
from train import synchronise

TARGET = 0.94


@torch.no_grad()
def predict_with_flip(model: nn.Module, loader, device: torch.device) -> dict:
    """Score every test image plainly and with flip averaging, in one pass.

    Returns the two sets of predictions, the targets, and the time taken by each
    kind of forward pass, measured separately so the cost of the augmentation is
    visible.
    """
    model.eval()
    plain_predictions, tta_predictions, all_targets = [], [], []
    plain_seconds = 0.0
    flip_seconds = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        synchronise(device)
        start = time.perf_counter()
        plain_probabilities = torch.softmax(model(images), dim=1)
        synchronise(device)
        plain_seconds += time.perf_counter() - start

        # Images are (batch, channels, height, width), so dimension 3 is the
        # horizontal axis. Normalisation is per channel, so flipping the
        # normalised tensor equals normalising the flipped image.
        start = time.perf_counter()
        flipped_probabilities = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        synchronise(device)
        flip_seconds += time.perf_counter() - start

        plain_predictions.append(plain_probabilities.argmax(dim=1).cpu())
        tta_predictions.append(((plain_probabilities + flipped_probabilities) / 2).argmax(dim=1).cpu())
        all_targets.append(targets)

    return {
        "plain": torch.cat(plain_predictions),
        "tta": torch.cat(tta_predictions),
        "targets": torch.cat(all_targets),
        "plain_seconds": plain_seconds,
        "flip_seconds": flip_seconds,
    }


def per_class(predictions: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """Accuracy of each class."""
    return [
        (predictions[targets == label] == label).float().mean().item()
        for label in range(len(CLASS_NAMES))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/amp/best.pt")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="training batch size; the test loader uses twice this, as in training")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None,
                        help="where to write the results as JSON (default: beside the checkpoint)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"no checkpoint at {checkpoint_path}. runs/ is not in git: the AMP checkpoint "
            "exists only on Rangpur, where slurm/train_amp.sh wrote it."
        )

    device = resolve_device(args.device)
    print("=" * 68)
    print("COMP3710 demonstration 2 - part 3.2c, test-time flip augmentation")
    print(f"job id : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"host   : {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"device : {describe_device(device)}")
    print("=" * 68)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trained_with = checkpoint.get("args", {})
    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"\nloaded {checkpoint_path}")
    print(f"  mixed precision in training: {trained_with.get('amp')}")
    print(f"  epoch            : {checkpoint['epoch']} of {trained_with.get('epochs')}")
    print(f"  recorded accuracy: {checkpoint['test_accuracy']:.2%}")
    if not trained_with.get("amp"):
        print("  WARNING: this checkpoint was not trained with mixed precision,"
              " so it cannot count towards part 3.2c")
    if checkpoint["epoch"] != trained_with.get("epochs"):
        print("  WARNING: not the final epoch - this checkpoint was selected on test"
              " accuracy, which would make any gain below optimistic")

    _, test_loader = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        download=False,
        pin_memory=device.type == "cuda",
    )

    print(f"\nscoring the {len(test_loader.dataset):,} test images, plain and flipped ...")
    result = predict_with_flip(model, test_loader, device)
    targets = result["targets"]
    plain_correct = result["plain"] == targets
    tta_correct = result["tta"] == targets

    n = targets.numel()
    plain_accuracy = plain_correct.float().mean().item()
    tta_accuracy = tta_correct.float().mean().item()
    fixed = int((~plain_correct & tta_correct).sum())
    broken = int((plain_correct & ~tta_correct).sum())
    standard_error = math.sqrt(tta_accuracy * (1 - tta_accuracy) / n)

    print(f"\n  images scored      : {n:,}")
    print(f"  plain accuracy     : {plain_accuracy:.2%}   (recorded {checkpoint['test_accuracy']:.2%})")
    print(f"  flip-averaged      : {tta_accuracy:.2%}   ({tta_accuracy - plain_accuracy:+.2%})")
    print(f"  target             : {TARGET:.0%}   -> {'MET' if tta_accuracy >= TARGET else 'NOT MET'}")
    print(f"\n  images the flip corrected: {fixed}")
    print(f"  images the flip broke    : {broken}")
    print(f"  net                      : {fixed - broken:+d} of {n:,}")
    print(f"  standard error of one accuracy figure: {standard_error:.2%}")

    if abs(plain_accuracy - checkpoint["test_accuracy"]) > 0.5 / n:
        print("\n  NOTE: the plain accuracy does not reproduce the recorded figure, so"
              " something differs from training's evaluation; read the result with care")

    print("\n  per class          plain     flipped")
    for name, before, after in zip(CLASS_NAMES, per_class(result["plain"], targets),
                                   per_class(result["tta"], targets)):
        print(f"    {name:<12}   {before:7.2%}   {after:7.2%}")

    print(f"\n  inference time, plain pass : {result['plain_seconds']:.2f} s")
    print(f"  inference time, flip pass  : {result['flip_seconds']:.2f} s"
          "  (training time is unchanged: no retraining)")

    out_path = Path(args.out) if args.out else checkpoint_path.with_name("tta.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "trained_with_amp": trained_with.get("amp"),
        "recorded_test_accuracy": checkpoint["test_accuracy"],
        "plain_test_accuracy": plain_accuracy,
        "flip_tta_test_accuracy": tta_accuracy,
        "target": TARGET,
        "target_met": tta_accuracy >= TARGET,
        "images": n,
        "fixed_by_flip": fixed,
        "broken_by_flip": broken,
        "standard_error": standard_error,
        "plain_inference_seconds": result["plain_seconds"],
        "flip_inference_seconds": result["flip_seconds"],
        "device": describe_device(device),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "method": "average of softmax over identity and horizontal flip, full precision",
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
