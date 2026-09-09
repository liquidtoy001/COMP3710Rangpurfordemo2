"""Turn a UNet run into the figures the demonstration needs (Part 4, Task 2).

The lab sheet requires the segmentation results to be visualised in order to
justify the DSC scores - a table of numbers is not evidence on its own. Run
locally, after copying a run directory back.

    python plot_unet.py runs/unet

Produces, in the run directory:

    dice.png            per-class DSC against the 0.9 requirement, over training
    curves.png          training and validation loss
    segmentations.png   image / ground truth / prediction / disagreement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DICE_TARGET = 0.9

# One colour per class, shared by the ground truth and prediction panels so the
# two can be compared at a glance. Class 0 is background and stays black.
CLASS_COLOURS = ListedColormap(["#000000", "#4c72b0", "#dd8452", "#55a868"])


def plot_dice(summary: dict, path: Path) -> None:
    """Per-class DSC over training, and the final test scores."""
    history = summary["history"]
    epochs = [row["epoch"] for row in history]
    num_classes = len(summary["test_dice"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for index in range(num_classes):
        axes[0].plot(
            epochs,
            [row[f"val_dice_c{index}"] for row in history],
            label=f"class {index}",
        )
    axes[0].axhline(DICE_TARGET, color="r", linestyle="--", alpha=0.7, label="0.9 requirement")
    best_epoch = summary["best_epoch"]
    axes[0].axvline(best_epoch, color="grey", linestyle=":", label=f"selected (epoch {best_epoch})")
    axes[0].set_title("Validation Dice per class")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("DSC")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    test_dice = summary["test_dice"]
    colours = ["tab:green" if value > DICE_TARGET else "tab:red" for value in test_dice]
    bars = axes[1].bar(range(num_classes), test_dice, color=colours, alpha=0.85)
    axes[1].axhline(DICE_TARGET, color="r", linestyle="--", alpha=0.7, label="0.9 requirement")
    for bar, value in zip(bars, test_dice):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, value + 0.01,
            f"{value:.4f}", ha="center", fontsize=10,
        )
    axes[1].set_title("Test set Dice per class")
    axes[1].set_xlabel("class")
    axes[1].set_ylabel("DSC")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xticks(range(num_classes))
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3, axis="y")

    verdict = "MET" if summary["target_met"] else "NOT MET"
    fig.suptitle(
        f"UNet on OASIS  -  worst class {summary['test_dice_min']:.4f},  "
        f"requirement (every label > {DICE_TARGET}) {verdict}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(summary: dict, path: Path) -> None:
    history = summary["history"]
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="validate")
    axes[0].set_title("Loss (Dice + cross entropy)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [r["val_dice_min"] for r in history], label="worst class")
    axes[1].plot(epochs, [r["val_dice_mean"] for r in history], label="mean")
    axes[1].axhline(DICE_TARGET, color="r", linestyle="--", alpha=0.7, label="0.9")
    axes[1].set_title("Worst class vs mean - the gap is why selection uses the worst")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("DSC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_segmentations(run_dir: Path, path: Path, count: int = 4) -> None:
    """Image, ground truth, prediction, and where the two disagree.

    The disagreement panel is the point: a prediction that looks right beside a
    mask still hides its errors, and at DSC around 0.96 the differences are a
    few pixels at the boundaries - invisible unless they are isolated.
    """
    data = np.load(run_dir / "segmentations.npz")
    images = data["images"]
    truth = data["ground_truth"]
    predictions = data["predictions"]
    count = min(count, images.shape[0])
    num_classes = int(max(truth.max(), predictions.max())) + 1

    fig, axes = plt.subplots(count, 4, figsize=(13, 3.2 * count))
    axes = np.atleast_2d(axes)

    for row in range(count):
        axes[row][0].imshow(images[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row][1].imshow(truth[row], cmap=CLASS_COLOURS, vmin=0, vmax=num_classes - 1)
        axes[row][2].imshow(predictions[row], cmap=CLASS_COLOURS, vmin=0, vmax=num_classes - 1)

        disagreement = truth[row] != predictions[row]
        axes[row][3].imshow(images[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row][3].imshow(
            np.ma.masked_where(~disagreement, disagreement), cmap="autumn", alpha=0.9
        )

        wrong = int(disagreement.sum())
        total = disagreement.size
        axes[row][3].set_xlabel(f"{wrong:,} of {total:,} pixels ({wrong / total:.2%})", fontsize=9)

        for column in range(4):
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])

    for column, title in enumerate(["input", "ground truth", "prediction", "disagreement"]):
        axes[0][column].set_title(title, fontsize=12)

    fig.suptitle("OASIS segmentation results", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarise(summary: dict) -> None:
    print("-" * 62)
    print(f"best epoch      : {summary['best_epoch']} of {len(summary['history'])}")
    print("test Dice per class:")
    for index, value in enumerate(summary["test_dice"]):
        status = "ok " if value > DICE_TARGET else "LOW"
        print(f"  class {index}      {value:.4f}  {status}  {'#' * int(value * 40)}")
    print(f"worst class     : {summary['test_dice_min']:.4f}")
    print(f"mean            : {summary['test_dice_mean']:.4f}")
    print(f"requirement     : {'MET' if summary['target_met'] else 'NOT MET'}"
          f"  (every label > {DICE_TARGET})")
    print(f"training time   : {summary['total_seconds']:.1f}s "
          f"({summary['total_seconds'] / 60:.1f} min) on {summary['device']}")
    print("-" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="a UNet run directory containing metrics.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    summarise(summary)

    for name, function in (
        ("dice.png", lambda p: plot_dice(summary, p)),
        ("curves.png", lambda p: plot_curves(summary, p)),
        ("segmentations.png", lambda p: plot_segmentations(run_dir, p)),
    ):
        path = run_dir / name
        function(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
