"""Turn a training run's ``metrics.json`` into figures for the demonstration.

``train.py`` calls this at the end of a run if matplotlib is available on the
cluster. It is also runnable on its own, which is the normal case: copy the run
directory back and render the figures on the machine that has matplotlib.

    scp -r s49133336@rangpur.compute.eait.uq.edu.au:~/COMP3710Rangpurfordemo2/runs/baseline runs/
    python plot_run.py runs/baseline

Produces ``curves.png`` (accuracy, loss, learning rate, per-epoch time) and
prints a short text summary suitable for reading out loud.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_history(summary: dict, output_path: Path) -> Path:
    """Render the four-panel training figure. Raises ImportError without matplotlib."""
    import matplotlib

    # Non-interactive backend, so this also works over SSH with no display.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = summary["history"]
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"ResNet-18 on CIFAR-10  -  best test accuracy {summary['best_test_accuracy']:.2%}"
        f"  in {summary['total_seconds'] / 60:.1f} min on {summary['device']}",
        fontsize=13,
    )

    # Accuracy. The 90% target line is the thing the demonstrator is checking.
    ax = axes[0][0]
    ax.plot(epochs, [row["train_accuracy"] for row in history], label="train")
    ax.plot(epochs, [row["test_accuracy"] for row in history], label="test")
    ax.axhline(0.90, color="r", linestyle="--", alpha=0.6, label="90% target")
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss. The gap between the curves is where overfitting shows up.
    ax = axes[0][1]
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["test_loss"] for row in history], label="test")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate: shows the one-cycle warm-up and decay actually happened.
    ax = axes[1][0]
    ax.plot(epochs, [row["lr"] for row in history], color="tab:green")
    ax.set_title("Learning rate (one-cycle)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lr")
    ax.grid(True, alpha=0.3)

    # Per-epoch wall clock, which is what the DAWNBench time budget is made of.
    ax = axes[1][1]
    seconds = [row["epoch_seconds"] for row in history]
    ax.bar(epochs, seconds, color="tab:purple", alpha=0.8)
    ax.set_title(f"Time per epoch (mean {sum(seconds) / len(seconds):.1f}s)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def summarise(summary: dict) -> None:
    """Print the numbers worth quoting to the demonstrator."""
    history = summary["history"]
    best = summary["best_test_accuracy"]
    total = summary["total_seconds"]
    seconds = [row["epoch_seconds"] for row in history]

    # First epoch at which the run crossed the target, if it did.
    crossed = next((row["epoch"] for row in history if row["test_accuracy"] > 0.90), None)

    print("-" * 60)
    print(f"device                : {summary['device']}")
    print(f"epochs                : {len(history)}")
    print(f"best test accuracy    : {best:.2%}")
    print(f"final test accuracy   : {summary['final_test_accuracy']:.2%}")
    print(f"total training time   : {total:.1f}s ({total / 60:.1f} min)")
    print(f"mean time per epoch   : {sum(seconds) / len(seconds):.1f}s")
    print(f"crossed 90% at epoch  : {crossed if crossed else 'never'}")
    print(f"DAWNBench >90% target : {'MET' if best > 0.90 else 'NOT MET'}")
    print(f"under 30 min          : {'yes' if total < 1800 else 'no'}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="a run directory containing metrics.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    summarise(summary)
    figure_path = plot_history(summary, run_dir / "curves.png")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
