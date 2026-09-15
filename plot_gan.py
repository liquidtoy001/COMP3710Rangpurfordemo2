"""Turn a GAN run and its evaluation into figures (Part 4, Task 3).

Run locally after copying the run directory back without its checkpoints;
matplotlib is not installed on Rangpur.

    python plot_gan.py runs/gan128

Writes, in the run directory:

    curves.png              losses, discriminator scores, diversity over training
    progress.png            the same noise vectors at successive steps
    samples.png             generated slices beside real test slices
    nearest_neighbours.png  generated slices above their nearest training slices
    distances.png           novelty and diversity distances, generated vs real
    latent.png              real and generated slices in the 2D VAE latent space
    tissue.png              tissue-class shares from the UNet, generated vs real
    quiz.png, quiz_key.png  real and generated mixed, with a separate answer key
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREY = dict(cmap="gray", vmin=0, vmax=255)
REAL_COLOUR = "#4C72B0"
GENERATED_COLOUR = "#DD8452"
REFERENCE_COLOUR = "#8C8C8C"
CLASS_NAMES = ("background", "class 1", "class 2", "class 3")


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_curves(run_dir: Path) -> None:
    history = read_csv(run_dir / "history.csv")
    diversity = read_csv(run_dir / "diversity.csv") if (run_dir / "diversity.csv").exists() else []
    steps = [row["step"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].plot(steps, [r["d_loss"] for r in history], label="discriminator (hinge)")
    axes[0].plot(steps, [r["g_loss"] for r in history], label="generator (hinge)")
    axes[0].set_title("Losses - they do not fall like a classifier's;\n"
                      "steady, bounded values are the healthy sign")

    axes[1].plot(steps, [r["d_real"] for r in history], color=REAL_COLOUR, alpha=0.5,
                 label="real slices, as trained on (augmented)")
    axes[1].plot(steps, [r["d_fake"] for r in history], color=GENERATED_COLOUR, alpha=0.5,
                 label="generated slices, as trained on (augmented)")
    axes[1].plot(steps, [r["d_train_clean"] for r in history], color="black",
                 label="training slices, unaugmented")
    axes[1].plot(steps, [r["d_validate_clean"] for r in history], color="black", linestyle=":",
                 label="validation slices, unaugmented (never trained on)")
    axes[1].set_title("Discriminator's mean score\n(black solid climbing away from dotted = memorising)")

    if diversity:
        axes[2].plot([r["step"] for r in diversity], [r["ratio"] for r in diversity], marker="o")
        axes[2].axhline(1.0, color=REFERENCE_COLOUR, linestyle="--", label="as diverse as real slices")
        axes[2].set_ylim(0, max(1.2, max(r["ratio"] for r in diversity) * 1.1))
    axes[2].set_title("Diversity: mean pairwise distance,\ngenerated / real (collapse drives it to 0)")

    for axis in axes:
        axis.set_xlabel("generator step")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)


def plot_progress(run_dir: Path, rows: int = 6, columns: int = 8) -> None:
    """A row per chosen step, showing the first ``columns`` fixed-noise samples."""
    frames = sorted((run_dir / "progress").glob("step_*.png"))
    if not frames:
        return
    chosen = [frames[round(i)] for i in np.linspace(0, len(frames) - 1, min(rows, len(frames)))]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(columns * 1.3, len(chosen) * 1.45))
    axes = np.atleast_1d(axes)
    for axis, frame in zip(axes, chosen):
        grid = np.array(Image.open(frame))
        tile = grid.shape[0] // 8
        axis.imshow(grid[:tile, : tile * columns], **GREY)
        axis.set_axis_off()
        axis.set_title(f"step {int(frame.stem.split('_')[1]):,}", fontsize=9, loc="left")
    fig.suptitle("The same eight noise vectors as training progresses", y=1.0)
    fig.tight_layout()
    fig.savefig(run_dir / "progress.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def mosaic(images: np.ndarray, columns: int) -> np.ndarray:
    rows = len(images) // columns
    return np.concatenate([np.concatenate(list(images[r * columns:(r + 1) * columns]), axis=1)
                           for r in range(rows)], axis=0)


def plot_samples(run_dir: Path, arrays: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4))
    axes[0].imshow(mosaic(arrays["grid_generated"], 8), **GREY)
    axes[0].set_title("Generated (the first 64 of a fixed seed - not selected)")
    axes[1].imshow(mosaic(arrays["grid_test"], 8), **GREY)
    axes[1].set_title("Real test slices (random 64)")
    for axis in axes:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(run_dir / "samples.png", dpi=120)
    plt.close(fig)


def plot_nearest(run_dir: Path, arrays: dict, report: dict) -> None:
    generated, train, distance = arrays["pairs_generated"], arrays["pairs_train"], arrays["pairs_distance"]
    count = len(generated)
    fig, axes = plt.subplots(2, count, figsize=(count * 1.25, 3.2))
    for i in range(count):
        axes[0, i].imshow(generated[i], **GREY)
        axes[1, i].imshow(train[i], **GREY)
        axes[1, i].set_title(f"{distance[i]:.3f}", fontsize=7)
        axes[0, i].set_axis_off()
        axes[1, i].set_axis_off()
    typical = report["novelty"]["test_to_train"]["median"]
    fig.suptitle(f"The {count} generated slices closest to the training set (top), above their nearest "
                 f"training slice with its RMS distance. A real test slice's median distance is {typical:.3f}.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "nearest_neighbours.png", dpi=120)
    plt.close(fig)


def plot_distances(run_dir: Path, arrays: dict, report: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    bins = 40
    axes[0].hist(arrays["test_to_train"], bins=bins, density=True, alpha=0.6, color=REAL_COLOUR,
                 label="real test slices")
    axes[0].hist(arrays["generated_to_train"], bins=bins, density=True, alpha=0.6, color=GENERATED_COLOUR,
                 label="generated slices")
    axes[0].axvline(report["novelty"]["copy_threshold"], color="black", linestyle="--",
                    label="5th percentile of real")
    axes[0].set_title("Distance to the nearest training slice\n(copies would pile up at the left)")

    axes[1].hist(arrays["real_to_real"], bins=bins, density=True, alpha=0.6, color=REAL_COLOUR,
                 label="random real training slices")
    axes[1].hist(arrays["generated_to_generated"], bins=bins, density=True, alpha=0.6, color=GENERATED_COLOUR,
                 label="generated slices")
    axes[1].set_title("Distance to the nearest other slice of the same set\n(collapse would pile up at the left)")
    for axis in axes:
        axis.set_xlabel("RMS pixel distance")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "distances.png", dpi=120)
    plt.close(fig)


def plot_latent(run_dir: Path, arrays: dict, report: dict) -> None:
    if "latent2_test" not in arrays:
        return
    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    axis.scatter(*arrays["latent2_test"].T, s=8, alpha=0.4, color=REFERENCE_COLOUR, label="real test slices")
    axis.scatter(*arrays["latent2_real"].T, s=8, alpha=0.6, color=REAL_COLOUR, label="random real training slices")
    axis.scatter(*arrays["latent2_generated"].T, s=8, alpha=0.6, color=GENERATED_COLOUR, label="generated slices")
    title = "Real and generated slices in the Task 1 VAE's 2D latent space"
    if "precision_recall" in report:
        pr = report["precision_recall"]
        title += (f"\n32D latent: precision {pr['generated']['precision']:.2f}, "
                  f"recall {pr['generated']['recall']:.2f} "
                  f"(random real training slices: {pr['reference']['precision']:.2f}, "
                  f"{pr['reference']['recall']:.2f})")
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("z1")
    axis.set_ylabel("z2")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "latent.png", dpi=120)
    plt.close(fig)


def plot_tissue(run_dir: Path, arrays: dict) -> None:
    if "fractions_real" not in arrays:
        return
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for c, axis in enumerate(axes):
        real, generated = arrays["fractions_real"][:, c], arrays["fractions_generated"][:, c]
        bins = np.linspace(min(real.min(), generated.min()), max(real.max(), generated.max()), 30)
        axis.hist(real, bins=bins, density=True, alpha=0.6, color=REAL_COLOUR, label="real training slices")
        axis.hist(generated, bins=bins, density=True, alpha=0.6, color=GENERATED_COLOUR, label="generated")
        axis.set_title(f"{CLASS_NAMES[c]}: share of the slice")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Tissue proportions, as segmented by the Task 2 UNet")
    fig.tight_layout()
    fig.savefig(run_dir / "tissue.png", dpi=120)
    plt.close(fig)


def plot_quiz(run_dir: Path, arrays: dict, count: int = 16, seed: int = 7) -> None:
    """Eight real and eight generated slices in a random order, and a separate key."""
    half = count // 2
    images = np.concatenate([arrays["grid_test"][:half], arrays["grid_generated"][:half]])
    is_generated = np.array([False] * half + [True] * half)
    order = np.random.default_rng(seed).permutation(count)
    images, is_generated = images[order], is_generated[order]

    for with_key, name in ((False, "quiz.png"), (True, "quiz_key.png")):
        fig, axes = plt.subplots(2, half, figsize=(half * 1.6, 3.8))
        for i, axis in enumerate(axes.flat):
            axis.imshow(images[i], **GREY)
            axis.set_axis_off()
            label = f"{i + 1}"
            if with_key:
                label += " generated" if is_generated[i] else " real"
            axis.set_title(label, fontsize=8, color=GENERATED_COLOUR if with_key and is_generated[i] else "black")
        fig.suptitle("Which are real?" if not with_key else "Answer key", fontsize=11)
        fig.tight_layout()
        fig.savefig(run_dir / name, dpi=120)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    run_dir = Path(parser.parse_args().run_dir)

    plot_curves(run_dir)
    plot_progress(run_dir)
    evaluation = run_dir / "evaluation"
    if (evaluation / "arrays.npz").exists():
        arrays = dict(np.load(evaluation / "arrays.npz"))
        report = json.loads((evaluation / "evaluation.json").read_text(encoding="utf-8"))
        plot_samples(run_dir, arrays)
        plot_nearest(run_dir, arrays, report)
        plot_distances(run_dir, arrays, report)
        plot_latent(run_dir, arrays, report)
        plot_tissue(run_dir, arrays)
        plot_quiz(run_dir, arrays)
    else:
        print(f"no {evaluation / 'arrays.npz'} yet - drew only the training figures")
    print(f"figures written to {run_dir}")


if __name__ == "__main__":
    main()
