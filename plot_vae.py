"""Turn a VAE run into the figures the demonstration needs (Part 4, Task 1).

Run locally, after copying a run directory back. The cluster has neither
matplotlib nor UMAP, and does not need them: ``train_vae.py`` saved the arrays.

    python plot_vae.py runs/vae
    python plot_vae.py runs/vae_latent2

Produces, in the run directory:

    curves.png          loss, split into its reconstruction and KL terms
    reconstructions.png test images above their rebuilds
    samples.png         brains decoded from z ~ N(0, I)
    latent.png          the manifold - a decoded grid for a 2D latent,
                        a PCA projection of the codes otherwise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display over SSH, and none needed to write PNGs
import matplotlib.pyplot as plt

GREY = dict(cmap="gray", vmin=0.0, vmax=1.0)


def plot_curves(summary: dict, path: Path) -> None:
    """Total loss, and the two terms it trades off against each other."""
    history = summary["history"]
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="validate")
    best = min(history, key=lambda r: r["val_loss"])
    axes[0].axvline(best["epoch"], color="grey", linestyle=":", label=f"best (epoch {best['epoch']})")
    axes[0].set_title("Total loss (reconstruction + KL)")

    axes[1].plot(epochs, [r["train_reconstruction"] for r in history], label="train")
    axes[1].plot(epochs, [r["val_reconstruction"] for r in history], label="validate")
    axes[1].set_title("Reconstruction term")

    axes[2].plot(epochs, [r["train_kl"] for r in history], label="train")
    axes[2].plot(epochs, [r["val_kl"] for r in history], label="validate")
    axes[2].set_title("KL term")

    for axis in axes:
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(True, alpha=0.3)

    fig.suptitle(
        f"VAE, latent dim {summary['args']['latent_dim']}, beta {summary['args']['beta']}"
        f"  -  best validation loss {summary['best_val_loss']:.1f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reconstructions(run_dir: Path, path: Path) -> None:
    """Inputs on the top row, the model's rebuilds directly beneath."""
    pairs = np.load(run_dir / "reconstructions.npy")  # (2, N, 1, H, W)
    originals, rebuilt = pairs[0], pairs[1]
    count = originals.shape[0]

    fig, axes = plt.subplots(2, count, figsize=(1.6 * count, 3.6))
    for index in range(count):
        axes[0][index].imshow(originals[index, 0], **GREY)
        axes[1][index].imshow(rebuilt[index, 0], **GREY)
        for row in (0, 1):
            axes[row][index].set_xticks([])
            axes[row][index].set_yticks([])
    axes[0][0].set_ylabel("input", fontsize=11)
    axes[1][0].set_ylabel("rebuilt", fontsize=11)

    fig.suptitle("Test images and their reconstructions")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_samples(run_dir: Path, path: Path) -> None:
    """Brains decoded from the prior - images the model invented."""
    samples = np.load(run_dir / "samples.npy")
    count = samples.shape[0]
    columns = 8
    rows = (count + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.7 * rows))
    for index, axis in enumerate(np.array(axes).ravel()):
        if index < count:
            axis.imshow(samples[index, 0], **GREY)
        axis.set_xticks([])
        axis.set_yticks([])

    # A collapsed VAE emits the same average for every draw, so the spread
    # across these samples is the quickest read on whether the latent space is
    # actually populated.
    spread = samples.reshape(count, -1).std(axis=0).mean()
    fig.suptitle(
        f"Decoded from z ~ N(0, I)  -  mean per-pixel spread across samples {spread:.4f}"
        + ("   (near zero would mean posterior collapse)" if spread < 0.01 else "")
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pca_project(codes: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Project latent codes to 2D with PCA, using the SVD.

    Implemented with numpy rather than a library so the script has no
    dependency beyond matplotlib - and it is the same eigen-decomposition as
    Part 2's eigenfaces, applied to latent codes instead of pixels.

    Returns the projection and the explained variance ratio per component.
    """
    centred = codes - codes.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)
    explained = singular_values ** 2 / (singular_values ** 2).sum()
    return centred @ right_vectors[:components].T, explained


def plot_latent(run_dir: Path, summary: dict, path: Path) -> None:
    """The manifold itself.

    With a two-dimensional latent this is a decoded sweep of the plane - the
    manifold directly, with no reduction in between. With more dimensions the
    codes are projected to 2D with PCA, and the per-dimension spread is shown
    alongside, because that is what reveals posterior collapse: dimensions the
    encoder has stopped using carry almost no variance.
    """
    latents = np.load(run_dir / "latents.npy")
    latent_dim = latents.shape[1]

    # Newer runs save the grid with the coordinates it covers; older ones saved
    # only the tiles, swept over a fixed +/-2.5 that often missed the codes
    # entirely.
    manifold_path = run_dir / "manifold.npz"
    legacy_path = run_dir / "manifold_grid.npy"
    z1 = z2 = None
    grid = None
    if manifold_path.exists():
        data = np.load(manifold_path)
        grid, z1, z2 = data["grid"], data["z1"], data["z2"]
    elif legacy_path.exists():
        grid = np.load(legacy_path)
        z1 = z2 = np.linspace(-2.5, 2.5, int(round(np.sqrt(grid.shape[0]))))

    if latent_dim == 2 and grid is not None:
        steps = int(round(np.sqrt(grid.shape[0])))
        tile = grid.shape[-1]

        # Stitch the decoded tiles into one image.
        canvas = np.zeros((steps * tile, steps * tile), dtype=np.float32)
        for index in range(grid.shape[0]):
            row, column = divmod(index, steps)
            canvas[row * tile:(row + 1) * tile, column * tile:(column + 1) * tile] = grid[index, 0]

        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        # extent labels the image with latent coordinates rather than pixels.
        axes[0].imshow(
            canvas, origin="lower",
            extent=(float(z1[0]), float(z1[-1]), float(z2[0]), float(z2[-1])),
            aspect="auto", **GREY,
        )
        axes[0].set_title("Manifold: the latent plane, decoded")
        axes[0].set_xlabel("z1")
        axes[0].set_ylabel("z2")

        axes[1].scatter(latents[:, 0], latents[:, 1], s=8, alpha=0.5, label="test images")
        # Show which part of the space the grid actually covered - the first run
        # swept a region far smaller than the codes occupied.
        axes[1].add_patch(
            plt.Rectangle(
                (float(z1[0]), float(z2[0])),
                float(z1[-1] - z1[0]), float(z2[-1] - z2[0]),
                fill=False, edgecolor="r", linestyle="--", label="grid extent",
            )
        )
        axes[1].scatter([0], [0], marker="+", s=200, color="k", label="prior mean")
        axes[1].set_title(f"Where the {len(latents)} test images land")
        axes[1].set_xlabel("z1")
        axes[1].set_ylabel("z2")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        projected, explained = pca_project(latents)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].scatter(projected[:, 0], projected[:, 1], s=8, alpha=0.5)
        axes[0].set_title(
            f"Latent codes, PCA to 2D "
            f"({explained[:2].sum():.1%} of variance in these two components)"
        )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].grid(True, alpha=0.3)

        spread = latents.std(axis=0)
        order = np.argsort(spread)[::-1]
        # A dimension the encoder has given up on sits at the prior, so its mu
        # barely varies. Counting those says how much of the latent space the
        # model is really using.
        alive = int((spread > 0.1).sum())
        axes[1].bar(range(latent_dim), spread[order], color="tab:purple", alpha=0.85)
        axes[1].axhline(0.1, color="r", linestyle="--", label="0.1 - effectively unused below this")
        axes[1].set_title(f"Spread of each latent dimension: {alive} of {latent_dim} in use")
        axes[1].set_xlabel("latent dimension (sorted)")
        axes[1].set_ylabel("standard deviation of mu")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarise(summary: dict, run_dir: Path) -> None:
    """Print the numbers worth quoting to the demonstrator."""
    latents = np.load(run_dir / "latents.npy")
    spread = latents.std(axis=0)

    print("-" * 62)
    print(f"latent dimensions   : {summary['args']['latent_dim']}")
    print(f"beta                : {summary['args']['beta']}")
    print(f"best validation loss: {summary['best_val_loss']:.1f} "
          f"(epoch {min(summary['history'], key=lambda r: r['val_loss'])['epoch']}"
          f" of {len(summary['history'])})")
    print(f"training time       : {summary['total_seconds']:.1f}s "
          f"({summary['total_seconds'] / 60:.1f} min) on {summary['device']}")
    print(f"dimensions in use   : {int((spread > 0.1).sum())} of {latents.shape[1]}"
          "   (std of mu above 0.1)")

    # A VAE's aggregate posterior should look like N(0, I). When the
    # reconstruction term is summed over 65,536 pixels and the KL over a handful
    # of latent dimensions, the KL is negligible unless beta compensates, and
    # the codes drift far outside the prior. Sampling z ~ N(0, I) then lands in
    # regions the encoder never visited, and the decoder returns nonsense.
    history = summary["history"]
    final = history[-1]
    kl_share = final["val_kl"] / (final["val_reconstruction"] + final["val_kl"])
    print(f"KL share of loss    : {kl_share:.3%}")
    print(f"mean |mu|           : {np.abs(latents).mean():.2f}"
          f"   max |mu|: {np.abs(latents).max():.2f}   (prior is N(0, 1))")
    if np.abs(latents).max() > 5:
        print("  -> codes sit far outside the prior; raise --beta to pull them in")
    print("-" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="a VAE run directory containing metrics.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    summarise(summary, run_dir)

    for name, function in (
        ("curves.png", lambda p: plot_curves(summary, p)),
        ("reconstructions.png", lambda p: plot_reconstructions(run_dir, p)),
        ("samples.png", lambda p: plot_samples(run_dir, p)),
        ("latent.png", lambda p: plot_latent(run_dir, summary, p)),
    ):
        path = run_dir / name
        function(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
