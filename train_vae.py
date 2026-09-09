"""Train the OASIS VAE (Part 4, Task 1).

Full marks require the model to be trained *and* the resulting manifold
visualised. The plotting itself happens locally, in ``plot_vae.py``, because
matplotlib and UMAP are not installed on the cluster - so this script saves
everything those figures need as ``.npy`` arrays:

    latents.npy         mu for every test image (the manifold, before reduction)
    reconstructions.npy a fixed batch of test images beside their rebuilds
    samples.npy         images decoded from z ~ N(0, I), i.e. novel brains
    manifold.npz        only when --latent-dim 2: a decoded sweep of the plane,
                        with the z1/z2 coordinates it covers

Smoke test first:

    python train_vae.py --epochs 1 --limit 64 --limit-batches 3 --out-dir runs/vae_smoke

Full run:

    sbatch slurm/train_vae.sh
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

from oasis import DEFAULT_ROOT, build_loaders
from vae import VAE, count_parameters, vae_loss

# How many test images to show as reconstruction pairs, and how many novel
# brains to sample from the prior.
NUM_RECONSTRUCTIONS = 8
NUM_SAMPLES = 16
MANIFOLD_STEPS = 20
# The grid is swept across this central percentile range of the codes the
# encoder actually produced, rather than a fixed +/-2.5. A VAE whose KL term
# is weak relative to its reconstruction term - which is the normal case when
# reconstruction is summed over 65,536 pixels - puts its codes nowhere near
# the unit scale of the prior. The first run here spanned z1 in [-7, 25], so a
# +/-2.5 sweep covered a small blob at the centre and every decoded tile came
# out looking identical.
MANIFOLD_PERCENTILES = (2.0, 98.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronise(device: torch.device) -> None:
    """CUDA is asynchronous; wait for queued work before reading the clock."""
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


def run_epoch(
    model: VAE,
    loader,
    device: torch.device,
    beta: float,
    optimiser: torch.optim.Optimizer | None = None,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """One pass. Training when an optimiser is given, evaluation otherwise."""
    training = optimiser is not None
    model.train(training)

    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    seen = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, images in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            images = images.to(device, non_blocking=True)

            reconstruction, mu, logvar = model(images)
            loss, reconstruction_loss, kl = vae_loss(
                reconstruction, images, mu, logvar, beta=beta
            )

            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            batch = images.size(0)
            totals["loss"] += loss.item() * batch
            totals["reconstruction"] += reconstruction_loss.item() * batch
            totals["kl"] += kl.item() * batch
            seen += batch

    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def save_visualisation_data(
    model: VAE, test_loader, device: torch.device, out_dir: Path
) -> None:
    """Save the arrays plot_vae.py turns into the manifold figures."""
    model.eval()

    # 1. Latent codes for the whole test set. This *is* the manifold; UMAP or
    #    PCA only projects it down to something plottable.
    latents = []
    for images in test_loader:
        mu, _ = model.encode(images.to(device, non_blocking=True))
        latents.append(mu.cpu().numpy())
    latents_array = np.concatenate(latents, axis=0)
    np.save(out_dir / "latents.npy", latents_array)

    # 2. Reconstructions beside their inputs - the qualitative evidence that
    #    the model learnt anything at all.
    images = next(iter(test_loader))[:NUM_RECONSTRUCTIONS].to(device)
    reconstruction, _, _ = model(images)
    np.save(
        out_dir / "reconstructions.npy",
        np.stack([images.cpu().numpy(), reconstruction.cpu().numpy()]),
    )

    # 3. Novel brains, decoded from the prior. A VAE that has collapsed
    #    produces the same blurry average here for every draw, so this is the
    #    quickest read on whether the latent space is actually populated.
    z = torch.randn(NUM_SAMPLES, model.latent_dim, device=device)
    np.save(out_dir / "samples.npy", model.decode(z).cpu().numpy())

    # 4. With a two-dimensional latent the manifold can be shown directly:
    #    sweep a grid over the plane and decode every point. No dimensionality
    #    reduction, and no approximation.
    if model.latent_dim == 2:
        low, high = np.percentile(latents_array, MANIFOLD_PERCENTILES, axis=0)
        z1 = torch.linspace(float(low[0]), float(high[0]), MANIFOLD_STEPS)
        z2 = torch.linspace(float(low[1]), float(high[1]), MANIFOLD_STEPS)
        grid = torch.stack(
            [torch.stack([x, y]) for y in z2 for x in z1]
        ).to(device)
        decoded = []
        for start in range(0, grid.size(0), 32):
            decoded.append(model.decode(grid[start:start + 32]).cpu().numpy())
        # Axes are saved alongside the tiles so the figure can be labelled with
        # the coordinates it actually covers.
        np.savez(
            out_dir / "manifold.npz",
            grid=np.concatenate(decoded, axis=0),
            z1=z1.numpy(),
            z2=z2.numpy(),
        )

    print(f"  latents        : {latents_array.shape} -> latents.npy")
    print(f"  reconstructions: {images.size(0)} pairs -> reconstructions.npy")
    print(f"  prior samples  : {NUM_SAMPLES} -> samples.npy")
    if model.latent_dim == 2:
        print(f"  manifold grid  : {MANIFOLD_STEPS}x{MANIFOLD_STEPS} over "
              f"z1 [{low[0]:.2f}, {high[0]:.2f}] z2 [{low[1]:.2f}, {high[1]:.2f}]"
              " -> manifold.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default="runs/vae")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--beta", type=float, default=1.0, help="weight on the KL term"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit", type=int, default=None, help="use only N images per split (smoke tests)"
    )
    parser.add_argument(
        "--limit-batches", type=int, default=None, help="stop each epoch after N batches"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"job id        : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"host          : {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"device        : {describe_device(device)}")
    print(f"torch         : {torch.__version__}")
    print(f"latent dim    : {args.latent_dim}   beta: {args.beta}")
    print(f"epochs        : {args.epochs}   batch size: {args.batch_size}   lr: {args.lr}")
    print("=" * 68)

    train_loader, validate_loader, test_loader = build_loaders(
        args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        with_masks=False,
        limit=args.limit,
    )
    print(f"train {len(train_loader.dataset):,} | "
          f"validate {len(validate_loader.dataset):,} | "
          f"test {len(test_loader.dataset):,} images")

    model = VAE(latent_dim=args.latent_dim).to(device)
    print(f"parameters    : {count_parameters(model):,}")

    # Adam rather than SGD: the usual choice for VAEs, and the loss surface
    # here is nothing like a classifier's.
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    csv_path = out_dir / "history.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "epoch",
        "train_loss", "train_reconstruction", "train_kl",
        "val_loss", "val_reconstruction", "val_kl",
        "epoch_seconds",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    history = []
    best_val_loss = float("inf")
    synchronise(device)
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        synchronise(device)
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model, train_loader, device, args.beta, optimiser, args.limit_batches
        )
        synchronise(device)
        epoch_seconds = time.perf_counter() - epoch_start

        val_metrics = run_epoch(
            model, validate_loader, device, args.beta, None, args.limit_batches
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reconstruction": train_metrics["reconstruction"],
            "train_kl": train_metrics["kl"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction": val_metrics["reconstruction"],
            "val_kl": val_metrics["kl"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)
        writer.writerow(record)
        csv_file.flush()

        marker = ""
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            marker = "  <- best"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "latent_dim": args.latent_dim,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train {train_metrics['loss']:9.1f} "
            f"(rec {train_metrics['reconstruction']:8.1f} kl {train_metrics['kl']:6.1f})  "
            f"val {val_metrics['loss']:9.1f}  {epoch_seconds:6.1f}s{marker}",
            flush=True,
        )

    total_seconds = time.perf_counter() - run_start
    csv_file.close()

    # Visualise from the best checkpoint, not the last epoch's weights.
    print("\nsaving visualisation data from the best checkpoint")
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    save_visualisation_data(model, test_loader, device, out_dir)

    summary = {
        "best_val_loss": best_val_loss,
        "total_seconds": total_seconds,
        "device": describe_device(device),
        "args": vars(args),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 68)
    print(f"best validation loss : {best_val_loss:.1f} (epoch {checkpoint['epoch']})")
    print(f"total wall clock     : {total_seconds:.1f}s ({total_seconds / 60:.1f} min)")
    print(f"checkpoint           : {out_dir / 'best.pt'}")
    print(f"metrics              : {out_dir / 'metrics.json'}")
    print("\nCopy this directory back and run:  python plot_vae.py " + str(out_dir))


if __name__ == "__main__":
    main()
