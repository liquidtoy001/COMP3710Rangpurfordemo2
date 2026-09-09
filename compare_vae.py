"""Compare VAE runs across the beta sweep (Part 4, Task 1).

A single run's figures show what one model does. The question beta answers is a
trade-off, and a trade-off needs the runs side by side:

    python compare_vae.py runs/vae runs/vae_l32_beta10 runs/vae_l32_beta30
    python compare_vae.py runs/vae_latent2 runs/vae_l2_beta50 runs/vae_l2_beta150

Prints a table and writes ``vae_beta_comparison.png`` in the working directory.

What to look for. Raising beta should:

* **increase** the reconstruction loss - the model is being asked to spend
  capacity on matching the prior instead of on detail;
* **decrease** the raw KL divergence, which is what beta asks for. Note that
  the KL's *share of the optimised loss* moves the other way, since that share
  is beta*KL. Reading one and calling it the other makes the sweep look
  self-contradictory;
* **decrease** ``max |mu|`` towards the unit scale of N(0, 1) - the codes being
  pulled back into the region the prior actually covers;
* **stop the noise in the prior samples** - because ``z ~ N(0, I)`` now lands
  where the encoder has been.

The right beta is the smallest one that achieves the third without ruining the
first. That judgement is the point of the sweep, and it cannot be made from a
loss number alone: the total loss is not comparable across betas, since beta
appears in it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREY = dict(cmap="gray", vmin=0.0, vmax=1.0)


def load(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    latents = np.load(run_dir / "latents.npy")
    samples = np.load(run_dir / "samples.npy")
    final = summary["history"][-1]

    return {
        "name": run_dir.name,
        "dir": run_dir,
        "latent_dim": summary["args"]["latent_dim"],
        "beta": summary["args"]["beta"],
        # The reconstruction term is the only loss component comparable across
        # betas: the total has beta baked into it, so a lower total at a lower
        # beta means nothing.
        "reconstruction": final["val_reconstruction"],
        "kl": final["val_kl"],
        # Two different quantities, easily confused. The raw KL says how far
        # the posterior sits from the prior - it should FALL as beta rises,
        # because that is what beta is asking the model to do. The weighted
        # share says how much of the optimised loss the KL accounts for.
        "kl_raw": final["val_kl"],
        "kl_weighted_share": (
            summary["args"]["beta"] * final["val_kl"]
            / (final["val_reconstruction"] + summary["args"]["beta"] * final["val_kl"])
        ),
        "mean_abs_mu": float(np.abs(latents).mean()),
        "max_abs_mu": float(np.abs(latents).max()),
        "sample_spread": float(samples.reshape(samples.shape[0], -1).std(axis=0).mean()),
        "samples": samples,
    }


def print_table(runs: list[dict]) -> None:
    header = (
        f"{'run':<24}{'latent':>7}{'beta':>8}{'recon':>10}"
        f"{'KL':>9}{'b*KL/loss':>11}{'mean|mu|':>10}{'max|mu|':>9}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        print(
            f"{run['name']:<24}{run['latent_dim']:>7}{run['beta']:>8.4g}"
            f"{run['reconstruction']:>10.1f}{run['kl_raw']:>9.2f}"
            f"{run['kl_weighted_share']:>10.2%}{run['mean_abs_mu']:>10.2f}{run['max_abs_mu']:>9.2f}"
        )
    print("-" * len(header))
    print("recon is the validation reconstruction term - the only loss component")
    print("comparable across betas, since beta itself appears in the total.")
    print("KL is the raw divergence in nats: it should FALL as beta rises, because")
    print("that is precisely what beta asks for. b*KL/loss is the share of the")
    print("optimised loss the KL accounts for, which rises instead. Confusing the")
    print("two makes the sweep look self-contradictory.")
    print("The prior is N(0, 1), so max|mu| far above ~3 means the codes sit")
    print("outside the region z ~ N(0, I) samples from.")


def plot_comparison(runs: list[dict], path: Path) -> None:
    runs = sorted(runs, key=lambda r: r["beta"])
    betas = [run["beta"] for run in runs]

    fig = plt.figure(figsize=(15, 4 + 2.2 * len(runs)))
    grid = fig.add_gridspec(len(runs) + 1, 4, height_ratios=[1.4] + [1] * len(runs))

    # Row 0: the trade-off, as three curves against beta.
    axis = fig.add_subplot(grid[0, 0])
    axis.plot(betas, [r["reconstruction"] for r in runs], "o-")
    axis.set_title("Reconstruction loss")
    axis.set_xlabel("beta")
    axis.set_xscale("symlog")
    axis.grid(True, alpha=0.3)

    axis = fig.add_subplot(grid[0, 1])
    axis.plot(betas, [r["max_abs_mu"] for r in runs], "o-", color="tab:orange")
    axis.axhline(3.0, color="r", linestyle="--", alpha=0.7, label="~3 = edge of the prior")
    axis.set_title("max |mu|")
    axis.set_xlabel("beta")
    axis.set_xscale("symlog")
    axis.legend()
    axis.grid(True, alpha=0.3)

    axis = fig.add_subplot(grid[0, 2])
    axis.plot(betas, [r["kl_raw"] for r in runs], "o-", color="tab:green")
    for run in runs:
        axis.annotate(
            f"{run['kl_weighted_share']:.1%} of loss",
            (run["beta"], run["kl_raw"]),
            textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center",
        )
    axis.set_title("KL divergence (nats)")
    axis.set_xlabel("beta")
    axis.set_xscale("symlog")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)

    axis = fig.add_subplot(grid[0, 3])
    axis.plot(betas, [r["sample_spread"] for r in runs], "o-", color="tab:purple")
    axis.set_title("Spread across prior samples")
    axis.set_xlabel("beta")
    axis.set_xscale("symlog")
    axis.grid(True, alpha=0.3)

    # One row per run: four brains decoded from the prior. This is the panel
    # that decides the question - numbers do not show whether a sample is a
    # brain or noise.
    for row, run in enumerate(runs, start=1):
        for column in range(4):
            axis = fig.add_subplot(grid[row, column])
            axis.imshow(run["samples"][column, 0], **GREY)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(
                    f"latent {run['latent_dim']}\nbeta {run['beta']:g}", fontsize=10
                )

    fig.suptitle(
        "VAE beta sweep - the trade-off, and what z ~ N(0, I) actually decodes to",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="VAE run directories to compare")
    parser.add_argument("--out", default="vae_beta_comparison.png")
    args = parser.parse_args()

    runs = [load(Path(directory)) for directory in args.run_dirs]
    runs.sort(key=lambda r: (r["latent_dim"], r["beta"]))

    print_table(runs)
    plot_comparison(runs, Path(args.out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
