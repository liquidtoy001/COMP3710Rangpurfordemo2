"""Evidence that the GAN's brains are realistic and not collapsed (Part 4, Task 3).

The lab sheet sets the bar for full marks: generated images must "look like
unique brains", and mode collapse "need[s] to be fully resolved". Looking at a
grid of samples cannot settle either - a grid can be chosen, and a generator
that copies its training set looks perfect. So this script measures, and every
measurement is paired with the same measurement on **real slices the GAN never
saw** (the test split), which is what the generated figure should resemble:

1. **Not copies.** For each generated slice, the nearest of the 9,664 training
   slices. Compared with the same distance for real test slices: a new brain
   from a new person sits about as far from the training set as a test slice
   does, while a memorised one sits much closer.
2. **Not collapsed.** For each generated slice, its nearest other generated
   slice, against the same for real slices. A collapsed generator draws
   near-duplicates, so its nearest-neighbour distances fall far below the real
   ones. This detects collapse and nothing else: noise is varied too, and a
   generator trained for 40 steps scored about 0.9 in testing. Realism is what
   measurements 3 and 4 are for.
3. **Covers the variety of real brains.** Real and generated slices are encoded
   with the Task 1 VAEs. In the 32-dimensional latent space, precision and
   recall in the sense of Kynkaanniemi et al. (2019): precision is the fraction
   of generated slices that land inside the region real slices occupy, recall the
   fraction of real slices that land inside the region generated ones occupy.
   Low precision means unrealistic slices; low recall is mode collapse measured
   directly. The two-dimensional latent is saved for a scatter plot.
4. **Anatomically plausible.** Real and generated slices are segmented with the
   Task 2 UNet, and the share of each tissue class is compared. A generator can
   paint brain-like texture with implausible anatomy; the tissue proportions
   catch that. The UNet's confidence is recorded too: it is trained only on real
   brains, so confident segmentations of generated ones are evidence they look
   like the real thing to a network that knows the anatomy.

Each figure for generated slices is shown beside the same figure for a random
sample of **real training slices**, which is what a perfect generator would
produce: it is trained to imitate the training distribution, so that sample is
the ceiling it can reach.

The VAE and UNet were trained here, on OASIS, not pre-trained elsewhere. The
usual GAN score, FID, needs an ImageNet-trained Inception network, which is a
pre-trained model the lab sheet does not allow without approval.

Fairness of the comparisons. The GAN may work below 256x256, while the VAEs and
UNet take 256x256 input. Generated slices are upsampled to 256; real slices are
first downsampled to the GAN's resolution and upsampled back the same way, so
both have lost exactly the same detail and the comparison is of content, not of
resolution. All distances are root-mean-square pixel differences in [0, 1], so
they read the same at every resolution.

Runs on the cluster, without matplotlib; ``plot_gan.py`` draws the figures.

    python evaluate_gan.py runs/gan128
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gan import EMA, Generator, recalibrate_batchnorm
from oasis import DEFAULT_ROOT, IMAGE_SIZE, NUM_CLASSES
from train_gan import describe_device, load_split, resolve_device
from unet import UNet
from vae import VAE

PRECISION_RECALL_K = 3
NEAREST_PAIRS = 16
GRID_IMAGES = 64


def load_generator(path: Path, device: torch.device) -> tuple[Generator, dict]:
    """The averaged generator, from final.pt or, for an unfinished run, last.pt."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "ema" in checkpoint:  # last.pt: rebuild the averaged generator and calibrate it
        args = checkpoint["args"]
        generator = Generator(args["resolution"], args["z_dim"], args["width"]).to(device)
        shadow = EMA(generator).shadow
        shadow.load_state_dict(checkpoint["ema"])
        generator = recalibrate_batchnorm(shadow, device)
        return generator, {"step": checkpoint["step"], "args": args, "source": "last.pt (unfinished run)"}
    generator = Generator(checkpoint["resolution"], checkpoint["z_dim"], checkpoint["width"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, {"step": checkpoint["step"], "args": checkpoint["args"], "source": "final.pt"}


@torch.no_grad()
def generate(generator: Generator, count: int, device: torch.device, seed: int, batch: int = 128) -> torch.Tensor:
    """``count`` generated slices in [0, 1], from a fixed seed so the evaluation repeats exactly."""
    noise = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for start in range(0, count, batch):
        z = torch.randn(min(batch, count - start), generator.z_dim, generator=noise).to(device)
        out.append((generator(z).clamp(-1, 1) + 1) / 2)
    return torch.cat(out)


@torch.no_grad()
def nearest(query: torch.Tensor, reference: torch.Tensor, exclude_self: bool = False,
            chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS pixel distance from each query image to its nearest reference image, and that image's index."""
    reference_flat = reference.flatten(1).float()
    pixels = reference_flat.size(1)
    distances, indices = [], []
    for start in range(0, len(query), chunk):
        block = torch.cdist(query[start:start + chunk].flatten(1).float(), reference_flat)
        if exclude_self:
            rows = torch.arange(block.size(0), device=block.device)
            block[rows, rows + start] = math.inf
        value, index = block.min(dim=1)
        distances.append(value / math.sqrt(pixels))
        indices.append(index)
    return torch.cat(distances), torch.cat(indices)


def to_256(images: torch.Tensor) -> torch.Tensor:
    """Upsample to the VAE and UNet input size; a no-op at 256."""
    if images.shape[-1] == IMAGE_SIZE:
        return images
    return F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).clamp(0, 1)


@torch.no_grad()
def encode(vae: VAE, images: torch.Tensor, batch: int = 64) -> torch.Tensor:
    return torch.cat([vae.encode(to_256(images[s:s + batch]))[0] for s in range(0, len(images), batch)])


@torch.no_grad()
def knn_radii(points: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest other point: the radius of its ball."""
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(math.inf)
    return distances.kthvalue(k, dim=1).values


@torch.no_grad()
def coverage(query: torch.Tensor, reference: torch.Tensor, radii: torch.Tensor) -> float:
    """Fraction of query points inside at least one reference point's k-NN ball."""
    return (torch.cdist(query, reference) <= radii.unsqueeze(0)).any(dim=1).float().mean().item()


def precision_recall(real: torch.Tensor, generated: torch.Tensor, k: int) -> dict[str, float]:
    return {
        "precision": coverage(generated, real, knn_radii(real, k)),
        "recall": coverage(real, generated, knn_radii(generated, k)),
    }


@torch.no_grad()
def tissue_statistics(unet: UNet, images: torch.Tensor, batch: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Per image: the share of each class, and the UNet's mean confidence on brain pixels."""
    fractions, confidences = [], []
    for start in range(0, len(images), batch):
        probabilities = torch.softmax(unet(to_256(images[start:start + batch])), dim=1)
        confidence, labels = probabilities.max(dim=1)
        fractions.append(torch.stack([(labels == c).float().mean(dim=(1, 2)) for c in range(NUM_CLASSES)], dim=1))
        brain = labels != 0
        brain_pixels = brain.sum(dim=(1, 2)).clamp(min=1)
        confidences.append((confidence * brain).sum(dim=(1, 2)) / brain_pixels)
    return torch.cat(fractions), torch.cat(confidences)


def wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Earth mover's distance between two equal-sized samples of one number: mean gap of the sorted values."""
    return (a.sort().values - b.sort().values).abs().mean().item()


def summarise(values: torch.Tensor) -> dict[str, float]:
    values = values.double().cpu()
    return {
        "median": values.median().item(),
        "p05": values.quantile(0.05).item(),
        "p95": values.quantile(0.95).item(),
    }


def to_uint8(images: torch.Tensor) -> np.ndarray:
    return (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default=None, help="default: final.pt, else last.pt, in run_dir")
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--vae2", default="runs/vae_l2_beta50/best.pt")
    parser.add_argument("--vae32", default="runs/vae_l32_beta10/best.pt")
    parser.add_argument("--unet", default="runs/unet/best.pt")
    parser.add_argument("--num-generated", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="N real slices per split (tests only)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        run_dir / "final.pt" if (run_dir / "final.pt").exists() else run_dir / "last.pt")
    if not checkpoint_path.exists():
        raise SystemExit(f"no final.pt or last.pt in {run_dir}")
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator, info = load_generator(checkpoint_path, device)
    resolution = generator.resolution

    print("=" * 78)
    print("COMP3710 demonstration 2 - Task 3, GAN evaluation")
    print(f"job id    : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"device    : {describe_device(device)}")
    print(f"generator : {checkpoint_path} ({info['source']}), step {info['step']:,}, {resolution}x{resolution}")
    print("=" * 78)

    train = load_split(args.data_root, "train", resolution, device, args.workers, args.limit).float() / 255
    test = load_split(args.data_root, "test", resolution, device, args.workers, args.limit).float() / 255
    generated = generate(generator, args.num_generated, device, args.seed)
    # Comparisons between sets use equal sizes, since nearest-neighbour distances
    # shrink as a set grows. The test split is the smallest, so it sets the size.
    size = min(len(test), len(generated))
    generated_matched = generated[:size]
    test_matched = test[:size]
    # The reference for "what a perfect generator would score": real training
    # slices drawn at random. The generator is trained to imitate the training
    # distribution, so a random sample of that distribution is exactly what an
    # ideal generator would produce - except that these slices are not new.
    # Validation slices would be the wrong reference: they come from other
    # people, and so differ from the test slices in ways a generator of the
    # training distribution is not expected to reproduce either.
    draw = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    reference = train[torch.randperm(len(train), generator=draw)[:size].to(device)]
    print(f"real: {len(train):,} train, {len(test):,} test | generated: {len(generated):,} | "
          f"matched set size: {size} (a random {size} training slices serve as the reference)")

    report: dict = {
        "checkpoint": str(checkpoint_path),
        "source": info["source"],
        "step": info["step"],
        "resolution": resolution,
        "num_generated": len(generated),
        "matched_size": size,
        "seed": args.seed,
        "device": describe_device(device),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    arrays: dict[str, np.ndarray] = {}

    # ---- 1. Not copies -------------------------------------------------------
    generated_to_train, generated_nearest_index = nearest(generated, train)
    test_to_train, _ = nearest(test, train)
    threshold = test_to_train.double().quantile(0.05).item()
    report["novelty"] = {
        "generated_to_train": summarise(generated_to_train),
        "test_to_train": summarise(test_to_train),
        "copy_threshold": threshold,
        "generated_closer_than_threshold": (generated_to_train < threshold).float().mean().item(),
        "closest_generated": generated_to_train.min().item(),
        "closest_test": test_to_train.min().item(),
    }
    arrays["generated_to_train"] = generated_to_train.cpu().numpy()
    arrays["test_to_train"] = test_to_train.cpu().numpy()
    # The pairs shown are the generated slices *closest* to the training set:
    # the most copy-like ones, not a flattering selection.
    closest = generated_to_train.argsort()[:NEAREST_PAIRS]
    arrays["pairs_generated"] = to_uint8(generated[closest])
    arrays["pairs_train"] = to_uint8(train[generated_nearest_index[closest]])
    arrays["pairs_distance"] = generated_to_train[closest].cpu().numpy()

    novelty = report["novelty"]
    print("\n1. Not copies - RMS distance to the nearest training slice")
    print(f"   generated slices : median {novelty['generated_to_train']['median']:.4f}   "
          f"closest {novelty['closest_generated']:.4f}")
    print(f"   real test slices : median {novelty['test_to_train']['median']:.4f}   "
          f"closest {novelty['closest_test']:.4f}   (new people, never trained on)")
    print(f"   generated slices nearer the training set than 95% of test slices: "
          f"{novelty['generated_closer_than_threshold']:.1%}   (5% of test slices, by definition)")
    print("   A generator of the training distribution can sit a little nearer it than new people do;")
    print("   copies show as distances near zero. nearest_neighbours.png shows the closest pairs.")

    # ---- 2. Not collapsed ----------------------------------------------------
    generated_to_generated, _ = nearest(generated_matched, generated_matched, exclude_self=True)
    reference_to_reference, _ = nearest(reference, reference, exclude_self=True)
    report["diversity"] = {
        "generated_to_generated": summarise(generated_to_generated),
        "reference_to_reference": summarise(reference_to_reference),
        "ratio_of_medians": (generated_to_generated.median() / reference_to_reference.median()).item(),
    }
    arrays["generated_to_generated"] = generated_to_generated.cpu().numpy()
    arrays["reference_to_reference"] = reference_to_reference.cpu().numpy()

    diversity = report["diversity"]
    print("\n2. Not collapsed - RMS distance to the nearest other slice in the same set")
    print(f"   generated                 : median {diversity['generated_to_generated']['median']:.4f}")
    print(f"   random real training slices: median {diversity['reference_to_reference']['median']:.4f}")
    print(f"   ratio                     : {diversity['ratio_of_medians']:.2f}   "
          "(near 1 is as varied as real data; collapse drives it towards 0)")
    print("   This detects collapse only. Noise is varied too - an untrained generator scores near 1 -")
    print("   so realism is judged by precision (3) and anatomy (4), not by this ratio.")

    # ---- 3. Coverage in the VAE latent spaces -------------------------------
    for key, path in (("vae32", args.vae32), ("vae2", args.vae2)):
        if not Path(path).exists():
            print(f"\n   {path} not found - skipping the {key} comparison")
            continue
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        vae = VAE(latent_dim=checkpoint["latent_dim"]).to(device)
        vae.load_state_dict(checkpoint["model_state"])
        vae.eval()
        latent_test = encode(vae, test_matched)
        latent_generated = encode(vae, generated_matched)
        latent_reference = encode(vae, reference)
        if key == "vae2":
            arrays["latent2_test"] = latent_test.cpu().numpy()
            arrays["latent2_generated"] = latent_generated.cpu().numpy()
            arrays["latent2_reference"] = latent_reference.cpu().numpy()
            report["latent2_checkpoint"] = path
            print(f"\n   two-dimensional latents saved for the scatter plot ({path})")
            continue
        report["precision_recall"] = {
            "checkpoint": path,
            "k": PRECISION_RECALL_K,
            "generated_vs_test": precision_recall(latent_test, latent_generated, PRECISION_RECALL_K),
            "reference_vs_test": precision_recall(latent_test, latent_reference, PRECISION_RECALL_K),
        }
        pr = report["precision_recall"]
        print(f"\n3. Coverage - precision and recall against test slices, "
              f"in the {checkpoint['latent_dim']}-dimensional VAE latent space")
        print(f"   generated slices           : precision {pr['generated_vs_test']['precision']:.3f}   "
              f"recall {pr['generated_vs_test']['recall']:.3f}")
        print(f"   random real training slices: precision {pr['reference_vs_test']['precision']:.3f}   "
              f"recall {pr['reference_vs_test']['recall']:.3f}   (what a perfect generator would score)")

    # ---- 4. Tissue proportions from the UNet --------------------------------
    if Path(args.unet).exists():
        checkpoint = torch.load(args.unet, map_location=device, weights_only=False)
        unet = UNet(num_classes=checkpoint["num_classes"], base_channels=checkpoint["base_channels"]).to(device)
        unet.load_state_dict(checkpoint["model_state"])
        unet.eval()
        statistics = {
            "test": tissue_statistics(unet, test_matched),
            "generated": tissue_statistics(unet, generated_matched),
            "reference": tissue_statistics(unet, reference),
        }
        for name, (fractions, confidence) in statistics.items():
            arrays[f"fractions_{name}"] = fractions.cpu().numpy()
            arrays[f"confidence_{name}"] = confidence.cpu().numpy()
        fractions_test = statistics["test"][0]
        report["tissue"] = {
            "checkpoint": args.unet,
            "mean_fraction": {name: fractions.mean(0).tolist() for name, (fractions, _) in statistics.items()},
            "wasserstein_to_test": {
                name: [wasserstein_1d(statistics[name][0][:, c], fractions_test[:, c]) for c in range(NUM_CLASSES)]
                for name in ("generated", "reference")
            },
            "mean_confidence": {name: confidence.mean().item() for name, (_, confidence) in statistics.items()},
        }
        tissue = report["tissue"]
        print("\n4. Anatomy - share of each class as segmented by the Task 2 UNet")
        print("   class                        " + "".join(f"{c:>9}" for c in range(NUM_CLASSES)))
        for name, label in (("test", "test slices"), ("reference", "random training slices"),
                            ("generated", "generated slices")):
            print(f"   mean share, {label:<18}" + "".join(f"{v:>9.3f}" for v in tissue["mean_fraction"][name]))
        for name, label in (("reference", "training vs test"), ("generated", "generated vs test")):
            print(f"   distance, {label:<20}" + "".join(f"{v:>9.4f}" for v in tissue["wasserstein_to_test"][name]))
        print("   (earth mover's distance between the per-slice shares; training vs test is the level a")
        print("    perfect generator would reach)")
        confidence = tissue["mean_confidence"]
        print(f"   UNet confidence on brain pixels: test {confidence['test']:.3f}, "
              f"training {confidence['reference']:.3f}, generated {confidence['generated']:.3f}")
    else:
        print(f"\n   {args.unet} not found - skipping the tissue comparison")

    # ---- Images for the figures ---------------------------------------------
    arrays["grid_generated"] = to_uint8(generated[:GRID_IMAGES])
    shuffle = torch.Generator(device="cpu").manual_seed(args.seed)
    arrays["grid_test"] = to_uint8(test[torch.randperm(len(test), generator=shuffle)[:GRID_IMAGES].to(device)])

    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'evaluation.json'} and {out_dir / 'arrays.npz'}")
    print(f"Copy {run_dir} back (without the .pt files) and run:  python plot_gan.py {run_dir}")


if __name__ == "__main__":
    main()
