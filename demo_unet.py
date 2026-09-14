"""Live demonstration for Part 4, Task 2: UNet inference on the OASIS test split.

The lab sheet requires this to happen in front of the demonstrator: "You must run
inference at demonstration on a test set and show the model is working correctly
during the demo." Training was done beforehand; this script does only the part that
has to be live, and finishes in well under a minute on an A100:

    1. load the trained checkpoint
    2. run inference over every slice of the test split, timed
    3. print the Dice coefficient for each class against the 0.9 requirement, beside
       the figures committed in results/unet/metrics.json
    4. write a PNG of chosen slices - input, ground truth, prediction, and the pixels
       where prediction and truth disagree

The demonstrator can choose which slices go in the picture, so it is not a
hand-picked best case:

    python demo_unet.py                         # four slices spread evenly across the test split
    python demo_unet.py --slices 12 200 431     # the demonstrator's choice
    python demo_unet.py --random 4              # four at random; the seed is printed

The picture is drawn with Pillow rather than matplotlib, because matplotlib is not
installed on Rangpur and Pillow is (oasis.py already reads the PNGs with it).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from oasis import DEFAULT_ROOT, OASIS
from unet import DiceScore, UNet, count_parameters

DICE_TARGET = 0.9

# The same colours as plot_unet.py, so the live picture matches the committed figures.
CLASS_COLOURS = np.array(
    [[0, 0, 0], [0x4C, 0x72, 0xB0], [0xDD, 0x84, 0x52], [0x55, 0xA8, 0x68]], dtype=np.uint8
)
DISAGREEMENT_COLOUR = np.array([230, 30, 30], dtype=np.uint8)


def resolve_device(prefer: str) -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronise(device: torch.device) -> None:
    """GPU calls return once work is queued; wait for it before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[UNet, dict]:
    """Rebuild the UNet with the width it was trained at, and load its weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(num_classes=checkpoint["num_classes"],
                 base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def recorded_test_dice(checkpoint_path: Path) -> list[float] | None:
    """The test Dice the training run wrote beside the checkpoint, if it is there."""
    metrics = checkpoint_path.parent / "metrics.json"
    if not metrics.exists():
        return None
    return json.loads(metrics.read_text(encoding="utf-8")).get("test_dice")


@torch.no_grad()
def evaluate(model: UNet, loader: DataLoader, device: torch.device, num_classes: int):
    """Per-class Dice over the whole split, accumulated before dividing."""
    score = DiceScore(num_classes, device=device)
    for images, masks in loader:
        predictions = model(images.to(device, non_blocking=True)).argmax(dim=1)
        score.update(predictions, masks.to(device, non_blocking=True))
    return score.compute()


def choose_slices(args: argparse.Namespace, count: int) -> tuple[list[int], str]:
    """Which slices to draw, and a sentence saying how they were chosen."""
    if args.slices:
        bad = [i for i in args.slices if not 0 <= i < count]
        if bad:
            raise SystemExit(f"slice indices must be between 0 and {count - 1}; got {bad}")
        return args.slices, "chosen on the command line"
    if args.random:
        seed = args.seed if args.seed is not None else int(time.time()) % 100_000
        rng = np.random.default_rng(seed)
        picks = sorted(rng.choice(count, size=min(args.random, count), replace=False).tolist())
        return picks, f"drawn at random with seed {seed} (rerun with --seed {seed} to reproduce)"
    picks = np.linspace(0, count - 1, num=min(4, count)).round().astype(int).tolist()
    return picks, "spread evenly across the test split, not hand-picked"


def colourise(labels: np.ndarray) -> np.ndarray:
    return CLASS_COLOURS[labels]


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:            # Pillow older than 10.1 has one fixed-size default font
        return ImageFont.load_default()


@torch.no_grad()
def draw_slices(model: UNet, dataset: OASIS, indices: list[int], device: torch.device,
                scale: int = 2) -> tuple[Image.Image, list[dict]]:
    """One row per slice: input | ground truth | prediction | disagreement."""
    titles = ["input", "ground truth", "prediction", "disagreement"]
    header, gap = 30, 8
    rows, notes = [], []

    for index in indices:
        image, mask = dataset[index]
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
        truth = mask.numpy()
        grey = (image[0].numpy() * 255).astype(np.uint8)
        grey_rgb = np.stack([grey] * 3, axis=-1)

        wrong = prediction != truth
        disagreement = grey_rgb.copy()
        disagreement[wrong] = DISAGREEMENT_COLOUR

        panels = [grey_rgb, colourise(truth), colourise(prediction), disagreement]
        panels = [np.kron(p, np.ones((scale, scale, 1), dtype=np.uint8)) for p in panels]
        rows.append(np.concatenate(
            [np.pad(p, ((0, 0), (0, gap), (0, 0)), constant_values=255) for p in panels], axis=1))
        notes.append({"index": index, "wrong": int(wrong.sum()), "total": wrong.size,
                      "name": dataset.image_paths[index].name})

    panel_width = rows[0].shape[1] // 4
    height = header + sum(r.shape[0] + header for r in rows) + 36
    canvas = Image.new("RGB", (rows[0].shape[1], height), "white")
    pen = ImageDraw.Draw(canvas)
    title_font, text_font = font(20), font(16)

    for column, title in enumerate(titles):
        pen.text((column * panel_width + 6, 6), title, fill="black", font=title_font)

    y = header
    for row, note in zip(rows, notes):
        pen.text((6, y + 6),
                 f"test slice {note['index']}  ({note['name']})  -  "
                 f"{note['wrong']:,} of {note['total']:,} pixels disagree "
                 f"({note['wrong'] / note['total']:.2%})",
                 fill="black", font=text_font)
        y += header
        canvas.paste(Image.fromarray(row), (0, y))
        y += row.shape[0]

    x = 6
    for label, colour in zip(["class 0 (background)", "class 1", "class 2", "class 3"],
                             CLASS_COLOURS):
        pen.rectangle([x, y + 10, x + 18, y + 28], fill=tuple(int(c) for c in colour),
                      outline="grey")
        pen.text((x + 24, y + 10), label, fill="black", font=text_font)
        x += 230
    pen.rectangle([x, y + 10, x + 18, y + 28], fill=tuple(int(c) for c in DISAGREEMENT_COLOUR))
    pen.text((x + 24, y + 10), "prediction differs from truth", fill="black", font=text_font)
    return canvas, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/unet/best.pt")
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--slices", type=int, nargs="+", help="test-split indices to draw")
    parser.add_argument("--random", type=int, help="draw this many slices at random")
    parser.add_argument("--seed", type=int, help="seed for --random")
    parser.add_argument("--out-dir", default="runs/demo")
    parser.add_argument("--limit", type=int, default=None, help="use only N test slices (smoke tests)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)

    print("=" * 72)
    print("COMP3710 demonstration 2 - Part 4 Task 2, live UNet inference")
    print(f"job id : {os.environ.get('SLURM_JOB_ID', 'not a Slurm job')}")
    print(f"host   : {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"device : {device}" + (f" - {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""))
    print("=" * 72)

    model, checkpoint = load_model(checkpoint_path, device)
    num_classes = checkpoint["num_classes"]
    print(f"\nloaded {checkpoint_path}")
    print(f"  selected at epoch {checkpoint['epoch']} on its worst validation class "
          f"({checkpoint['val_dice_min']:.4f})")
    print(f"  {count_parameters(model):,} parameters, base width {checkpoint['base_channels']}")

    dataset = OASIS(args.data_root, split="test", with_masks=True, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    # --- 1. Inference over the whole test split ------------------------------
    scope = "all" if args.limit is None else "the first"
    print(f"\n[1/2] inference over {scope} {len(dataset)} test slices")
    synchronise(device)
    start = time.perf_counter()
    dice = evaluate(model, loader, device, num_classes)
    synchronise(device)
    elapsed = time.perf_counter() - start
    print(f"  {elapsed:.1f} s, {len(dataset) / elapsed:.0f} slices per second"
          " (including reading the PNGs from disk)")

    recorded = recorded_test_dice(checkpoint_path) if args.limit is None else None
    print()
    print(f"  {'class':<8}{'live DSC':>10}{'recorded':>10}   {'vs 0.9':<8}")
    for c, value in enumerate(dice):
        before = f"{recorded[c]:.4f}" if recorded else "-"
        status = "ok" if value > DICE_TARGET else "LOW"
        print(f"  {c:<8}{value:>10.4f}{before:>10}   {status:<8}{'#' * int(value * 40)}")
    print(f"  worst class {min(dice):.4f}, mean {sum(dice) / len(dice):.4f}")
    print(f"  requirement, every class above {DICE_TARGET}: "
          f"{'MET' if min(dice) > DICE_TARGET else 'NOT MET'}")
    if recorded:
        largest = max(abs(a - b) for a, b in zip(dice, recorded))
        print(f"  largest difference from the committed results: {largest:.1e}"
              + ("  - the live run reproduces them" if largest < 5e-4 else ""))
    elif args.limit is not None:
        print("  (a --limit run covers only part of the split, so it is not compared"
              " with the committed figures)")

    # --- 2. The picture --------------------------------------------------------
    indices, how = choose_slices(args, len(dataset))
    print(f"\n[2/2] drawing test slices {indices}, {how}")
    canvas, notes = draw_slices(model, dataset, indices, device)
    for note in notes:
        print(f"  slice {note['index']:>4}: {note['wrong']:>5,} of {note['total']:,} pixels"
              f" disagree ({note['wrong'] / note['total']:.2%})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"unet_slices_{stamp}.png"
    canvas.save(out_path)
    print(f"\n  wrote {out_path.resolve()}")
    print("  to view it on the laptop:")
    print(f"    scp {getpass.getuser()}@rangpur.compute.eait.uq.edu.au:{out_path.resolve().as_posix()} .")


if __name__ == "__main__":
    main()
