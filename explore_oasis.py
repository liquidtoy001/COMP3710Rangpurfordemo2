"""Report what the OASIS dataset on Rangpur actually contains.

Part 4's tasks all read the preprocessed OASIS brain MR data from
``/home/groups/comp3710/``, but the lab sheet does not say how it is laid out:
directory structure, file format, image size, how many images, whether a
train/test split already exists, or how many segmentation labels there are.
Every one of those decides how the dataset class is written, so this runs first.

Read-only. It opens a handful of files to report their shape and value range and
writes nothing.

    sbatch slurm/explore_oasis.sh          # on a CPU node
    python explore_oasis.py --root /home/groups/comp3710
"""

from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path

MAX_TREE_ENTRIES = 12
SAMPLE_FILES = 3


def human(size: float) -> str:
    """Bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def show_tree(root: Path, max_depth: int = 3) -> None:
    """Print the directory structure, collapsing long listings."""
    print(f"\n{'=' * 70}\nDIRECTORY TREE (depth {max_depth})\n{'=' * 70}")

    def walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            print(f"{prefix}[permission denied]")
            return
        except OSError as error:
            print(f"{prefix}[{error}]")
            return

        directories = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        for directory in directories[:MAX_TREE_ENTRIES]:
            try:
                child_count = sum(1 for _ in directory.iterdir())
            except OSError:
                child_count = -1
            print(f"{prefix}{directory.name}/  ({child_count} entries)")
            walk(directory, prefix + "    ", depth + 1)
        if len(directories) > MAX_TREE_ENTRIES:
            print(f"{prefix}... and {len(directories) - MAX_TREE_ENTRIES} more directories")

        if files:
            # Files are summarised by extension rather than listed: these
            # directories can hold thousands.
            by_extension = collections.Counter(f.suffix.lower() or "(no extension)" for f in files)
            summary = ", ".join(f"{count}x {ext}" for ext, count in by_extension.most_common(6))
            print(f"{prefix}[{len(files)} files: {summary}]")

    walk(root, "  ", 1)


def summarise_files(root: Path) -> dict[str, list[Path]]:
    """Count every file under root by extension, and remember examples."""
    print(f"\n{'=' * 70}\nFILE INVENTORY\n{'=' * 70}")

    counts: collections.Counter = collections.Counter()
    total_bytes: collections.Counter = collections.Counter()
    examples: dict[str, list[Path]] = collections.defaultdict(list)
    total_files = 0

    for directory_path, _, file_names in os.walk(root):
        for name in file_names:
            path = Path(directory_path) / name
            extension = path.suffix.lower() or "(no extension)"
            counts[extension] += 1
            total_files += 1
            try:
                total_bytes[extension] += path.stat().st_size
            except OSError:
                pass
            if len(examples[extension]) < SAMPLE_FILES:
                examples[extension].append(path)

    print(f"{'extension':<18}{'count':>10}{'total size':>14}")
    print("-" * 42)
    for extension, count in counts.most_common():
        print(f"{extension:<18}{count:>10,}{human(total_bytes[extension]):>14}")
    print("-" * 42)
    print(f"{'TOTAL':<18}{total_files:>10,}{human(sum(total_bytes.values())):>14}")

    return examples


def inspect_images(examples: dict[str, list[Path]]) -> None:
    """Open a few sample files and report shape, dtype and value range.

    The shapes and the set of distinct values are what decide the model: the
    input size for the VAE, and how many segmentation classes the UNet's
    one-hot output needs.
    """
    print(f"\n{'=' * 70}\nSAMPLE FILE CONTENTS\n{'=' * 70}")

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    volume_extensions = {".nii", ".gz", ".mhd", ".nrrd"}

    for extension, paths in sorted(examples.items()):
        if extension not in image_extensions | volume_extensions | {".npy", ".npz"}:
            continue
        for path in paths:
            print(f"\n  {path}")
            try:
                if extension in image_extensions:
                    from PIL import Image
                    import numpy as np

                    with Image.open(path) as image:
                        array = np.array(image)
                        print(f"    PIL mode   : {image.mode}   size: {image.size}")
                    print(f"    numpy shape: {array.shape}   dtype: {array.dtype}")
                    print(f"    value range: {array.min()} .. {array.max()}")
                    unique = np.unique(array)
                    if len(unique) <= 16:
                        # A small set of distinct values means this is a
                        # segmentation label map, not an intensity image - and
                        # its length is the number of classes.
                        print(f"    distinct values ({len(unique)}): {unique.tolist()}")
                        print("    -> looks like a LABEL MAP")
                    else:
                        print(f"    distinct values: {len(unique)} -> looks like an INTENSITY image")

                elif extension in {".npy", ".npz"}:
                    import numpy as np

                    data = np.load(path)
                    if extension == ".npz":
                        print(f"    keys: {list(data.keys())}")
                    else:
                        print(f"    shape: {data.shape}  dtype: {data.dtype}")
                        print(f"    value range: {data.min()} .. {data.max()}")

                else:
                    try:
                        import nibabel

                        volume = nibabel.load(str(path))
                        print(f"    nibabel shape: {volume.shape}")
                        print(f"    voxel sizes  : {volume.header.get_zooms()}")
                    except ImportError:
                        print("    (nibabel not installed - "
                              "`pip install --no-cache-dir nibabel` to inspect)")
            except Exception as error:  # noqa: BLE001 - a probe should not die on one bad file
                print(f"    could not read: {type(error).__name__}: {error}")


# --- OASIS-specific analysis -------------------------------------------------
#
# The generic inventory says how many files exist, but three things decide the
# Part 4 code and none are visible from a file listing: the image size (the
# VAE's input layer), the set of label values (the width of the UNet's one-hot
# output), and how an image filename maps to its mask's.

OASIS_SPLITS = ("train", "validate", "test")
LABEL_SAMPLE_FILES = 25


def analyse_oasis(root: Path) -> None:
    """Report image size, label classes and filename pairing for OASIS."""
    import numpy as np
    from PIL import Image

    image_dirs = {s: root / f"keras_png_slices_{s}" for s in OASIS_SPLITS}
    label_dirs = {s: root / f"keras_png_slices_seg_{s}" for s in OASIS_SPLITS}

    if not all(d.is_dir() for d in list(image_dirs.values()) + list(label_dirs.values())):
        return  # not an OASIS-shaped directory; the generic report is enough

    print("")
    print("=" * 70)
    print("OASIS DETAIL")
    print("=" * 70)

    print("")
    print("split sizes")
    print(f"  {'split':<10}{'images':>10}{'labels':>10}   paired")
    for split in OASIS_SPLITS:
        images = sorted(image_dirs[split].iterdir())
        labels = sorted(label_dirs[split].iterdir())
        paired = "yes" if len(images) == len(labels) else "NO - counts differ"
        print(f"  {split:<10}{len(images):>10,}{len(labels):>10,}   {paired}")

    # Image geometry and dtype: what the VAE's encoder has to accept.
    print("")
    print("image samples (keras_png_slices_train)")
    for path in sorted(image_dirs["train"].iterdir())[:3]:
        with Image.open(path) as image:
            array = np.array(image)
            mode = image.mode
        print(f"  {path.name}")
        print(f"    mode {mode}  shape {array.shape}  dtype {array.dtype}"
              f"  range {array.min()}..{array.max()}"
              f"  distinct {len(np.unique(array))}")

    # Label values accumulated across many files. A single slice need not
    # contain every class, so the union over a sample is the trustworthy answer.
    print("")
    print(f"label values, union over {LABEL_SAMPLE_FILES} files (keras_png_slices_seg_train)")
    seen: set[int] = set()
    per_file_counts = []
    for path in sorted(label_dirs["train"].iterdir())[:LABEL_SAMPLE_FILES]:
        with Image.open(path) as image:
            values = np.unique(np.array(image))
        seen.update(int(v) for v in values)
        per_file_counts.append(len(values))
    print(f"  distinct values across the sample: {sorted(seen)}")
    print(f"  -> {len(seen)} classes; per-file class counts ranged "
          f"{min(per_file_counts)}..{max(per_file_counts)}")
    print("  (this is the width the UNet one-hot output needs)")

    # Filename mapping: the dataset class must turn an image path into its mask
    # path, and the two directories need not use the same names.
    print("")
    print("filename pairing")
    image_names = sorted(p.name for p in image_dirs["train"].iterdir())[:3]
    label_names = sorted(p.name for p in label_dirs["train"].iterdir())[:3]
    print(f"  image names: {image_names}")
    print(f"  label names: {label_names}")
    label_set = {p.name for p in label_dirs["train"].iterdir()}
    if any(n in label_set for n in image_names):
        print("  -> names match exactly; pair by filename")
    else:
        print("  -> names differ; derive the rule from the samples above")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/home/groups/comp3710",
        help="directory to explore (the lab sheet points at /home/groups/comp3710)",
    )
    parser.add_argument("--depth", type=int, default=3, help="tree depth to print")
    args = parser.parse_args()

    root = Path(args.root)
    print(f"exploring: {root}")
    if not root.exists():
        print("does not exist, or is not readable from this node.")
        print("Check the path in the lab sheet's appendix A and your group membership.")
        return

    show_tree(root, args.depth)
    examples = summarise_files(root)
    inspect_images(examples)
    analyse_oasis(root)

    print(f"\n{'=' * 70}")
    print("Questions this output should answer before any Part 4 code is written:")
    print("  1. Where exactly is OASIS, and is it already split into train/test?")
    print("  2. What format and image size? (decides the VAE's input layer)")
    print("  3. Are segmentation labels present, and how many classes?")
    print("     (the UNet needs one-hot output over exactly that many)")
    print("  4. How many images in total? (decides batch size and epoch length)")
    print("=" * 70)


if __name__ == "__main__":
    main()
