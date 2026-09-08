"""The preprocessed OASIS brain MR dataset on Rangpur.

Layout, confirmed with ``explore_oasis.py`` on 9 September 2026:

    /home/groups/comp3710/OASIS/
        keras_png_slices_train/          9,664 PNG   256x256 greyscale uint8
        keras_png_slices_validate/       1,120 PNG
        keras_png_slices_test/             544 PNG
        keras_png_slices_seg_train/      9,664 PNG   segmentation masks
        keras_png_slices_seg_validate/   1,120 PNG
        keras_png_slices_seg_test/         544 PNG

The split is already made, so no train/test division happens here.

Two details are not obvious from the directory listing and both will break a
model silently if missed. They are handled in :class:`OASIS` and explained at
:data:`LABEL_VALUES` and :func:`mask_path_for`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = "/home/groups/comp3710/OASIS"

IMAGE_SIZE = 256
NUM_CLASSES = 4

# The masks are stored with their four classes spread evenly across the 8-bit
# range - 0, 85, 170, 255 - rather than as 0, 1, 2, 3. That makes them viewable
# as ordinary greyscale PNGs, but it means the raw pixel values cannot be used
# as class indices: cross entropy would treat them as indices into a 256-class
# output and either raise an index error or train on nonsense. Every mask is
# mapped back to 0..3 on load.
LABEL_VALUES = (0, 85, 170, 255)

SPLITS = ("train", "validate", "test")


def image_dir(root: Path | str, split: str) -> Path:
    return Path(root) / f"keras_png_slices_{split}"


def mask_dir(root: Path | str, split: str) -> Path:
    return Path(root) / f"keras_png_slices_seg_{split}"


def mask_path_for(image_path: Path, root: Path | str, split: str) -> Path:
    """Locate the mask belonging to an image.

    The two directories do not use identical filenames: an image is
    ``case_001_slice_0.nii.png`` and its mask is ``seg_001_slice_0.nii.png``.
    Only the leading ``case_`` differs, so the substitution is anchored to the
    start of the name rather than applied globally.
    """
    return mask_dir(root, split) / image_path.name.replace("case_", "seg_", 1)


def encode_mask(array: np.ndarray) -> np.ndarray:
    """Map raw mask pixels (0/85/170/255) to class indices (0/1/2/3).

    Uses the known value set rather than integer division so that an unexpected
    value is caught here, at load time, instead of surfacing later as a strange
    loss curve.
    """
    lookup = np.searchsorted(np.array(LABEL_VALUES), array)
    if not np.array_equal(np.array(LABEL_VALUES)[lookup], array):
        unexpected = sorted(set(np.unique(array).tolist()) - set(LABEL_VALUES))
        raise ValueError(
            f"mask contains values outside {LABEL_VALUES}: {unexpected[:10]}"
        )
    return lookup.astype(np.int64)


class OASIS(Dataset):
    """OASIS slices, optionally paired with their segmentation masks.

    Returns a float image in ``[0, 1]`` shaped ``(1, 256, 256)``. With
    ``with_masks=True`` it returns ``(image, mask)`` where the mask is an int64
    tensor of class indices shaped ``(256, 256)`` - the layout
    ``nn.CrossEntropyLoss`` expects, from which one-hot is one call away.

    Task 1's VAE is unsupervised and uses ``with_masks=False``; Task 2's UNet
    needs the pairs.
    """

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        split: str = "train",
        with_masks: bool = False,
        limit: int | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.with_masks = with_masks

        directory = image_dir(self.root, split)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} does not exist. On Rangpur the dataset is at "
                f"{DEFAULT_ROOT}; pass --data-root if it has moved."
            )

        # Sorted so that a run is reproducible: the filesystem's own order is
        # not guaranteed to be stable between nodes.
        self.image_paths = sorted(directory.glob("*.png"))
        if limit is not None:
            self.image_paths = self.image_paths[:limit]

        if not self.image_paths:
            raise FileNotFoundError(f"no PNG files under {directory}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            # Convert to "L" defensively: the files are already greyscale, but
            # a stray RGB one would otherwise change the tensor's shape and
            # break the batch collation with a confusing error.
            array = np.array(image.convert("L"), dtype=np.uint8)

        # Scale to [0, 1]. The decoder ends in a sigmoid and the reconstruction
        # loss is binary cross entropy, both of which assume this range.
        image_tensor = torch.from_numpy(array).float().div_(255.0).unsqueeze(0)

        if not self.with_masks:
            return image_tensor

        mask_path = mask_path_for(image_path, self.root, self.split)
        if not mask_path.exists():
            raise FileNotFoundError(f"no mask for {image_path.name} at {mask_path}")
        with Image.open(mask_path) as mask_image:
            mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)

        return image_tensor, torch.from_numpy(encode_mask(mask_array))


def build_loaders(
    root: Path | str = DEFAULT_ROOT,
    batch_size: int = 32,
    workers: int = 4,
    with_masks: bool = False,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders for the dataset's own train / validate / test split."""
    loaders = []
    for split in SPLITS:
        dataset = OASIS(root, split=split, with_masks=with_masks, limit=limit)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=split == "train",
                persistent_workers=workers > 0,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]


def describe(root: Path | str = DEFAULT_ROOT) -> None:
    """Print split sizes and one sample's shape, for the run log."""
    print(f"OASIS root: {root}")
    for split in SPLITS:
        dataset = OASIS(root, split=split)
        print(f"  {split:<10}{len(dataset):>8,} images")
    sample = OASIS(root, split="train", with_masks=True)[0]
    image, mask = sample
    print(f"  image {tuple(image.shape)} {image.dtype} "
          f"range {image.min():.3f}..{image.max():.3f}")
    print(f"  mask  {tuple(mask.shape)} {mask.dtype} "
          f"classes {sorted(mask.unique().tolist())}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    describe(parser.parse_args().data_root)
