"""CIFAR-10 loading, augmentation and normalisation.

Downloading is kept separate from training on purpose (see ``prepare_data.py``).
A GPU job that downloads is a GPU job that holds an A100 idle while it waits on
the network, and that fails outright if the download does.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Per-channel statistics of the CIFAR-10 training set. Normalising with these
# centres each channel near zero with unit variance, which keeps the first
# layer's activations in a sensible range and makes optimisation better behaved.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_transforms(train: bool) -> transforms.Compose:
    """Training gets augmentation; evaluation must not.

    Random crop from a 4-pixel zero padding and a horizontal flip are the
    standard CIFAR-10 augmentation pair. They cost nothing and are worth several
    points of accuracy, because 50,000 images is small enough that a ResNet-18
    will otherwise memorise the training set.
    """
    steps = []
    if train:
        steps += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(steps)


def build_datasets(data_dir: str, download: bool = False):
    """Return ``(train_dataset, test_dataset)``.

    ``download`` defaults to False so that training jobs fail loudly with a
    clear error if the data was never prepared, rather than silently starting a
    download on a GPU node.
    """
    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=build_transforms(train=True)
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=build_transforms(train=False)
    )
    return train_set, test_set


def build_loaders(
    data_dir: str,
    batch_size: int = 128,
    workers: int = 4,
    download: bool = False,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Data loaders for training and evaluation.

    The test loader uses a larger batch because no gradients or activations are
    stored during evaluation, so more images fit in memory at once.
    """
    train_set, test_set = build_datasets(data_dir, download=download)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        # Dropping the last short batch keeps every step the same size, which
        # matters for the one-cycle schedule's step count and for AMP later.
        drop_last=True,
        persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    return train_loader, test_loader


def describe(data_dir: str) -> None:
    """Print dataset sizes, for the run log."""
    train_set, test_set = build_datasets(data_dir, download=False)
    print(f"data dir      : {data_dir}")
    print(f"train images  : {len(train_set):,}")
    print(f"test images   : {len(test_set):,}")
    print(f"classes       : {len(CLASS_NAMES)} {CLASS_NAMES}")


def resolve_device(prefer: str = "auto") -> torch.device:
    """Pick a device.

    CUDA on Rangpur's A100 nodes; MPS so the same code runs on the MacBook the
    demonstration is given from; CPU as the fallback for a login node.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Human-readable device name for the run log."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type
