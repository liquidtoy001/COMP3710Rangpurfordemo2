"""A generative adversarial network for OASIS brain slices (Part 4, Task 3).

Two networks are trained against each other. The **generator** turns a vector
of Gaussian noise into an image; the **discriminator** scores an image for how
real it looks. The discriminator learns to tell real slices from generated ones,
and the generator learns to make slices the discriminator scores as real. The
generator never sees a real image: everything it knows about brains arrives as
the gradient of the discriminator's score.

The lab sheet warns that GANs converge chaotically, so every design choice here
is made for stability first, and each one is a published, standard remedy:

* **Spectral normalisation** on every discriminator layer (Miyato et al., 2018).
  It divides each weight matrix by its largest singular value, which bounds how
  sharply the discriminator's score can change with its input. An unbounded
  discriminator can become so confident that the generator's gradient vanishes
  or explodes; a bounded one keeps giving the generator a usable signal.
* **Hinge loss** (Lim & Ye, 2017; used with spectral normalisation by SNGAN and
  BigGAN). The discriminator stops being pushed on an image once it scores it
  beyond a margin, so it spends its capacity on the images it still gets wrong.
* **No BatchNorm in the discriminator.** Its batches are all-real or all-fake,
  so batch statistics would leak which kind of batch it is looking at.
* **Upsample-then-convolve in the generator** rather than transposed
  convolutions, which overlap unevenly and paint checkerboard artefacts
  (Odena et al., 2016) - visible and distracting on smooth brain tissue.
* **An exponential moving average of the generator's weights** for sampling
  (Karras et al., 2018; Yazici et al., 2019). The live weights keep jittering
  as the two networks chase each other; their average is smoother and gives
  steadier images.
* **DiffAugment** (Zhao et al., 2020), optional: the same random translation
  and cutout applied to real and generated images before the discriminator
  sees them. With 9,664 training slices a discriminator can start memorising
  the training set; augmenting both sides stops that without teaching the
  generator to draw the augmentations.

Images are scaled to [-1, 1], matching the generator's tanh output.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

RESOLUTIONS = (64, 128, 256)
MAX_CHANNELS = 512


def channels_at(size: int, resolution: int, width: int) -> int:
    """Feature channels at a given spatial size.

    Channels double each time the resolution halves, starting from ``width`` at
    full resolution and capped at MAX_CHANNELS. Both networks use the same rule,
    so neither is given an unfair share of the capacity.
    """
    return min(MAX_CHANNELS, width * resolution // size)


def check_resolution(resolution: int) -> int:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution}")
    return int(math.log2(resolution)) - 2  # stages between 4x4 and full size


class UpBlock(nn.Module):
    """Double the resolution: nearest-neighbour upsample, then two 3x3 convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Generator(nn.Module):
    """Noise ``(batch, z_dim)`` -> image ``(batch, 1, resolution, resolution)`` in [-1, 1]."""

    def __init__(self, resolution: int = 128, z_dim: int = 128, width: int = 32):
        super().__init__()
        stages = check_resolution(resolution)
        self.resolution = resolution
        self.z_dim = z_dim

        start = channels_at(4, resolution, width)
        # The noise vector becomes a 4x4 feature map, which the stages grow.
        self.project = nn.Linear(z_dim, start * 4 * 4, bias=False)
        self.project_norm = nn.Sequential(nn.BatchNorm2d(start), nn.ReLU(inplace=True))
        self.start_channels = start

        blocks = []
        size = 4
        for _ in range(stages):
            blocks.append(UpBlock(channels_at(size, resolution, width),
                                  channels_at(size * 2, resolution, width)))
            size *= 2
        self.blocks = nn.Sequential(*blocks)

        # To one greyscale channel. tanh bounds the output to [-1, 1], the range
        # the real images are scaled to.
        self.to_image = nn.Conv2d(channels_at(resolution, resolution, width), 1, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, self.start_channels, 4, 4)
        x = self.blocks(self.project_norm(x))
        return torch.tanh(self.to_image(x))


class DownBlock(nn.Module):
    """Halve the resolution: a 3x3 convolution, then a stride-2 4x4 convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Discriminator(nn.Module):
    """Image -> one unbounded realness score per image (hinge loss wants no sigmoid)."""

    def __init__(self, resolution: int = 128, width: int = 32):
        super().__init__()
        stages = check_resolution(resolution)
        self.resolution = resolution

        # LeakyReLU rather than ReLU, so a unit that is off still passes some
        # gradient back to the generator.
        self.from_image = nn.Sequential(
            spectral_norm(nn.Conv2d(1, channels_at(resolution, resolution, width), 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        blocks = []
        size = resolution
        for _ in range(stages):
            blocks.append(DownBlock(channels_at(size, resolution, width),
                                    channels_at(size // 2, resolution, width)))
            size //= 2
        self.blocks = nn.Sequential(*blocks)

        end = channels_at(4, resolution, width)
        self.head = nn.Sequential(
            spectral_norm(nn.Conv2d(end, end, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            spectral_norm(nn.Linear(end * 4 * 4, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.from_image(x))).squeeze(1)


def discriminator_hinge_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Push real scores above +1 and fake scores below -1; no push beyond the margin."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def generator_hinge_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Raise the discriminator's score of generated images."""
    return -fake_scores.mean()


class EMA:
    """Exponential moving average of a generator's weights.

    ``shadow`` is a full copy of the generator whose parameters follow
    ``shadow = decay * shadow + (1 - decay) * live`` after every training step.
    With decay 0.999 it averages over roughly the last thousand steps.

    BatchNorm running statistics are not averaged: they belong to the live
    weights, not the averaged ones, so :func:`recalibrate_batchnorm` recomputes
    them for the shadow before it is used.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow, live in zip(self.shadow.parameters(), model.parameters()):
            shadow.lerp_(live, 1.0 - self.decay)
        for shadow, live in zip(self.shadow.buffers(), model.buffers()):
            shadow.copy_(live)


@torch.no_grad()
def recalibrate_batchnorm(generator: Generator, device: torch.device,
                          batches: int = 32, batch_size: int = 64) -> Generator:
    """Recompute a generator's BatchNorm statistics for its own weights.

    Returns a copy; the generator passed in is untouched. Each BatchNorm layer's
    running mean and variance are reset and re-estimated as a plain average over
    ``batches`` batches of fresh noise, so sampling in eval mode normalises with
    statistics that match these weights. Without this, the averaged generator
    would be normalised with the live generator's statistics and could produce
    washed-out or saturated images.
    """
    calibrated = copy.deepcopy(generator).to(device)
    norms = [m for m in calibrated.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None  # cumulative average rather than exponential
    calibrated.train()
    for _ in range(batches):
        calibrated(torch.randn(batch_size, calibrated.z_dim, device=device))
    calibrated.eval()
    return calibrated


def diff_augment(images: torch.Tensor, policy: str) -> torch.Tensor:
    """DiffAugment: the same differentiable random augmentations for real and fake.

    ``policy`` is a comma-separated subset of ``translation`` and ``cutout``.
    Both are made of indexing and masking, so gradients flow through them to the
    generator. Each image gets its own random draw.
    """
    if not policy:
        return images
    for name in policy.split(","):
        if name == "translation":
            images = _translate(images, ratio=0.125)
        elif name == "cutout":
            images = _cutout(images, ratio=0.5)
        else:
            raise ValueError(f"unknown DiffAugment operation {name!r}")
    return images


def _translate(images: torch.Tensor, ratio: float) -> torch.Tensor:
    """Shift each image by up to ``ratio`` of its size, filling with background."""
    batch, _, height, width = images.shape
    shift = int(height * ratio + 0.5)
    dy = torch.randint(-shift, shift + 1, (batch, 1, 1), device=images.device)
    dx = torch.randint(-shift, shift + 1, (batch, 1, 1), device=images.device)
    rows = torch.arange(height, device=images.device).view(1, height, 1)
    cols = torch.arange(width, device=images.device).view(1, 1, width)
    # Pad with one border of -1 (black background) and clamp indices into it.
    padded = F.pad(images, [1, 1, 1, 1], value=-1.0)
    source_rows = (rows - dy + 1).clamp(0, height + 1).expand(batch, height, width)
    source_cols = (cols - dx + 1).clamp(0, width + 1).expand(batch, height, width)
    index = torch.arange(batch, device=images.device).view(batch, 1, 1)
    return padded[index, :, source_rows, source_cols].permute(0, 3, 1, 2)


def _cutout(images: torch.Tensor, ratio: float) -> torch.Tensor:
    """Blank a random square of ``ratio`` of the side length in each image."""
    batch, _, height, width = images.shape
    size = int(height * ratio + 0.5)
    top = torch.randint(0, height - size + 1, (batch, 1, 1), device=images.device)
    left = torch.randint(0, width - size + 1, (batch, 1, 1), device=images.device)
    rows = torch.arange(height, device=images.device).view(1, height, 1)
    cols = torch.arange(width, device=images.device).view(1, 1, width)
    inside = (rows >= top) & (rows < top + size) & (cols >= left) & (cols < left + size)
    return torch.where(inside.unsqueeze(1), torch.full_like(images, -1.0), images)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape check, runnable anywhere: no GPU and no dataset needed.
    for resolution in RESOLUTIONS:
        generator = Generator(resolution)
        discriminator = Discriminator(resolution)
        z = torch.randn(2, generator.z_dim)
        images = generator(z)
        scores = discriminator(diff_augment(images, "translation,cutout"))
        print(f"resolution {resolution}: image {tuple(images.shape)} "
              f"range [{images.min():.2f}, {images.max():.2f}] -> scores {tuple(scores.shape)}")
        print(f"  generator {count_parameters(generator):,} parameters, "
              f"discriminator {count_parameters(discriminator):,}")
