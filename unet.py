"""UNet for OASIS brain MR segmentation, and the losses and metrics it needs.

Part 4, Task 2. The lab sheet requires categorical (one-hot) output and a Dice
similarity coefficient above 0.9 **for every label**, not merely on average.

The architecture is Ronneberger, Fischer and Brox, "U-Net: Convolutional
Networks for Biomedical Image Segmentation" (MICCAI 2015). Its shape is an
encoder that halves the resolution four times, a bottleneck, and a decoder that
doubles it back - plus the part that gives the network its name and its power:
**skip connections** that carry each encoder stage's feature map straight across
to the matching decoder stage.

Why the skips matter, and why a plain encoder-decoder is not enough:
downsampling is what buys the network a large receptive field, so that a pixel's
label can depend on context far away. But it destroys spatial precision - by the
bottleneck each position covers a 16x16 patch of the original image. The decoder
can recover *what* is there from the bottleneck, yet not *exactly where* its
boundaries are. The skip connections hand it back the high-resolution features
it needs to place edges to the pixel. Segmentation is a per-pixel task, so this
is the difference between a usable mask and a blurry blob.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU.

    The repeating unit of a UNet stage. Padding 1 keeps the resolution
    unchanged, so only the explicit pooling and upsampling steps alter it -
    which is what lets an encoder feature map concatenate with its decoder
    counterpart without any cropping.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """UNet producing one output channel per class.

    ``forward`` returns raw logits shaped ``(batch, num_classes, H, W)``. The
    softmax that makes them a categorical distribution over the classes lives in
    the loss (for numerical stability) and in :func:`predict`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth

        # Encoder. Each stage doubles the channels; the pooling between stages
        # halves the resolution. 256 -> 128 -> 64 -> 32 -> 16.
        self.encoders = nn.ModuleList()
        channels = in_channels
        stage_channels = []
        for level in range(depth):
            out_channels = base_channels * (2 ** level)
            self.encoders.append(DoubleConv(channels, out_channels))
            stage_channels.append(out_channels)
            channels = out_channels

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels, channels * 2)
        channels = channels * 2

        # Decoder. Each stage upsamples, concatenates the matching encoder
        # feature map, then convolves. The DoubleConv's input width is
        # upsampled + skip, hence the doubling.
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for out_channels in reversed(stage_channels):
            self.upsamples.append(
                nn.ConvTranspose2d(channels, out_channels, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(out_channels * 2, out_channels))
            channels = out_channels

        # 1x1 convolution to one channel per class: the categorical output the
        # lab sheet requires. Each pixel ends up with a score per class, which
        # softmax turns into a distribution over the four labels.
        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            # Guard against odd input sizes. With 256x256 the shapes always
            # match exactly, but a mismatch here would otherwise surface as an
            # inscrutable concatenation error.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = decoder(torch.cat([skip, x], dim=1))

        return self.head(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Class indices per pixel, shaped ``(batch, H, W)``."""
        return self.forward(x).argmax(dim=1)


def one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``(batch, H, W)`` class indices -> ``(batch, num_classes, H, W)`` one-hot.

    The lab sheet asks for categorical output explicitly. The network already
    produces one channel per class; this puts the *targets* in the same form so
    the Dice terms can be computed per class.
    """
    return F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int, smooth: float = 1.0
) -> torch.Tensor:
    """1 - mean Dice over classes, computed on softmax probabilities.

    "Soft" because it uses the probabilities rather than a hard argmax: argmax
    has zero gradient almost everywhere and cannot be trained through. The
    reported metric in :class:`DiceScore` uses the hard prediction, which is
    what is actually being asked for - so the two differ by design and the
    training loss will look slightly better than the true DSC.

    Dice is used rather than cross entropy alone because the classes are very
    unbalanced. Background covers most of a brain slice, so a model that
    predicted background everywhere would score well on pixel accuracy and on
    unweighted cross entropy while being useless. Dice normalises each class by
    its own size, so a small structure counts as much as a large one.
    """
    probabilities = F.softmax(logits, dim=1)
    targets_one_hot = one_hot(targets, num_classes)

    # Sum over batch and space, leaving one number per class.
    dims = (0, 2, 3)
    intersection = (probabilities * targets_one_hot).sum(dims)
    cardinality = probabilities.sum(dims) + targets_one_hot.sum(dims)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Dice plus cross entropy.

    Cross entropy gives strong, well-conditioned gradients early on, when the
    predictions are near-uniform and Dice's gradient is weak. Dice directly
    optimises the quantity being marked. Using both trains faster and more
    stably than either alone, which is the usual practice in segmentation.
    """

    def __init__(self, num_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = soft_dice_loss(logits, targets, self.num_classes)
        cross_entropy = self.cross_entropy(logits, targets)
        return self.dice_weight * dice + self.ce_weight * cross_entropy


class DiceScore:
    """Per-class Dice, accumulated over a whole dataset.

    Intersections and cardinalities are summed across every batch and the
    coefficient is computed once at the end, rather than averaging per-batch
    Dice scores. That matters here: many brain slices do not contain every
    class, and a per-slice Dice for an absent class is 0/0 - undefined, and
    usually fudged to either 0 or 1, both of which distort the average badly.
    Accumulating first sidesteps the question entirely.

    This uses the hard argmax prediction, so it is the coefficient the lab
    sheet's 0.9 threshold refers to.
    """

    def __init__(self, num_classes: int, device: torch.device | None = None):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes, dtype=torch.float64, device=device)
        self.cardinality = torch.zeros(num_classes, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """``predictions`` and ``targets`` are ``(batch, H, W)`` class indices."""
        predicted_one_hot = one_hot(predictions, self.num_classes).double()
        targets_one_hot = one_hot(targets, self.num_classes).double()
        dims = (0, 2, 3)
        self.intersection += (predicted_one_hot * targets_one_hot).sum(dims)
        self.cardinality += predicted_one_hot.sum(dims) + targets_one_hot.sum(dims)

    def compute(self) -> list[float]:
        """Dice per class. A class absent from the entire dataset scores 0."""
        dice = torch.where(
            self.cardinality > 0,
            2.0 * self.intersection / self.cardinality,
            torch.zeros_like(self.cardinality),
        )
        return dice.cpu().tolist()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape and metric check, runnable anywhere: no GPU and no dataset needed.
    model = UNet(num_classes=4)
    images = torch.rand(2, 1, 256, 256)
    targets = torch.randint(0, 4, (2, 256, 256))

    logits = model(images)
    print(f"input  {tuple(images.shape)} -> logits {tuple(logits.shape)}")
    print(f"parameters: {count_parameters(model):,}")

    loss = CombinedLoss(num_classes=4)(logits, targets)
    print(f"combined loss: {loss.item():.4f}")

    score = DiceScore(num_classes=4)
    score.update(model.predict(images), targets)
    print("per-class dice (random weights, so near chance): "
          + ", ".join(f"{d:.3f}" for d in score.compute()))

    # A perfect prediction must score exactly 1.0 for every present class.
    perfect = DiceScore(num_classes=4)
    perfect.update(targets, targets)
    print("per-class dice (perfect prediction): "
          + ", ".join(f"{d:.3f}" for d in perfect.compute()))
