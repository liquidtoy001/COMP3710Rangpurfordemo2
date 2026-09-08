"""ResNet-18, adapted for 32x32 CIFAR-10 inputs.

Written out here rather than imported from ``torchvision.models``, because the
lab sheet states that pre-built models are generally not allowed without the
demonstrator's approval.

The architecture follows He et al., "Deep Residual Learning for Image
Recognition" (CVPR 2016), with the standard CIFAR stem substitution described in
``ResNet18.__init__``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual shortcut around them.

    The block computes ``relu(F(x) + shortcut(x))``. Because the shortcut is an
    identity whenever the shapes already match, the convolutions only have to
    learn the *residual* correction to their input rather than the whole
    mapping, which is what lets very deep stacks stay trainable.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # The shortcut must match the main path's shape before the two can be
        # added. A 1x1 projection is only needed when this block downsamples
        # (stride != 1) or changes the channel count; otherwise the cheaper and
        # better-behaved identity path is used.
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # Convolutions carry no bias because the BatchNorm immediately after
        # them has its own learnable shift, which would make the bias redundant.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-sized images.

    Four stages of two :class:`BasicBlock` s each (2+2+2+2 blocks, two
    convolutions per block, plus the stem and the classifier = 18 weighted
    layers, hence the name).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        # CIFAR stem: a single 3x3 stride-1 convolution, and no max pool.
        #
        # torchvision's ImageNet stem is a 7x7 stride-2 convolution followed by
        # a stride-2 max pool, which is sized for 224x224 inputs. Applied to a
        # 32x32 CIFAR image it would reduce the feature map to 8x8 before the
        # first residual block even runs, throwing away most of the spatial
        # detail the network needs. Keeping stride 1 here means the first stage
        # still sees the image at full 32x32 resolution.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Each stage doubles the channels and halves the resolution, so the
        # compute per stage stays roughly constant.
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)   # 32x32
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)  # 16x16
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)  # 8x8
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)  # 4x4

        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialise_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Build one stage. Only its first block downsamples."""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(BasicBlock(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def _initialise_weights(self) -> None:
        """He initialisation for convolutions, unit scale/zero shift for norms."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Global average pooling: 4x4 spatial -> 1x1, so the classifier sees one
        # number per channel and the network is insensitive to input size.
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters, for the record in the run log."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape check, runnable on the login node: no GPU and no dataset needed.
    model = ResNet18()
    dummy = torch.randn(2, 3, 32, 32)
    logits = model(dummy)
    print(f"input  {tuple(dummy.shape)}")
    print(f"output {tuple(logits.shape)}")
    print(f"trainable parameters: {count_parameters(model):,}")
