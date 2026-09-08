"""A convolutional variational autoencoder for 256x256 OASIS brain slices.

Part 4, Task 1. An autoencoder learns to compress and rebuild its input. What
makes it *variational* is that the encoder emits a distribution over the latent
code rather than a single point, and the loss pushes that distribution towards a
standard normal. The result is a latent space that is continuous and populated
everywhere, so a code drawn at random decodes to a plausible brain - which is
what makes the manifold visualisation the task asks for meaningful.

The loss has two terms:

* **reconstruction** - how well the decoder rebuilds the input, here binary
  cross entropy over pixels in [0, 1];
* **KL divergence** - how far the encoder's distribution has drifted from
  N(0, I). Without it the model would shrink every variance to zero and
  degenerate into an ordinary autoencoder with a latent space full of holes.

``beta`` weights the second term. Raising it buys a smoother, more disentangled
latent space at the cost of blurrier reconstructions; lowering it does the
reverse. At beta=0 this is a plain autoencoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Five stride-2 stages take 256 -> 128 -> 64 -> 32 -> 16 -> 8.
ENCODER_CHANNELS = (32, 64, 128, 256, 256)
BOTTLENECK_SIZE = 8

# The KL term contains exp(logvar). Left unbounded, a few bad steps early in
# training can drive logvar high enough that exp() overflows and the loss goes
# to 1e12 or NaN, taking the whole run with it. exp(10) is about 22,000 and
# exp(-10) about 4.5e-5, so this range covers every variance a trained model
# plausibly wants while making the blow-up impossible.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Downsample by two. BatchNorm before the activation, no bias."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Upsample by two.

    kernel 4 with stride 2 and padding 1 exactly doubles the resolution and,
    unlike kernel 3, divides evenly by the stride - which is what avoids the
    checkerboard artefacts transposed convolutions are notorious for.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        encoder_layers = []
        channels = in_channels
        for out_channels in ENCODER_CHANNELS:
            encoder_layers.append(conv_block(channels, out_channels))
            channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        self.flat_features = ENCODER_CHANNELS[-1] * BOTTLENECK_SIZE * BOTTLENECK_SIZE

        # Two heads on the same trunk: the mean and the log-variance of the
        # approximate posterior q(z|x). log-variance rather than variance so the
        # network can output any real number and the variance stays positive.
        self.fc_mu = nn.Linear(self.flat_features, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flat_features)

        decoder_layers = []
        reversed_channels = list(reversed(ENCODER_CHANNELS))
        for index in range(len(reversed_channels) - 1):
            decoder_layers.append(
                deconv_block(reversed_channels[index], reversed_channels[index + 1])
            )
        self.decoder = nn.Sequential(*decoder_layers)

        # Final layer back to image channels. No BatchNorm and no ReLU here -
        # the output is a per-pixel probability, produced by the sigmoid in
        # decode().
        self.to_image = nn.ConvTranspose2d(
            reversed_channels[-1], in_channels, kernel_size=4, stride=2, padding=1
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).flatten(1)
        logvar = self.fc_logvar(features).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return self.fc_mu(features), logvar

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) in a way gradients can pass through.

        Sampling is not differentiable, so the randomness is moved out of the
        computational path: draw eps ~ N(0, I), which does not depend on any
        parameter, and compute z = mu + sigma * eps. The gradient then flows
        into mu and sigma while eps is treated as a constant. This is the
        reparameterisation trick, and it is the whole reason a VAE can be
        trained by ordinary backpropagation.

        At evaluation time the mean is used directly: a deterministic encoding
        is what makes reconstructions and latent plots reproducible.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.fc_decode(z)
        features = features.view(-1, ENCODER_CHANNELS[-1], BOTTLENECK_SIZE, BOTTLENECK_SIZE)
        return torch.sigmoid(self.to_image(self.decoder(features)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, reconstruction, kl)``, each averaged per image.

    Both terms are summed over pixels and latent dimensions and then divided by
    the batch size, rather than averaged over everything. Averaging over pixels
    would shrink the reconstruction term by a factor of 65,536 relative to the
    KL term, and the model would collapse to producing the dataset mean.
    """
    batch_size = target.size(0)

    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, target, reduction="sum"
    ) / batch_size

    # Closed form of KL( N(mu, sigma^2) || N(0, I) ), summed over latent dims.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    return reconstruction_loss + beta * kl, reconstruction_loss, kl


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape check, runnable anywhere: no GPU and no dataset needed.
    for latent_dim in (2, 32):
        model = VAE(latent_dim=latent_dim)
        dummy = torch.rand(2, 1, 256, 256)
        reconstruction, mu, logvar = model(dummy)
        total, rec, kl = vae_loss(reconstruction, dummy, mu, logvar)
        print(f"latent_dim={latent_dim}")
        print(f"  input  {tuple(dummy.shape)} -> reconstruction {tuple(reconstruction.shape)}")
        print(f"  mu {tuple(mu.shape)}  logvar {tuple(logvar.shape)}")
        print(f"  loss {total.item():.1f} (rec {rec.item():.1f}, kl {kl.item():.1f})")
        print(f"  parameters: {count_parameters(model):,}")
