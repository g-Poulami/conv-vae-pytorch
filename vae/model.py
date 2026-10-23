"""Convolutional variational autoencoder for 28x28 images.

Encoder q(z|x): two strided convolutions, a hidden dense layer, then two
linear heads giving the mean and log-variance of a diagonal Gaussian.
Decoder p(x|z): the mirror image, ending in per-pixel logits so the
reconstruction term can be computed with a numerically stable
binary cross-entropy.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        if image_size != 28:
            raise ValueError("ConvVAE is built for 28x28 inputs (MNIST / MedMNIST size 28).")
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        feat = 64 * 7 * 7  # spatial size after two stride-2 convs: 28 -> 14 -> 7

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(feat, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 28x28
        )

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I); keeps sampling differentiable."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return per-pixel logits of p(x|z)."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Draw n images from the generative model p(z) p(x|z) (mean intensities)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction through the posterior mean."""
        mu, _ = self.encode(x)
        return torch.sigmoid(self.decode(mu))

    def config(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "image_size": self.image_size,
        }
