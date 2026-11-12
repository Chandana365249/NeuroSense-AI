"""FusionNet: EEGNet-style spatial-frequency encoder + temporal convolutional network + binary valence head.

(batch, 30, 500)
  temporal conv (16 filters, 0.5 s) -> depthwise spatial conv (32 maps) -> average pool x4       EEGNet-style
  3 dilated residual temporal blocks (dilation 1, 2, 4)                                          TCN
  mean + log-variance over time -> linear                                                        latent (64)
  linear                                                                                         valence logit
  gradient reversal -> linear                                                                    domain head (DANN only)
  linear projection                                                                              self-supervised pretraining only
"""

from __future__ import annotations

import torch
from torch import nn

LATENT = 64


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels), nn.ELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, n_channels: int = 30, dropout: float = 0.3) -> None:
        super().__init__()
        self.spectral_spatial = nn.Sequential(
            nn.Conv2d(1, 16, (1, 63), padding=(0, 31), bias=False), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False), nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.temporal = nn.Sequential(*(TemporalBlock(32, d, dropout) for d in (1, 2, 4)))
        self.latent = nn.Sequential(nn.Linear(64, LATENT), nn.ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(self.spectral_spatial(x[:, None]).squeeze(2))
        pooled = torch.cat([h.mean(dim=-1), torch.log(h.var(dim=-1) + 1e-6)], dim=1)
        return self.latent(pooled)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight  # type: ignore[attr-defined]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * grad, None  # type: ignore[attr-defined]


class FusionNet(nn.Module):
    def __init__(self, n_channels: int = 30, n_domains: int = 16) -> None:
        super().__init__()
        self.encoder = Encoder(n_channels)
        self.head = nn.Linear(LATENT, 1)
        self.domain_head = nn.Linear(LATENT, n_domains)
        self.projection = nn.Sequential(nn.Linear(LATENT, LATENT), nn.ELU(), nn.Linear(LATENT, 32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.head(z).squeeze(1), z

    def domain_logits(self, z: torch.Tensor, weight: float) -> torch.Tensor:
        return self.domain_head(GradientReversal.apply(z, weight))


def parameter_counts(model: FusionNet) -> dict[str, int]:
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    return {"encoder": count(model.encoder), "valence_head": count(model.head), "domain_head": count(model.domain_head),
            "projection_head": count(model.projection), "total": count(model)}


def coral_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance between the covariance of two participant-disjoint groups of latents."""
    def covariance(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        return x.T @ x / max(len(x) - 1, 1)
    return (covariance(a) - covariance(b)).pow(2).sum() / (4 * a.shape[1] ** 2)


def mmd_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multi-bandwidth RBF maximum mean discrepancy between two participant-disjoint groups."""
    joined = torch.cat([a, b])
    distances = torch.cdist(joined, joined).pow(2)
    bandwidth = distances.detach().mean().clamp_min(1e-6)
    kernel = sum(torch.exp(-distances / (bandwidth * s)) for s in (0.5, 1.0, 2.0))
    n = len(a)
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()


def contrastive_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent: row i of `a` should match row i of `b`."""
    a, b = nn.functional.normalize(a, dim=1), nn.functional.normalize(b, dim=1)
    logits = a @ b.T / temperature
    targets = torch.arange(len(a))
    return (nn.functional.cross_entropy(logits, targets) + nn.functional.cross_entropy(logits.T, targets)) / 2
