"""Compact convolutional network for multi-channel EEG segments.

Architecture follows the EEGNet design (Lawhern et al., 2018, J. Neural Eng. 15:056013) because it is a
small, widely used EEG baseline that later extends naturally to CNN-LSTM. It is not claimed to be
optimal for this task.

Input: (batch, 1, channels=32, samples=512)  -- one 4 s segment at 128 Hz, z-scored per channel.

1. temporal conv  Conv2d(1 -> F1, kernel (1, K_t), 'same')           frequency-selective filters
   BatchNorm
2. spatial conv   depthwise Conv2d(F1 -> F1*D, kernel (C, 1), groups=F1)  spatial filters per frequency
   BatchNorm, ELU, AvgPool(1, P1), Dropout
3. separable conv depthwise Conv2d(F1*D, kernel (1, K_s), 'same') + pointwise Conv2d(-> F2, 1x1)
   BatchNorm, ELU, AvgPool(1, P2), Dropout
4. classifier     Flatten -> Linear(F2 * samples / (P1*P2) -> n_classes)  (logits)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CNNConfig:
    n_channels: int = 32
    n_samples: int = 512
    n_classes: int = 2
    temporal_filters: int = 8  # F1
    depth_multiplier: int = 2  # D
    separable_filters: int = 16  # F2
    temporal_kernel: int = 64  # K_t: 0.5 s at 128 Hz, resolves >= 2 Hz
    separable_kernel: int = 16  # K_s
    pool1: int = 4
    pool2: int = 8
    dropout: float = 0.5

    def __post_init__(self) -> None:
        if self.n_samples % (self.pool1 * self.pool2):
            raise ValueError("n_samples must be divisible by pool1 * pool2")
        if min(self.n_channels, self.n_samples, self.n_classes, self.temporal_filters,
               self.depth_multiplier, self.separable_filters) < 1:
            raise ValueError("All sizes must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _same_pad(kernel: int) -> nn.ZeroPad2d:
    """Pads the time axis so a stride-1 conv keeps its length (handles even kernels explicitly)."""
    return nn.ZeroPad2d(((kernel - 1) // 2, kernel // 2, 0, 0))


class EEGCNN(nn.Module):
    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        c = config or CNNConfig()
        self.config = c
        f1, fd, f2 = c.temporal_filters, c.temporal_filters * c.depth_multiplier, c.separable_filters
        self.temporal = nn.Sequential(
            _same_pad(c.temporal_kernel),
            nn.Conv2d(1, f1, (1, c.temporal_kernel), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, fd, (c.n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(fd),
            nn.ELU(),
            nn.AvgPool2d((1, c.pool1)),
            nn.Dropout(c.dropout),
        )
        self.separable = nn.Sequential(
            _same_pad(c.separable_kernel),
            nn.Conv2d(fd, fd, (1, c.separable_kernel), groups=fd, bias=False),
            nn.Conv2d(fd, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, c.pool2)),
            nn.Dropout(c.dropout),
        )
        self.classifier = nn.Linear(f2 * (c.n_samples // (c.pool1 * c.pool2)), c.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (1, self.config.n_channels, self.config.n_samples)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"EEGCNN expects input of shape (batch, {', '.join(map(str, expected))}), got {tuple(x.shape)}")
        x = self.separable(self.spatial(self.temporal(x)))
        return self.classifier(torch.flatten(x, 1))


def build_cnn(config: CNNConfig, seed: int) -> EEGCNN:
    """Construct with deterministic weight initialization."""
    torch.manual_seed(seed)
    return EEGCNN(config)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_summary(model: EEGCNN) -> list[dict[str, Any]]:
    """Output shape and parameter count of each top-level stage for a single input."""
    c = model.config
    x = torch.zeros(1, 1, c.n_channels, c.n_samples)
    rows = [{"stage": "input", "output_shape": list(x.shape), "parameters": 0}]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for name in ("temporal", "spatial", "separable"):
            stage = getattr(model, name)
            x = stage(x)
            rows.append({"stage": name, "output_shape": list(x.shape), "parameters": parameter_count(stage)})
        x = model.classifier(torch.flatten(x, 1))
        rows.append({"stage": "classifier", "output_shape": list(x.shape), "parameters": parameter_count(model.classifier)})
    model.train(was_training)
    return rows
