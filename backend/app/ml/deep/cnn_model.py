from __future__ import annotations

import torch
import torch.nn as nn


class EEGCNN(nn.Module):
    """A simple CNN model for EEG-derived feature vectors."""

    def __init__(self, input_dim: int = 9, num_classes: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
