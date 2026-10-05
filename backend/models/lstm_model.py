from __future__ import annotations

import torch
import torch.nn as nn


class EEGLSTM(nn.Module):
    """A small LSTM model for sequence classification."""

    def __init__(self, input_dim: int = 9, hidden_dim: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
