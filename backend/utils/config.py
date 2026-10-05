from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Application configuration for training and serving."""

    data_dir: str = os.environ.get("NEUROSENSE_DATA_DIR", "Dataset")
    model_dir: str = os.environ.get("NEUROSENSE_MODEL_DIR", "outputs/models")
    graph_dir: str = os.environ.get("NEUROSENSE_GRAPH_DIR", "outputs/graphs")
    prediction_dir: str = os.environ.get("NEUROSENSE_PREDICTION_DIR", "outputs/predictions")
    sampling_rate: int = int(os.environ.get("NEUROSENSE_SAMPLING_RATE", "128"))
    window_size: int = int(os.environ.get("NEUROSENSE_WINDOW_SIZE", "128"))


CONFIG = AppConfig()
