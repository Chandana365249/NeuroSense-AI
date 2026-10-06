"""Inference wrapper giving the CNN the same interface as the sklearn models (classes_, predict_proba)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.ml.deep.cnn_model import CNNConfig, EEGCNN
from app.ml.deep.data import ChannelNormalizer

ARTIFACT_FORMAT = "neurosense-cnn-v1"


class CNNClassifier:
    def __init__(self, model: EEGCNN, classes: list[str], normalizer: ChannelNormalizer) -> None:
        if len(classes) != model.config.n_classes:
            raise ValueError("Number of class labels does not match the network output size")
        self.model = model.eval()
        self.classes_ = np.asarray(classes)
        self.normalizer = normalizer

    @property
    def config(self) -> CNNConfig:
        return self.model.config

    def predict_proba(self, segments: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """segments: (n, channels, samples) preprocessed EEG -> (n, n_classes); columns follow classes_."""
        segments = np.asarray(segments, dtype=np.float32)
        expected = (self.config.n_channels, self.config.n_samples)
        if segments.ndim != 3 or segments.shape[1:] != expected:
            raise ValueError(f"CNN expects segments of shape (n, {expected[0]}, {expected[1]}), got {segments.shape}")
        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(segments), batch_size):
                x = torch.from_numpy(self.normalizer.apply(segments[start : start + batch_size])).unsqueeze(1)
                out.append(torch.softmax(self.model(x), dim=1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes_)), dtype=np.float32)

    def predict(self, segments: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(segments), axis=1)]

    def save(self, path: Path) -> None:
        torch.save(
            {
                "format": ARTIFACT_FORMAT,
                "config": self.config.to_dict(),
                "classes": [str(c) for c in self.classes_],
                "normalizer": {"mean": self.normalizer.mean.tolist(), "std": self.normalizer.std.tolist()},
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> CNNClassifier:
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"Unsupported CNN artifact format: {payload.get('format')}")
        model = EEGCNN(CNNConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return cls(model, payload["classes"], ChannelNormalizer.from_dict(payload["normalizer"]))
