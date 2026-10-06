"""Deep-learning data representation for DEAP.

Each example is one preprocessed 4 s EEG segment: a float32 array of shape (32 channels, 512 samples)
produced by the same EEGPreprocessor as the Review 2 feature baseline (baseline removal, 4-45 Hz
band-pass, robust clipping, segment rejection). Rows, labels and participant IDs therefore match the
feature table row for row, so the CNN and the feature models share one participant-aware split.

Tensor fed to the CNN: (batch, 1, 32, 512), i.e. a single-"image" plane with EEG channels as rows
and time as columns. Labels are integer codes of the sorted class names (same order as the
baseline's LabelEncoder). Normalization: per-channel mean/std computed on training participants only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.core.logging import get_logger
from app.ml.dataset import DEAPDataset
from app.ml.feature_table import dataset_fingerprint
from app.ml.labels import LabelConfig, make_labels
from app.preprocessing.eeg_preprocessor import EEGPreprocessor

logger = get_logger("deep.data")

SEGMENT_CACHE_VERSION = "seg-v1"


@dataclass
class SegmentTable:
    """Preprocessed segments (rows, channels, samples) plus per-row metadata."""

    X: np.ndarray
    ratings: np.ndarray
    participant: np.ndarray
    trial: np.ndarray
    segment: np.ndarray
    fingerprint: str
    rejected_segments: int

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.X.shape[1]), int(self.X.shape[2])

    def labels(self, config: LabelConfig) -> np.ndarray:
        return make_labels(self.ratings, config)


def build_segment_table(dataset: DEAPDataset, preprocessor: EEGPreprocessor, cache_dir: Path | None = None) -> SegmentTable:
    fingerprint = dataset_fingerprint(dataset, preprocessor)
    data_path = cache_dir / f"segments_{fingerprint}_{SEGMENT_CACHE_VERSION}.npy" if cache_dir else None
    meta_path = cache_dir / f"segments_{fingerprint}_{SEGMENT_CACHE_VERSION}_meta.npz" if cache_dir else None

    if data_path and meta_path and data_path.exists() and meta_path.exists():
        logger.info("Loading cached segment table", extra={"path": str(data_path)})
        meta = np.load(meta_path, allow_pickle=False)
        return SegmentTable(
            X=np.load(data_path, mmap_mode="r"),
            ratings=meta["ratings"], participant=meta["participant"], trial=meta["trial"], segment=meta["segment"],
            fingerprint=fingerprint, rejected_segments=int(meta["rejected_segments"]),
        )

    chunks: list[np.ndarray] = []
    ratings, participant, trial, segment = [], [], [], []
    rejected = 0
    for record in dataset.iter_participants():
        logger.info("Segmenting participant", extra={"participant": record.participant_id})
        for t in range(record.n_trials):
            result = preprocessor.preprocess(record.eeg[t], has_baseline=True)
            rejected += len(result.rejected_segment_indices)
            n = result.segments.shape[0]
            if n == 0:
                continue
            chunks.append(result.segments.astype(np.float32))
            ratings.append(np.repeat(record.ratings[t][None, :], n, axis=0))
            participant.extend([record.participant_id] * n)
            trial.extend([t] * n)
            segment.extend(result.kept_segment_indices)

    table = SegmentTable(
        X=np.concatenate(chunks),
        ratings=np.concatenate(ratings),
        participant=np.array(participant),
        trial=np.array(trial, dtype=np.int32),
        segment=np.array(segment, dtype=np.int32),
        fingerprint=fingerprint,
        rejected_segments=rejected,
    )
    if data_path and meta_path:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(data_path, table.X)
        np.savez(meta_path, ratings=table.ratings, participant=table.participant, trial=table.trial,
                 segment=table.segment, rejected_segments=rejected)
        logger.info("Saved segment cache", extra={"path": str(data_path), "rows": int(table.X.shape[0])})
        table.X = np.load(data_path, mmap_mode="r")
    return table


@dataclass(frozen=True)
class ChannelNormalizer:
    """Per-channel standardization; statistics come only from the rows it was fitted on."""

    mean: np.ndarray  # (channels,)
    std: np.ndarray  # (channels,)

    @classmethod
    def fit(cls, X: np.ndarray, rows: np.ndarray, chunk: int = 512) -> ChannelNormalizer:
        n_ch = X.shape[1]
        total = np.zeros(n_ch, dtype=np.float64)
        total_sq = np.zeros(n_ch, dtype=np.float64)
        count = 0
        for start in range(0, len(rows), chunk):
            block = np.asarray(X[np.sort(rows[start : start + chunk])], dtype=np.float64)
            total += block.sum(axis=(0, 2))
            total_sq += (block**2).sum(axis=(0, 2))
            count += block.shape[0] * block.shape[2]
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-12))
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean[:, None]) / self.std[:, None]).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"method": "per-channel z-score (training participants)", "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelNormalizer:
        return cls(np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32))


def encode_labels(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[str(c)] for c in labels], dtype=np.int64)


class EEGSegmentDataset(Dataset):
    """Yields (tensor (1, channels, samples), label code) for the given row indices."""

    def __init__(self, X: np.ndarray, rows: np.ndarray, label_codes: np.ndarray, normalizer: ChannelNormalizer) -> None:
        self.X = X
        self.rows = np.asarray(rows)
        self.label_codes = np.asarray(label_codes, dtype=np.int64)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.rows[i])
        x = self.normalizer.apply(self.X[row])
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(self.label_codes[row])


def make_loader(ds: EEGSegmentDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def describe(table: SegmentTable) -> str:
    return json.dumps({"rows": int(table.X.shape[0]), "channels": table.shape[0], "samples": table.shape[1]})
