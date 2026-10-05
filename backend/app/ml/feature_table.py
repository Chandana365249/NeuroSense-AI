from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.ml.dataset import DEAPDataset
from app.ml.labels import LabelConfig, make_labels
from app.preprocessing.eeg_preprocessor import EEGPreprocessor
from app.preprocessing.features import FEATURE_VERSION, FeatureExtractor

logger = get_logger("features")


@dataclass
class FeatureTable:
    """One row per kept EEG segment. Ratings are kept so labels can be re-derived without reprocessing."""

    X: np.ndarray  # (rows, features)
    ratings: np.ndarray  # (rows, 4)
    participant: np.ndarray  # (rows,) participant id strings
    trial: np.ndarray  # (rows,) trial index within participant
    segment: np.ndarray  # (rows,) segment index within trial
    feature_names: list[str]
    fingerprint: str
    rejected_segments: int

    def labels(self, config: LabelConfig) -> np.ndarray:
        return make_labels(self.ratings, config)

    def subset(self, mask: np.ndarray) -> FeatureTable:
        return FeatureTable(
            self.X[mask], self.ratings[mask], self.participant[mask], self.trial[mask], self.segment[mask],
            self.feature_names, self.fingerprint, self.rejected_segments,
        )


def dataset_fingerprint(dataset: DEAPDataset, preprocessor: EEGPreprocessor) -> str:
    files = [(p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in dataset.discover()]
    payload = json.dumps(
        {"files": files, "preprocessing": asdict(preprocessor.config), "features": FEATURE_VERSION}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_feature_table(
    dataset: DEAPDataset,
    preprocessor: EEGPreprocessor,
    extractor: FeatureExtractor,
    cache_dir: Path | None = None,
) -> FeatureTable:
    fingerprint = dataset_fingerprint(dataset, preprocessor)
    cache_path = cache_dir / f"features_{fingerprint}.npz" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached feature table", extra={"path": str(cache_path)})
        z = np.load(cache_path, allow_pickle=False)
        return FeatureTable(
            z["X"], z["ratings"], z["participant"], z["trial"], z["segment"],
            [str(n) for n in z["feature_names"]], fingerprint, int(z["rejected_segments"]),
        )

    X, ratings, participant, trial, segment = [], [], [], [], []
    rejected = 0
    for record in dataset.iter_participants():
        logger.info("Extracting features", extra={"participant": record.participant_id, "trials": record.n_trials})
        for t in range(record.n_trials):
            result = preprocessor.preprocess(record.eeg[t], has_baseline=True)
            rejected += len(result.rejected_segment_indices)
            if result.segments.shape[0] == 0:
                continue
            feats = extractor.extract(result.segments)
            n = feats.shape[0]
            X.append(feats)
            ratings.append(np.repeat(record.ratings[t][None, :], n, axis=0))
            participant.extend([record.participant_id] * n)
            trial.extend([t] * n)
            segment.extend(result.kept_segment_indices)

    table = FeatureTable(
        X=np.concatenate(X).astype(np.float32),
        ratings=np.concatenate(ratings),
        participant=np.array(participant),
        trial=np.array(trial, dtype=np.int32),
        segment=np.array(segment, dtype=np.int32),
        feature_names=extractor.feature_names,
        fingerprint=fingerprint,
        rejected_segments=rejected,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, X=table.X, ratings=table.ratings, participant=table.participant, trial=table.trial,
            segment=table.segment, feature_names=np.array(table.feature_names), rejected_segments=rejected,
        )
        logger.info("Saved feature cache", extra={"path": str(cache_path), "rows": int(table.X.shape[0])})
    return table
