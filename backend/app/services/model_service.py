from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.ml.labels import LabelConfig, LabelStrategy
from app.ml.registry import ModelNotAvailableError, ModelRegistry
from app.models import MLModel
from app.preprocessing.eeg_preprocessor import EEGPreprocessor, PreprocessingConfig, SignalValidationError
from app.preprocessing.features import FeatureExtractor
from app.repositories import MetricRepository, ModelRepository

logger = get_logger("model_service")

_METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc")


@dataclass
class SignalPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    segments_used: int
    segments_rejected: int


class ModelService:
    """Holds the active model in memory. Loaded once at startup and after each training run."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ModelRegistry(settings.model_dir)
        self._lock = threading.RLock()
        self._model: Any = None
        self._meta: dict[str, Any] | None = None
        self._preprocessor: EEGPreprocessor | None = None
        self._extractor: FeatureExtractor | None = None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self._meta

    def load_active(self) -> bool:
        with self._lock:
            try:
                model, meta = self.registry.load()
            except ModelNotAvailableError as exc:
                self._model = self._meta = self._preprocessor = self._extractor = None
                logger.warning("No model loaded", extra={"reason": str(exc)})
                return False
            pre_cfg = PreprocessingConfig(**meta["preprocessing"])
            extractor: FeatureExtractor | None = None
            if input_type(meta) == "features":
                extractor = FeatureExtractor(sampling_rate=pre_cfg.sampling_rate)
                if extractor.feature_names != meta["feature_schema"]["names"]:
                    logger.error("Feature schema mismatch; refusing to load model", extra={"version": meta["version"]})
                    self._model = self._meta = None
                    return False
            self._model, self._meta = model, meta
            self._preprocessor, self._extractor = EEGPreprocessor(pre_cfg), extractor
            logger.info("Model loaded", extra={"version": meta["version"], "algorithm": meta["algorithm"],
                                               "framework": self.registry.framework_of(meta)})
            return True

    def require(self) -> tuple[Any, dict[str, Any], EEGPreprocessor, FeatureExtractor | None]:
        with self._lock:
            if self._model is None or self._meta is None or self._preprocessor is None:
                raise ModelNotAvailableError("No trained model available. Train a model first.")
            return self._model, self._meta, self._preprocessor, self._extractor

    def label_config(self) -> LabelConfig | None:
        if not self._meta:
            return None
        label = self._meta["label"]
        return LabelConfig(LabelStrategy(label["strategy"]), label["valence_threshold"], label["arousal_threshold"])

    def predict_signal(self, eeg: np.ndarray | list, includes_baseline: bool) -> SignalPrediction:
        model, _, preprocessor, extractor = self.require()
        result = preprocessor.preprocess(eeg, has_baseline=includes_baseline)
        if result.segments.shape[0] == 0:
            raise SignalValidationError("All segments were rejected as artifact-contaminated")
        model_input = result.segments if extractor is None else extractor.extract(result.segments)
        proba = model.predict_proba(model_input).mean(axis=0)
        classes = [str(c) for c in model.classes_]
        best = int(np.argmax(proba))
        return SignalPrediction(
            label=classes[best],
            confidence=float(proba[best]),
            probabilities={c: float(p) for c, p in zip(classes, proba, strict=True)},
            segments_used=int(result.segments.shape[0]),
            segments_rejected=len(result.rejected_segment_indices),
        )

    def sync_to_db(self, db: Session) -> MLModel | None:
        """Register the active model and its metrics in the database (idempotent)."""
        if not self._meta:
            return None
        return _sync_metadata(db, self._meta, active=True)

    def sync_registry_to_db(self, db: Session) -> list[MLModel]:
        """Register every saved model version; only the active one is flagged active."""
        active = self.registry.active_version()
        records = [_sync_metadata(db, meta, active=False) for meta in self.registry.list_versions() if meta["version"] != active]
        current = self.sync_to_db(db)
        return records + ([current] if current else [])


def input_type(meta: dict[str, Any]) -> str:
    """'features' for Review 2 feature models (default), 'segments' for the CNN."""
    return meta.get("input", {}).get("type", "features")


def input_requirements(meta: dict[str, Any]) -> dict[str, Any]:
    pre = meta["preprocessing"]
    req: dict[str, Any] = {
        "api_input": "raw EEG [channels][samples] or a DEAP reference",
        "channels": pre["n_channels"],
        "channel_order": "DEAP (Geneva) order",
        "sampling_rate_hz": pre["sampling_rate"],
        "minimum_seconds": pre["segment_seconds"],
        "segment_seconds": pre["segment_seconds"],
    }
    if input_type(meta) == "segments":
        req["model_input"] = f"preprocessed segments, tensor {meta['input']['shape']}, per-channel z-scored"
    else:
        req["model_input"] = f"{meta['feature_schema']['count']} engineered features per segment"
    return req


def _sync_metadata(db: Session, meta: dict[str, Any], active: bool) -> MLModel:
    repo = ModelRepository(db)
    record = repo.upsert_from_metadata(meta)
    if active:
        repo.set_active(record)
    rows: list[tuple[str, str, str, float | None]] = []
    chosen_val = meta["validation"][meta["algorithm"]]
    for split, block in (("validation", chosen_val), ("test", meta["test"]), ("test_reference", meta["test_reference_majority"])):
        for level in ("segment", "trial"):
            if not block.get(level):
                continue
            for name in _METRIC_NAMES:
                rows.append((split, level, name, block[level].get(name)))
    cv = meta.get("cross_validation")
    if cv:
        for name in ("segment_accuracy_mean", "segment_accuracy_std", "segment_f1_macro_mean", "segment_f1_macro_std"):
            rows.append(("cross_validation", "segment", name, cv[name]))
    MetricRepository(db).replace_for_model(record.id, rows)
    return record
