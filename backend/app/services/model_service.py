from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import AppError, ConflictError, InvalidInputError, NotFoundError
from app.core.logging import get_logger
from app.ml.labels import LabelConfig, LabelStrategy
from app.ml.registry import ModelNotAvailableError, ModelRegistry
from app.models import MLModel
from app.preprocessing.eeg_preprocessor import EEGPreprocessor, PreprocessingConfig, SignalValidationError
from app.preprocessing.features import FeatureExtractor
from app.repositories import AuditRepository, MetricRepository, ModelRepository

logger = get_logger("model_service")

_METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc")


ACTIVATION_FAILED_MESSAGE = "Model activation failed; the previous model remains active. Details are in the server log."
ACTIVATION_RESTORE_FAILED_MESSAGE = ("Model activation failed and the previous state could not be fully restored. "
                                     "Check the server logs before serving predictions.")


class ModelValidationError(Exception):
    """A saved model is not compatible with this server (feature schema, classes or output shape)."""


class ActivationFailedError(AppError):
    status_code = 500
    code = "activation_failed"


@dataclass
class ActivationResult:
    version: str
    previous_version: str | None
    changed: bool


@dataclass
class SignalPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    segments_used: int
    segments_rejected: int


@dataclass(frozen=True)
class ModelSnapshot:
    """The active model together with its own metadata and preprocessing, captured at one instant.

    A request takes one snapshot and uses it for everything (version, labels, split, prediction), so an activation
    that happens meanwhile cannot mix two models. The service never mutates these objects; it only swaps them."""
    model: Any
    meta: dict[str, Any]
    preprocessor: EEGPreprocessor
    extractor: FeatureExtractor | None

    @property
    def version(self) -> str:
        return self.meta["version"]

    def label_config(self) -> LabelConfig:
        label = self.meta["label"]
        return LabelConfig(LabelStrategy(label["strategy"]), label["valence_threshold"], label["arousal_threshold"])

    def predict_signal(self, eeg: np.ndarray | list, includes_baseline: bool) -> SignalPrediction:
        result = self.preprocessor.preprocess(eeg, has_baseline=includes_baseline)
        if result.segments.shape[0] == 0:
            raise SignalValidationError("All segments were rejected as artifact-contaminated")
        model_input = result.segments if self.extractor is None else self.extractor.extract(result.segments)
        proba = self.model.predict_proba(model_input).mean(axis=0)
        classes = [str(c) for c in self.model.classes_]
        best = int(np.argmax(proba))
        return SignalPrediction(
            label=classes[best],
            confidence=float(proba[best]),
            probabilities={c: float(p) for c, p in zip(classes, proba, strict=True)},
            segments_used=int(result.segments.shape[0]),
            segments_rejected=len(result.rejected_segment_indices),
        )


class ModelService:
    """Holds the active model in memory. Loaded at startup and replaced only by an explicit activation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ModelRegistry(settings.model_dir)
        self._lock = threading.RLock()
        self._model: Any = None
        self._meta: dict[str, Any] | None = None
        self._preprocessor: EEGPreprocessor | None = None
        self._extractor: FeatureExtractor | None = None
        self._activation_lock = threading.Lock()
        self._activating = False

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
            try:
                preprocessor, extractor = _prepare(model, meta, smoke_test=False)
            except ModelValidationError:
                logger.error("Feature schema mismatch; refusing to load model", extra={"version": meta["version"]})
                self._model = self._meta = None
                return False
            self._model, self._meta = model, meta
            self._preprocessor, self._extractor = preprocessor, extractor
            logger.info("Model loaded", extra={"version": meta["version"], "algorithm": meta["algorithm"],
                                               "framework": self.registry.framework_of(meta)})
            return True

    @property
    def activation_in_progress(self) -> bool:
        return self._activating

    def activate_version(self, db: Session, version: str, user_id: int, reason: str, ip: str | None,
                         training_running: Callable[[], bool]) -> ActivationResult:
        """Make `version` the active model: validate it, then switch the pointer, the in-memory model and the
        database flag together, with an audit entry. On any failure the previous state is restored.

        Raises NotFoundError (unknown version), ConflictError (training or another activation running),
        InvalidInputError (invalid model, integrity or schema failure) or ActivationFailedError."""
        if not self.registry.has_version(version):
            raise NotFoundError("Model version not found")
        if not self._activation_lock.acquire(blocking=False):
            raise ConflictError("Another model activation is in progress")
        self._activating = True
        try:
            if training_running():
                raise ConflictError("A training run is in progress; activate the model after it finishes")
            previous = self.registry.active_version()
            if self._already_active(db, version):
                logger.info("Model already active; nothing changed", extra={"version": version, "user_id": user_id})
                return ActivationResult(version=version, previous_version=previous, changed=False)
            try:
                model, meta = self.registry.load(version)
                preprocessor, extractor = _prepare(model, meta, smoke_test=True)
            except Exception as exc:  # integrity, schema, unreadable or incompatible artifacts
                logger.warning("Model failed activation checks", extra={"version": version}, exc_info=exc)
                raise InvalidInputError(_validation_message(exc)) from exc
            return self._switch(db, version, previous, model, meta, preprocessor, extractor, user_id, reason, ip)
        finally:
            self._activating = False
            self._activation_lock.release()

    def _already_active(self, db: Session, version: str) -> bool:
        if self.registry.active_version() != version or not self._meta or self._meta.get("version") != version:
            return False
        return [m.version for m in ModelRepository(db).list() if m.is_active] == [version]

    def _switch(self, db: Session, version: str, previous: str | None, model: Any, meta: dict[str, Any],
                preprocessor: EEGPreprocessor, extractor: FeatureExtractor | None, user_id: int, reason: str,
                ip: str | None) -> ActivationResult:
        # Order: database changes (flushed, not committed) -> atomic pointer write -> in-memory swap -> commit.
        # A failing step undoes the steps before it, so the previous model stays active everywhere.
        pointer_before = self.registry.pointer_bytes()
        try:
            _sync_metadata(db, meta, active=True)
            AuditRepository(db).add("model.activate", user_id, "model", version,
                                    {"previous_version": previous, "new_version": version, "administrator_id": user_id,
                                     "reason": reason}, ip)
            db.flush()
        except Exception as exc:
            db.rollback()
            logger.error("Model activation failed at the database step", extra={"version": version}, exc_info=exc)
            raise ActivationFailedError(ACTIVATION_FAILED_MESSAGE) from exc

        try:
            self.registry.activate(version)
        except Exception as exc:
            db.rollback()
            restored = self._restore_pointer(pointer_before)
            logger.error("Model activation failed writing the active pointer", extra={"version": version}, exc_info=exc)
            raise ActivationFailedError(ACTIVATION_FAILED_MESSAGE if restored else ACTIVATION_RESTORE_FAILED_MESSAGE) from exc

        with self._lock:
            before = (self._model, self._meta, self._preprocessor, self._extractor)
            self._model, self._meta, self._preprocessor, self._extractor = model, meta, preprocessor, extractor
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            with self._lock:
                self._model, self._meta, self._preprocessor, self._extractor = before
            restored = self._restore_pointer(pointer_before)
            logger.error("Model activation failed committing to the database", extra={"version": version}, exc_info=exc)
            raise ActivationFailedError(ACTIVATION_FAILED_MESSAGE if restored else ACTIVATION_RESTORE_FAILED_MESSAGE) from exc

        logger.info("Model activated", extra={"previous_version": previous, "version": version, "user_id": user_id})
        return ActivationResult(version=version, previous_version=previous, changed=True)

    def _restore_pointer(self, content: bytes | None) -> bool:
        if self.registry.pointer_bytes() == content:
            return True
        try:
            self.registry.restore_pointer(content)
            return True
        except Exception:
            logger.critical("Could not restore the active model pointer after a failed activation", exc_info=True)
            return False

    def snapshot(self) -> ModelSnapshot:
        """The active model and its matching metadata, read together under the lock (the lock is not held while
        the caller predicts). Raises ModelNotAvailableError if no model is loaded."""
        with self._lock:
            if self._model is None or self._meta is None or self._preprocessor is None:
                raise ModelNotAvailableError("No trained model available. Train a model first.")
            return ModelSnapshot(self._model, self._meta, self._preprocessor, self._extractor)

    def require(self) -> tuple[Any, dict[str, Any], EEGPreprocessor, FeatureExtractor | None]:
        snap = self.snapshot()
        return snap.model, snap.meta, snap.preprocessor, snap.extractor

    def label_config(self) -> LabelConfig | None:
        try:
            return self.snapshot().label_config()
        except ModelNotAvailableError:
            return None

    def predict_signal(self, eeg: np.ndarray | list, includes_baseline: bool) -> SignalPrediction:
        return self.snapshot().predict_signal(eeg, includes_baseline)

    def sync_to_db(self, db: Session) -> MLModel | None:
        """Register the active model and its metrics in the database (idempotent)."""
        if not self._meta:
            return None
        return _sync_metadata(db, self._meta, active=True)

    def ensure_registered(self, db: Session, snap: ModelSnapshot) -> MLModel:
        """The database row for the snapshot's model, registering it if missing. Only flagged active if it is still
        the active model (an activation may have happened since the snapshot was taken)."""
        with self._lock:
            still_active = self._meta is snap.meta
        return _sync_metadata(db, snap.meta, active=still_active)

    def register_candidate(self, db: Session, meta: dict[str, Any]) -> MLModel:
        """Register a newly trained version and its metrics without changing which model is active."""
        return _sync_metadata(db, meta, active=False)

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


def _prepare(model: Any, meta: dict[str, Any], smoke_test: bool) -> tuple[EEGPreprocessor, FeatureExtractor | None]:
    """Build the preprocessing for a loaded model and check it is compatible with this server.

    Raises ModelValidationError on a feature-schema mismatch. With smoke_test (before activation, not at startup) the
    class labels must match the metadata and one neutral input of the expected shape must give a valid probability row."""
    pre_cfg = PreprocessingConfig(**meta["preprocessing"])
    extractor: FeatureExtractor | None = None
    if input_type(meta) == "features":
        extractor = FeatureExtractor(sampling_rate=pre_cfg.sampling_rate)
        if extractor.feature_names != (meta.get("feature_schema") or {}).get("names"):
            raise ModelValidationError("feature schema mismatch")
    if smoke_test:
        classes = [str(c) for c in getattr(model, "classes_", [])]
        if not classes or classes != [str(c) for c in meta.get("class_labels", [])]:
            raise ModelValidationError("class labels do not match the metadata")
        if extractor is not None:
            sample = np.zeros((1, len(extractor.feature_names)))
        else:
            sample = np.zeros((1, pre_cfg.n_channels, pre_cfg.segment_samples), dtype=np.float32)
        proba = np.asarray(model.predict_proba(sample), dtype=float)
        if proba.shape != (1, len(classes)) or not np.all(np.isfinite(proba)) or abs(float(proba.sum()) - 1.0) > 1e-3:
            raise ModelValidationError("model output does not match the expected classes")
    return EEGPreprocessor(pre_cfg), extractor


def _validation_message(exc: Exception) -> str:
    """A safe client message for a failed activation check (details go to the server log only)."""
    text = str(exc)
    if "integrity" in text:
        return "The model file failed its integrity check; it was not activated."
    if "Model file missing" in text:
        return "The model file is missing; it was not activated."
    if "feature schema" in text:
        return "The model's feature schema is not compatible with this server; it was not activated."
    return "The model is not valid for this server; it was not activated."


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
