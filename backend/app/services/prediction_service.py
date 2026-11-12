from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import InvalidInputError, NotFoundError, ServiceUnavailableError
from app.core.logging import get_logger
from app.ml.dataset import RATING_NAMES, DatasetError, DEAPDataset
from app.ml.labels import make_labels
from app.ml.registry import ModelNotAvailableError
from app.ml.training import DISCLAIMER
from app.models import Prediction, User
from app.preprocessing.eeg_preprocessor import SignalValidationError
from app.repositories import AuditRepository, ModelRepository, PredictionRepository, SessionRepository
from app.schemas.schemas import ModelRef, PredictionOut, PredictionRequest
from app.services.model_service import ModelService
from app.services.upload_store import save_upload

logger = get_logger("prediction")

FACED_SOURCE = "faced_reference"


def _model_ref(item: Prediction) -> ModelRef | None:
    """DEAP records point at the model registry. FACED models are never registered, so their record carries its own."""
    if item.model:
        return ModelRef(version=item.model.version, algorithm=item.model.algorithm, label_strategy=item.model.label_strategy)
    ref = item.source_reference or {}
    if item.input_source == FACED_SOURCE and ref.get("model_version"):
        return ModelRef(version=ref["model_version"], algorithm=ref.get("algorithm", "unknown"), label_strategy=ref.get("label_rule", "unknown"))
    return None


def to_prediction_out(item: Prediction) -> PredictionOut:
    return PredictionOut(
        prediction_id=item.id,
        session_id=item.session_id,
        created_at=item.created_at,
        prediction=item.predicted_label,
        confidence=item.confidence,
        probabilities=item.probabilities,
        segments_used=item.segments_used,
        segments_rejected=item.segments_rejected,
        input_source=item.input_source,
        source_reference=item.source_reference,
        model=_model_ref(item),
        disclaimer=DISCLAIMER,
    )


def store_faced_prediction(db: Session, user: User, result: dict[str, Any], session_id: int | None, ip: str | None) -> Prediction:
    """Save a FACED result in the shared history. model_id stays empty: the FACED model is not in the DEAP registry."""
    if session_id is not None and not SessionRepository(db).get_for_user(session_id, user.id):
        raise NotFoundError(f"Session {session_id} not found")
    p_high = result["probability_high_valence"]
    item = PredictionRepository(db).create(
        user_id=user.id,
        session_id=session_id,
        model_id=None,
        predicted_label=result["predicted_label"],
        confidence=max(p_high, 1 - p_high),
        probabilities={"high_valence": p_high, "low_valence": 1 - p_high},
        segments_used=result["segments"],
        segments_rejected=result["segments_rejected"],
        input_source=FACED_SOURCE,
        source_reference=result["provenance"],
    )
    AuditRepository(db).add("prediction.create", user.id, "prediction", str(item.id), {"source": FACED_SOURCE}, ip)
    logger.info("Prediction stored", extra={"prediction_id": item.id, "user_id": user.id, "source": FACED_SOURCE,
                                            "label": item.predicted_label, "segments": item.segments_used})
    db.flush()
    db.refresh(item)
    return item


class PredictionService:
    def __init__(self, db: Session, settings: Settings, model_service: ModelService, dataset: DEAPDataset) -> None:
        self.db = db
        self.settings = settings
        self.model_service = model_service
        self.dataset = dataset
        self.predictions = PredictionRepository(db)
        self.sessions = SessionRepository(db)
        self.models = ModelRepository(db)
        self.audit = AuditRepository(db)

    def predict(self, user: User, request: PredictionRequest, ip: str | None) -> PredictionOut:
        try:
            # One consistent snapshot for the whole request: an activation during this request cannot mix the
            # output of one model with another model's version, labels or split.
            snap = self.model_service.snapshot()
        except ModelNotAvailableError as exc:
            raise ServiceUnavailableError(str(exc)) from exc
        meta = snap.meta

        if request.session_id is not None and not self.sessions.get_for_user(request.session_id, user.id):
            raise NotFoundError(f"Session {request.session_id} not found")

        source_reference: dict[str, Any] | None = None
        if request.eeg is not None:
            expected_sr = meta["preprocessing"]["sampling_rate"]
            if request.eeg.sampling_rate != expected_sr:
                raise InvalidInputError(f"sampling_rate must be {expected_sr} Hz (resampling is not supported yet)")
            signal = np.asarray(request.eeg.signal, dtype=np.float64)
            includes_baseline = request.eeg.includes_baseline
            source = "upload"
            source_reference = {"channels": int(signal.shape[0]), "samples": int(signal.shape[1]),
                                "sampling_rate": request.eeg.sampling_rate}
            if request.eeg.synthetic_test_signal:
                source_reference["synthetic_test_signal"] = True
        else:
            ref = request.deap_reference
            assert ref is not None
            if ref.participant_id not in self.settings.development_participants:
                # Only authorized development participants may be read; frozen test participants never are.
                raise NotFoundError(f"DEAP participant {ref.participant_id} is not available on this server")
            try:
                signal, ratings = self.dataset.load_trial(ref.participant_id, ref.trial_index)
            except DatasetError as exc:
                raise NotFoundError(str(exc)) from exc
            includes_baseline = True
            source = "deap_reference"
            label_cfg = snap.label_config()
            split = meta["split"]
            membership = next((p for p in ("train", "validation", "test") if ref.participant_id in split[p]), "not_used")
            source_reference = {
                "participant_id": ref.participant_id,
                "trial_index": ref.trial_index,
                "self_reported_ratings": {n: round(float(v), 2) for n, v in zip(RATING_NAMES, ratings, strict=True)},
                "self_reported_label": str(make_labels(ratings[None, :], label_cfg)[0]) if label_cfg else None,
                "participant_split": membership,
            }

        try:
            result = snap.predict_signal(signal, includes_baseline)
        except SignalValidationError as exc:
            raise InvalidInputError(str(exc)) from exc

        model_row = self.models.get_by_version(snap.version) or self.model_service.ensure_registered(self.db, snap)
        item = self.predictions.create(
            user_id=user.id,
            session_id=request.session_id,
            model_id=model_row.id if model_row else None,
            predicted_label=result.label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            segments_used=result.segments_used,
            segments_rejected=result.segments_rejected,
            input_source=source,
            source_reference=source_reference,
        )
        if source == "upload" and not (source_reference or {}).get("synthetic_test_signal"):
            # Kept privately so the result can be explained later without re-sending the signal.
            save_upload(self.settings, user.id, item.id, signal, request.eeg.sampling_rate, includes_baseline)
        self.audit.add("prediction.create", user.id, "prediction", str(item.id), {"source": source}, ip)
        logger.info("Prediction stored", extra={"prediction_id": item.id, "user_id": user.id, "source": source,
                                                "label": result.label, "segments": result.segments_used})
        self.db.refresh(item)
        return to_prediction_out(item)
