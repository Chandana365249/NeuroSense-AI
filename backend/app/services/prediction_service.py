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

logger = get_logger("prediction")


def to_prediction_out(item: Prediction) -> PredictionOut:
    model = item.model
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
        model=ModelRef(version=model.version, algorithm=model.algorithm, label_strategy=model.label_strategy) if model else None,
        disclaimer=DISCLAIMER,
    )


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
        self.audit.add("prediction.create", user.id, "prediction", str(item.id), {"source": source}, ip)
        logger.info("Prediction stored", extra={"prediction_id": item.id, "user_id": user.id, "source": source,
                                                "label": result.label, "segments": result.segments_used})
        self.db.refresh(item)
        return to_prediction_out(item)
