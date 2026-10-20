from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ConflictError, ServiceUnavailableError
from app.core.logging import get_logger
from app.database.session import Database
from app.ml.dataset import DatasetError
from app.ml.factory import make_dataset
from app.ml.training import BaselineTrainer
from app.repositories import AuditRepository, ReportRepository
from app.services.model_service import ModelService

logger = get_logger("training_service")

# Shown in the status response after a failed run. Exception details can contain paths or database errors, so they
# go to the server log only.
TRAINING_FAILED_MESSAGE = "Training failed. Check the server logs or contact an administrator."


class TrainingService:
    """Runs one training job at a time in a background thread and tracks its status in memory."""

    def __init__(self, settings: Settings, db: Database, model_service: ModelService) -> None:
        self.settings = settings
        self.db = db
        self.model_service = model_service
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {"state": "idle", "progress": 0.0}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _update(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)

    def is_running(self) -> bool:
        with self._lock:
            return self._status["state"] == "running"

    def start(self, user_id: int, run_cross_validation: bool) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] == "running":
                raise ConflictError("A training run is already in progress")
            if self.model_service.activation_in_progress:
                raise ConflictError("A model activation is in progress; start training after it finishes")
            try:
                make_dataset(self.settings).discover()
            except DatasetError as exc:
                raise ServiceUnavailableError(f"Dataset not available: {exc}") from exc
            self._status = {"state": "running", "stage": "Queued", "progress": 0.0,
                            "started_at": datetime.now(UTC), "started_by": user_id}
            self._thread = threading.Thread(target=self._run, args=(user_id, run_cross_validation), daemon=True,
                                            name="neurosense-training")
            self._thread.start()
            return dict(self._status)

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _run(self, user_id: int, run_cross_validation: bool) -> None:
        logger.info("Training started", extra={"user_id": user_id})
        try:
            # The new version is saved as an inactive candidate: training never changes the active model pointer,
            # the model in memory or the database active flag. Activation is a separate administrator action.
            trainer = BaselineTrainer(self.settings, run_cross_validation=run_cross_validation, activate=False)
            result = trainer.run(progress=lambda stage, frac: self._update(stage=stage, progress=round(frac, 3)))
            with self.db.session() as session:
                meta = result.metadata
                record = self.model_service.register_candidate(session, meta)
                ReportRepository(session).create(
                    "training",
                    f"Training report {result.version}",
                    {k: meta[k] for k in ("version", "algorithm", "label", "split", "dataset", "validation", "test",
                                          "test_reference_majority", "cross_validation", "training_seconds", "disclaimer")},
                    user_id,
                    record.id,
                )
                AuditRepository(session).add("training.completed", user_id, "model", result.version, {"activated": False})
                session.commit()
            self._update(state="succeeded", stage="Done", progress=1.0, finished_at=datetime.now(UTC),
                         model_version=result.version)
            logger.info("Training completed", extra={"version": result.version})
        except Exception as exc:
            logger.error("Training failed", exc_info=exc)  # full detail: server log only
            at_failure = self.status()
            self._update(state="failed", error=TRAINING_FAILED_MESSAGE, finished_at=datetime.now(UTC))
            # The audit record gets a safe description and status only, never the exception text (which can contain
            # paths or database errors). Entries written before this change may still hold that text.
            details = {"event": "Training run failed; details are in the server log", "stage": at_failure.get("stage"),
                       "progress": at_failure.get("progress"), "error_type": type(exc).__name__}
            try:
                with self.db.session() as session:
                    AuditRepository(session).add("training.failed", user_id, "model", None, details)
                    session.commit()
            except Exception:
                logger.exception("Could not write training failure audit log")
