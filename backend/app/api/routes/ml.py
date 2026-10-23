from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request, status
from sqlalchemy.engine import make_url

from app.api.deps import AdminUser, ClientIP, DatasetDep, DbSession, ModelServiceDep, SettingsDep, TrainingServiceDep
from app.core.config import Settings
from app.core.exceptions import InvalidInputError, ServiceUnavailableError
from app.core.logging import get_logger
from app.ml.dataset import DatasetError, DEAPDataset
from app.ml.participants import is_frozen
from app.ml.registry import ModelNotAvailableError
from app.repositories import AuditRepository, ModelRepository
from app.schemas.schemas import (DatasetSummaryOut, DemoParticipant, DemoTrialsOut, DemoUnavailable, GlobalImportanceOut, MetricsDatasetOut, MetricsOut,
                                 ModelActivateRequest,
                                 ModelActivationOut, ModelInfoOut, ModelSummary, ModelSyncOut, ParticipantStructure,
                                 TrainingStatusOut, TrainRequest)
from app.services.explainability import global_importance
from app.services.model_service import input_requirements
from app.services.public_metadata import (public_dataset_facts, public_evaluation, public_protocol,
                                          training_provenance)

router = APIRouter(tags=["model"])
logger = get_logger("api")

NOT_TRAINED = "No trained model available. Train a model first."


@router.post("/train", response_model=TrainingStatusOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Start baseline training on the DEAP dataset (administrator only, runs in background)")
def train(payload: TrainRequest, user: AdminUser, training: TrainingServiceDep, db: DbSession, request: Request) -> TrainingStatusOut:
    state = training.start(user.id, payload.run_cross_validation)
    AuditRepository(db).add("training.start", user.id, "model", None, {"cv": payload.run_cross_validation},
                            request.client.host if request.client else None)
    return TrainingStatusOut(**state)


@router.get("/training/status", response_model=TrainingStatusOut, summary="Training job status (administrator only)")
def training_status(_: AdminUser, training: TrainingServiceDep) -> TrainingStatusOut:
    return TrainingStatusOut(**training.status())


@router.get("/model-info", response_model=ModelInfoOut, summary="Active model metadata")
def model_info(model_service: ModelServiceDep) -> ModelInfoOut:
    meta = model_service.metadata
    if not meta:
        return ModelInfoOut(available=False, message=NOT_TRAINED)
    schema = meta["feature_schema"]
    return ModelInfoOut(
        available=True,
        version=meta["version"],
        name=meta["name"],
        algorithm=meta["algorithm"],
        algorithm_description=meta["algorithm_description"],
        model_type=meta.get("model_type", meta["algorithm"]),
        framework=meta.get("framework", "sklearn"),
        framework_version=meta.get("framework_version"),
        input_requirements=input_requirements(meta),
        architecture={k: v for k, v in meta["architecture"].items() if k != "layers"} if meta.get("architecture") else None,
        trained_at=meta["trained_at"],
        label=meta["label"],
        class_labels=meta["class_labels"],
        feature_schema={"version": schema["version"], "count": schema["count"], "example_names": schema["names"][:12]},
        preprocessing=meta["preprocessing"],
        split=meta["split"],
        test_metrics={level: {k: v for k, v in m.items() if k not in ("confusion_matrix",)} for level, m in meta["test"].items()},
        provenance=training_provenance(meta),
        disclaimer=meta["disclaimer"],
    )


@router.get("/explain/global", response_model=GlobalImportanceOut, tags=["explainability"],
            summary="Global feature importance of the active Random Forest (model-level, not a per-prediction explanation)")
def explain_global(model_service: ModelServiceDep, top: int = Query(20, ge=1, le=100)) -> GlobalImportanceOut:
    try:
        snap = model_service.snapshot()  # one consistent model + metadata
    except ModelNotAvailableError:
        snap = None
    return GlobalImportanceOut(**global_importance(snap, top))


@router.get("/models", response_model=list[ModelSummary], summary="All registered model versions (read-only)")
def list_models(db: DbSession, model_service: ModelServiceDep) -> list[ModelSummary]:
    # Read-only: the database registry is synchronised at startup, after training, and by POST /models/sync.
    registry_meta = {m["version"]: m for m in model_service.registry.list_versions()}
    summaries = []
    for m in ModelRepository(db).list():
        meta = registry_meta.get(m.version, {})
        test_segment = (meta.get("test") or {}).get("segment") or {}
        summaries.append(ModelSummary(
            version=m.version, name=m.name, algorithm=m.algorithm, label_strategy=m.label_strategy,
            class_labels=m.class_labels, feature_count=m.feature_count, trained_at=m.trained_at, is_active=m.is_active,
            model_type=meta.get("model_type", m.algorithm) if meta else None,
            framework=meta.get("framework", "sklearn") if meta else None,
            test_metrics={k: test_segment.get(k) for k in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")} if test_segment else None,
            evaluation=public_evaluation(meta.get("evaluation"), include_folds=False),
            provenance=training_provenance(meta) if meta else None,
        ))
    return summaries


@router.get("/metrics", response_model=MetricsOut, summary="Evaluation metrics of the active model (real held-out results)")
def metrics(model_service: ModelServiceDep, request: Request) -> MetricsOut:
    meta = model_service.metadata
    if not meta:
        return MetricsOut(available=False, message=NOT_TRAINED)
    prefix = request.app.state.settings.api_prefix
    graphs = [f"{prefix}/graphs/{Path(p).name}" for p in ((meta.get("artifacts") or {}).get("graphs") or {}).values() if p]
    return MetricsOut(
        available=True,
        model_version=meta["version"],
        label=meta["label"],
        split=meta["split"],
        validation=meta["validation"],
        test=meta["test"],
        test_reference_majority=meta["test_reference_majority"],
        cross_validation=meta.get("cross_validation"),
        evaluation=public_evaluation(meta.get("evaluation")),
        protocol=public_protocol(meta.get("protocol")),
        dataset=MetricsDatasetOut(**public_dataset_facts(meta.get("dataset"))),
        graphs=graphs,
    )



@router.post("/models/sync", response_model=ModelSyncOut, tags=["admin"],
             summary="Synchronise saved model versions into the database registry (administrator only)")
def sync_models(user: AdminUser, db: DbSession, model_service: ModelServiceDep, ip: ClientIP) -> ModelSyncOut:
    # Same idempotent operation as at startup. The registry's active pointer decides the active model; nothing is
    # trained, activated or deleted here.
    records = model_service.sync_registry_to_db(db)
    active = model_service.registry.active_version()
    AuditRepository(db).add("models.sync", user.id, "model", active, {"synced": len(records)}, ip)
    return ModelSyncOut(synced=len(records), active_version=active)


@router.post("/models/{version}/activate", response_model=ModelActivationOut, tags=["admin"],
             summary="Make a registered model version the active model (administrator only, audited)")
def activate_model(version: str, payload: ModelActivateRequest, user: AdminUser, db: DbSession,
                   model_service: ModelServiceDep, training: TrainingServiceDep, settings: SettingsDep,
                   ip: ClientIP) -> ModelActivationOut:
    # The model is validated (integrity, feature schema, a neutral test prediction) before anything is switched;
    # pointer, in-memory model, database flag and audit entry change together or not at all.
    if payload.confirm_version != version:
        raise InvalidInputError("confirm_version must repeat the version being activated")
    if _contains_configured_secret(payload.reason, settings):
        raise InvalidInputError("Reason appears to contain a credential or secret; remove it and try again")
    result = model_service.activate_version(db, version, user.id, payload.reason, ip, training.is_running)
    return ModelActivationOut(version=result.version, previous_version=result.previous_version, changed=result.changed,
                              active_version=model_service.registry.active_version() or result.version)


def _contains_configured_secret(text: str, settings: Settings) -> bool:
    secrets = [settings.jwt_secret_key.get_secret_value()]
    try:
        secrets.append(make_url(settings.database_url).password or "")
    except Exception:  # an unparsable URL has no password to compare against
        pass
    return any(len(s) >= 6 and s in text for s in secrets)


def _dataset_summary(dataset: DEAPDataset, request: Request) -> dict[str, Any]:
    """Full structural summary of the local files, cached per process. Raises DatasetError if unavailable."""
    cached = getattr(request.app.state, "dataset_summary", None)
    if cached is None:
        cached = dataset.summarize().to_dict()
        request.app.state.dataset_summary = cached
    return cached


@router.get("/dataset/summary", response_model=DatasetSummaryOut, tags=["dataset"],
            summary="Aggregate structure of the working DEAP dataset (no paths, file names or rating statistics)")
def dataset_summary(dataset: DatasetDep, request: Request, settings: SettingsDep) -> DatasetSummaryOut:
    try:
        full = _dataset_summary(dataset, request)
    except DatasetError as exc:
        logger.warning("Dataset summary unavailable", extra={"error": str(exc)})
        raise ServiceUnavailableError("Dataset not available on this server") from exc
    return DatasetSummaryOut(
        sampling_rate_hz=full["sampling_rate_hz"],
        baseline_seconds=full["baseline_seconds"],
        participants_found=full["participants_found"],
        participants_valid=full["participants_valid"],
        total_trials=full["total_trials"],
        eeg_channels_used=full["eeg_channels_used"],
        eeg_channel_names=full["eeg_channel_names"],
        participants=[ParticipantStructure(participant_id=f["participant_id"], readable=f["ok"], trials=f["trials"],
                                           channels=f["channels"], samples=f["samples"]) for f in full["files"]],
        authorized_participants=list(settings.development_participants),
    )


@router.get("/admin/dataset/summary", tags=["admin"],
            summary="Full dataset diagnostics, including location, files and rating statistics (administrator only)")
def admin_dataset_summary(_: AdminUser, dataset: DatasetDep, request: Request) -> dict[str, Any]:
    try:
        return _dataset_summary(dataset, request)
    except DatasetError as exc:
        raise ServiceUnavailableError(f"Dataset not available: {exc}") from exc


@router.get("/dataset/demo-trials", tags=["dataset"], response_model=DemoTrialsOut,
            summary="Authorized DEAP participants readable on this server, their trials, and each one's role for the active model")
def demo_trials(model_service: ModelServiceDep, dataset: DatasetDep, settings: SettingsDep,
                request: Request) -> DemoTrialsOut:
    """Which input data can be analysed (authorized, never frozen, readable) is independent of the active model's
    evaluation split. Each participant's role for that model comes only from its recorded provenance."""
    meta = model_service.metadata
    provenance = training_provenance(meta)
    authorized = [p for p in settings.development_participants if not is_frozen(p)]
    try:
        files = {f["participant_id"]: f for f in _dataset_summary(dataset, request)["files"]}
        dataset_available = True
    except DatasetError as exc:
        logger.warning("Dataset unavailable for demo trials", extra={"error": str(exc)})
        files, dataset_available = {}, False

    def role(pid: str) -> str:
        if not provenance or not provenance["recorded"]:
            return "unknown"
        if pid in provenance["selection_participants"]:
            return "selection"
        if pid in provenance["fit_participants"]:
            return "fit"
        if pid in provenance["held_out_participants"]:
            return "held_out"
        return "not_used"

    participants, unavailable = [], []
    for pid in authorized:
        f = files.get(pid)
        if f and f["ok"] and f["trials"]:
            participants.append(DemoParticipant(participant_id=pid, trial_indices=list(range(f["trials"])), model_role=role(pid)))
        elif dataset_available:
            unavailable.append(DemoUnavailable(participant_id=pid, reason="file not found or not readable on this server"))
    trial_counts = {len(p.trial_indices) for p in participants}
    if not dataset_available:
        note = "The DEAP dataset is not available on this server."
    elif not participants:
        note = "No authorized participant file is readable on this server."
    else:
        note = ("Authorized participants readable on this server. Each one's role for the active model comes from the "
                "model's recorded metadata; a participant used in training gives an in-sample demonstration only.")
    return DemoTrialsOut(
        model_version=meta["version"] if meta else None,
        held_out_participants=[p.participant_id for p in participants if p.model_role == "held_out"],
        trials_per_participant=trial_counts.pop() if len(trial_counts) == 1 else None,
        participants=participants,
        unavailable=unavailable,
        dataset_available=dataset_available,
        provenance=provenance,
        note=note,
    )