from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, status

from app.api.deps import AdminUser, DatasetDep, DbSession, ModelServiceDep, TrainingServiceDep
from app.core.exceptions import ServiceUnavailableError
from app.ml.dataset import DatasetError
from app.repositories import AuditRepository, ModelRepository
from app.schemas.schemas import DemoTrialsOut, MetricsOut, ModelInfoOut, ModelSummary, TrainingStatusOut, TrainRequest
from app.services.model_service import input_requirements

router = APIRouter(tags=["model"])

NOT_TRAINED = "No trained model available. Train a model first."


@router.post("/train", response_model=TrainingStatusOut, status_code=status.HTTP_202_ACCEPTED,
             summary="Start baseline training on the DEAP dataset (administrator only, runs in background)")
def train(payload: TrainRequest, user: AdminUser, training: TrainingServiceDep, db: DbSession, request: Request) -> TrainingStatusOut:
    state = training.start(user.id, payload.run_cross_validation)
    AuditRepository(db).add("training.start", user.id, "model", None, {"cv": payload.run_cross_validation},
                            request.client.host if request.client else None)
    return TrainingStatusOut(**state)


@router.get("/training/status", response_model=TrainingStatusOut)
def training_status(training: TrainingServiceDep) -> TrainingStatusOut:
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
        disclaimer=meta["disclaimer"],
    )


@router.get("/models", response_model=list[ModelSummary], summary="All registered model versions")
def list_models(db: DbSession, model_service: ModelServiceDep) -> list[ModelSummary]:
    model_service.sync_registry_to_db(db)
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
        ))
    return summaries


@router.get("/metrics", response_model=MetricsOut, summary="Evaluation metrics of the active model (real held-out results)")
def metrics(model_service: ModelServiceDep, request: Request) -> MetricsOut:
    meta = model_service.metadata
    if not meta:
        return MetricsOut(available=False, message=NOT_TRAINED)
    prefix = request.app.state.settings.api_prefix
    graphs = [f"{prefix}/graphs/{Path(p).name}" for p in meta["artifacts"]["graphs"].values() if p]
    return MetricsOut(
        available=True,
        model_version=meta["version"],
        label=meta["label"],
        split=meta["split"],
        validation=meta["validation"],
        test=meta["test"],
        test_reference_majority=meta["test_reference_majority"],
        cross_validation=meta.get("cross_validation"),
        dataset=meta["dataset"],
        graphs=graphs,
    )


@router.get("/dataset/summary", tags=["dataset"], summary="Structural summary of the local DEAP files")
def dataset_summary(dataset: DatasetDep, request: Request) -> dict[str, Any]:
    cached = getattr(request.app.state, "dataset_summary", None)
    if cached is None:
        try:
            cached = dataset.summarize().to_dict()
        except DatasetError as exc:
            raise ServiceUnavailableError(f"Dataset not available: {exc}") from exc
        request.app.state.dataset_summary = cached
    return cached


@router.get("/dataset/demo-trials", tags=["dataset"], response_model=DemoTrialsOut,
            summary="Participants held out from training, usable for honest demo predictions")
def demo_trials(model_service: ModelServiceDep) -> DemoTrialsOut:
    meta = model_service.metadata
    return DemoTrialsOut(
        model_version=meta["version"] if meta else None,
        held_out_participants=meta["split"]["test"] if meta else [],
        trials_per_participant=40,
        note="These participants were never seen during training or model selection.",
    )
