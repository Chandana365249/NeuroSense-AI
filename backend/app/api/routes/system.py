from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import DatasetDep, ModelServiceDep, SettingsDep
from app.core.logging import get_logger
from app.ml.dataset import DatasetError
from app.schemas.schemas import HealthOut

router = APIRouter(tags=["system"])
logger = get_logger("health")


@router.get("/health", response_model=HealthOut)
def health(request: Request, settings: SettingsDep, model_service: ModelServiceDep, dataset: DatasetDep) -> HealthOut:
    db = request.app.state.db
    try:
        db.ping()
        database = {"connected": True, "dialect": db.dialect}
    except Exception as exc:
        logger.error("Database ping failed", extra={"error": type(exc).__name__})
        database = {"connected": False, "dialect": db.dialect}
    try:
        dataset.discover()
        dataset_ok = True
    except DatasetError:
        dataset_ok = False
    meta = model_service.metadata
    model = {"loaded": model_service.available, "version": meta["version"] if meta else None}
    return HealthOut(
        status="ok" if database["connected"] else "degraded",
        version=request.app.version,
        environment=settings.environment,
        database=database,
        model=model,
        dataset_available=dataset_ok,
    )
