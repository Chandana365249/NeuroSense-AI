from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.api.deps import AdminUser, DatasetDep, ModelServiceDep, SettingsDep
from app.core.logging import get_logger
from app.ml.dataset import DatasetError
from app.schemas.schemas import DiagnosticsOut, HealthOut

router = APIRouter(tags=["system"])
logger = get_logger("health")


def _database_status(request: Request) -> dict[str, Any]:
    db = request.app.state.db
    try:
        db.ping()
        return {"connected": True, "dialect": db.dialect}
    except Exception as exc:
        logger.error("Database ping failed", extra={"error": type(exc).__name__})
        return {"connected": False, "dialect": db.dialect}


@router.get("/health", response_model=HealthOut, summary="Public liveness check (status and API version only)")
def health(request: Request) -> HealthOut:
    database = _database_status(request)
    return HealthOut(status="ok" if database["connected"] else "degraded", version=request.app.version)


@router.get("/admin/diagnostics", response_model=DiagnosticsOut, tags=["admin"],
            summary="Detailed system diagnostics (administrator only)")
def diagnostics(request: Request, _: AdminUser, settings: SettingsDep, model_service: ModelServiceDep,
                dataset: DatasetDep) -> DiagnosticsOut:
    database = _database_status(request)
    try:
        dataset.discover()
        dataset_ok = True
    except DatasetError:
        dataset_ok = False
    meta = model_service.metadata
    return DiagnosticsOut(
        status="ok" if database["connected"] else "degraded",
        version=request.app.version,
        environment=settings.environment,
        database=database,  # connection state and engine name only; never the URL or credentials
        model={"loaded": model_service.available, "version": meta["version"] if meta else None},
        dataset_available=dataset_ok,
    )
