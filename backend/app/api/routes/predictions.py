from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import AdminUser, ClientIP, CurrentUser, DatasetDep, DbSession, ModelServiceDep, SettingsDep
from app.core.exceptions import NotFoundError
from app.repositories import AuditRepository, PredictionRepository, ReportRepository, SessionRepository
from app.schemas.schemas import PredictionListOut, PredictionOut, PredictionRequest, ReportOut, SessionCreate, SessionOut
from app.services.prediction_service import PredictionService, to_prediction_out
from app.services.public_metadata import sanitize_report_content

router = APIRouter()

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]


@router.post("/predict", response_model=PredictionOut, status_code=status.HTTP_201_CREATED, tags=["prediction"],
             summary="Predict the operational affect class for one EEG recording and store the result")
def predict(payload: PredictionRequest, user: CurrentUser, db: DbSession, settings: SettingsDep,
            model_service: ModelServiceDep, dataset: DatasetDep, ip: ClientIP) -> PredictionOut:
    return PredictionService(db, settings, model_service, dataset).predict(user, payload, ip)


@router.get("/predictions", response_model=PredictionListOut, tags=["history"], summary="Current user's prediction history")
def list_predictions(user: CurrentUser, db: DbSession, limit: Limit = 50, offset: Offset = 0,
                     session_id: int | None = None) -> PredictionListOut:
    repo = PredictionRepository(db)
    items = repo.list_for_user(user.id, limit, offset, session_id)
    return PredictionListOut(total=repo.count_for_user(user.id), items=[to_prediction_out(p) for p in items])


@router.get("/predictions/{prediction_id}", response_model=PredictionOut, tags=["history"])
def get_prediction(prediction_id: int, user: CurrentUser, db: DbSession) -> PredictionOut:
    item = PredictionRepository(db).get_for_user(prediction_id, user.id)
    if item is None:
        raise NotFoundError("Prediction not found")
    return to_prediction_out(item)


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED, tags=["history"])
def create_session(payload: SessionCreate, user: CurrentUser, db: DbSession, ip: ClientIP) -> SessionOut:
    item = SessionRepository(db).create(user.id, payload.name, payload.description)
    AuditRepository(db).add("session.create", user.id, "session", str(item.id), None, ip)
    return SessionOut.model_validate(item)


@router.get("/sessions", response_model=list[SessionOut], tags=["history"], summary="Current user's analysis sessions")
def list_sessions(user: CurrentUser, db: DbSession, limit: Limit = 50, offset: Offset = 0) -> list[SessionOut]:
    repo = SessionRepository(db)
    items = repo.list_for_user(user.id, limit, offset)
    counts = repo.prediction_counts([s.id for s in items])
    return [SessionOut.model_validate(s).model_copy(update={"prediction_count": counts.get(s.id, 0)}) for s in items]


@router.get("/reports", response_model=list[ReportOut], tags=["reports"],
            summary="Training/evaluation reports (administrator only)")
def list_reports(_: AdminUser, db: DbSession, limit: Limit = 20, offset: Offset = 0) -> list[ReportOut]:
    # Stored content keeps internal details; responses get a sanitised copy (the ORM objects are not modified).
    return [ReportOut(id=r.id, report_type=r.report_type, title=r.title, content=sanitize_report_content(r.content),
                      model_id=r.model_id, user_id=r.user_id, created_at=r.created_at)
            for r in ReportRepository(db).list(limit, offset)]
