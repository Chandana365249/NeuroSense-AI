"""FACED research endpoints (signed-in users). Separate from the DEAP model: FACED pipeline and FACED model only.
Results are saved in the user's history as FACED records, with their own provenance."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import ClientIP, CurrentUser, DbSession
from app.services.faced_service import FacedService
from app.services.prediction_service import store_faced_prediction

router = APIRouter(prefix="/faced", tags=["faced"])


def get_faced_service(request: Request) -> FacedService:
    return request.app.state.faced_service


FacedDep = Annotated[FacedService, Depends(get_faced_service)]


class FacedPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant_id: str = Field(pattern=r"^sub-\d{3}$")
    video_index: int = Field(ge=1, le=28)
    session_id: int | None = None


@router.get("/status", summary="FACED installation, validation and model status")
def faced_status(_: CurrentUser, faced: FacedDep) -> dict[str, Any]:
    return faced.status()


@router.get("/evidence", summary="Measured FACED results, read from the experiment output files")
def faced_evidence(_: CurrentUser, faced: FacedDep) -> dict[str, Any]:
    return faced.evidence()


@router.get("/participants", summary="Validated FACED participants on this server (searchable)")
def faced_participants(_: CurrentUser, faced: FacedDep,
                       q: Annotated[str, Query(max_length=40)] = "") -> dict[str, Any]:
    rows = faced.participants(q)
    return {"count": len(rows), "participants": rows}


@router.get("/participants/{participant_id}/trials", summary="Video trials of one validated FACED participant")
def faced_trials(participant_id: str, _: CurrentUser, faced: FacedDep) -> dict[str, Any]:
    return faced.trials(participant_id)


@router.post("/predict", summary="Run the FACED model on one FACED trial and save the result to the user's history")
def faced_predict(payload: FacedPredictRequest, user: CurrentUser, db: DbSession, faced: FacedDep, ip: ClientIP) -> dict[str, Any]:
    result = faced.predict(payload.participant_id, payload.video_index)
    item = store_faced_prediction(db, user, result, payload.session_id, ip)
    return {**result, "prediction_id": item.id, "session_id": item.session_id, "created_at": item.created_at, "stored": True}
