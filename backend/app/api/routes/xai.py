"""Explanations (XAI) of the signed-in user's own results, and of each model as a whole. Read-only."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import NotFoundError
from app.repositories import PredictionRepository
from app.schemas.schemas import EEGInput
from app.xai.service import XaiService

router = APIRouter(tags=["explainability"])


def get_xai_service(request: Request) -> XaiService:
    return request.app.state.xai_service


XaiDep = Annotated[XaiService, Depends(get_xai_service)]


class UploadExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eeg: EEGInput


def _own(db: Any, prediction_id: int, user_id: int) -> Any:
    item = PredictionRepository(db).get_for_user(prediction_id, user_id)
    if item is None:
        raise NotFoundError(f"Prediction {prediction_id} not found")
    return item


@router.get("/predictions/{prediction_id}/explanation", summary="Why the model made this prediction (recomputed and verified)")
def explain_prediction(prediction_id: int, user: CurrentUser, db: DbSession, xai: XaiDep) -> dict[str, Any]:
    return xai.explain(_own(db, prediction_id, user.id), None)


@router.post("/predictions/{prediction_id}/explanation", summary="Explain an uploaded-signal prediction (the same signal is re-sent)")
def explain_upload(prediction_id: int, payload: UploadExplanationRequest, user: CurrentUser, db: DbSession, xai: XaiDep) -> dict[str, Any]:
    return xai.explain(_own(db, prediction_id, user.id), payload.eeg.model_dump())


@router.post("/predictions/{prediction_id}/personal", summary="Personalised estimates from the same person's other rated trials")
def personal_estimate(prediction_id: int, user: CurrentUser, db: DbSession, xai: XaiDep) -> dict[str, Any]:
    return xai.personal(_own(db, prediction_id, user.id))


@router.get("/explain/global/{dataset}", summary="Model-level explanation computed on the model's own training data")
def explain_global(dataset: Literal["deap", "faced"], _: CurrentUser, xai: XaiDep) -> dict[str, Any]:
    return xai.global_view(dataset)
