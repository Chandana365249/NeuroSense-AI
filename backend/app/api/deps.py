from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import AuthenticationError, PermissionDeniedError
from app.core.security import InvalidTokenError, decode_access_token
from app.database.session import get_db
from app.ml.dataset import DEAPDataset
from app.models import User
from app.repositories import UserRepository
from app.services.model_service import ModelService
from app.services.training_service import TrainingService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_training_service(request: Request) -> TrainingService:
    return request.app.state.training_service


def get_dataset(request: Request) -> DEAPDataset:
    return request.app.state.dataset


def client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
TrainingServiceDep = Annotated[TrainingService, Depends(get_training_service)]
DatasetDep = Annotated[DEAPDataset, Depends(get_dataset)]
ClientIP = Annotated[str | None, Depends(client_ip)]


def get_current_user(
    db: DbSession, settings: SettingsDep, token: Annotated[str | None, Depends(oauth2_scheme)]
) -> User:
    if not token:
        raise AuthenticationError("Not authenticated")
    try:
        user_id = int(decode_access_token(token, settings))
    except (InvalidTokenError, ValueError) as exc:
        raise AuthenticationError("Invalid or expired token") from exc
    user = UserRepository(db).get(user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_admin_user(user: CurrentUser) -> User:
    if not user.is_admin:
        raise PermissionDeniedError("Administrator privileges required")
    return user


AdminUser = Annotated[User, Depends(get_admin_user)]
