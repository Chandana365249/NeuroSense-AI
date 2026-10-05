from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import ClientIP, CurrentUser, DbSession, SettingsDep
from app.schemas.schemas import RegisterRequest, TokenOut, UserOut
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED,
             summary="Create an account (the first account becomes administrator)")
def register(payload: RegisterRequest, db: DbSession, settings: SettingsDep, ip: ClientIP) -> UserOut:
    user = AuthService(db, settings).register(payload.email, payload.password, payload.full_name, ip)
    return UserOut.model_validate(user)


@router.post("/login", response_model=TokenOut, summary="OAuth2 password login (form fields: username=email, password)")
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession, settings: SettingsDep, ip: ClientIP) -> TokenOut:
    token, expires_in, user = AuthService(db, settings).login(form.username, form.password, ip)
    return TokenOut(access_token=token, expires_in=expires_in, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
