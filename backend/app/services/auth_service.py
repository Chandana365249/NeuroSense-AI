from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User
from app.repositories import AuditRepository, UserRepository

logger = get_logger("auth")

# Constant-time-ish defence: verify against a dummy hash when the email is unknown.
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


class AuthService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.users = UserRepository(db)
        self.audit = AuditRepository(db)
        self.settings = settings

    def register(self, email: str, password: str, full_name: str | None, ip: str | None) -> User:
        if self.users.get_by_email(email):
            raise ConflictError("An account with this email already exists")
        is_first = self.users.count() == 0
        user = self.users.create(email, hash_password(password), full_name, is_admin=is_first)
        self.audit.add("auth.register", user.id, "user", str(user.id), {"is_admin": is_first}, ip)
        logger.info("User registered", extra={"user_id": user.id, "is_admin": is_first})
        return user

    def login(self, email: str, password: str, ip: str | None) -> tuple[str, int, User]:
        user = self.users.get_by_email(email)
        valid = verify_password(password, user.hashed_password if user else _DUMMY_HASH)
        if not user or not valid or not user.is_active:
            self.audit.add("auth.login_failed", user.id if user else None, "user", None, None, ip)
            logger.info("Login failed", extra={"user_id": user.id if user else None})
            raise AuthenticationError("Incorrect email or password")
        token, expires_in = create_access_token(str(user.id), self.settings)
        self.users.touch_login(user, datetime.now(UTC))
        self.audit.add("auth.login", user.id, "user", str(user.id), None, ip)
        logger.info("Login succeeded", extra={"user_id": user.id})
        return token, expires_in, user
