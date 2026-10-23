from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import AuthenticationError, ConflictError, InvalidInputError
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User
from app.repositories import AuditRepository, UserRepository

logger = get_logger("auth")

PASSWORD_MIN, PASSWORD_MAX_BYTES = 8, 72


def validate_password(password: str) -> None:
    """The same rule as public registration: 8-72 characters, at most 72 bytes (bcrypt limit)."""
    if len(password) < PASSWORD_MIN or len(password.encode("utf-8")) > PASSWORD_MAX_BYTES:
        raise InvalidInputError("Password must be 8-72 characters (at most 72 bytes)")


# Constant-time-ish defence: verify against a dummy hash when the email is unknown.
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


class AuthService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.users = UserRepository(db)
        self.audit = AuditRepository(db)
        self.settings = settings

    def register(self, email: str, password: str, full_name: str | None, ip: str | None) -> User:
        """Public self-registration. Always creates a standard (non-administrator) account, including the very
        first account in an empty database. Administrators are created only by bootstrap_admin()."""
        if self.users.get_by_email(email):
            raise ConflictError("An account with this email already exists")
        user = self.users.create(email, hash_password(password), full_name, is_admin=False)
        self.audit.add("auth.register", user.id, "user", str(user.id), {"is_admin": False}, ip)
        logger.info("User registered", extra={"user_id": user.id, "is_admin": False})
        return user

    def bootstrap_admin(self, email: str, password: str, full_name: str | None) -> User:
        """Create the first administrator (command-line bootstrap only; never reachable from the web API).

        Refuses if any administrator already exists, or if the email already belongs to an account: it never
        creates a second administrator and never promotes an existing account."""
        validate_password(password)
        if self.users.admin_count() > 0:
            raise ConflictError("An administrator already exists; bootstrap refused. Use password recovery instead.")
        if self.users.get_by_email(email):
            raise ConflictError("An account with this email already exists; bootstrap never promotes existing accounts.")
        user = self.users.create(email, hash_password(password), full_name, is_admin=True)
        self.audit.add("auth.admin_bootstrap", user.id, "user", str(user.id), {"is_admin": True}, "local-cli")
        logger.info("Administrator bootstrapped", extra={"user_id": user.id})
        return user

    def reset_admin_password(self, email: str, password: str) -> User:
        """Recovery: set a new password for an existing, active administrator account (command line only)."""
        validate_password(password)
        user = self.users.get_by_email(email)
        if user is None or not user.is_admin:
            raise ConflictError("No administrator account with this email exists.")
        self.users.set_password(user, hash_password(password))
        self.audit.add("auth.admin_password_reset", user.id, "user", str(user.id), None, "local-cli")
        logger.info("Administrator password reset", extra={"user_id": user.id})
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
