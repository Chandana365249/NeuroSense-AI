"""Create the first administrator, or reset an administrator's password. Local command line only.

Usage (from backend/):
    python -m scripts.create_admin --email admin@your-org.example [--full-name "Name"]
    python -m scripts.create_admin --email admin@your-org.example --reset-password

The password is asked for twice without being shown (or read from standard input with --password-stdin, for
automation). It is never printed or logged.

Safety rules:
- Bootstrap refuses if any administrator already exists, so it can never create a second administrator.
- Bootstrap refuses if the email already belongs to an account: existing accounts are never promoted.
- Public web registration always creates standard accounts; this command is the only way to create an administrator.
- --reset-password only changes the password of an existing administrator (recovery); it creates no account.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from typing import TextIO

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.database.session import Database
from app.services.auth_service import AuthService

EXIT_OK, EXIT_REFUSED, EXIT_DATABASE = 0, 1, 2


def _read_password(from_stdin: bool, stdin: TextIO) -> str:
    if from_stdin:
        return stdin.readline().rstrip("\r\n")
    first = getpass.getpass("New administrator password (8-72 characters): ")
    if getpass.getpass("Repeat the password: ") != first:
        raise AppError("The passwords do not match")
    return first


def main(argv: list[str] | None = None, settings: Settings | None = None, stdin: TextIO | None = None,
         out: TextIO | None = None) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(description="Create the first NeuroSense AI administrator, or reset its password.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--full-name", default=None)
    parser.add_argument("--reset-password", action="store_true", help="set a new password for an existing administrator")
    parser.add_argument("--password-stdin", action="store_true", help="read the password from standard input")
    args = parser.parse_args(argv)

    try:
        email = TypeAdapter(EmailStr).validate_python(args.email)
    except ValidationError:
        print("Error: enter a valid email address.", file=out)
        return EXIT_REFUSED

    settings = settings or get_settings()
    db = Database(settings.database_url)
    try:
        password = _read_password(args.password_stdin, stdin or sys.stdin)
        with db.session() as session:
            service = AuthService(session, settings)
            if args.reset_password:
                user = service.reset_admin_password(email, password)
                message = f"Password updated for administrator account id {user.id}."
            else:
                user = service.bootstrap_admin(email, password, args.full_name)
                message = f"Administrator account created (id {user.id}). Sign in with this email address."
            session.commit()
        print(message, file=out)
        return EXIT_OK
    except AppError as exc:
        print(f"Refused: {exc.message}", file=out)
        return EXIT_REFUSED
    except SQLAlchemyError:
        # No connection string, host or driver details: they can contain credentials.
        print("Error: the database could not be reached or is not initialised. Check that PostgreSQL is running, "
              "NEUROSENSE_DATABASE_URL in backend/.env is correct, and migrations have been applied "
              "(alembic upgrade head).", file=out)
        return EXIT_DATABASE
    finally:
        db.dispose()


if __name__ == "__main__":
    sys.exit(main())
