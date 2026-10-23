"""Registration never creates administrators; administrators come only from the command-line bootstrap, which
refuses to create a second one or promote an existing account. Every test uses its own temporary SQLite database."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.database.base import Base
from app.main import create_app
from scripts.create_admin import EXIT_DATABASE, EXIT_OK, EXIT_REFUSED
from scripts.create_admin import main as create_admin
from tests.conftest import make_settings, register_and_login

PASSWORD = "a-strong-test-password"


def _users(client: TestClient) -> list[tuple]:
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        return [tuple(r) for r in conn.execute(text("select email, is_admin from users order by id"))]


def _audit(client: TestClient, action: str) -> int:
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        return conn.execute(text("select count(*) from audit_logs where action = :a"), {"a": action}).scalar()


def _cli(client: TestClient, *args: str, password: str = PASSWORD) -> tuple[int, str]:
    out = io.StringIO()
    code = create_admin([*args, "--password-stdin"], settings=client.app.state.settings,  # type: ignore[attr-defined]
                        stdin=io.StringIO(password + "\n"), out=out)
    return code, out.getvalue()


# ---------- public registration ----------
def test_first_registered_account_is_not_an_administrator(client: TestClient) -> None:
    r = client.post("/api/auth/register", json={"email": "first@example.com", "password": PASSWORD})
    assert r.status_code == 201 and r.json()["is_admin"] is False
    assert _users(client) == [("first@example.com", False)]


@pytest.mark.parametrize("extra", [{"is_admin": True}, {"role": "admin"}, {"is_admin": "true", "is_active": True}])
def test_registration_rejects_role_fields(client: TestClient, extra: dict) -> None:
    r = client.post("/api/auth/register", json={"email": "sneaky@example.com", "password": PASSWORD, **extra})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert _users(client) == []  # nothing was created


def test_registered_users_cannot_reach_administrator_endpoints(client: TestClient) -> None:
    headers = register_and_login(client, "plain@example.com", PASSWORD, admin=False)
    assert client.get("/api/auth/me", headers=headers).json()["is_admin"] is False
    for method, path, body in (("post", "/api/train", {}), ("get", "/api/training/status", None),
                               ("get", "/api/reports", None), ("get", "/api/admin/diagnostics", None),
                               ("get", "/api/admin/dataset/summary", None), ("post", "/api/models/sync", None),
                               ("post", "/api/models/some-version/activate",
                                {"confirm_version": "some-version", "reason": "trying without rights"})):
        r = getattr(client, method)(path, headers=headers, **({"json": body} if body is not None else {}))
        assert r.status_code == 403, (path, r.status_code)
        assert r.json()["error"]["code"] == "forbidden"


# ---------- command-line bootstrap ----------
def test_bootstrap_creates_the_first_administrator(client: TestClient) -> None:
    client.post("/api/auth/register", json={"email": "someone@example.com", "password": PASSWORD})
    code, output = _cli(client, "--email", "Admin@Example.com", "--full-name", "Research Admin")
    assert code == EXIT_OK, output
    assert PASSWORD not in output  # the password is never printed
    assert _users(client) == [("someone@example.com", False), ("admin@example.com", True)]
    assert _audit(client, "auth.admin_bootstrap") == 1
    r = client.post("/api/auth/login", data={"username": "admin@example.com", "password": PASSWORD})
    assert r.status_code == 200 and r.json()["user"]["is_admin"] is True


def test_bootstrap_refuses_when_an_administrator_exists(client: TestClient) -> None:
    assert _cli(client, "--email", "admin@example.com")[0] == EXIT_OK
    code, output = _cli(client, "--email", "second-admin@example.com")
    assert code == EXIT_REFUSED and "already exists" in output
    assert [u for u in _users(client) if u[1]] == [("admin@example.com", True)]  # still exactly one administrator


def test_bootstrap_never_promotes_an_existing_account(client: TestClient) -> None:
    client.post("/api/auth/register", json={"email": "user@example.com", "password": PASSWORD})
    code, output = _cli(client, "--email", "user@example.com")
    assert code == EXIT_REFUSED and "never promotes" in output
    assert _users(client) == [("user@example.com", False)]


@pytest.mark.parametrize(("email", "password"), [("not-an-email", PASSWORD), ("admin@example.com", "short")])
def test_bootstrap_validates_input(client: TestClient, email: str, password: str) -> None:
    code, output = _cli(client, "--email", email, password=password)
    assert code == EXIT_REFUSED and password not in output
    assert _users(client) == []


def test_password_reset_only_for_existing_administrators(client: TestClient) -> None:
    assert _cli(client, "--email", "admin@example.com")[0] == EXIT_OK
    client.post("/api/auth/register", json={"email": "user@example.com", "password": PASSWORD})
    assert _cli(client, "--email", "user@example.com", "--reset-password", password="another-password-1")[0] == EXIT_REFUSED
    code, output = _cli(client, "--email", "admin@example.com", "--reset-password", password="another-password-1")
    assert code == EXIT_OK and "another-password-1" not in output
    assert client.post("/api/auth/login", data={"username": "admin@example.com", "password": PASSWORD}).status_code == 401
    assert client.post("/api/auth/login", data={"username": "admin@example.com", "password": "another-password-1"}).status_code == 200
    assert len(_users(client)) == 2 and _audit(client, "auth.admin_password_reset") == 1


def test_database_errors_reveal_no_connection_details(tmp_path: Path, fake_deap_dir: Path) -> None:
    settings = make_settings(fake_deap_dir, tmp_path / "out", tmp_path / "missing" / "nested" / "db.sqlite3")
    out = io.StringIO()
    code = create_admin(["--email", "admin@example.com", "--password-stdin"], settings=settings,
                        stdin=io.StringIO(PASSWORD + "\n"), out=out)
    text_out = out.getvalue()
    assert code == EXIT_DATABASE and "database could not be reached" in text_out
    assert "sqlite" not in text_out and str(tmp_path) not in text_out and "Traceback" not in text_out


def test_bootstrap_works_on_a_fresh_database(tmp_path: Path, fake_deap_dir: Path) -> None:
    app = create_app(make_settings(fake_deap_dir, tmp_path / "out", tmp_path / "fresh.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        assert _cli(client, "--email", "admin@example.com")[0] == EXIT_OK
        assert _users(client) == [("admin@example.com", True)]
