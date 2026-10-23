"""API hardening: read-only model registry listing, minimal public responses, administrator-only diagnostics.

Every test runs against its own temporary SQLite database (see conftest)."""
from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import event, text

from app.database.base import Base
from app.main import create_app
from tests.conftest import PARTICIPANTS, make_settings, register_and_login

def plain(value: object) -> str:
    """The underlying text of a setting, unwrapping pydantic SecretStr values."""
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


WRITE_SQL = re.compile(r"^\s*(INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER)\b", re.IGNORECASE)


@contextmanager
def captured_sql(client: TestClient) -> Iterator[list[str]]:
    """Records every SQL statement the app sends to its database while the block runs."""
    engine = client.app.state.db.engine  # type: ignore[attr-defined]
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:  # noqa: ANN001
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


def registry_snapshot(client: TestClient) -> tuple[list[tuple], list[tuple]]:
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        models = conn.execute(text("select id, version, is_active from models order by id")).fetchall()
        metrics = conn.execute(text("select id, model_id, split, level, name, value from metrics order by id")).fetchall()
    return [tuple(r) for r in models], [tuple(r) for r in metrics]


def admin_and_user(client: TestClient) -> tuple[dict[str, str], dict[str, str]]:
    admin = register_and_login(client)  # the first account is the administrator (existing behaviour)
    user = register_and_login(client, email="user@example.com")
    return admin, user


# ---------- 1. GET /models is read-only ----------
def test_get_models_performs_no_database_writes(trained_client: TestClient) -> None:
    before = registry_snapshot(trained_client)
    assert before[0], "startup sync should have registered the trained model"
    with captured_sql(trained_client) as statements:
        for _ in range(3):
            r = trained_client.get("/api/models")
            assert r.status_code == 200 and len(r.json()) == 1
    writes = [s for s in statements if WRITE_SQL.match(s)]
    assert statements, "the listing should have queried the database"
    assert writes == [], writes
    assert registry_snapshot(trained_client) == before


def test_get_models_does_not_resynchronise(trained_client: TestClient) -> None:
    # Clear this test database's registry rows: a read-only GET must not recreate them; the explicit sync must.
    with trained_client.app.state.db.engine.begin() as conn:  # type: ignore[attr-defined]
        conn.execute(text("delete from metrics"))
        conn.execute(text("delete from models"))
    assert trained_client.get("/api/models").json() == []
    admin, _ = admin_and_user(trained_client)
    r = trained_client.post("/api/models/sync", headers=admin)
    assert r.status_code == 200, r.text
    models = trained_client.get("/api/models").json()
    assert len(models) == 1 and models[0]["is_active"] is True
    assert models[0]["version"] == r.json()["active_version"]
    assert models[0]["test_metrics"]["balanced_accuracy"] is not None


def test_models_sync_requires_administrator(trained_client: TestClient) -> None:
    admin, user = admin_and_user(trained_client)
    before = registry_snapshot(trained_client)
    assert trained_client.post("/api/models/sync").status_code == 401
    assert trained_client.post("/api/models/sync", headers={"Authorization": "Bearer not-a-token"}).status_code == 401
    assert trained_client.post("/api/models/sync", headers=user).status_code == 403
    assert registry_snapshot(trained_client) == before  # rejected calls changed nothing


def test_models_sync_keeps_registry_and_active_model(trained_client: TestClient) -> None:
    admin, _ = admin_and_user(trained_client)
    active = trained_client.get("/api/model-info").json()["version"]
    models_before, metrics_before = registry_snapshot(trained_client)
    r = trained_client.post("/api/models/sync", headers=admin)
    assert r.status_code == 200 and r.json() == {"synced": 1, "active_version": active}
    models_after, metrics_after = registry_snapshot(trained_client)
    assert models_after == models_before  # same rows, same active flags
    assert [m[1:] for m in metrics_after] == [m[1:] for m in metrics_before]  # same metric values
    assert trained_client.get("/api/model-info").json()["version"] == active
    with trained_client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        actions = [row[0] for row in conn.execute(text("select action from audit_logs"))]
    assert "models.sync" in actions


# ---------- 2. Dataset summary: public aggregate, administrator detail ----------
def test_public_dataset_summary_is_aggregate_only(client: TestClient, fake_deap_dir: Path) -> None:
    r = client.get("/api/dataset/summary")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"sampling_rate_hz", "baseline_seconds", "participants_found", "participants_valid", "total_trials",
                         "eeg_channels_used", "eeg_channel_names", "participants", "authorized_participants"}
    assert body["authorized_participants"] == client.app.state.settings.development_participants
    assert body["participants_found"] == body["participants_valid"] == len(PARTICIPANTS)
    assert [p["participant_id"] for p in body["participants"]] == PARTICIPANTS
    assert all(set(p) == {"participant_id", "readable", "trials", "channels", "samples"} for p in body["participants"])
    raw = r.text
    for forbidden in (str(fake_deap_dir), fake_deap_dir.as_posix(), ".dat", "dataset_dir", "rating", "size_bytes", "error"):
        assert forbidden not in raw, forbidden


def test_public_dataset_error_does_not_reveal_location(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-dataset"
    app = create_app(make_settings(missing, tmp_path / "outputs", tmp_path / "db.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        r = client.get("/api/dataset/summary")
        assert r.status_code == 503
        assert r.json()["error"]["message"] == "Dataset not available on this server"
        assert str(missing) not in r.text and missing.as_posix() not in r.text and "no-such-dataset" not in r.text


def test_admin_dataset_summary_requires_administrator(client: TestClient, fake_deap_dir: Path) -> None:
    admin, user = admin_and_user(client)
    assert client.get("/api/admin/dataset/summary").status_code == 401
    assert client.get("/api/admin/dataset/summary", headers=user).status_code == 403
    r = client.get("/api/admin/dataset/summary", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert Path(body["dataset_dir"]) == fake_deap_dir
    assert [f["file"] for f in body["files"]] == [f"{p}.dat" for p in PARTICIPANTS]


# ---------- 3. Public health vs administrator diagnostics ----------
def test_public_health_is_minimal(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "version"}
    settings = client.app.state.settings  # type: ignore[attr-defined]
    for forbidden in ("sqlite", "database", "dialect", "environment", "model", "dataset", plain(settings.jwt_secret_key)):
        assert forbidden not in r.text, forbidden


def test_admin_diagnostics_rejects_unauthenticated_and_non_admin(client: TestClient) -> None:
    _, user = admin_and_user(client)
    assert client.get("/api/admin/diagnostics").status_code == 401
    assert client.get("/api/admin/diagnostics", headers={"Authorization": "Bearer not-a-token"}).status_code == 401
    r = client.get("/api/admin/diagnostics", headers=user)
    assert r.status_code == 403
    assert "dialect" not in r.text and "sqlite" not in r.text


def test_admin_diagnostics_for_administrator(client: TestClient) -> None:
    admin, _ = admin_and_user(client)
    r = client.get("/api/admin/diagnostics", headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["environment"] == "test"
    assert body["database"] == {"connected": True, "dialect": "sqlite"}
    assert body["dataset_available"] is True
    assert body["model"] == {"loaded": False, "version": None}
    settings = client.app.state.settings  # type: ignore[attr-defined]
    for secret in (plain(settings.jwt_secret_key), plain(settings.database_url), "sqlite:///", "password"):
        assert secret not in r.text, secret


def test_admin_diagnostics_reports_loaded_model(trained_client: TestClient) -> None:
    admin, _ = admin_and_user(trained_client)
    body = trained_client.get("/api/admin/diagnostics", headers=admin).json()
    assert body["model"] == {"loaded": True, "version": trained_client.get("/api/model-info").json()["version"]}


# ---------- 4. /metrics: public research metrics, no filesystem details ----------
PUBLIC_DATASET_FIELDS = {"participant_count", "segments", "rejected_segments", "class_balance", "fingerprint"}


def _active_metadata(output_dir: Path) -> tuple[Path, dict]:
    import json

    version = json.loads((output_dir / "models" / "active_model.json").read_text(encoding="utf-8"))["version"]
    path = output_dir / "models" / version / "metadata.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_public_metrics_keep_research_fields_without_paths(trained_client: TestClient, trained_output_dir: Path,
                                                          fake_deap_dir: Path) -> None:
    meta_path, meta = _active_metadata(trained_output_dir)
    stored_before = meta_path.read_bytes()
    r = trained_client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True and body["model_version"] == meta["version"]
    for field in ("label", "split", "validation", "test", "test_reference_majority", "cross_validation", "graphs"):
        assert body[field], field
    assert body["test"] == meta["test"]  # research metrics unchanged
    assert set(body["dataset"]) == PUBLIC_DATASET_FIELDS
    assert body["dataset"] == {"participant_count": len(meta["dataset"]["participants"]), "segments": meta["dataset"]["segments"],
                               "rejected_segments": meta["dataset"]["rejected_segments"],
                               "class_balance": meta["dataset"]["class_balance"], "fingerprint": meta["dataset"]["fingerprint"]}
    assert all(g.startswith("/api/graphs/") and "/" not in g.removeprefix("/api/graphs/") for g in body["graphs"])
    raw = r.text
    for forbidden in (str(fake_deap_dir), fake_deap_dir.as_posix(), str(trained_output_dir), trained_output_dir.as_posix(),
                      meta["dataset"]["dir"], '"dir"', "model.joblib", ".dat", "model_path"):
        assert forbidden not in raw, forbidden
    assert meta_path.read_bytes() == stored_before  # internal metadata file untouched


def test_metrics_errors_do_not_expose_paths(trained_settings, fake_deap_dir: Path, monkeypatch) -> None:  # noqa: ANN001
    from app.services.model_service import ModelService

    app = create_app(trained_settings)
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app, raise_server_exceptions=False) as client:
        def broken(self: ModelService) -> dict:
            raise RuntimeError(f"cannot read {fake_deap_dir / 's01.dat'}")

        monkeypatch.setattr(ModelService, "metadata", property(broken))
        r = client.get("/api/metrics")
        assert r.status_code == 500
        assert r.json()["error"]["message"] == "Internal server error"
        assert str(fake_deap_dir) not in r.text and fake_deap_dir.as_posix() not in r.text and "s01.dat" not in r.text


def test_metrics_without_model_has_no_paths(client: TestClient, fake_deap_dir: Path) -> None:
    r = client.get("/api/metrics")
    assert r.json() == {"available": False, "message": "No trained model available. Train a model first.", "model_version": None,
                        "label": None, "split": None, "validation": None, "test": None, "test_reference_majority": None,
                        "cross_validation": None, "evaluation": None, "protocol": None, "dataset": None, "graphs": []}


# ---------- 5. /training/status: administrators only, generic failures ----------
def test_training_status_requires_administrator(client: TestClient) -> None:
    admin, user = admin_and_user(client)
    assert client.get("/api/training/status").status_code == 401
    assert client.get("/api/training/status", headers={"Authorization": "Bearer not-a-token"}).status_code == 401
    r = client.get("/api/training/status", headers=user)
    assert r.status_code == 403 and "state" not in r.json()
    r = client.get("/api/training/status", headers=admin)
    assert r.status_code == 200 and r.json()["state"] == "idle"


def test_failed_training_is_polled_and_reported_generically(client: TestClient, fake_deap_dir: Path, monkeypatch) -> None:  # noqa: ANN001
    import threading

    import app.services.training_service as training_module

    proceed = threading.Event()
    secret_path = fake_deap_dir / "s01.dat"

    class FailingTrainer:
        def __init__(self, settings, run_cross_validation: bool, activate: bool) -> None:  # noqa: ANN001
            assert activate is False  # the training service never activates

        def run(self, progress):  # noqa: ANN001, ANN201
            progress("Loading participants", 0.25)
            proceed.wait(timeout=30)
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{secret_path}'")

    logged: list[tuple[str, object]] = []

    class RecordingLogger:
        def info(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def exception(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def error(self, msg: str, *args, exc_info=None, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
            logged.append((msg, exc_info))

    monkeypatch.setattr(training_module, "ProtocolTrainer", FailingTrainer)
    monkeypatch.setattr(training_module, "logger", RecordingLogger())
    admin, user = admin_and_user(client)
    assert client.post("/api/train", json={"run_cross_validation": False}, headers=admin).status_code == 202

    # Polling while the run is in progress (as the Training page does).
    for _ in range(2):
        running = client.get("/api/training/status", headers=admin).json()
        assert running["state"] == "running" and running["stage"] == "Loading participants" and running["progress"] == 0.25
    assert client.get("/api/training/status", headers=user).status_code == 403

    proceed.set()
    client.app.state.training_service.wait(timeout=30)  # type: ignore[attr-defined]
    r = client.get("/api/training/status", headers=admin)
    body = r.json()
    assert body["state"] == "failed"
    assert body["error"] == training_module.TRAINING_FAILED_MESSAGE
    for forbidden in (str(secret_path), secret_path.as_posix(), "s01.dat", "FileNotFoundError", "Errno", "No such file"):
        assert forbidden not in r.text, forbidden
    # The detail stays in the server log.
    assert logged and logged[0][0] == "Training failed" and str(secret_path) in str(logged[0][1])


def test_metrics_and_training_status_reads_do_not_write(trained_client: TestClient) -> None:
    admin, _ = admin_and_user(trained_client)
    before = registry_snapshot(trained_client)
    with captured_sql(trained_client) as statements:
        for _ in range(2):
            assert trained_client.get("/api/metrics").status_code == 200
            assert trained_client.get("/api/training/status", headers=admin).status_code == 200
    assert [s for s in statements if WRITE_SQL.match(s)] == []
    assert registry_snapshot(trained_client) == before
