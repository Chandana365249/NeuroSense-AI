"""Safe model lifecycle: training registers inactive candidates, and only an explicit, audited administrator
activation changes the active model.

Every test copies the shared trained fixture into its own temporary output directory and uses its own temporary
SQLite database, so no test changes a shared registry, the real registry or the real database."""
from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.database.base import Base
from app.main import create_app
from app.ml.registry import ModelRegistry
from app.ml.training import BaselineTrainer
from tests.conftest import make_settings
from tests.test_api_hardening import WRITE_SQL, admin_and_user, captured_sql

REASON = "Candidate passed review on the trusted participants"


# ---------- helpers ----------
def _models_dir(client: TestClient) -> Path:
    return client.app.state.model_service.registry.base_dir  # type: ignore[attr-defined]


def make_candidate(models_dir: Path, name: str, mutate=None) -> str:  # noqa: ANN001
    """A second saved version: a copy of the active model with its own version name (and optional changes)."""
    active = json.loads((models_dir / ModelRegistry.ACTIVE_FILE).read_text(encoding="utf-8"))["version"]
    shutil.copytree(models_dir / active, models_dir / name)
    meta_path = models_dir / name / ModelRegistry.METADATA_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(version=name, model_path=str(models_dir / name / "model.joblib"), trained_at="2030-01-01T00:00:00+00:00")
    if mutate:
        mutate(meta)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return name


@pytest.fixture
def lifecycle_client(tmp_path: Path, fake_deap_dir: Path, trained_output_dir: Path) -> Iterator[TestClient]:
    out = tmp_path / "outputs"
    shutil.copytree(trained_output_dir, out)  # private copy: the shared fixture's pointer is never touched
    make_candidate(out / "models", "candidate-ok")
    app = create_app(make_settings(fake_deap_dir, out, tmp_path / "lifecycle.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        yield client


def state(client: TestClient) -> dict:
    """Everything an activation may change: pointer file, in-memory model, database rows (models, metrics, audit)."""
    models_dir = _models_dir(client)
    service = client.app.state.model_service  # type: ignore[attr-defined]
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        rows = {
            "models": [tuple(r) for r in conn.execute(text("select id, version, is_active from models order by id"))],
            "metrics": [tuple(r) for r in conn.execute(text("select id, model_id, split, level, name, value from metrics order by id"))],
            "audit": [tuple(r) for r in conn.execute(text("select id, action, details from audit_logs order by id"))],
        }
    return {"pointer": (models_dir / ModelRegistry.ACTIVE_FILE).read_bytes(),
            "loaded": service.metadata["version"] if service.metadata else None,
            "model_object": id(service._model),
            "stray_files": sorted(p.name for p in models_dir.glob(".active_model.*")),
            **rows}


def activate(client: TestClient, version: str, headers: dict | None = None, **body) -> object:  # noqa: ANN003
    payload = {"confirm_version": version, "reason": REASON, **body}
    return client.post(f"/api/models/{version}/activate", json=payload, headers=headers or {})


def active_version(client: TestClient) -> str:
    return client.get("/api/model-info").json()["version"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- Stage A: training never activates ----------
def test_registry_and_trainer_default_to_inactive(tmp_path: Path, trained_output_dir: Path) -> None:
    assert inspect.signature(ModelRegistry.save).parameters["activate"].default is False
    assert inspect.signature(BaselineTrainer).parameters["activate"].default is False
    model, meta = ModelRegistry(trained_output_dir / "models").load()
    registry = ModelRegistry(tmp_path / "models")
    registry.save("v1", model, {k: v for k, v in meta.items() if k not in ("version", "model_path", "sha256")})
    assert registry.active_version() is None and registry.pointer_bytes() is None  # saved, not activated
    registry.save("v2", model, {k: v for k, v in meta.items() if k not in ("version", "model_path", "sha256")}, activate=True)
    assert registry.active_version() == "v2"
    assert not list((tmp_path / "models").glob(".active_model.*"))  # atomic write left no temporary file


def test_registry_rejects_versions_outside_the_registry(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    (tmp_path / ModelRegistry.METADATA_FILE).write_text("{}", encoding="utf-8")  # a metadata file one level up
    for bad in ("", ".", "..", "../models", "a/b", "a\\b", "missing"):
        assert registry.has_version(bad) is False, bad


@pytest.mark.parametrize(("argv", "expected"), [([], False), (["--activate"], True), (["--no-cv"], False)])
def test_training_script_activates_only_with_flag(monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: bool) -> None:
    import scripts.train_baseline as script

    seen: dict = {}

    class Stop(Exception):
        pass

    class FakeTrainer:
        def __init__(self, settings, run_cross_validation: bool, activate: bool) -> None:  # noqa: ANN001
            seen.update(activate=activate, cv=run_cross_validation)

        def run(self, progress):  # noqa: ANN001, ANN201
            raise Stop

    monkeypatch.setattr(script, "BaselineTrainer", FakeTrainer)
    monkeypatch.setattr(script, "get_settings", lambda: SimpleNamespace(log_dir=None))  # never the real settings
    monkeypatch.setattr(script, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["train_baseline", *argv])
    with pytest.raises(Stop):
        script.main()
    assert seen["activate"] is expected


def test_training_via_api_registers_an_inactive_candidate(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    models_dir = _models_dir(client)
    active = active_version(client)
    artifact_hashes = {p.relative_to(models_dir): _sha(p) for p in (models_dir / active).rglob("*") if p.is_file()}
    admin, _ = admin_and_user(client)
    before = state(client)

    # With the exploratory evaluation on, the candidate's protocol metrics are recorded (a run without it has none).
    assert client.post("/api/train", json={"run_cross_validation": True}, headers=admin).status_code == 202
    client.app.state.training_service.wait(timeout=600)  # type: ignore[attr-defined]
    status = client.get("/api/training/status", headers=admin).json()
    assert status["state"] == "succeeded", status
    new_version = status["model_version"]

    after = state(client)
    assert after["pointer"] == before["pointer"] and after["loaded"] == active == active_version(client)
    assert after["model_object"] == before["model_object"]  # the in-memory model was not reloaded
    flags = {v: a for _, v, a in after["models"]}
    assert flags[new_version] is False or flags[new_version] == 0
    assert [v for v, a in flags.items() if a] == [active]
    assert [r for r in after["models"] if r[1] != new_version] == before["models"]  # existing rows untouched
    assert set(before["metrics"]) <= set(after["metrics"])  # historical metrics preserved
    new_id = next(i for i, v, _ in after["models"] if v == new_version)
    assert any(r[1] == new_id for r in after["metrics"])  # candidate metrics recorded
    assert {p.relative_to(models_dir): _sha(p) for p in (models_dir / active).rglob("*") if p.is_file()} == artifact_hashes
    completed = [r for r in after["audit"] if r[1] == "training.completed"]
    assert len(completed) == 1 and json.loads(completed[0][2]) == {"activated": False}
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        assert conn.execute(text("select model_id from reports")).scalar() == new_id


# ---------- Stage B: authorization and request validation ----------
def test_activation_requires_an_administrator(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    _, user = admin_and_user(client)
    before = state(client)
    assert activate(client, "candidate-ok").status_code == 401
    assert activate(client, "candidate-ok", {"Authorization": "Bearer not-a-token"}).status_code == 401
    r = activate(client, "candidate-ok", user)
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    r = client.post("/api/models/candidate-ok/activate", json={}, headers=user)  # authorization before validation
    assert r.status_code == 403
    assert state(client) == before


@pytest.mark.parametrize("body", [
    {"confirm_version": "something-else"},
    {"confirm_version": ""},
    {"reason": "   too short   "},
    {"reason": "x" * 501},
    {"reason": " " * 40},
    {"reason": "rotating after incident, password=hunter2 was used"},
    {"reason": "use header Bearer abcdefghijklmnopqrstuvwxyz0123"},
    {"reason": "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl attached"},
    {"reason": "db at postgresql://neurosense:s3cretpw@localhost:5432/x"},
])
def test_activation_rejects_invalid_confirmation_or_reason(lifecycle_client: TestClient, body: dict) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    before = state(client)
    r = activate(client, "candidate-ok", admin, **body)
    assert r.status_code == 422, r.text
    for secret in ("hunter2", "abcdefghijklmnopqrstuvwxyz0123", "eyJhbGciOiJIUzI1NiJ9", "s3cretpw"):
        assert secret not in r.text  # rejected values are not echoed back
    assert state(client) == before


def test_activation_rejects_a_reason_containing_the_configured_secret(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    secret = client.app.state.settings.jwt_secret_key.get_secret_value()  # type: ignore[attr-defined]
    before = state(client)
    r = activate(client, "candidate-ok", admin, reason=f"note {secret}")
    assert r.status_code == 422 and secret not in r.text
    assert state(client) == before


def test_activation_trims_and_records_the_reason(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    r = activate(client, "candidate-ok", admin, reason=f"   {REASON}   ")
    assert r.status_code == 200, r.text
    audit = [r for r in state(client)["audit"] if r[1] == "model.activate"]
    assert json.loads(audit[-1][2])["reason"] == REASON
    boundary = "y" * 500  # 500 characters is allowed; 10 is the minimum
    assert activate(client, "candidate-ok", admin, reason=boundary).json()["changed"] is False  # already active


def test_unknown_version_returns_404(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    before = state(client)
    r = activate(client, "no-such-version", admin)
    assert r.status_code == 404 and "no-such-version" not in r.json()["error"]["message"]
    assert state(client) == before


def test_activation_is_refused_while_training_runs(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    training = client.app.state.training_service  # type: ignore[attr-defined]
    before = state(client)
    training._status = {"state": "running", "stage": "Fitting", "progress": 0.3}
    try:
        r = activate(client, "candidate-ok", admin)
    finally:
        training._status = {"state": "idle", "progress": 0.0}
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    assert state(client) == before


def test_training_and_second_activation_are_refused_during_an_activation(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    service = client.app.state.model_service  # type: ignore[attr-defined]
    before = state(client)
    service._activation_lock.acquire()
    service._activating = True
    try:
        assert activate(client, "candidate-ok", admin).status_code == 409
        assert client.post("/api/train", json={"run_cross_validation": False}, headers=admin).status_code == 409
    finally:
        service._activating = False
        service._activation_lock.release()
    assert state(client) == before


# ---------- Stage B: invalid candidates are rejected before anything changes ----------
def _reverse_feature_names(meta: dict) -> None:
    meta["feature_schema"]["names"] = list(reversed(meta["feature_schema"]["names"]))


def _swap_class_labels(meta: dict) -> None:
    meta["class_labels"] = list(reversed(meta["class_labels"]))


@pytest.mark.parametrize(("name", "mutate", "corrupt", "message"), [
    ("candidate-corrupt", None, True, "integrity"),
    ("candidate-schema", _reverse_feature_names, False, "feature schema"),
    ("candidate-classes", _swap_class_labels, False, "not valid"),
])
def test_invalid_candidates_are_rejected(lifecycle_client: TestClient, name: str, mutate, corrupt: bool,  # noqa: ANN001
                                         message: str) -> None:
    client = lifecycle_client
    models_dir = _models_dir(client)
    make_candidate(models_dir, name, mutate)
    if corrupt:
        (models_dir / name / "model.joblib").write_bytes(b"corrupted artifact")
    admin, _ = admin_and_user(client)
    before = state(client)
    r = activate(client, name, admin)
    assert r.status_code == 422, r.text
    assert message in r.json()["error"]["message"]
    assert str(models_dir) not in r.text and "joblib" not in r.text  # no paths or internal file names
    assert state(client) == before


def test_missing_artifact_is_rejected(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    models_dir = _models_dir(client)
    make_candidate(models_dir, "candidate-missing")
    (models_dir / "candidate-missing" / "model.joblib").unlink()
    admin, _ = admin_and_user(client)
    before = state(client)
    r = activate(client, "candidate-missing", admin)
    assert r.status_code == 422 and "missing" in r.json()["error"]["message"]
    assert state(client) == before


# ---------- Stage B: failures during the switch restore the previous state ----------
def test_pointer_write_failure_keeps_the_previous_model(lifecycle_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ml.registry as registry_module

    client = lifecycle_client
    admin, _ = admin_and_user(client)
    before = state(client)

    def failing_replace(src, dst):  # noqa: ANN001, ANN202
        raise OSError("simulated disk failure")

    monkeypatch.setattr(registry_module.os, "replace", failing_replace)
    r = activate(client, "candidate-ok", admin)
    monkeypatch.undo()
    assert r.status_code == 500 and r.json()["error"]["code"] == "activation_failed"
    assert "simulated" not in r.text and "previous model remains active" in r.text
    assert state(client) == before  # pointer, memory, database flags and audit log unchanged; no temporary file left


def test_database_failure_before_the_switch_changes_nothing(lifecycle_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.model_service as service_module

    client = lifecycle_client
    admin, _ = admin_and_user(client)
    before = state(client)

    class FailingAudit:
        def __init__(self, db) -> None:  # noqa: ANN001
            pass

        def add(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise OperationalError("INSERT INTO audit_logs", {}, Exception("simulated database failure"))

    monkeypatch.setattr(service_module, "AuditRepository", FailingAudit)
    r = activate(client, "candidate-ok", admin)
    monkeypatch.undo()
    assert r.status_code == 500 and r.json()["error"]["code"] == "activation_failed" and "simulated" not in r.text
    assert state(client) == before


def test_commit_failure_after_the_pointer_write_is_rolled_back(lifecycle_client: TestClient,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    registry = client.app.state.model_service.registry  # type: ignore[attr-defined]
    before = state(client)
    seen_pointer: list[str | None] = []
    original_commit = Session.commit

    def failing_commit(self: Session) -> None:
        seen_pointer.append(registry.active_version())  # the pointer had already been switched at this point
        monkeypatch.setattr(Session, "commit", original_commit)  # only the activation's commit fails
        raise OperationalError("COMMIT", {}, Exception("simulated commit failure"))

    monkeypatch.setattr(Session, "commit", failing_commit)
    r = activate(client, "candidate-ok", admin)
    monkeypatch.undo()
    assert seen_pointer == ["candidate-ok"]
    assert r.status_code == 500 and r.json()["error"]["code"] == "activation_failed" and "simulated" not in r.text
    assert state(client) == before  # pointer restored byte for byte, previous model back in memory, no rows written
    assert client.post("/api/predict", headers=admin, json={"deap_reference": {"participant_id": "s01", "trial_index": 0}}
                       ).json()["model"]["version"] == before["loaded"]


# ---------- Stage B: successful activation, reactivation and the already-active case ----------
def test_activation_switches_everything_and_can_be_reversed(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    original = active_version(client)
    before = state(client)
    admin_id = client.get("/api/auth/me", headers=admin).json()["id"]

    r = activate(client, "candidate-ok", admin)
    assert r.status_code == 200, r.text
    assert r.json() == {"version": "candidate-ok", "previous_version": original, "changed": True,
                        "active_version": "candidate-ok"}
    after = state(client)
    assert json.loads(after["pointer"]) == {"version": "candidate-ok"} and after["stray_files"] == []
    assert active_version(client) == "candidate-ok"
    assert [v for _, v, a in after["models"] if a] == ["candidate-ok"]
    assert set(before["metrics"]) <= set(after["metrics"])  # historical metrics untouched
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        user_id, resource_id, details, ip = conn.execute(text(
            "select user_id, resource_id, details, ip_address from audit_logs where action = 'model.activate'")).one()
    assert (user_id, resource_id) == (admin_id, "candidate-ok") and ip
    assert json.loads(details) == {"previous_version": original, "new_version": "candidate-ok",
                                   "administrator_id": admin_id, "reason": REASON}
    prediction = client.post("/api/predict", headers=admin, json={"deap_reference": {"participant_id": "s01", "trial_index": 0}})
    assert prediction.status_code == 201 and prediction.json()["model"]["version"] == "candidate-ok"
    assert {m["version"]: m["is_active"] for m in client.get("/api/models").json()}["candidate-ok"] is True

    r = activate(client, original, admin, reason="Reverting to the previous model")
    assert r.status_code == 200 and r.json()["previous_version"] == "candidate-ok" and r.json()["changed"] is True
    restored = state(client)
    assert restored["pointer"] == before["pointer"] and active_version(client) == original
    assert [v for _, v, a in restored["models"] if a] == [original]
    assert len([a for a in restored["audit"] if a[1] == "model.activate"]) == 2


def test_activating_the_active_version_writes_nothing(lifecycle_client: TestClient) -> None:
    client = lifecycle_client
    admin, _ = admin_and_user(client)
    current = active_version(client)
    pointer = _models_dir(client) / ModelRegistry.ACTIVE_FILE
    mtime = pointer.stat().st_mtime_ns
    before = state(client)
    with captured_sql(client) as statements:
        r = activate(client, current, admin)
    assert r.status_code == 200 and r.json() == {"version": current, "previous_version": current, "changed": False,
                                                 "active_version": current}
    assert [s for s in statements if WRITE_SQL.match(s)] == []
    assert pointer.stat().st_mtime_ns == mtime and state(client) == before
