"""Report responses are sanitised and training-failure audit entries hold no exception text.

Every test runs against its own temporary SQLite database (see conftest)."""
from __future__ import annotations

import copy
import getpass
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text

from app.repositories import AuditRepository, ReportRepository
from app.services.public_metadata import sanitize_report_content
from tests.conftest import register_and_login
from tests.test_api_hardening import WRITE_SQL, _active_metadata, admin_and_user, captured_sql

APPROVED_DATASET_FIELDS = {"participant_count", "segments", "rejected_segments", "class_balance", "fingerprint"}
# The keys the training service stores in a training report.
TRAINING_REPORT_KEYS = ("version", "algorithm", "label", "split", "dataset", "validation", "test",
                        "test_reference_majority", "cross_validation", "training_seconds", "disclaimer")


def _insert_report(client: TestClient, content: dict) -> int:
    with client.app.state.db.session() as session:  # type: ignore[attr-defined]
        report = ReportRepository(session).create("training", "Test report", content, None, None)
        session.commit()
        return report.id


def _report_rows(client: TestClient) -> list[tuple]:
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        return [tuple(r) for r in conn.execute(text("select id, title, content from reports order by id"))]


# ---------- sanitiser ----------
def test_sanitize_report_content_keeps_research_values_and_drops_paths() -> None:
    content = {
        "version": "valence_binary-random_forest-20260101-000000",
        "split": {"train": ["s02", "s03"], "test": ["s01"], "seed": 42},
        "test": {"segment": {"accuracy": 0.54, "confusion_matrix": [[1, 2], [3, 4]]}},
        "disclaimer": "Research prototype; not a diagnosis.",
        "dataset": {"dir": "C:\\Users\\someone\\Data\\DEAP", "participants": ["s01", "s02", "s03"], "segments": 10,
                    "rejected_segments": 1, "class_balance": {"high_valence": 6, "low_valence": 4}, "fingerprint": "abc"},
        "model_path": "/home/someone/outputs/models/v1/model.joblib",
        "artifacts": {"graphs": {"roc": "C:\\out\\roc.png"}},
        "notes": ["could not read C:\\Users\\someone\\Data\\DEAP\\s05.dat", "share \\\\server\\share\\x",
                  "copied from /srv/data/deap/raw ok"],
        "nested": {"files": [{"file": "s01.dat"}], "path": "/tmp/x", "keep": 3},
    }
    original = copy.deepcopy(content)
    clean = sanitize_report_content(content)
    assert content == original  # the stored structure is never modified
    assert clean["dataset"] == {"participant_count": 3, "segments": 10, "rejected_segments": 1,
                                "class_balance": {"high_valence": 6, "low_valence": 4}, "fingerprint": "abc"}
    for key in ("version", "split", "test", "disclaimer"):
        assert clean[key] == content[key], key
    assert "model_path" not in clean and "artifacts" not in clean
    assert clean["nested"] == {"keep": 3}
    out = json.dumps(clean)
    for forbidden in ("someone", "C:\\\\", "server\\\\share", "/srv/data", ".dat", "model.joblib", "/tmp/x"):
        assert forbidden not in out, forbidden


# ---------- GET /reports ----------
def test_reports_endpoint_returns_sanitised_content(trained_client: TestClient, trained_output_dir: Path,
                                                   fake_deap_dir: Path) -> None:
    _, meta = _active_metadata(trained_output_dir)
    stored = {k: meta[k] for k in TRAINING_REPORT_KEYS}  # exactly what the training service stores
    legacy = {"version": "legacy", "dataset": {"dir": str(fake_deap_dir), "files": ["s01.dat"]},
              "model_path": str(trained_output_dir / "models" / "legacy" / "model.joblib"),
              "message": f"evaluation read {fake_deap_dir / 's01.dat'}"}
    ids = (_insert_report(trained_client, stored), _insert_report(trained_client, legacy))
    rows_before = _report_rows(trained_client)
    headers = register_and_login(trained_client)

    with captured_sql(trained_client) as statements:
        r = trained_client.get("/api/reports", headers=headers)
    assert r.status_code == 200
    reports = {item["id"]: item for item in r.json()}
    assert set(ids) <= set(reports)

    current = reports[ids[0]]["content"]
    assert set(current["dataset"]) == APPROVED_DATASET_FIELDS
    assert current["dataset"]["participant_count"] == len(meta["dataset"]["participants"])
    for key in TRAINING_REPORT_KEYS:
        if key != "dataset":
            assert current[key] == json.loads(json.dumps(meta[key])), key  # research information preserved
    old = reports[ids[1]]["content"]
    assert old["version"] == "legacy" and "model_path" not in old and set(old["dataset"]) == APPROVED_DATASET_FIELDS

    raw = r.text
    for forbidden in (str(fake_deap_dir), fake_deap_dir.as_posix(), str(trained_output_dir), trained_output_dir.as_posix(),
                      meta["dataset"]["dir"], getpass.getuser(), '"dir"', "model.joblib", ".dat", "model_path", "artifacts"):
        assert forbidden not in raw, forbidden
    assert not re.search(r"[A-Za-z]:\\\\", raw), "Windows drive path in response"
    assert [s for s in statements if WRITE_SQL.match(s)] == []  # reading reports writes nothing
    assert _report_rows(trained_client) == rows_before  # stored records unchanged


def test_reports_require_an_administrator(trained_client: TestClient) -> None:
    _insert_report(trained_client, {"version": "v1", "disclaimer": "Research prototype."})
    rows_before = _report_rows(trained_client)
    assert trained_client.get("/api/reports").status_code == 401
    assert trained_client.get("/api/reports", headers={"Authorization": "Bearer not-a-token"}).status_code == 401
    admin, user = admin_and_user(trained_client)
    r = trained_client.get("/api/reports", headers=user)
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    assert "Research prototype" not in r.text  # no report content in the refusal
    r = trained_client.get("/api/reports", headers=admin)
    assert r.status_code == 200 and [item["content"]["version"] for item in r.json()] == ["v1"]
    assert _report_rows(trained_client) == rows_before  # stored reports are not modified


# ---------- training-failure audit entries ----------
def test_training_failure_audit_entry_has_no_exception_text(client: TestClient, fake_deap_dir: Path, monkeypatch) -> None:  # noqa: ANN001
    import app.services.training_service as training_module

    secret_path = fake_deap_dir / "s01.dat"

    class FailingTrainer:
        def __init__(self, settings, run_cross_validation: bool, activate: bool) -> None:  # noqa: ANN001
            assert activate is False  # the training service never activates

        def run(self, progress):  # noqa: ANN001, ANN201
            progress("Fitting random_forest on training participants", 0.4)
            raise OSError(f"[Errno 2] No such file or directory: '{secret_path}' (user=neurosense password=hunter2)")

    monkeypatch.setattr(training_module, "BaselineTrainer", FailingTrainer)
    # An entry written by the previous code; it must stay exactly as it is.
    with client.app.state.db.session() as session:  # type: ignore[attr-defined]
        AuditRepository(session).add("training.failed", None, "model", None, {"error": "OSError: legacy detail C:\\legacy\\path"})
        session.commit()
    query = text("select id, details from audit_logs where action = 'training.failed' order by id")
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        legacy_before = [tuple(r) for r in conn.execute(query)]

    admin, _ = admin_and_user(client)
    assert client.post("/api/train", json={"run_cross_validation": False}, headers=admin).status_code == 202
    client.app.state.training_service.wait(timeout=30)  # type: ignore[attr-defined]
    assert client.get("/api/training/status", headers=admin).json()["state"] == "failed"

    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        rows = [tuple(r) for r in conn.execute(query)]
    assert len(rows) == 2 and rows[0] == legacy_before[0]  # the historical entry is not rewritten
    details = rows[1][1] if isinstance(rows[1][1], dict) else json.loads(rows[1][1])
    assert details == {"event": "Training run failed; details are in the server log",
                       "stage": "Fitting random_forest on training participants", "progress": 0.4, "error_type": "OSError"}
    raw = json.dumps(details)
    for forbidden in (str(secret_path), secret_path.as_posix(), "s01.dat", "Errno", "No such file", "hunter2", "password"):
        assert forbidden not in raw, forbidden
