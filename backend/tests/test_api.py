from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import register_and_login


# ---------- health & auth ----------
def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["database"] == {"connected": True, "dialect": "sqlite"}
    assert body["dataset_available"] is True
    assert body["model"]["loaded"] is False


def test_openapi_docs_available(client: TestClient) -> None:
    assert client.get("/api/docs").status_code == 200
    paths = client.get("/api/openapi.json").json()["paths"]
    for path in ("/api/health", "/api/auth/login", "/api/predict", "/api/metrics", "/api/predictions", "/api/reports"):
        assert path in paths


def test_register_login_me(client: TestClient) -> None:
    headers = register_and_login(client)
    me = client.get("/api/auth/me", headers=headers).json()
    assert me["email"] == "admin@example.com" and me["is_admin"] is True
    assert "hashed_password" not in me


def test_second_user_is_not_admin(client: TestClient) -> None:
    register_and_login(client)
    headers = register_and_login(client, "user@example.com")
    assert client.get("/api/auth/me", headers=headers).json()["is_admin"] is False
    r = client.post("/api/train", json={}, headers=headers)
    assert r.status_code == 403


def test_duplicate_registration_conflicts(client: TestClient) -> None:
    register_and_login(client)
    r = client.post("/api/auth/register", json={"email": "ADMIN@example.com", "password": "another-password"})
    assert r.status_code == 409


def test_wrong_password_and_unknown_user(client: TestClient) -> None:
    register_and_login(client)
    for user, pw in (("admin@example.com", "wrong-password"), ("ghost@example.com", "whatever-pass")):
        r = client.post("/api/auth/login", data={"username": user, "password": pw})
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "not_authenticated"


def test_short_password_rejected(client: TestClient) -> None:
    r = client.post("/api/auth/register", json={"email": "a@example.com", "password": "short"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_protected_routes_require_token(client: TestClient) -> None:
    for method, path in (("get", "/api/auth/me"), ("post", "/api/predict"), ("get", "/api/predictions"),
                         ("get", "/api/sessions"), ("get", "/api/reports"), ("post", "/api/train")):
        assert getattr(client, method)(path).status_code == 401, path


def test_invalid_token_rejected(client: TestClient) -> None:
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_password_is_hashed_in_database(client: TestClient) -> None:
    from app.models import User

    register_and_login(client)
    with client.app.state.db.session() as s:  # type: ignore[attr-defined]
        user = s.query(User).one()
        assert user.hashed_password.startswith("$2") and "correct-horse" not in user.hashed_password


# ---------- no trained model ----------
def test_model_endpoints_without_model(client: TestClient) -> None:
    assert client.get("/api/model-info").json()["available"] is False
    assert client.get("/api/metrics").json()["available"] is False
    headers = register_and_login(client)
    r = client.post("/api/predict", json={"deap_reference": {"participant_id": "s01", "trial_index": 0}}, headers=headers)
    assert r.status_code == 503
    assert "No trained model" in r.json()["error"]["message"]


# ---------- with trained model ----------
def test_model_info_and_metrics(trained_client: TestClient) -> None:
    info = trained_client.get("/api/model-info").json()
    assert info["available"] is True
    assert info["class_labels"] == ["high_valence", "low_valence"]
    assert info["feature_schema"]["count"] == 440
    metrics = trained_client.get("/api/metrics").json()
    assert metrics["available"] is True
    assert 0 <= metrics["test"]["segment"]["accuracy"] <= 1
    assert metrics["graphs"] and trained_client.get(metrics["graphs"][0]).status_code == 200
    models = trained_client.get("/api/models").json()
    assert len(models) == 1 and models[0]["is_active"] is True


def test_predict_from_deap_reference_is_stored(trained_client: TestClient) -> None:
    headers = register_and_login(trained_client)
    session_id = trained_client.post("/api/sessions", json={"name": "demo"}, headers=headers).json()["id"]
    r = trained_client.post("/api/predict", headers=headers,
                            json={"deap_reference": {"participant_id": "s01", "trial_index": 0}, "session_id": session_id})
    assert r.status_code == 201, r.text
    body = r.json()
    assert set(body["probabilities"]) == {"high_valence", "low_valence"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6
    assert body["confidence"] == max(body["probabilities"].values())
    assert body["prediction"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["source_reference"]["self_reported_label"] == "high_valence"
    assert body["model"]["version"] and body["prediction_id"]
    assert body["created_at"].endswith(("Z", "+00:00")), "timestamps must be timezone-aware UTC (SQLite regression)"

    history = trained_client.get("/api/predictions", headers=headers).json()
    assert history["total"] == 1 and history["items"][0]["prediction_id"] == body["prediction_id"]
    sessions = trained_client.get("/api/sessions", headers=headers).json()
    assert sessions[0]["prediction_count"] == 1
    assert trained_client.get(f"/api/predictions/{body['prediction_id']}", headers=headers).status_code == 200


def test_predict_from_uploaded_signal(trained_client: TestClient) -> None:
    headers = register_and_login(trained_client)
    t = np.arange(128 * 8) / 128
    signal = (5 * np.sin(2 * np.pi * 10 * t) + np.random.default_rng(0).normal(0, 1, (32, t.size))).tolist()
    r = trained_client.post("/api/predict", headers=headers, json={"eeg": {"signal": signal, "sampling_rate": 128}})
    assert r.status_code == 201, r.text
    assert r.json()["segments_used"] == 2
    assert r.json()["input_source"] == "upload"


def test_predict_validation_errors(trained_client: TestClient) -> None:
    headers = register_and_login(trained_client)
    cases = [
        ({}, 422),
        ({"eeg": {"signal": [[0.0] * 600]}, "deap_reference": {"participant_id": "s01", "trial_index": 0}}, 422),
        ({"eeg": {"signal": [[1.0, 2.0]] * 32}}, 422),
        ({"eeg": {"signal": [[1.0] * 600] * 32, "sampling_rate": 256}}, 422),
        ({"eeg": {"signal": [[1.0] * 600, [1.0] * 599]}}, 422),
        ({"deap_reference": {"participant_id": "s99", "trial_index": 0}}, 404),
        ({"deap_reference": {"participant_id": "s01", "trial_index": 0}, "session_id": 999}, 404),
    ]
    for payload, expected in cases:
        r = trained_client.post("/api/predict", json=payload, headers=headers)
        assert r.status_code == expected, (payload.keys(), r.text)
        assert "error" in r.json()


def test_users_cannot_see_each_others_history(trained_client: TestClient) -> None:
    alice = register_and_login(trained_client, "alice@example.com")
    bob = register_and_login(trained_client, "bob@example.com")
    pid = trained_client.post("/api/predict", headers=alice,
                              json={"deap_reference": {"participant_id": "s02", "trial_index": 1}}).json()["prediction_id"]
    assert trained_client.get("/api/predictions", headers=bob).json()["total"] == 0
    assert trained_client.get(f"/api/predictions/{pid}", headers=bob).status_code == 404


def test_training_via_api_creates_report(client: TestClient) -> None:
    headers = register_and_login(client)
    r = client.post("/api/train", json={"run_cross_validation": False}, headers=headers)
    assert r.status_code == 202 and r.json()["state"] == "running"
    client.app.state.training_service.wait(timeout=300)  # type: ignore[attr-defined]
    status = client.get("/api/training/status").json()
    assert status["state"] == "succeeded", status
    assert client.get("/api/model-info").json()["version"] == status["model_version"]
    reports = client.get("/api/reports", headers=headers).json()
    assert reports[0]["report_type"] == "training"
    assert reports[0]["content"]["version"] == status["model_version"]
