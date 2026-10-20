"""A prediction always uses one consistent model snapshot: the model that produced the output, the version stored
with the prediction and the metadata used for labels and split all belong to the same model, even when an
administrator activates another model while the prediction is running.

Two test-only models with fixed, different outputs make any mix-up visible. Every test copies the shared trained
fixture into its own temporary output directory and uses its own temporary SQLite database."""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.database.base import Base
from app.main import create_app
from app.ml.registry import ModelRegistry
from app.models import User
from app.schemas.schemas import PredictionRequest
from app.services.prediction_service import PredictionService
from tests.conftest import make_settings
from tests.test_api_hardening import admin_and_user

CLASSES = ["high_valence", "low_valence"]
# Each version always outputs its own probabilities, so a result shows which model really produced it.
OUTPUT = {"model-a": 0.9, "model-b": 0.2}


class ConstantModel:
    """Test-only classifier with a fixed output. When `hook` is set it runs once in the middle of a prediction."""

    hook = None

    def __init__(self, p_first: float) -> None:
        self.p_first = p_first
        self.classes_ = np.asarray(CLASSES)

    def predict_proba(self, X) -> np.ndarray:  # noqa: ANN001, N803
        hook, ConstantModel.hook = ConstantModel.hook, None
        if hook:
            hook()
        time.sleep(0.002)  # widen the window in which a concurrent activation could interleave
        return np.tile([self.p_first, 1 - self.p_first], (len(X), 1))


@pytest.fixture
def two_model_app(tmp_path: Path, fake_deap_dir: Path, trained_output_dir: Path) -> Iterator[TestClient]:
    out = tmp_path / "outputs"
    shutil.copytree(trained_output_dir, out)
    registry = ModelRegistry(out / "models")
    _, meta = registry.load()
    base = {k: v for k, v in meta.items() if k not in ("version", "model_path", "sha256")}
    for version, p in OUTPUT.items():
        registry.save(version, ConstantModel(p), {**base, "class_labels": CLASSES, "name": f"Test {version}"},
                      activate=version == "model-a")
    app = create_app(make_settings(fake_deap_dir, out, tmp_path / "consistency.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        ConstantModel.hook = None
        yield client
    ConstantModel.hook = None


def _signal() -> list:
    t = np.arange(128 * 8) / 128
    return (5 * np.sin(2 * np.pi * 10 * t) + np.random.default_rng(0).normal(0, 1, (32, t.size))).tolist()


def _predict(client: TestClient, user_id: int, request: PredictionRequest) -> dict:
    app = client.app
    with app.state.db.session() as db:  # type: ignore[attr-defined]
        user = db.get(User, user_id)
        out = PredictionService(db, app.state.settings, app.state.model_service, app.state.dataset).predict(  # type: ignore[attr-defined]
            user, request, None)
        db.commit()
        return out.model_dump()


def _activate(client: TestClient, version: str, admin_id: int) -> None:
    app = client.app
    with app.state.db.session() as db:  # type: ignore[attr-defined]
        app.state.model_service.activate_version(db, version, admin_id, "Concurrency regression test switch", None,  # type: ignore[attr-defined]
                                                 lambda: False)


def _assert_consistent(result: dict) -> None:
    version = result["model"]["version"]
    assert result["probabilities"]["high_valence"] == pytest.approx(OUTPUT[version]), result


def _stored(client: TestClient) -> list[tuple[str, float]]:
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        rows = conn.execute(text("select m.version, p.probabilities from predictions p join models m on m.id = p.model_id"))
        return [(v, (json.loads(pr) if isinstance(pr, str) else pr)["high_valence"]) for v, pr in rows]


def _ids(client: TestClient) -> tuple[int, int]:
    admin, _ = admin_and_user(client)
    admin_id = client.get("/api/auth/me", headers=admin).json()["id"]
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        user_id = conn.execute(text("select id from users where email = 'user@example.com'")).scalar()
    return admin_id, user_id


def test_activation_during_a_prediction_does_not_mix_models(two_model_app: TestClient) -> None:
    """Deterministic: model B is activated while model A is computing its output (DEAP reference, so the
    participant split and self-reported label also come from the metadata)."""
    client = two_model_app
    admin_id, user_id = _ids(client)
    assert client.get("/api/model-info").json()["version"] == "model-a"
    ConstantModel.hook = lambda: _activate(client, "model-b", admin_id)

    request = PredictionRequest(deap_reference={"participant_id": "s01", "trial_index": 0})
    result = _predict(client, user_id, request)

    assert ConstantModel.hook is None  # the activation really ran mid-prediction
    assert client.get("/api/model-info").json()["version"] == "model-b"
    assert result["model"]["version"] == "model-a"  # stored and reported with the model that produced it
    _assert_consistent(result)
    assert result["source_reference"]["participant_split"] in ("train", "validation", "test", "not_used")
    assert _stored(client) == [("model-a", pytest.approx(OUTPUT["model-a"]))]
    with client.app.state.db.engine.connect() as conn:  # type: ignore[attr-defined]
        active = [v for (v,) in conn.execute(text("select version from models where is_active"))]
    assert active == ["model-b"]  # the prediction did not re-flag the old model as active

    after = _predict(client, user_id, PredictionRequest(eeg={"signal": _signal(), "sampling_rate": 128}))
    assert after["model"]["version"] == "model-b"  # later predictions use the newly active model
    _assert_consistent(after)


def test_activation_while_loading_the_trial_does_not_mix_models(two_model_app: TestClient,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic: model B is activated after the request has started (while the DEAP trial is read) but before
    the model runs. The request must still use model A throughout."""
    client = two_model_app
    admin_id, user_id = _ids(client)
    dataset = client.app.state.dataset  # type: ignore[attr-defined]
    original_load = dataset.load_trial
    switched: list[bool] = []

    def load_then_activate(participant_id: str, trial_index: int):  # noqa: ANN202
        data = original_load(participant_id, trial_index)
        if not switched:
            _activate(client, "model-b", admin_id)
            switched.append(True)
        return data

    monkeypatch.setattr(dataset, "load_trial", load_then_activate)
    result = _predict(client, user_id, PredictionRequest(deap_reference={"participant_id": "s01", "trial_index": 0}))
    assert switched and client.get("/api/model-info").json()["version"] == "model-b"
    assert result["model"]["version"] == "model-a"
    _assert_consistent(result)
    assert _stored(client) == [("model-a", pytest.approx(OUTPUT["model-a"]))]


def test_concurrent_predictions_and_activations_stay_consistent(two_model_app: TestClient) -> None:
    client = two_model_app
    admin_id, user_id = _ids(client)
    requests = [PredictionRequest(eeg={"signal": _signal(), "sampling_rate": 128}),
                PredictionRequest(deap_reference={"participant_id": "s01", "trial_index": 0})]
    stop = threading.Event()
    errors: list[BaseException] = []
    results: list[dict] = []
    switches = 0

    def predictor() -> None:
        try:
            i = 0
            while not stop.is_set():
                results.append(_predict(client, user_id, requests[i % 2]))
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=predictor) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for i in range(12):
            _activate(client, "model-b" if i % 2 == 0 else "model-a", admin_id)
            switches += 1
            time.sleep(0.02)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=60)

    assert not errors, errors
    assert switches == 12 and len(results) >= 8
    for result in results:
        _assert_consistent(result)
    assert {r["model"]["version"] for r in results} == {"model-a", "model-b"}  # both models really served requests
    for version, p in _stored(client):
        assert p == pytest.approx(OUTPUT[version])  # stored records are consistent too


def test_model_service_snapshot_is_one_model(two_model_app: TestClient) -> None:
    service = two_model_app.app.state.model_service  # type: ignore[attr-defined]
    snap = service.snapshot()
    assert snap.version == "model-a" and snap.model.p_first == OUTPUT["model-a"]
    with pytest.raises(AttributeError):
        snap.meta = {}  # frozen
