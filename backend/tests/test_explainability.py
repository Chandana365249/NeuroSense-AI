"""Global feature importance: computed from the real trained forest, aligned with the feature schema, and honestly
unavailable for unsupported models or mismatched schemas. Temporary SQLite databases only."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ml.models import candidate_models
from app.services.explainability import describe_feature, global_importance
from app.services.model_service import ModelSnapshot
from tests.test_api_hardening import WRITE_SQL, captured_sql


def _snapshot(client: TestClient) -> ModelSnapshot:
    return client.app.state.model_service.snapshot()  # type: ignore[attr-defined]


def test_global_importance_matches_the_trained_forest(trained_client: TestClient) -> None:
    with captured_sql(trained_client) as statements:
        r = trained_client.get("/api/explain/global")
    assert r.status_code == 200
    body = r.json()
    snap = _snapshot(trained_client)
    forest = snap.model.estimator_.steps[-1][1]
    names = snap.meta["feature_schema"]["names"]
    assert body["available"] is True and body["scope"] == "global"
    assert body["model_version"] == snap.meta["version"] and body["n_features"] == len(names) == 440
    assert body["n_trees"] == len(forest.estimators_)
    top = body["top"]
    assert len(top) == 20 and [t["rank"] for t in top] == list(range(1, 21))
    assert all(a["importance"] >= b["importance"] for a, b in zip(top, top[1:], strict=False))
    expected = {names[i]: v for i, v in enumerate(forest.feature_importances_)}
    for t in top:
        assert t["importance"] == pytest.approx(expected[t["name"]])  # values come from the model itself
    assert max(expected.values()) == pytest.approx(top[0]["importance"])
    assert sum(g["importance"] for g in body["by_family"]) == pytest.approx(1.0)
    assert sum(g["importance"] for g in body["by_band"]) == pytest.approx(1.0)
    assert "does not explain any single prediction" in body["interpretation"]
    assert [s for s in statements if WRITE_SQL.match(s)] == []  # read-only


@pytest.mark.parametrize(("top", "status", "count"), [(5, 200, 5), (100, 200, 100), (0, 422, None), (101, 422, None)])
def test_top_parameter_is_bounded(trained_client: TestClient, top: int, status: int, count: int | None) -> None:
    r = trained_client.get(f"/api/explain/global?top={top}")
    assert r.status_code == status
    if count:
        assert len(r.json()["top"]) == count


def test_no_model_reports_unavailable(client: TestClient) -> None:
    body = client.get("/api/explain/global").json()
    assert body["available"] is False and "No trained model" in body["reason"] and body["top"] == []


def test_response_has_no_paths(trained_client: TestClient, trained_output_dir) -> None:  # noqa: ANN001
    raw = trained_client.get("/api/explain/global?top=100").text
    assert str(trained_output_dir) not in raw and "joblib" not in raw and ":\\\\" not in raw


def test_unsupported_models_are_not_explained(trained_client: TestClient) -> None:
    snap = _snapshot(trained_client)
    cnn = ModelSnapshot(model=object(), meta={**snap.meta, "input": {"type": "segments"}, "algorithm": "cnn"},
                        preprocessor=snap.preprocessor, extractor=None)
    out = global_importance(cnn)
    assert out["available"] is False and "neural network" in out["reason"] and "top" not in out

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 440)), np.array(["high_valence", "low_valence"] * 20)
    dummy = candidate_models(0)["majority_baseline"]().fit(X, y)
    out = global_importance(ModelSnapshot(dummy, snap.meta, snap.preprocessor, snap.extractor))
    assert out["available"] is False and "only supported for the Random Forest" in out["reason"]


def test_schema_mismatch_is_not_explained(trained_client: TestClient) -> None:
    snap = _snapshot(trained_client)
    schema = dict(snap.meta["feature_schema"])
    schema["names"] = schema["names"][:-1]  # one name missing: values can no longer be attributed
    out = global_importance(ModelSnapshot(snap.model, {**snap.meta, "feature_schema": schema}, snap.preprocessor, snap.extractor))
    assert out["available"] is False and "do not match" in out["reason"]


@pytest.mark.parametrize(("name", "expected"), [
    ("Fp1_logpow_alpha", {"channel": "Fp1", "family": "logpow", "band": "alpha"}),
    ("O2_relpow_gamma", {"channel": "O2", "family": "relpow", "band": "gamma"}),
    ("Cz_hjorth_mobility", {"channel": "Cz", "family": "hjorth_mobility", "band": None}),
    ("asym_Fp2-Fp1_theta", {"channel": "Fp2-Fp1", "family": "asym", "band": "theta"}),
])
def test_feature_names_are_parsed(name: str, expected: dict) -> None:
    assert describe_feature(name) == expected


def test_every_schema_name_is_parseable(trained_client: TestClient) -> None:
    names = _snapshot(trained_client).meta["feature_schema"]["names"]
    families = {describe_feature(n)["family"] for n in names}
    assert families == {"logpow", "relpow", "spectral_entropy", "hjorth_mobility", "hjorth_complexity", "log_variance", "asym"}
    json.dumps([describe_feature(n) for n in names])
