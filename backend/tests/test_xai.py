"""Explanations (app/xai): exact attribution, feature order, class direction, verification against stored
predictions, refusal for synthetic signals and leakage-free global reference data. Synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.dataset import EEG_CHANNEL_NAMES
from app.ml.faced import pipeline as fp
from app.ml.faced.dataset import FACED_CHANNELS
from app.ml.models import EncodedLabelClassifier
from app.preprocessing.features import FeatureExtractor
from app.xai import attribution as A
from tests.conftest import register_and_login, synthetic_trial
from tests.test_faced_api import faced_client, faced_predict  # noqa: F401  (fixture reused)

pytest.importorskip("shap")


def _data(n: int = 200, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)) * 5 + 3
    return X, (X[:, 0] > 3).astype(int)


def test_feature_info_follows_the_extractor_order_for_both_datasets() -> None:
    for channels, names in ((EEG_CHANNEL_NAMES, FeatureExtractor().feature_names), (FACED_CHANNELS, fp.feature_names())):
        info = A.feature_info(names)
        assert [f["name"] for f in info] == names
        assert {f["channel"] for f in info if f["kind"] == "channel"} == set(channels)
        assert info[0] == {"name": f"{channels[0]}_logpow_theta", "channel": channels[0], "family": "logpow", "band": "theta", "kind": "channel"}
        entropy = next(f for f in info if f["family"] == "spectral_entropy")
        assert entropy["band"] == "broadband"
        asym = next(f for f in info if f["kind"] == "pair")
        assert asym["family"] == "asym" and asym["band"] in A.BAND_LABELS and "-" in asym["channel"]
    assert len(FeatureExtractor().feature_names) == 32 * 12 + 14 * 4 == 440
    assert len(fp.feature_names()) == 30 * 12 + 13 * 4 == 412


def test_tree_shap_adds_up_to_the_probability_and_points_the_right_way() -> None:
    X, y = _data()
    labels = np.where(y == 1, "high_valence", "low_valence")
    model = EncodedLabelClassifier(Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(50, random_state=0))])).fit(X, labels)
    att = A.explain(model, X[:20])
    hi = list(model.classes_).index("high_valence")
    assert att.output == "probability" and att.max_abs_error < 1e-9
    assert np.allclose(att.base_value + att.values.sum(axis=1), model.predict_proba(X[:20])[:, hi])
    high_rows = X[:20, 0] > 6
    assert (att.values[high_rows, 0] > 0).all() and (att.values[X[:20, 0] < 0, 0] < 0).all()
    again = A.explain(model, X[:20])
    assert np.array_equal(again.values, att.values)  # reproducible


def test_linear_attribution_reconstructs_the_log_odds_including_scaling_and_intercept() -> None:
    X, y = _data()
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]).fit(X, y)
    att = A.explain(model, X[:30])
    assert att.output == "log_odds" and att.max_abs_error < 1e-9
    assert att.base_value == pytest.approx(float(model[-1].intercept_[0]))
    assert np.allclose(att.base_value + att.values.sum(axis=1), model.decision_function(X[:30]))
    z = (X[:30] - model[0].mean_) / model[0].scale_
    assert np.allclose(att.values, model[-1].coef_[0] * z)  # scaling accounted for exactly
    assert (att.values[X[:30, 0] > 6, 0] > 0).all()  # feature 0 drives class 1 = high valence


def test_an_unsupported_model_is_refused_rather_than_explained() -> None:
    from sklearn.svm import SVC

    X, y = _data()
    model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]).fit(X, y)
    with pytest.raises(TypeError):
        A.explain(model, X[:3])


# ---------------------------------------------------------------- API, DEAP (synthetic data, trained Random Forest)

def test_deap_explanation_matches_the_stored_prediction_and_leaves_it_unchanged(trained_client: TestClient) -> None:
    h = register_and_login(trained_client)
    pred = trained_client.post("/api/predict", json={"deap_reference": {"participant_id": "s01", "trial_index": 0}}, headers=h).json()
    r = trained_client.get(f"/api/predictions/{pred['prediction_id']}/explanation", headers=h)
    assert r.status_code == 200
    e = r.json()
    assert e["available"] and e["dataset"] == "DEAP" and e["output_kind"] == "probability" and e["target_class"] == "high_valence"
    v = e["verification"]
    assert v["matches_stored_prediction"] and v["max_segment_reconstruction_error"] < 1e-9
    assert e["probability_high_valence"] == pytest.approx(pred["probabilities"]["high_valence"], abs=1e-12)
    assert e["trial_output"] == pytest.approx(pred["probabilities"]["high_valence"], abs=1e-9)
    total = sum(g["contribution"] for g in e["by_band"])
    assert e["base_value"] + total == pytest.approx(e["trial_output"], abs=1e-9)  # every feature has exactly one band
    assert [x["name"] for x in e["ratings"]] == ["Valence", "Arousal", "Dominance", "Liking"]
    assert len(e["trial_view"]["log_band_power"]) == 32 and len(e["trial_view"]["signal"]["segments"][0]["data"]) == 32
    again = trained_client.get(f"/api/predictions/{pred['prediction_id']}", headers=h).json()
    assert again["probabilities"] == pred["probabilities"]  # explaining did not change the stored result


def test_synthetic_uploads_are_never_explained_and_real_uploads_need_the_same_signal(trained_client: TestClient) -> None:
    h = register_and_login(trained_client)
    signal = synthetic_trial(np.random.default_rng(3), True)[:32].tolist()
    body = {"signal": signal, "sampling_rate": 128, "includes_baseline": True}
    synth = trained_client.post("/api/predict", json={"eeg": {**body, "synthetic_test_signal": True}}, headers=h).json()
    e = trained_client.get(f"/api/predictions/{synth['prediction_id']}/explanation", headers=h).json()
    assert not e["available"] and "synthetic" in e["reason"]
    e = trained_client.post(f"/api/predictions/{synth['prediction_id']}/explanation", json={"eeg": body}, headers=h).json()
    assert not e["available"]

    up = trained_client.post("/api/predict", json={"eeg": body}, headers=h).json()
    e = trained_client.get(f"/api/predictions/{up['prediction_id']}/explanation", headers=h).json()
    assert e["available"]  # the upload was stored privately, so it can be explained without re-sending
    e = trained_client.post(f"/api/predictions/{up['prediction_id']}/explanation", json={"eeg": body}, headers=h).json()
    assert e["available"] and e["probability_high_valence"] == pytest.approx(up["probabilities"]["high_valence"], abs=1e-12)
    other = {**body, "signal": [row[:-128] for row in signal]}
    e = trained_client.post(f"/api/predictions/{up['prediction_id']}/explanation", json={"eeg": other}, headers=h).json()
    assert not e["available"]


def test_a_stored_result_that_cannot_be_reproduced_is_not_explained(trained_client: TestClient) -> None:
    h = register_and_login(trained_client)
    pred = trained_client.post("/api/predict", json={"deap_reference": {"participant_id": "s01", "trial_index": 0}}, headers=h).json()
    from app.models import Prediction

    with trained_client.app.state.db.session() as s:  # type: ignore[attr-defined]
        item = s.get(Prediction, pred["prediction_id"])
        item.probabilities = {"high_valence": 0.123, "low_valence": 0.877}
        s.commit()
    e = trained_client.get(f"/api/predictions/{pred['prediction_id']}/explanation", headers=h).json()
    assert not e["available"] and "reproduce" in e["reason"]


def test_explanations_belong_to_their_owner(trained_client: TestClient) -> None:
    h = register_and_login(trained_client)
    pred = trained_client.post("/api/predict", json={"deap_reference": {"participant_id": "s01", "trial_index": 0}}, headers=h).json()
    other = register_and_login(trained_client, "other@example.com", admin=False)
    assert trained_client.get(f"/api/predictions/{pred['prediction_id']}/explanation", headers=other).status_code == 404
    assert trained_client.get(f"/api/predictions/{pred['prediction_id']}/explanation").status_code == 401


# ---------------------------------------------------------------- API, FACED (synthetic recordings, logistic regression)

def test_faced_explanation_reconstructs_the_log_odds(faced_client: TestClient) -> None:  # noqa: F811
    h = register_and_login(faced_client)
    pred = faced_predict(faced_client, h, "sub-000", 3)
    e = faced_client.get(f"/api/predictions/{pred['prediction_id']}/explanation", headers=h).json()
    assert e["available"] and e["dataset"] == "FACED" and e["output_kind"] == "log_odds"
    assert e["verification"]["max_segment_reconstruction_error"] < 1e-9
    assert e["probability_high_valence"] == pytest.approx(pred["probability_high_valence"], abs=1e-12)
    assert {x["name"] for x in e["ratings"]} <= {"Valence", "Arousal", "Liking", "Familiarity"}
    assert len(e["trial_view"]["log_band_power"]) == 30


def test_global_explanation_uses_only_development_rows_and_reports_when_missing(faced_client: TestClient, tmp_path: Path) -> None:  # noqa: F811
    h = register_and_login(faced_client)
    assert not faced_client.get("/api/explain/global/faced", headers=h).json()["available"]
    settings = faced_client.app.state.settings  # type: ignore[attr-defined]
    names = fp.feature_names()
    rng = np.random.default_rng(1)
    p = np.array(["sub-000"] * 10 + ["sub-001"] * 10)
    v = np.array([3, 4, 5, 6, 7, 1, 2, 8, 9, 10] * 2)  # sub-001 is a test participant; clips 1 and 2 are test clips
    np.savez(settings.faced_output_dir / f"features_{fp.PIPELINE_VERSION}.npz", X=rng.standard_normal((20, len(names))),
             participant=p, video=v, feature_names=np.array(names))
    from app.xai.build_global import build_faced

    build_faced(settings)
    g = faced_client.get("/api/explain/global/faced", headers=h).json()
    assert g["available"] and g["reference"]["participants"] == ["sub-000"]
    assert g["n_segments"] == 8  # sub-000 without clips 1 and 2
    assert {b["group"] for b in g["by_band"]} == {"theta", "alpha", "beta", "gamma", "broadband"}
    assert json.dumps(g).count("sub-001") == 0


def test_deap_global_explanation_uses_the_models_training_split_only(trained_client: TestClient) -> None:
    h = register_and_login(trained_client)
    missing = trained_client.get("/api/explain/global/deap", headers=h)
    assert missing.status_code == 200 and not missing.json()["available"]
    from app.xai.build_global import build_deap

    settings = trained_client.app.state.settings  # type: ignore[attr-defined]
    build_deap(settings)
    g = trained_client.get("/api/explain/global/deap", headers=h).json()
    train = trained_client.app.state.model_service.metadata["split"]["train"]  # type: ignore[attr-defined]
    assert g["available"] and g["reference"]["participants"] == sorted(train)
    assert g["output_kind"] == "probability" and len(g["channels"]) == 32 and len(g["channel_band_map"]) == 32
    assert all(r["mean_abs"] >= 0 and r["sd_across_participants"] >= 0 for r in g["by_band"])
