"""Affect-dimension models and calibrated confidence (app/affect), stored uploads, and XGBoost / CNN explanations.
Synthetic data only."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.affect import build as B
from app.affect import core as C
from app.affect.service import AffectService
from app.services.upload_store import upload_path
from app.xai import attribution as A
from tests.conftest import register_and_login, synthetic_trial
from tests.test_faced_api import faced_client, faced_predict  # noqa: F401  (fixture reused)

pytest.importorskip("shap")


def test_platt_scaling_recovers_a_known_miscalibration_and_lowers_the_error() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(len(s)) < C.apply_platt({"a": 0.5, "b": 0.3}, s)).astype(int)  # truth: weaker and shifted
    params = C.fit_platt(s, y)
    assert params["a"] == pytest.approx(0.5, abs=0.05) and params["b"] == pytest.approx(0.3, abs=0.05)
    assert C.ece(C.apply_platt(params, s), y) < 0.02 < C.ece(s, y)


def test_out_of_person_scores_never_see_the_person_and_personal_models_never_see_the_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    people, trials, segs = 4, 6, 3
    p = np.repeat([f"p{i}" for i in range(people)], trials * segs)
    t = np.tile(np.repeat(np.arange(trials), segs), people)
    y = np.tile(np.repeat(np.arange(trials) % 2, segs), people)
    # Column 0 = person, column 1 = trial, so every fit can be checked for what it was trained on.
    X = np.c_[np.char.lstrip(p, "p").astype(float), t, y + rng.normal(0, 1, (len(y), 1)), rng.normal(size=(len(y), 3))]
    calls = []
    real = B._fit_score

    def spy(make, Xtr, ytr, Xte):  # noqa: ANN001, ANN202
        calls.append((set(Xtr[:, 0].tolist()), set(Xtr[:, 1].tolist()), set(Xte[:, 0].tolist()), set(Xte[:, 1].tolist())))
        return real(make, Xtr, ytr, Xte)

    monkeypatch.setattr(B, "_fit_score", spy)
    monkeypatch.setattr(B, "JOBS", 1)  # in-process, so the spy sees every fit
    from sklearn.linear_model import LogisticRegression

    card = B.run_dimension("test", lambda: LogisticRegression(), None, X, y, p, t, [[f"p{i}"] for i in range(people)], None)
    lopo, personal = calls[:people], calls[people:]
    assert all(test_p.isdisjoint(train_p) for train_p, _, test_p, _ in lopo)
    assert len(personal) == people * trials
    for train_p, train_t, test_p, test_t in personal:
        assert train_p == test_p and len(train_p) == 1 and train_t.isdisjoint(test_t) and len(test_t) == 1
    assert sorted(card["out_of_person_scores"]) == [f"p{i}" for i in range(people)]
    assert {"a", "b"} <= set(card["calibration"]) and {"a", "b"} <= set(card["personal"]["calibration"])
    assert card["evaluation"]["participants"] == people and card["evaluation"]["trials"] == people * trials


def _card_with_model(tmp_path, model) -> AffectService:  # noqa: ANN001
    from types import SimpleNamespace

    out = tmp_path / "affect"
    out.mkdir()
    joblib.dump(model, out / "faced_arousal.joblib")
    ev = {"balanced_accuracy_per_participant": {"mean": 0.55, "ci_low": 0.5, "ci_high": 0.6}, "participants": 3, "pooled_auc": 0.56,
          "calibrated_accuracy": 0.6, "majority_accuracy": 0.6, "ece_raw": 0.1, "ece_calibrated": 0.02}
    dim = {"evaluation": ev, "calibration": {"a": 1.0, "b": 0.0}, "personal": {"evaluation": ev, "calibration": {"a": 1.0, "b": 0.0}}}
    card = {"faced": {"scale": "0-7", "dimensions": {
        "valence": dim,
        "arousal": {**dim, "model_file": "faced_arousal.joblib", "sha256": B.sha256(out / "faced_arousal.joblib"),
                    "out_of_person_scores": {"sub-000": {"3": 0.9}}}}}}
    (out / "affect_card.json").write_text(json.dumps(card))
    return AffectService(SimpleNamespace(output_dir=tmp_path))  # type: ignore[arg-type]


def test_estimates_use_the_held_out_score_for_a_training_participant_and_check_the_model_file(tmp_path) -> None:  # noqa: ANN001
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(2)
    model = LogisticRegression().fit(rng.normal(size=(40, 3)), np.r_[np.zeros(20), np.ones(20)].astype(int))
    svc = _card_with_model(tmp_path, model)
    X = rng.normal(size=(5, 3))
    held = svc.estimate("FACED", X, 0.3, "sub-000", 3, {"valence": 6.0, "arousal": 1.0})
    by = {d["key"]: d for d in held["dimensions"]}
    assert by["valence"]["raw_score"] == 0.3 and by["valence"]["predicted"] == "lower" and by["valence"]["own_label"] == "higher"
    assert by["arousal"]["raw_score"] == 0.9 and "held-out" in by["arousal"]["source"]
    assert by["arousal"]["probability_high"] == pytest.approx(0.9)  # identity calibration (a=1, b=0)
    new = {d["key"]: d for d in svc.estimate("FACED", X, 0.3, "sub-999", 3)["dimensions"]}
    assert new["arousal"]["raw_score"] == pytest.approx(model.predict_proba(X)[:, 1].mean())
    (tmp_path / "affect" / "faced_arousal.joblib").write_bytes(b"tampered")
    svc._models.clear()
    with pytest.raises(FileNotFoundError):
        svc.estimate("FACED", X, 0.3, "sub-999", 3)


def test_xgboost_tree_shap_adds_up_to_the_log_odds() -> None:
    from app.ml.models import candidate_models

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 8))
    labels = np.where(X[:, 0] > 0, "high_valence", "low_valence")
    model = candidate_models(0)["xgboost"]().fit(X, labels)
    att = A.explain(model, X[:25])
    p_high = model.predict_proba(X[:25])[:, A.high_class_index(model)]
    assert "XGBoost" in att.method and att.output == "log_odds" and att.max_abs_error < A.XGB_TOLERANCE
    assert np.allclose(1 / (1 + np.exp(-(att.base_value + att.values.sum(axis=1)))), p_high, atol=1e-5)
    assert (att.values[X[:25, 0] > 1, 0] > 0).all()  # feature 0 drives high valence


def test_cnn_integrated_gradients_are_complete_and_tied_to_the_network_output() -> None:
    import torch

    from app.ml.deep.classifier import CNNClassifier
    from app.ml.deep.cnn_model import EEGCNN, CNNConfig
    from app.ml.deep.data import ChannelNormalizer

    torch.manual_seed(0)
    cfg = CNNConfig(n_channels=4, n_samples=64, temporal_kernel=8, separable_kernel=4, pool1=2, pool2=4)
    model = CNNClassifier(EEGCNN(cfg), ["high_valence", "low_valence"], ChannelNormalizer(np.zeros(4), np.ones(4)))
    X = np.random.default_rng(4).normal(size=(3, 4, 64)).astype(np.float32)
    att = A.explain(model, X)
    p = model.predict_proba(X)
    assert att.method.startswith("Integrated gradients") and att.values.shape == (3, 4, 64)
    assert np.allclose(att.segment_outputs, np.log(p[:, 0] / p[:, 1]), atol=1e-5)
    assert att.max_abs_error < A.CNN_TOLERANCE


# ---------------------------------------------------------------- API (synthetic DEAP files, the shared trained model)

def test_trial_estimates_and_personal_models_through_the_api(trained_client: TestClient) -> None:
    settings = trained_client.app.state.settings  # type: ignore[attr-defined]
    out = settings.output_dir / "affect"
    out.mkdir(exist_ok=True)
    card = {"deap": B.build_deap(settings, out)}
    (out / "affect_card.json").write_text(json.dumps(card))
    h = register_and_login(trained_client)
    pred = trained_client.post("/api/predict", json={"deap_reference": {"participant_id": "s01", "trial_index": 0}}, headers=h).json()
    e = trained_client.get(f"/api/predictions/{pred['prediction_id']}/explanation", headers=h).json()
    dims = {d["key"]: d for d in e["affect"]["dimensions"]}
    assert list(dims) == ["valence", "arousal", "dominance", "liking"]
    assert dims["valence"]["raw_score"] == pytest.approx(pred["probabilities"]["high_valence"], abs=1e-12)
    assert all("held-out" in dims[k]["source"] for k in ("arousal", "dominance", "liking"))
    assert all(0 <= d["probability_high"] <= 1 and d["own_rating"] is not None for d in dims.values())
    personal = trained_client.post(f"/api/predictions/{pred['prediction_id']}/personal", headers=h).json()
    assert personal["available"] and personal["trial"] == 0
    assert all(d.get("training_trials") == 5 or not d["available"] for d in personal["dimensions"])  # 6 trials, one held out
    other = register_and_login(trained_client, "other@example.com", admin=False)
    assert trained_client.post(f"/api/predictions/{pred['prediction_id']}/personal", headers=other).status_code == 404


def test_real_uploads_are_stored_privately_and_explained_later_synthetic_ones_are_not(trained_client: TestClient) -> None:
    settings = trained_client.app.state.settings  # type: ignore[attr-defined]
    h = register_and_login(trained_client)
    body = {"signal": synthetic_trial(np.random.default_rng(5), True)[:32].tolist(), "sampling_rate": 128, "includes_baseline": True}
    up = trained_client.post("/api/predict", json={"eeg": body}, headers=h).json()
    me = trained_client.get("/api/auth/me", headers=h).json()
    assert upload_path(settings, me["id"], up["prediction_id"]).exists()
    e = trained_client.get(f"/api/predictions/{up['prediction_id']}/explanation", headers=h).json()
    assert e["available"] and e["probability_high_valence"] == pytest.approx(up["probabilities"]["high_valence"], abs=1e-12)
    personal = trained_client.post(f"/api/predictions/{up['prediction_id']}/personal", headers=h).json()
    assert not personal["available"]  # no ratings of other trials exist for an upload
    synth = trained_client.post("/api/predict", json={"eeg": {**body, "synthetic_test_signal": True}}, headers=h).json()
    assert not upload_path(settings, me["id"], synth["prediction_id"]).exists()


def test_personalisation_says_so_when_the_recordings_are_not_installed(faced_client: TestClient) -> None:  # noqa: F811
    h = register_and_login(faced_client)
    pred = faced_predict(faced_client, h, "sub-000", 3)
    r = faced_client.post(f"/api/predictions/{pred['prediction_id']}/personal", headers=h)
    assert r.status_code == 200 and not r.json()["available"] and "not available on this server" in r.json()["reason"]
