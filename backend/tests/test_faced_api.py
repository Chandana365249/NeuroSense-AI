"""FACED API: sign-in required, only validated participants listed, honest evaluation status per trial, a FACED-only
prediction path whose model must match the FACED pipeline, results saved per user with their provenance, and history
filtering across datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.database.base import Base
from app.main import create_app
from app.core.config import PROJECT_ROOT
from app.ml.faced import pipeline as fp
from app.repositories import PredictionRepository
from app.services.research_service import DEAP_ZERO_SHOT, deap_summary
from tests.conftest import make_settings, register_and_login
from tests.test_faced import make_participant


@pytest.fixture
def faced_client(tmp_path: Path, fake_deap_dir: Path) -> Iterator[TestClient]:
    root = tmp_path / "faced"
    root.mkdir()
    (root / "participants.tsv").write_text("participant_id\tage\tsex\nsub-000\t21\tF\nsub-001\t22\tM\nsub-002\t20\tF\n")
    (root / "stimuli").mkdir()
    (root / "stimuli" / "stimuli.tsv").write_text("stim_id\tsource_title\n" + "".join(f"{v}\tFilm {v}\n" for v in range(1, 29)))
    for i, pid in enumerate(("sub-000", "sub-001")):
        make_participant(root, pid, fs=250, seed=i)
    out = tmp_path / "outputs"
    faced_out = out / "faced"
    (faced_out / "models").mkdir(parents=True)
    trials = [{"video_index": v, "used": True, "segments": 7, "valence": float(v % 8), "duration_s": 35.0,
               "targeted_emotion": "Joy"} for v in range(1, 29)]
    report = {"summary": {"participants_declared": 3, "participants_valid": 2, "participants_failed": [["sub-002", "missing eeg.bdf"]],
                          "sampling_rates": {"250.0": 2}, "channels": ["Fp1"] * 30, "trials_used": 56, "segments": 392},
              "participants": [{"participant_id": "sub-000", "ok": True, "sampling_rate": 250.0, "trials": trials},
                               {"participant_id": "sub-001", "ok": True, "sampling_rate": 250.0, "trials": trials},
                               {"participant_id": "sub-002", "ok": False, "error": "missing eeg.bdf"}]}
    (faced_out / "validation_report.json").write_text(json.dumps(report))
    (faced_out / "protocol_split_v1.json").write_text(json.dumps({"test_participants": ["sub-001"], "dev_participants": ["sub-000"],
                                                                  "test_videos": [1, 2], "dev_videos": list(range(3, 29))}))
    rng = np.random.default_rng(0)
    n = len(fp.feature_names())
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]).fit(rng.standard_normal((40, n)), np.r_[np.zeros(20), np.ones(20)].astype(int))
    joblib.dump(model, faced_out / "models" / "faced_S1_lr.joblib")
    ev = {"model": "lr", "test_trials": 2, "trial_balanced_accuracy": 0.55, "trial_balanced_accuracy_ci95": [0.4, 0.7],
          "trial_accuracy": 0.56, "trial_accuracy_ci95": [0.4, 0.7], "trial_macro_f1": 0.5, "trial_roc_auc": 0.6,
          "majority_reference_accuracy": 0.5, "segment_balanced_accuracy": 0.52}
    card = {"name": "faced-valence-lr-S1", "algorithm": {"name": "lr"}, "artifact": "faced_S1_lr.joblib", "pipeline": fp.PIPELINE_VERSION,
            "dataset": {"id": "nm000112"}, "feature_schema": {"n_features": n, "sha256": hashlib.sha256("\n".join(fp.feature_names()).encode()).hexdigest()},
            "dataset_compatibility": "FACED only", "trained_on": {"participants": 1}, "target_80_reached": False,
            "evaluation": {"S1_strict_primary": ev, "S2_unseen_people_seen_clips": ev, "S3_strict_unsupervised_calibration": ev},
            "status": "EXPERIMENTAL", "limitations": []}
    (faced_out / "models" / "model_card.json").write_text(json.dumps(card))
    settings = make_settings(fake_deap_dir, out, tmp_path / "test.sqlite3").model_copy(update={"faced_dataset_dir": root})
    app = create_app(settings)
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        yield client


def test_faced_endpoints_require_sign_in(faced_client: TestClient) -> None:
    for method, path in (("get", "/api/faced/status"), ("get", "/api/faced/participants"), ("get", "/api/faced/evidence"),
                         ("get", "/api/faced/participants/sub-000/trials")):
        assert getattr(faced_client, method)(path).status_code == 401
    assert faced_client.post("/api/faced/predict", json={"participant_id": "sub-000", "video_index": 1}).status_code == 401


def test_status_reports_validated_counts_and_honest_model_state(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    s = faced_client.get("/api/faced/status", headers=h).json()
    assert s["validated"] and s["participants_valid"] == 2 and s["participants_declared"] == 3 and s["participants_failed"] == 1
    assert s["model"]["available"] and s["model"]["target_80_reached"] is False and "FACED only" in s["model"]["dataset_compatibility"]


def test_only_validated_participants_are_listed_and_searchable(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    r = faced_client.get("/api/faced/participants", headers=h).json()
    assert [p["participant_id"] for p in r["participants"]] == ["sub-000", "sub-001"]
    assert {p["participant_id"]: p["role"] for p in r["participants"]} == {"sub-000": "development", "sub-001": "final_test"}
    assert [p["participant_id"] for p in faced_client.get("/api/faced/participants?q=001", headers=h).json()["participants"]] == ["sub-001"]
    assert faced_client.get("/api/faced/participants/sub-002/trials", headers=h).status_code == 404


def test_trials_carry_ratings_labels_and_evaluation_status(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    t = {x["video_index"]: x for x in faced_client.get("/api/faced/participants/sub-001/trials", headers=h).json()["trials"]}
    assert len(t) == 28 and t[1]["evaluation_status"] == "strict_held_out" and t[5]["evaluation_status"] == "unseen_person_seen_clip"
    assert t[4]["self_reported_valence"] == 4.0 and t[4]["valence_label"] == "high_valence" and t[3]["valence_label"] == "low_valence"
    d = {x["video_index"]: x for x in faced_client.get("/api/faced/participants/sub-000/trials", headers=h).json()["trials"]}
    assert d[1]["evaluation_status"] == "seen_person_unseen_clip" and d[5]["evaluation_status"] == "in_sample"


def faced_predict(client: TestClient, headers: dict[str, str], participant: str = "sub-001", video: int = 1, **extra: object) -> dict:
    r = client.post("/api/faced/predict", headers=headers, json={"participant_id": participant, "video_index": video, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_prediction_uses_the_faced_path_and_is_saved_with_its_provenance(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    body = faced_predict(faced_client, h)
    assert body["stored"] is True and body["pipeline"] == fp.PIPELINE_VERSION and body["segments"] == 7
    assert 0 <= body["probability_high_valence"] <= 1 and body["evaluation_status"] == "strict_held_out"
    assert body["source"] == "FACED recording (real EEG)"

    saved = faced_client.get(f"/api/predictions/{body['prediction_id']}", headers=h).json()
    assert saved["input_source"] == "faced_reference" and saved["prediction"] == body["predicted_label"]
    assert saved["probabilities"]["high_valence"] == body["probability_high_valence"]
    assert abs(sum(saved["probabilities"].values()) - 1) < 1e-9 and saved["confidence"] >= 0.5
    assert saved["segments_used"] == 7 and saved["created_at"]
    assert saved["model"]["version"] == "faced-valence-lr-S1"
    ref = saved["source_reference"]
    assert ref["dataset"] == "FACED" and ref["dataset_version"] == "1.1.4" and ref["participant_id"] == "sub-001"
    assert ref["video_index"] == 1 and ref["model_version"] == "faced-valence-lr-S1" and ref["pipeline"] == fp.PIPELINE_VERSION
    assert ref["evaluation_status"] == "strict_held_out" and "Strict held-out" in ref["evaluation_text"]
    assert ref["self_reported_label"] in ("high_valence", "low_valence") and len(ref["segment_probabilities"]) == 7
    assert ref["model_strict_balanced_accuracy"] == 0.55


def test_each_user_sees_only_their_own_faced_records(faced_client: TestClient) -> None:
    a = register_and_login(faced_client, "a@example.com", admin=False)
    b = register_and_login(faced_client, "b@example.com", admin=False)
    pid = faced_predict(faced_client, a)["prediction_id"]
    assert faced_client.get(f"/api/predictions/{pid}", headers=b).status_code == 404
    assert faced_client.get("/api/predictions", headers=b).json()["total"] == 0
    assert faced_client.get("/api/predictions?dataset=faced&q=sub-001", headers=b).json()["total"] == 0
    assert faced_client.get("/api/predictions", headers=a).json()["total"] == 1
    assert faced_client.get("/api/predictions?dataset=faced").status_code == 401
    assert faced_client.get(f"/api/predictions/{pid}").status_code == 401


def test_faced_records_follow_analysis_sessions_and_refuse_another_users_session(faced_client: TestClient) -> None:
    a = register_and_login(faced_client, "a@example.com", admin=False)
    b = register_and_login(faced_client, "b@example.com", admin=False)
    session = faced_client.post("/api/sessions", headers=a, json={"name": "first"}).json()["id"]
    other = faced_client.post("/api/sessions", headers=a, json={"name": "second"}).json()["id"]
    in_session = faced_predict(faced_client, a, session_id=session)
    faced_predict(faced_client, a, video=2)
    assert in_session["session_id"] == session
    listed = faced_client.get(f"/api/predictions?session_id={session}", headers=a).json()
    assert [p["prediction_id"] for p in listed["items"]] == [in_session["prediction_id"]] and listed["total"] == 1
    assert faced_client.get(f"/api/predictions?session_id={other}", headers=a).json()["total"] == 0
    r = faced_client.post("/api/faced/predict", headers=b, json={"participant_id": "sub-001", "video_index": 1, "session_id": session})
    assert r.status_code == 404
    assert faced_client.get("/api/predictions", headers=b).json()["total"] == 0


def add_record(client: TestClient, email_id: int, source: str, ref: dict, label: str = "high_valence") -> None:
    with client.app.state.db.session() as db:  # type: ignore[attr-defined]
        PredictionRepository(db).create(user_id=email_id, predicted_label=label, confidence=0.6, probabilities={"high_valence": 0.6, "low_valence": 0.4},
                                        segments_used=1, segments_rejected=0, input_source=source, source_reference=ref)
        db.commit()


def test_history_filters_by_dataset_and_searches_by_participant(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    uid = faced_client.get("/api/auth/me", headers=h).json()["id"]
    add_record(faced_client, uid, "deap_reference", {"participant_id": "s07", "trial_index": 3})
    add_record(faced_client, uid, "upload", {"channels": 32, "samples": 512, "sampling_rate": 128})
    add_record(faced_client, uid, "upload", {"channels": 32, "samples": 512, "sampling_rate": 128, "synthetic_test_signal": True})
    faced_predict(faced_client, h, "sub-001", 2)

    def sources(query: str) -> list[str]:
        return [p["input_source"] for p in faced_client.get(f"/api/predictions?{query}", headers=h).json()["items"]]

    assert len(sources("")) == 4
    assert sources("dataset=deap") == ["deap_reference"] and sources("dataset=faced") == ["faced_reference"]
    synthetic = faced_client.get("/api/predictions?dataset=synthetic", headers=h).json()
    assert synthetic["total"] == 1 and synthetic["items"][0]["source_reference"]["synthetic_test_signal"] is True
    uploaded = faced_client.get("/api/predictions?dataset=upload", headers=h).json()
    assert uploaded["total"] == 1 and "synthetic_test_signal" not in uploaded["items"][0]["source_reference"]
    assert sources("q=s07") == ["deap_reference"] and sources("q=7") == ["deap_reference"]
    assert sources("q=SUB-001") == ["faced_reference"] and sources("q=1") == ["faced_reference"]
    assert sources("q=zzz") == [] and sources("q=%25") == [] and sources("q=_") == []
    assert sources("dataset=deap&q=sub-001") == []
    assert faced_client.get("/api/predictions?dataset=other", headers=h).status_code == 422
    assert faced_client.get("/api/predictions?dataset=faced&limit=1", headers=h).json()["total"] == 1


def test_deap_research_summary_comes_from_the_saved_file_and_reports_a_missing_one(faced_client: TestClient, tmp_path: Path) -> None:
    assert faced_client.get("/api/research/deap-summary").status_code == 401
    h = register_and_login(faced_client, "u@example.com", admin=False)
    body = faced_client.get("/api/research/deap-summary", headers=h).json()
    saved = json.loads((PROJECT_ROOT / "docs/results" / DEAP_ZERO_SHOT).read_text(encoding="utf-8"))["group"]["balanced_accuracy"]["personal"]
    assert body["available"] and body["value"] == saved["mean"] and body["ci95"] == [saved["ci_low"], saved["ci_high"]]
    assert body["metric"] == "trial_balanced_accuracy" and body["level"] == "trial" and body["n_participants"] == 12
    assert body["independent_test"] is False

    settings = faced_client.app.state.settings  # type: ignore[attr-defined]
    assert deap_summary(settings.model_copy(update={"research_results_dir": tmp_path})) == {
        "available": False, "message": "The saved DEAP result file could not be read."}
    broken = tmp_path / DEAP_ZERO_SHOT
    broken.parent.mkdir(parents=True)
    broken.write_text('{"group": {}}')
    assert deap_summary(settings.model_copy(update={"research_results_dir": tmp_path}))["available"] is False


def test_prediction_rejects_bad_input_and_unknown_participants(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    assert faced_client.post("/api/faced/predict", headers=h, json={"participant_id": "sub-001", "video_index": 29}).status_code == 422
    assert faced_client.post("/api/faced/predict", headers=h, json={"participant_id": "../x", "video_index": 1}).status_code == 422
    assert faced_client.post("/api/faced/predict", headers=h, json={"participant_id": "sub-002", "video_index": 1}).status_code == 404


def test_a_model_that_does_not_match_the_faced_pipeline_is_refused(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    path = faced_client.app.state.settings.faced_output_dir / "models" / "model_card.json"  # type: ignore[attr-defined]
    card = json.loads(path.read_text())
    card["feature_schema"]["n_features"] = 440  # a DEAP-sized schema
    path.write_text(json.dumps(card))
    assert faced_client.post("/api/faced/predict", headers=h, json={"participant_id": "sub-001", "video_index": 1}).status_code == 503


@pytest.mark.parametrize("change", [{"dataset": {"id": "deap"}}, {"pipeline": "deap-v1"}])
def test_a_model_from_another_dataset_or_pipeline_is_refused(faced_client: TestClient, change: dict) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    path = faced_client.app.state.settings.faced_output_dir / "models" / "model_card.json"  # type: ignore[attr-defined]
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    assert faced_client.get("/api/faced/status", headers=h).json()["model"]["available"] is False
    assert faced_client.post("/api/faced/predict", headers=h, json={"participant_id": "sub-001", "video_index": 1}).status_code == 503


def test_a_changed_feature_schema_with_the_same_count_is_refused(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    path = faced_client.app.state.settings.faced_output_dir / "models" / "model_card.json"  # type: ignore[attr-defined]
    card = json.loads(path.read_text())
    card["feature_schema"]["sha256"] = "0" * 64
    path.write_text(json.dumps(card))
    assert faced_client.post("/api/faced/predict", headers=h, json={"participant_id": "sub-001", "video_index": 1}).status_code == 503


def test_the_deap_prediction_path_refuses_faced_participants(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    r = faced_client.post("/api/predict", headers=h, json={"deap_reference": {"participant_id": "sub-001", "trial_index": 0}})
    assert r.status_code == 422


def test_evidence_reports_missing_results_instead_of_inventing_them(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    body = faced_client.get("/api/faced/evidence", headers=h).json()
    assert body["available"] is False and "settings" not in body


def test_evidence_returns_the_measured_values_from_the_result_files(faced_client: TestClient) -> None:
    h = register_and_login(faced_client, "u@example.com", admin=False)
    trial = {"n": 2, "accuracy": 0.5, "balanced_accuracy": 0.566, "f1_macro": 0.5, "roc_auc": 0.5}
    block = {"selected": {"name": "lr"}, "selected_on": "S1", "train_trials": 1, "test_trials": 2, "train_high_share": 0.4,
             "majority_reference": trial, "trial_pooled": trial, "trial_pooled_ci95_participant_bootstrap": {"balanced_accuracy": [0.5, 0.6]},
             "trial_confusion": {"matrix": [[1, 0], [0, 1]]}, "per_participant_balanced_accuracy": {"mean": 0.5},
             "per_participant_accuracy": {"mean": 0.5}, "segment_level": trial, "selection_scores_inner": {"lr": 0.55},
             "per_participant": {"sub-001": trial}}
    out = faced_client.app.state.settings.faced_output_dir  # type: ignore[attr-defined]
    (out / "final_results.json").write_text(json.dumps({"protocol": "p", "split_sha256": "x", "settings": {"S1": block}}))
    body = faced_client.get("/api/faced/evidence", headers=h).json()
    assert body["available"] and body["settings"]["S1"]["trial"]["balanced_accuracy"] == 0.566
    assert body["strict_per_participant"][0]["participant_id"] == "sub-001"
    assert body["exploratory"] == {"stimulus_category": None, "noise_removal": None}
