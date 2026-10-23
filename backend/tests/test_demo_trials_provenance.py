"""Demo participants and training provenance come from recorded metadata and the readable dataset only.

Regression tests for the Analyze EEG participant list: it must match the backend's eligible, available data, never
include frozen test participants, never claim a participant was excluded from training without recorded provenance,
and explain (not fake) an unavailable dataset."""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.routes.ml import demo_trials
from app.database.base import Base
from app.main import create_app
from app.ml.factory import make_dataset
from app.ml.participants import FROZEN_TEST_PARTICIPANTS
from app.services.public_metadata import training_provenance
from tests.conftest import N_TRIALS, PARTICIPANTS, make_settings, register_and_login


def _fixed_split_meta(**split_overrides):
    split = {"train": ["s02", "s03"], "validation": ["s04"], "test": ["s01", "s05"],
             "segments": {"train": 20, "validation": 10, "test": 20}}
    split.update(split_overrides)
    return {"version": "v-test", "split": split, "training_config": {"final_fit": "train + validation participants"}}


def _request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


# ---------- training_provenance (pure) ----------
def test_fixed_split_provenance_uses_recorded_final_fit():
    p = training_provenance(_fixed_split_meta())
    assert p["recorded"] and p["kind"] == "fixed_split"
    assert p["fit_participants"] == ["s02", "s03", "s04"]
    assert p["selection_participants"] == ["s04"] and p["held_out_participants"] == ["s01", "s05"]
    assert p["fit_segments"] == 30
    assert p["fit_trials"] is None  # not recorded for fixed-split models; never derived


def test_unrecorded_final_fit_reports_nothing_rather_than_guessing():
    meta = _fixed_split_meta()
    meta["training_config"] = {}
    p = training_provenance(meta)
    assert p["recorded"] is False and p["kind"] == "unknown"
    assert p["fit_participants"] == [] and p["held_out_participants"] == []
    assert "does not record" in p["note"]


def test_unknown_final_fit_value_is_not_interpreted():
    meta = _fixed_split_meta()
    meta["training_config"] = {"final_fit": "train only"}
    assert training_provenance(meta)["recorded"] is False


def test_protocol_provenance_counts_recorded_trials():
    meta = {"split": {"train": ["s01", "s02"], "validation": [], "test": [], "segments": {"train": 50}},
            "training_config": {"final_fit": "all development participants"},
            "protocol": {"trial_groups": {"s01": [0, 1, 2], "s02": [0, 1]}}}
    p = training_provenance(meta)
    assert p["kind"] == "protocol_all_development" and p["fit_participants"] == ["s01", "s02"]
    assert p["held_out_participants"] == [] and p["fit_trials"] == 5 and p["fit_segments"] == 50


def test_no_metadata_has_no_provenance():
    assert training_provenance(None) is None and training_provenance({}) is None


# ---------- /dataset/demo-trials ----------
def test_demo_offers_every_authorized_readable_participant_with_its_recorded_role(trained_client):
    info = trained_client.get("/api/model-info").json()
    prov = info["provenance"]
    assert prov["recorded"] and prov["kind"] == "fixed_split"
    assert prov["held_out_participants"] == sorted(info["split"]["test"])
    assert prov["fit_participants"] == sorted(info["split"]["train"] + info["split"]["validation"])

    demo = trained_client.get("/api/dataset/demo-trials").json()
    authorized = trained_client.app.state.settings.development_participants
    # Input availability is the authorized, readable data, not the model's evaluation split.
    assert [p["participant_id"] for p in demo["participants"]] == authorized == PARTICIPANTS
    for p in demo["participants"]:
        assert p["trial_indices"] == list(range(N_TRIALS))  # actual trials on disk, not an assumed 40
        pid = p["participant_id"]
        expected = ("selection" if pid in prov["selection_participants"] else "fit" if pid in prov["fit_participants"]
                    else "held_out" if pid in prov["held_out_participants"] else "not_used")
        assert p["model_role"] == expected
    assert demo["held_out_participants"] == prov["held_out_participants"]
    assert demo["trials_per_participant"] == N_TRIALS and demo["unavailable"] == []
    assert demo["dataset_available"] is True and demo["model_version"] == info["version"]


def test_demo_lists_only_authorized_participants_and_predict_refuses_others(trained_settings):
    s = trained_settings.model_copy(update={"development_participants": ["s01", "s02"]})
    app = create_app(s)
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        demo = client.get("/api/dataset/demo-trials").json()
        assert [p["participant_id"] for p in demo["participants"]] == ["s01", "s02"]
        headers = register_and_login(client)
        ok = client.post("/api/predict", headers=headers, json={"deap_reference": {"participant_id": "s02", "trial_index": 1}})
        refused = client.post("/api/predict", headers=headers, json={"deap_reference": {"participant_id": "s03", "trial_index": 1}})
    assert ok.status_code == 201
    assert refused.status_code == 404  # a file on disk is not enough; the participant must be authorized


def test_demo_explains_missing_dataset_without_inventing_participants(tmp_path, trained_output_dir):
    empty = tmp_path / "no_data"
    empty.mkdir()
    app = create_app(make_settings(empty, trained_output_dir, tmp_path / "db.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        demo = client.get("/api/dataset/demo-trials").json()
    assert demo["dataset_available"] is False
    assert demo["participants"] == [] and demo["held_out_participants"] == [] and demo["trials_per_participant"] is None
    assert "not available" in demo["note"]
    assert demo["provenance"]["recorded"] is True  # provenance is still reported; only the data is missing


def test_demo_with_unrecorded_provenance_marks_every_role_unknown(settings, fake_deap_dir):
    meta = _fixed_split_meta()
    meta["training_config"] = {}
    out = demo_trials(SimpleNamespace(metadata=meta), make_dataset(settings), settings, _request())
    assert [p.participant_id for p in out.participants] == PARTICIPANTS
    assert {p.model_role for p in out.participants} == {"unknown"}  # never guessed from the split
    assert out.held_out_participants == [] and out.provenance.recorded is False


def test_frozen_participants_never_offered_even_if_recorded_as_held_out(settings):
    frozen = sorted(FROZEN_TEST_PARTICIPANTS)[:2]
    meta = _fixed_split_meta(test=["s01", *frozen])
    out = demo_trials(SimpleNamespace(metadata=meta), make_dataset(settings), settings, _request())
    offered = {p.participant_id for p in out.participants} | {u.participant_id for u in out.unavailable}
    assert not offered & FROZEN_TEST_PARTICIPANTS and not set(out.held_out_participants) & FROZEN_TEST_PARTICIPANTS


def test_frozen_participant_file_is_not_served_for_prediction(tmp_path, trained_output_dir, fake_deap_dir):
    data = tmp_path / "deap"
    shutil.copytree(fake_deap_dir, data)
    frozen = sorted(FROZEN_TEST_PARTICIPANTS)[0]
    shutil.copy(Path(data) / "s01.dat", Path(data) / f"{frozen}.dat")
    app = create_app(make_settings(data, trained_output_dir, tmp_path / "db.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        assert frozen not in client.get("/api/dataset/demo-trials").json()["held_out_participants"]
        assert frozen not in {p["participant_id"] for p in client.get("/api/dataset/summary").json()["participants"]}
        headers = register_and_login(client)
        r = client.post("/api/predict", headers=headers, json={"deap_reference": {"participant_id": frozen, "trial_index": 0}})
        assert r.status_code == 404


def test_registry_lists_provenance_per_model(trained_client):
    models = trained_client.get("/api/models").json()
    assert models and all(m["provenance"]["recorded"] for m in models)
