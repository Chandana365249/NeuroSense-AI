"""Protocol-aware candidate training (Evaluation Protocol v1.1) on synthetic DEAP-format data.

Checks the frozen-participant guard, whole-participant folds with disjoint trial groups, that training never touches
the active-model pointer, honest metadata labels, and that the API, registry and activation handle protocol
candidates. Every test uses temporary directories and temporary SQLite databases."""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import text

from app.core.config import Settings
from app.database.base import Base
from app.main import create_app
from app.ml.dataset import DatasetError, DEAPDataset
from app.ml.participants import FROZEN_TEST_PARTICIPANTS, TRUSTED_DEVELOPMENT_PARTICIPANTS
from app.ml.protocol_training import ProtocolTrainer
from app.ml.registry import ModelRegistry
from app.services.model_service import metric_rows
from tests.conftest import PARTICIPANTS, make_settings, register_and_login, write_fake_participant


def protocol_settings(dataset_dir: Path, out: Path, db: Path, participants: list[str] | None = None) -> Settings:
    return make_settings(dataset_dir, out, db).model_copy(update={"development_participants": participants or PARTICIPANTS})


@pytest.fixture(scope="session")
def protocol_run(tmp_path_factory: pytest.TempPathFactory, fake_deap_dir: Path) -> tuple[Settings, dict]:
    out = tmp_path_factory.mktemp("protocol_outputs")
    settings = protocol_settings(fake_deap_dir, out, out / "unused.sqlite3")
    result = ProtocolTrainer(settings, run_cross_validation=True).run()
    return settings, result.metadata


# ---------- frozen-participant guard ----------
def test_frozen_participants_are_never_loaded(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    write_fake_participant(tmp_path / "s01.dat", rng)
    write_fake_participant(tmp_path / "s12.dat", rng)  # a frozen final-test id (synthetic content)
    dataset = DEAPDataset(tmp_path)
    assert dataset.participant_ids() == ["s01"]
    with pytest.raises(DatasetError, match="frozen"):
        dataset.load_participant(tmp_path / "s12.dat")
    with pytest.raises(DatasetError):
        dataset.load_trial("s12", 0)
    assert {r.participant_id for r in dataset.iter_participants()} == {"s01"}


def test_settings_reject_frozen_development_participants(tmp_path: Path) -> None:
    assert set(TRUSTED_DEVELOPMENT_PARTICIPANTS).isdisjoint(FROZEN_TEST_PARTICIPANTS)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, jwt_secret_key="x" * 40, development_participants=["s01", "s17"])  # type: ignore[call-arg]
    assert Settings(_env_file=None, jwt_secret_key="x" * 40).development_participants == sorted(TRUSTED_DEVELOPMENT_PARTICIPANTS)  # type: ignore[call-arg]


def test_non_development_participants_are_excluded(tmp_path: Path, fake_deap_dir: Path) -> None:
    allowed = [p for p in PARTICIPANTS if p != "s04"]
    settings = protocol_settings(fake_deap_dir, tmp_path / "out", tmp_path / "db.sqlite3", allowed)
    meta = ProtocolTrainer(settings, run_cross_validation=False).run().metadata
    assert meta["protocol"]["development_participants"] == allowed
    assert "s04" not in meta["dataset"]["participants"] and meta["split"]["train"] == allowed


def test_too_few_development_participants_fail_safely(tmp_path: Path, fake_deap_dir: Path) -> None:
    settings = protocol_settings(fake_deap_dir, tmp_path / "out", tmp_path / "db.sqlite3", ["s01", "s02"])
    with pytest.raises(DatasetError, match="at least 3"):
        ProtocolTrainer(settings).run()
    assert not list((tmp_path / "out" / "models").glob("*/metadata.json"))  # nothing saved


# ---------- folds, groups and metadata ----------
def test_lopo_folds_are_whole_participant_and_group_disjoint(protocol_run: tuple[Settings, dict]) -> None:
    _, meta = protocol_run
    folds = meta["evaluation"]["folds"]
    assert [f["held_out"] for f in folds] == sorted(PARTICIPANTS)  # every participant held out exactly once
    for f in folds:
        assert f["held_out"] not in f["train_participants"]
        assert set(f["train_participants"]) | {f["held_out"]} == set(PARTICIPANTS)
        # the held-out person's trials (videos) are evaluated, never trained on
        assert f["trial_groups"] == meta["protocol"]["trial_groups"][f["held_out"]]
        assert f["n_trials"] == len(f["trial_groups"])
        assert 0 <= f["trial"]["balanced_accuracy"] <= 1 and 0 <= f["predicted_positive_share_trial"] <= 1
    agg = meta["evaluation"]["aggregate"]
    assert agg["trial_balanced_accuracy"]["n"] == len(PARTICIPANTS)
    assert agg["majority_trial_balanced_accuracy"]["mean"] == pytest.approx(0.5)  # a constant label scores 0.5


def test_protocol_metadata_is_labelled_honestly(protocol_run: tuple[Settings, dict]) -> None:
    settings, meta = protocol_run
    assert "-protocol-" in meta["version"]
    assert meta["split"]["test"] == [] and meta["split"]["validation"] == []
    assert meta["test"] == {} and meta["validation"] == {}  # no fixed-split test result is claimed
    p = meta["protocol"]
    assert p["version"] == "1.1" and p["frozen_participants_used"] is False and p["algorithm_fixed_in_advance"]
    assert "not an independent or held-out test result" in p["limitations"]
    assert meta["evaluation"]["label"].startswith("exploratory")
    assert meta["feature_schema"]["count"] == len(meta["feature_schema"]["names"]) == 440
    assert Path(meta["artifacts"]["lopo_trial_predictions"]).exists()
    assert (settings.metrics_dir / f"{meta['version']}.json").exists()


def test_protocol_training_never_activates(protocol_run: tuple[Settings, dict]) -> None:
    settings, meta = protocol_run
    registry = ModelRegistry(settings.model_dir)
    assert registry.active_version() is None and registry.pointer_bytes() is None
    assert registry.has_version(meta["version"])


def test_protocol_training_leaves_an_existing_pointer_byte_identical(tmp_path: Path, fake_deap_dir: Path,
                                                                    trained_output_dir: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(trained_output_dir, out)
    pointer = out / "models" / ModelRegistry.ACTIVE_FILE
    before, mtime = pointer.read_bytes(), pointer.stat().st_mtime_ns
    ProtocolTrainer(protocol_settings(fake_deap_dir, out, tmp_path / "db.sqlite3"), run_cross_validation=False).run()
    assert pointer.read_bytes() == before and pointer.stat().st_mtime_ns == mtime


def test_metric_rows_for_protocol_candidates(protocol_run: tuple[Settings, dict]) -> None:
    _, meta = protocol_run
    rows = metric_rows(meta)
    assert rows and all(split == "lopo_protocol" and level == "trial" for split, level, _, _ in rows)
    names = {name for _, _, name, _ in rows}
    assert {"balanced_accuracy_mean", "f1_macro_mean", "within_person_auc_mean", "single_label_participants"} <= names


# ---------- API: training, listing, activation ----------
def test_training_api_creates_a_protocol_candidate(tmp_path: Path, fake_deap_dir: Path, trained_output_dir: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(trained_output_dir, out)
    app = create_app(protocol_settings(fake_deap_dir, out, tmp_path / "api.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        active = client.get("/api/model-info").json()["version"]
        metrics_before = client.get("/api/metrics").json()
        admin = register_and_login(client)
        assert client.post("/api/train", json={"run_cross_validation": True}, headers=admin).status_code == 202
        app.state.training_service.wait(timeout=600)
        status = client.get("/api/training/status", headers=admin).json()
        assert status["state"] == "succeeded", status
        version = status["model_version"]
        assert "-protocol-" in version
        assert client.get("/api/model-info").json()["version"] == active  # still the same active model
        assert client.get("/api/metrics").json() == metrics_before
        listed = {m["version"]: m for m in client.get("/api/models").json()}
        assert listed[version]["is_active"] is False and listed[version]["test_metrics"] is None
        evaluation = listed[version]["evaluation"]
        assert evaluation["kind"] == "lopo_exploratory" and evaluation["n_participants"] == len(PARTICIPANTS)
        assert "folds" not in evaluation  # the list view carries the summary only
        with app.state.db.engine.connect() as conn:
            splits = {r[0] for r in conn.execute(text(
                "select distinct m.split from metrics m join models x on x.id = m.model_id where x.version = :v"), {"v": version})}
        assert splits == {"lopo_protocol"}
        report = client.get("/api/reports", headers=admin).json()[0]["content"]
        assert report["evaluation"]["label"].startswith("exploratory") and "lopo_trial_predictions" not in json.dumps(report)


def test_activated_protocol_candidate_is_served_consistently(tmp_path: Path, fake_deap_dir: Path,
                                                             protocol_run: tuple[Settings, dict]) -> None:
    settings, meta = protocol_run
    out = tmp_path / "out"
    shutil.copytree(settings.output_dir, out)
    app = create_app(protocol_settings(fake_deap_dir, out, tmp_path / "act.sqlite3"))
    Base.metadata.create_all(app.state.db.engine)
    with TestClient(app) as client:
        admin = register_and_login(client)
        r = client.post(f"/api/models/{meta['version']}/activate", headers=admin,
                        json={"confirm_version": meta["version"], "reason": "Protocol candidate activation test"})
        assert r.status_code == 200, r.text
        info = client.get("/api/model-info").json()
        assert info["version"] == meta["version"] and info["split"]["test"] == []
        metrics = client.get("/api/metrics").json()
        assert metrics["available"] and metrics["test"] == {} and metrics["graphs"] == []
        assert metrics["evaluation"]["n_participants"] == len(PARTICIPANTS) and len(metrics["evaluation"]["folds"]) == len(PARTICIPANTS)
        assert metrics["protocol"]["frozen_participants_used"] is False
        raw = json.dumps(metrics)
        assert str(out) not in raw and str(fake_deap_dir) not in raw and "trial_groups" not in raw
        assert client.get("/api/dataset/demo-trials").json()["held_out_participants"] == []
        pred = client.post("/api/predict", headers=admin, json={"deap_reference": {"participant_id": "s01", "trial_index": 0}})
        assert pred.status_code == 201 and pred.json()["source_reference"]["participant_split"] == "train"
