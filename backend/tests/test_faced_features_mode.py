"""Online FACED: without the recordings, predictions and explanations run from the saved feature table and give the
same numbers as the recordings. Synthetic recordings only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.ml.faced import dataset as fd
from app.ml.faced import pipeline as fp
from tests.conftest import register_and_login
from tests.test_faced_api import faced_client, faced_predict  # noqa: F401  (fixture reused)


def _save_feature_table(service) -> None:  # noqa: ANN001
    """The feature table the real pipeline saves, built from the fixture's synthetic recordings."""
    X, part, video, seg, ratings = [], [], [], [], []
    for pid in ("sub-000", "sub-001"):
        participant = fd.load_participant(service.root, pid)
        for trial in participant.trials:
            feats, kept, _ = fp.trial_features(participant, trial)
            X.append(feats)
            part += [pid] * len(kept)
            video += [trial.video_index] * len(kept)
            seg += kept
            ratings += [[trial.ratings.get(k, np.nan) for k in fd.RATING_KEYS]] * len(kept)
    np.savez(service.out / f"features_{fp.PIPELINE_VERSION}.npz", X=np.concatenate(X).astype(np.float32),
             participant=np.array(part), video=np.array(video), segment=np.array(seg), ratings=np.array(ratings),
             rating_keys=np.array(fd.RATING_KEYS), y=np.zeros(len(part), int), feature_names=np.array(fp.feature_names()))
    (service.out / "stimuli.json").write_text(json.dumps({str(v): {"source_title": f"Film {v}"} for v in range(1, 29)}))


def test_without_recordings_faced_runs_from_the_saved_features_with_the_same_result(faced_client: TestClient, tmp_path: Path) -> None:  # noqa: F811
    service = faced_client.app.state.faced_service
    _save_feature_table(service)
    h = register_and_login(faced_client)
    local = faced_predict(faced_client, h, "sub-000", 3)
    local_exp = faced_client.get(f"/api/predictions/{local['prediction_id']}/explanation", headers=h).json()
    assert faced_client.get("/api/faced/status", headers=h).json()["source"] == "recordings"
    assert local_exp["available"] and local_exp["trial_view"]["signal"] is not None

    service.root = tmp_path / "no-recordings"  # the online server: no recordings installed
    status = faced_client.get("/api/faced/status", headers=h).json()
    assert status["validated"] and status["source"] == "features"
    assert len(faced_client.get("/api/faced/participants", headers=h).json()["participants"]) == 2
    trials = faced_client.get("/api/faced/participants/sub-000/trials", headers=h).json()["trials"]
    assert trials[2]["title"] == "Film 3"

    online = faced_predict(faced_client, h, "sub-000", 3)
    assert abs(online["probability_high_valence"] - local["probability_high_valence"]) < 1e-6
    assert online["segment_probabilities"] == local["segment_probabilities"]
    assert online["provenance"]["input"] == "saved_features"
    exp = faced_client.get(f"/api/predictions/{online['prediction_id']}/explanation", headers=h).json()
    assert exp["available"] and exp["trial_view"]["signal"] is None  # no raw EEG online, everything else identical
    assert exp["by_band"] == local_exp["by_band"] and exp["verification"]["matches_stored_prediction"]
    assert exp["trial_view"]["kept_segments"] == local_exp["trial_view"]["kept_segments"]


def test_without_recordings_or_features_faced_is_reported_unavailable(faced_client: TestClient, tmp_path: Path) -> None:  # noqa: F811
    faced_client.app.state.faced_service.root = tmp_path / "no-recordings"
    h = register_and_login(faced_client)
    status = faced_client.get("/api/faced/status", headers=h).json()
    assert status["validated"] is False and status["source"] is None
    assert faced_client.get("/api/faced/participants", headers=h).status_code == 503
