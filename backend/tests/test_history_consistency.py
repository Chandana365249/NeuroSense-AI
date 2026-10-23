"""History records match what was requested and stored: participant, trial index, label, probability, order, totals."""
from __future__ import annotations

import numpy as np

from tests.conftest import register_and_login


def test_history_reflects_requests_in_newest_first_order(trained_client):
    headers = register_and_login(trained_client)
    held_out = trained_client.get("/api/model-info").json()["split"]["test"]
    requested = [(held_out[0], 0), (held_out[-1], 5), (held_out[0], 3)]
    ids = []
    for pid, trial in requested:
        r = trained_client.post("/api/predict", headers=headers, json={"deap_reference": {"participant_id": pid, "trial_index": trial}})
        assert r.status_code == 201, r.text
        ids.append(r.json()["prediction_id"])

    data = trained_client.get("/api/predictions", headers=headers).json()
    assert data["total"] == 3
    assert [p["prediction_id"] for p in data["items"]] == ids[::-1]
    for item, (pid, trial) in zip(data["items"], requested[::-1], strict=True):
        ref = item["source_reference"]
        assert (ref["participant_id"], ref["trial_index"]) == (pid, trial)
        assert ref["participant_split"] == "test"
        assert item["prediction"] == max(item["probabilities"], key=item["probabilities"].get)
        assert abs(item["confidence"] - max(item["probabilities"].values())) < 1e-9


def test_session_filtered_total_counts_only_that_session(trained_client):
    headers = register_and_login(trained_client)
    pid = trained_client.get("/api/model-info").json()["split"]["test"][0]
    session_id = trained_client.post("/api/sessions", json={"name": "s"}, headers=headers).json()["id"]
    body = {"deap_reference": {"participant_id": pid, "trial_index": 1}}
    assert trained_client.post("/api/predict", headers=headers, json=body).status_code == 201
    assert trained_client.post("/api/predict", headers=headers, json={**body, "session_id": session_id}).status_code == 201

    filtered = trained_client.get(f"/api/predictions?session_id={session_id}", headers=headers).json()
    assert filtered["total"] == 1 and len(filtered["items"]) == 1
    assert trained_client.get("/api/predictions", headers=headers).json()["total"] == 2


def test_synthetic_upload_is_recorded_as_synthetic(trained_client):
    headers = register_and_login(trained_client)
    signal = np.random.default_rng(1).normal(0, 10, size=(32, 128 * 8)).round(3).tolist()
    plain = trained_client.post("/api/predict", headers=headers, json={"eeg": {"signal": signal}})
    synth = trained_client.post("/api/predict", headers=headers, json={"eeg": {"signal": signal, "synthetic_test_signal": True}})
    assert plain.status_code == synth.status_code == 201
    assert "synthetic_test_signal" not in plain.json()["source_reference"]
    assert synth.json()["source_reference"]["synthetic_test_signal"] is True
    stored = trained_client.get(f"/api/predictions/{synth.json()['prediction_id']}", headers=headers).json()
    assert stored["source_reference"]["synthetic_test_signal"] is True
