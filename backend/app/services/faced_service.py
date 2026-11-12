"""FACED research workflow: validated participants, trials and a FACED-only prediction path.

Everything here is separate from the DEAP model and its endpoints:
- participants and trials come only from the local validation report (outputs/faced/validation_report.json), so the
  UI shows what was actually validated on this server, never the published count;
- predictions run FACED's own pipeline (app.ml.faced.pipeline) on the participant's recording and FACED's own model
  (outputs/faced/models). The result carries the provenance saved with it in the shared history;
- every trial carries its evaluation status under faced-protocol-v1 relative to the model's training data.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.exceptions import InvalidInputError, NotFoundError, ServiceUnavailableError
from app.ml.faced import dataset as fd
from app.ml.faced import pipeline as fpipe

STATUS_TEXT = {
    "strict_held_out": "Strict held-out test: neither this person nor this clip was used to train the model.",
    "unseen_person_seen_clip": "This person was held out, but other people's responses to this clip were used in training.",
    "seen_person_unseen_clip": "This person's other clips were used in training; this clip was held out.",
    "in_sample": "In-sample: this person and this clip were both used to train the model. Not an independent test.",
}


class FacedService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root: Path = settings.faced_dataset_dir
        self.out: Path = settings.faced_output_dir
        self._lock = threading.Lock()
        self._model: Any = None
        self._model_file: str | None = None

    # ------------------------------------------------------------ cached files
    def _json(self, name: str) -> dict[str, Any] | None:
        path = self.out / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _report(self) -> dict[str, Any]:
        report = self._json("validation_report.json")
        if report is None or not (self.root / "participants.tsv").exists():
            raise ServiceUnavailableError("FACED is not installed or not validated on this server")
        return report

    def _split(self) -> dict[str, Any] | None:
        return self._json("protocol_split_v1.json")

    def card(self) -> dict[str, Any] | None:
        return self._json("models/model_card.json")

    def _stimuli(self) -> dict[int, dict[str, str]]:
        try:
            return fd.stimuli(self.root)
        except OSError:
            return {}

    # ------------------------------------------------------------ public views
    def status(self) -> dict[str, Any]:
        report = self._json("validation_report.json")
        installed = (self.root / "participants.tsv").exists()
        card = self.card()
        summary = (report or {}).get("summary", {})
        return {
            "dataset_id": fd.DATASET_ID, "version": fd.DATASET_VERSION,
            "name": "FACED - Finer-grained Affective Computing EEG Dataset",
            "license": "CC-BY-4.0",
            "citation": "Chen, J., Wang, X., Huang, C., Hu, X., Shen, X., & Zhang, D. (2023). A Large Finer-grained "
                        "Affective Computing EEG Dataset. Scientific Data, 10, 740. doi:10.1038/s41597-023-02650-w",
            "installed": installed, "validated": report is not None and installed,
            "participants_declared": summary.get("participants_declared"),
            "participants_valid": summary.get("participants_valid"),
            "participants_failed": len(summary.get("participants_failed", [])),
            "sampling_rates": summary.get("sampling_rates", {}),
            "channels": summary.get("channels", []),
            "trials_used": summary.get("trials_used"), "segments": summary.get("segments"),
            "pipeline": fpipe.PIPELINE_VERSION, "window_seconds": fpipe.WINDOW_S, "target_sampling_rate": fpipe.TARGET_FS,
            "label_rule": f"high valence if the participant's own valence rating > {fpipe.VALENCE_THRESHOLD} (0-7 scale)",
            "protocol": "faced-protocol-v1" if self._split() else None,
            "model": self._model_view(card),
        }

    def incompatibility(self, card: dict[str, Any]) -> str | None:
        """Why this model must not be used with the FACED pipeline, or None. Nothing else is ever substituted for it."""
        names = fpipe.feature_names()
        if card.get("dataset", {}).get("id") != fd.DATASET_ID:
            return "The model was not trained on FACED"
        if card.get("pipeline") != fpipe.PIPELINE_VERSION:
            return "The model was built for a different FACED pipeline version"
        schema = card.get("feature_schema", {})
        if schema.get("n_features") != len(names) or schema.get("sha256") != hashlib.sha256("\n".join(names).encode()).hexdigest():
            return "The model's feature schema does not match the FACED pipeline"
        return None

    def _model_view(self, card: dict[str, Any] | None) -> dict[str, Any]:
        if not card:
            return {"available": False, "message": "No FACED model has been trained and evaluated on this server."}
        path = self.out / "models" / card["artifact"] if card.get("artifact") else None
        problem = self.incompatibility(card)
        return {"available": bool(path and path.exists()) and problem is None, "message": problem,
                "name": card["name"], "algorithm": card["algorithm"], "pipeline": card.get("pipeline"),
                "dataset_compatibility": card["dataset_compatibility"], "trained_on": card["trained_on"],
                "evaluation": card["evaluation"], "target_80_reached": card["target_80_reached"],
                "status": card["status"], "limitations": card["limitations"]}

    def evidence(self) -> dict[str, Any]:
        """Measured results, read from the experiment output files; a missing file is reported as missing, never filled in."""
        final = self._json("final_results.json")
        if final is None:
            return {"available": False, "message": "The FACED held-out evaluation has not been run on this server."}

        def setting(r: dict[str, Any]) -> dict[str, Any]:
            return {"model": r["selected"]["name"], "selected_on": r["selected_on"], "train_trials": r["train_trials"],
                    "test_trials": r["test_trials"], "train_high_share": r["train_high_share"],
                    "majority_reference": r["majority_reference"], "trial": r["trial_pooled"],
                    "ci95": r["trial_pooled_ci95_participant_bootstrap"], "confusion": r["trial_confusion"],
                    "per_participant_balanced_accuracy": r["per_participant_balanced_accuracy"],
                    "per_participant_accuracy": r["per_participant_accuracy"], "segment": r["segment_level"],
                    "selection_scores_inner": r["selection_scores_inner"]}

        settings = {k: setting(v) for k, v in final["settings"].items()}
        s1 = final["settings"]["S1"]
        rows = [{"participant_id": pid, "trials": m["n"], "accuracy": m["accuracy"],
                 "balanced_accuracy": m["balanced_accuracy"]} for pid, m in s1["per_participant"].items()]
        stim = self._json("stimulus_category/final_results.json")
        clean = self._json("cleaning/results.json")
        return {
            "available": True, "protocol": final["protocol"], "split_sha256": final["split_sha256"], "evaluated_once": True,
            "settings": settings, "strict_per_participant": rows,
            "exploratory": {
                "stimulus_category": None if stim is None else {
                    "task": "Intended positive vs negative category of a known video, not the person's own rating",
                    "model": stim["selected"]["name"], "test_participants": stim["test_participants"],
                    "test_trials": stim["test_trials"], "trial": stim["trial_pooled"],
                    "ci95": stim["trial_pooled_ci95_participant_bootstrap"],
                    "document": "docs/FACED_STIMULUS_CATEGORY_EXPERIMENT.md"},
                "noise_removal": None if clean is None else {
                    "task": "Current preprocessing vs enhanced cleaning, 98 development participants, identical folds",
                    "current": {"trial": clean["conditions"]["A_current"]["trial_pooled"],
                                "ci95": clean["conditions"]["A_current"]["trial_pooled_ci95_participant_bootstrap"]},
                    "enhanced": {"trial": clean["conditions"]["B_enhanced"]["trial_pooled"],
                                 "ci95": clean["conditions"]["B_enhanced"]["trial_pooled_ci95_participant_bootstrap"]},
                    "paired": clean["paired"], "document": "docs/FACED_NOISE_REMOVAL_EXPERIMENT.md"},
            },
        }

    def participants(self, query: str = "") -> list[dict[str, Any]]:
        report = self._report()
        split = self._split() or {}
        test = set(split.get("test_participants", []))
        q = query.strip().lower()
        rows = []
        for r in report["participants"]:
            if not r.get("ok"):
                continue
            pid = r["participant_id"]
            label = f"Participant {pid.split('-')[-1]}"
            if q and q not in pid.lower() and q not in label.lower():
                continue
            rows.append({"participant_id": pid, "label": label, "sampling_rate": r["sampling_rate"],
                         "usable_trials": sum(1 for t in r["trials"] if t["used"]),
                         "role": "final_test" if pid in test else "development" if split else "unassigned"})
        return rows

    def _participant_row(self, pid: str) -> dict[str, Any]:
        for r in self._report()["participants"]:
            if r["participant_id"] == pid and r.get("ok"):
                return r
        raise NotFoundError(f"FACED participant {pid} is not available on this server")

    def evaluation_status(self, pid: str, video: int) -> str | None:
        split = self._split()
        if not split:
            return None
        p_test, v_test = pid in split["test_participants"], video in split["test_videos"]
        return {(True, True): "strict_held_out", (True, False): "unseen_person_seen_clip",
                (False, True): "seen_person_unseen_clip", (False, False): "in_sample"}[(p_test, v_test)]

    def trials(self, pid: str) -> dict[str, Any]:
        row = self._participant_row(pid)
        stim = self._stimuli()
        out = []
        for t in row["trials"]:
            v = t["video_index"]
            s = stim.get(v, {})
            status = self.evaluation_status(pid, v)
            out.append({"video_index": v, "title": s.get("source_title"), "targeted_emotion": t.get("targeted_emotion"),
                        "duration_s": t.get("duration_s"), "usable": t["used"], "segments": t.get("segments", 0),
                        "self_reported_valence": t.get("valence"),
                        "valence_label": None if t.get("valence") is None else
                        ("high_valence" if fpipe.valence_label(t["valence"]) else "low_valence"),
                        "evaluation_status": status, "evaluation_text": STATUS_TEXT.get(status or "")})
        return {"participant_id": pid, "label": f"Participant {pid.split('-')[-1]}",
                "sampling_rate": row["sampling_rate"], "trials": out}

    # ------------------------------------------------------------ prediction
    def _load_model(self, card: dict[str, Any]) -> Any:
        import joblib

        with self._lock:
            if self._model is None or self._model_file != card["artifact"]:
                path = self.out / "models" / card["artifact"]
                if not path.exists():
                    raise ServiceUnavailableError("The FACED model file is missing")
                self._model = joblib.load(path)
                self._model_file = card["artifact"]
            return self._model

    def predict(self, pid: str, video: int) -> dict[str, Any]:
        if not 1 <= video <= fd.N_VIDEOS:
            raise InvalidInputError("video_index must be between 1 and 28")
        card = self.card()
        if not card or not card.get("artifact"):
            raise ServiceUnavailableError("No FACED model is available")
        problem = self.incompatibility(card)
        if problem:
            raise ServiceUnavailableError(problem)
        row = self._participant_row(pid)
        trial_row = next((t for t in row["trials"] if t["video_index"] == video), None)
        if trial_row is None or not trial_row["used"]:
            raise NotFoundError(f"Video {video} is not a usable trial for {pid}")
        participant = fd.load_participant(self.root, pid)
        trial = next(t for t in participant.trials if t.video_index == video)
        feats, kept, rejected = fpipe.trial_features(participant, trial)
        if feats.shape[0] == 0:
            raise NotFoundError("Every segment of this trial was rejected by the artifact check")
        model = self._load_model(card)
        seg = model.predict_proba(feats.astype(np.float64))[:, 1]
        p_high = float(seg.mean())
        status = self.evaluation_status(pid, video)
        actual = "high_valence" if fpipe.valence_label(trial.valence) else "low_valence"
        strict = card.get("evaluation", {}).get("S1_strict_primary", {})
        provenance = {
            "dataset": "FACED", "dataset_id": fd.DATASET_ID, "dataset_version": fd.DATASET_VERSION,
            "participant_id": pid, "video_index": video, "video_title": trial.title, "targeted_emotion": trial.targeted_emotion,
            "model_version": card["name"], "algorithm": card["algorithm"].get("name", "unknown"), "pipeline": fpipe.PIPELINE_VERSION,
            "label_rule": f"valence>{fpipe.VALENCE_THRESHOLD}", "protocol": card.get("protocol"), "split_sha256": card.get("split_sha256"),
            "evaluation_status": status, "evaluation_text": STATUS_TEXT.get(status or ""),
            "model_strict_balanced_accuracy": strict.get("trial_balanced_accuracy"),
            "self_reported_valence": round(float(trial.valence), 2), "self_reported_label": actual,
            "segment_probabilities": [round(float(x), 4) for x in seg],
        }
        return {"participant_id": pid, "video_index": video, "model": card["name"], "pipeline": fpipe.PIPELINE_VERSION,
                "probability_high_valence": p_high,
                "predicted_label": "high_valence" if p_high > 0.5 else "low_valence",
                "segments": int(feats.shape[0]), "segment_probabilities": [float(x) for x in seg],
                "self_reported_valence": trial.valence,
                "actual_label": actual, "segments_rejected": rejected, "provenance": provenance,
                "evaluation_status": status, "evaluation_text": STATUS_TEXT.get(status or ""),
                "source": "FACED recording (real EEG)"}
