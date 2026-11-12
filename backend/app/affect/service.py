"""Calibrated estimates of valence, arousal, dominance and liking for one analysed trial, from the same EEG features.

- Valence: the stored output of the model that made the prediction, passed through the calibration fitted on
  out-of-person scores. The valence model itself is unchanged.
- Arousal, dominance (DEAP only) and liking: separate models from `python -m app.affect.build`. For a participant
  those models were trained on, the stored out-of-person score (from the fold where that person was held out) is used
  instead of the final model, so an estimate is never made by a model that saw the person.
- Personalised: on request, one model per dimension trained on the same person's other rated trials only.

Every number shown carries the evaluation it comes from. Nothing here writes to the database or changes a model.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.affect import build as B
from app.affect import core as C
from app.core.config import Settings

DATASETS = {"DEAP": ("deap", C.DEAP_DIMENSIONS), "FACED": ("faced", C.FACED_DIMENSIONS)}


def _eval_view(e: dict[str, Any]) -> dict[str, Any]:
    ba = e["balanced_accuracy_per_participant"]
    # Better than chance only if the whole 95% interval over people is above 50%.
    return {"balanced_accuracy": ba.get("mean"), "ci_low": ba.get("ci_low"), "ci_high": ba.get("ci_high"),
            "better_than_chance": bool(ba.get("ci_low") is not None and ba["ci_low"] > 0.5),
            "participants": e["participants"], "auc": e["pooled_auc"], "calibrated_accuracy": e["calibrated_accuracy"],
            "majority_accuracy": e["majority_accuracy"], "ece_raw": e["ece_raw"], "ece_calibrated": e["ece_calibrated"]}


def _estimate(dim: Any, raw: float, platt: dict[str, float], source: str, evaluation: dict[str, Any],
              rating: float | None) -> dict[str, Any]:
    # The decision is the model's own (score > 0.5, as evaluated by balanced accuracy); the calibrated probability
    # says how often trials with this score were actually rated high, so it can lean the other way for weak models.
    # With a base-rate-only calibration the score has no usable direction, so no decision is shown at all.
    p = float(C.apply_platt(platt, raw))
    higher = raw > 0.5
    decided = not platt.get("base_rate_only")
    return {"key": dim.key, "label": dim.label, "probability_high": p,
            "predicted": ("higher" if higher else "lower") if decided else None,
            "probability_predicted": (p if higher else 1 - p) if decided else None, "raw_score": float(raw), "source": source,
            "evaluation": _eval_view(evaluation),
            "own_rating": None if rating is None else round(float(rating), 2),
            "own_label": None if rating is None else ("higher" if rating > dim.threshold else "lower"),
            "threshold": dim.threshold}


class AffectService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dir = settings.output_dir / "affect"
        self._lock = threading.Lock()
        self._card: dict[str, Any] | None = None
        self._card_mtime = 0.0
        self._models: dict[str, Any] = {}

    def card(self) -> dict[str, Any] | None:
        path = self.dir / "affect_card.json"
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        with self._lock:
            if self._card is None or mtime != self._card_mtime:
                self._card, self._card_mtime, self._models = json.loads(path.read_text(encoding="utf-8")), mtime, {}
            return self._card

    def _model(self, file: str, sha: str) -> Any:
        with self._lock:
            if file not in self._models:
                path: Path = self.dir / file
                if not path.exists() or B.sha256(path) != sha:
                    raise FileNotFoundError(f"{file} is missing or failed its integrity check")
                self._models[file] = joblib.load(path)
            return self._models[file]

    def estimate(self, dataset: str, X: np.ndarray, valence_score: float, participant: str | None = None,
                 trial: int | None = None, ratings: dict[str, float] | None = None) -> dict[str, Any] | None:
        """Calibrated estimates for one trial. X: the trial's segment features (the exact prediction input)."""
        key, dims = DATASETS[dataset]
        card = (self.card() or {}).get(key)
        if not card:
            return None
        ratings = ratings or {}
        out = []
        for dim in dims:
            d = card["dimensions"].get(dim.key)
            if d is None:
                continue
            if dim.key == "valence":
                raw, source = valence_score, "the model that made this prediction"
            else:
                held = d["out_of_person_scores"].get(str(participant), {}).get(str(trial)) if participant is not None else None
                if held is not None:
                    raw, source = float(held), "held-out model (this person was not in its training data)"
                else:
                    model = self._model(d["model_file"], d["sha256"])
                    raw, source = float(model.predict_proba(X.astype(np.float64))[:, 1].mean()), "model trained on other people"
            out.append(_estimate(dim, raw, d["calibration"], source, d["evaluation"], ratings.get(dim.key)))
        return {"dataset": dataset, "dimensions": out, "scale": card["scale"],
                "personal_available": participant is not None,
                "method": "Platt-calibrated probability from out-of-person scores of the development participants",
                "note": (self.card() or {}).get("note")}

    def personal(self, dataset: str, X_person: np.ndarray, trial_ids: np.ndarray, ratings_by_trial: dict[int, dict[str, float]],
                 trial: int) -> dict[str, Any]:
        """Personalised estimates for one trial: per dimension, a model trained only on the same person's other trials."""
        key, dims = DATASETS[dataset]
        card = (self.card() or {}).get(key)
        if not card:
            return {"available": False, "reason": "The affect models have not been built on this server."}
        factory = (lambda: B.deap_model(-1)) if dataset == "DEAP" else B.faced_model
        train, test = trial_ids != trial, trial_ids == trial
        if not test.any():
            return {"available": False, "reason": "This trial has no usable segments."}
        out = []
        for dim in dims:
            d = card["dimensions"].get(dim.key)
            if d is None:
                continue
            y_trial = {t: int(r[dim.key] > dim.threshold) for t, r in ratings_by_trial.items() if dim.key in r}
            y = np.array([y_trial[int(t)] for t in trial_ids[train]])
            scores = B._fit_score(factory, X_person[train], y, X_person[test])
            n_other = len({int(t) for t in trial_ids[train]})
            if scores is None:
                out.append({"key": dim.key, "label": dim.label, "available": False,
                            "reason": f"All of this person's other {n_other} ratings are on the same side, so nothing can be learned."})
                continue
            est = _estimate(dim, float(scores.mean()), d["personal"]["calibration"],
                            f"personal model trained on this person's other {n_other} rated trials",
                            d["personal"]["evaluation"], ratings_by_trial.get(trial, {}).get(dim.key))
            out.append({**est, "available": True, "training_trials": n_other})
        return {"available": True, "dataset": dataset, "trial": trial, "dimensions": out,
                "method": "Per-dimension model trained only on this person's other rated trials, Platt-calibrated."}
