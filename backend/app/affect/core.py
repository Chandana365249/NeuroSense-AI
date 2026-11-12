"""Shared definitions for the affect-dimension models: rating columns, labels, calibration and trial metrics.

A model output becomes a calibrated probability by Platt scaling: p = sigmoid(a * logit(score) + b), with a and b
fitted on out-of-person (leave-one-participant-out) trial scores of the development participants. The calibrated
probability is the estimated frequency with which trials given that score were rated high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

EPS = 1e-4
ECE_BINS = 10


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    column: int  # column in the dataset's ratings array
    threshold: float  # rating > threshold => high


# DEAP self-assessment, 1-9 scale; columns: valence, arousal, dominance, liking.
DEAP_DIMENSIONS = (Dimension("valence", "Valence", 0, 5.0), Dimension("arousal", "Arousal", 1, 5.0),
                   Dimension("dominance", "Dominance", 2, 5.0), Dimension("liking", "Liking", 3, 5.0))
# FACED ratings, 0-7 scale; dominance was not collected. Same split point as the FACED valence label.
FACED_DIMENSIONS = (Dimension("valence", "Valence", -1, 3.5), Dimension("arousal", "Arousal", -1, 3.5),
                    Dimension("liking", "Liking", -1, 3.5))
FACED_RATING_KEYS = {"valence": "Valence", "arousal": "Arousal", "liking": "Liking"}


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(score: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """A negative slope would mean higher model scores go with fewer high ratings; such a score carries no usable
    direction for new people, so the calibrated probability falls back to the base rate instead of being inverted."""
    y = np.asarray(y, int)
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(logit(score)[:, None], y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if a < 0:
        return {"a": 0.0, "b": float(logit(y.mean())), "base_rate_only": True}
    return {"a": a, "b": b}


def apply_platt(params: dict[str, float], score: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(params["a"] * logit(score) + params["b"])))


def ece(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error of P(high): bin-weighted |mean predicted - observed frequency|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 5) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"low": float(edges[b]), "high": float(edges[b + 1]), "n": int((idx == b).sum()),
             "mean_predicted": float(p[idx == b].mean()), "observed_high": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def trial_scores(groups: np.ndarray, seg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean segment score per trial group, in sorted group order."""
    keys, inv = np.unique(groups, return_inverse=True)
    return keys, np.bincount(inv, weights=seg_scores) / np.bincount(inv)


def summarise_scores(participant: np.ndarray, y: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    """Out-of-person evaluation of trial scores, with a nested (leave-one-participant-out) Platt calibration so the
    calibrated figures are never fitted on the person they describe."""
    from app.ml.stats import mean_ci

    people = sorted(set(participant.tolist()))
    calibrated = np.empty_like(raw)
    for q in people:
        held = participant == q
        calibrated[held] = apply_platt(fit_platt(raw[~held], y[~held]), raw[held])
    per_ba, single = [], []
    for q in people:
        m = participant == q
        if len(set(y[m].tolist())) < 2:
            single.append(q)
            continue
        per_ba.append(float(balanced_accuracy_score(y[m], raw[m] > 0.5)))
    base = float(max(y.mean(), 1 - y.mean()))
    return {
        "trials": int(len(y)), "participants": len(people), "share_high": float(y.mean()),
        "balanced_accuracy_per_participant": mean_ci(per_ba), "single_label_participants": single,
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y, raw > 0.5)),
        "pooled_auc": float(roc_auc_score(y, raw)) if len(set(y.tolist())) == 2 else None,
        "calibrated_accuracy": float(((calibrated > 0.5) == y).mean()), "majority_accuracy": base,
        "ece_raw": ece(raw, y), "ece_calibrated": ece(calibrated, y),
        "brier_raw": float(np.mean((raw - y) ** 2)), "brier_calibrated": float(np.mean((calibrated - y) ** 2)),
        "reliability_calibrated": reliability(calibrated, y),
    }
