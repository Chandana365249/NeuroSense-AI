"""Global (model-level) feature importance for the active Random Forest.

What this is: the forest's impurity-based importance (mean decrease in Gini impurity, averaged over all trees),
read from the trained model. It summarises which input features the whole model used most across its training
data.

What this is NOT:
- not an explanation of an individual prediction;
- not evidence about a participant's actual emotion or internal experience;
- not a causal statement: important features can be correlated with other features or with person-level offsets.

It is only produced when the model and its feature schema demonstrably align (a scikit-learn pipeline whose final
step is a RandomForestClassifier with exactly one importance value per named feature). Otherwise the response says
explanation is unavailable and why, instead of guessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.services.model_service import ModelSnapshot, input_type

METHOD = "Mean decrease in impurity (Gini), averaged over the trees of the Random Forest"
INTERPRETATION = (
    "Global importance describes the model as a whole: how much each input feature reduced impurity across all "
    "decision trees during training. It does not explain any single prediction, and it says nothing about what a "
    "person actually felt."
)
LIMITATIONS = (
    "Impurity-based importance is computed on the training data and can favour features with many distinct values.",
    "Correlated features (for example neighbouring channels in the same band) share importance, so a low value does "
    "not mean a feature carries no information.",
    "Importance reflects what the model relied on, not why valence ratings differ; given near-chance cross-participant "
    "results, it can reflect person-specific signal rather than a general marker of valence.",
    "High-frequency (gamma) EEG features are easily contaminated by muscle (EMG) and eye-movement activity, so high "
    "gamma importance should not be read as evidence of a neural valence signal.",
)
FAMILY_LABELS = {
    "logpow": "Log band power", "relpow": "Relative band power", "spectral_entropy": "Spectral entropy",
    "hjorth_mobility": "Hjorth mobility", "hjorth_complexity": "Hjorth complexity", "log_variance": "Log variance",
    "asym": "Hemispheric asymmetry",
}
BAND_LABELS = {"theta": "Theta (4–8 Hz)", "alpha": "Alpha (8–13 Hz)", "beta": "Beta (13–30 Hz)", "gamma": "Gamma (30–45 Hz)"}


def describe_feature(name: str) -> dict[str, Any]:
    """Split a feature name from FeatureExtractor into channel, family and band."""
    if name.startswith("asym_"):
        _, pair, band = name.split("_", 2)
        return {"channel": pair, "family": "asym", "band": band}
    channel, rest = name.split("_", 1)
    for family in ("logpow", "relpow"):
        if rest.startswith(family + "_"):
            return {"channel": channel, "family": family, "band": rest[len(family) + 1:]}
    return {"channel": channel, "family": rest, "band": None}


def _unavailable(snap: ModelSnapshot | None, reason: str) -> dict[str, Any]:
    meta = snap.meta if snap else {}
    return {"available": False, "reason": reason, "model_version": meta.get("version"), "algorithm": meta.get("algorithm"),
            "scope": "global", "method": None, "interpretation": INTERPRETATION, "limitations": list(LIMITATIONS)}


def _forest(model: Any) -> tuple[RandomForestClassifier, Pipeline] | None:
    pipeline = getattr(model, "estimator_", None)
    if not isinstance(pipeline, Pipeline):
        return None
    final = pipeline.steps[-1][1]
    return (final, pipeline) if isinstance(final, RandomForestClassifier) else None


def global_importance(snap: ModelSnapshot | None, top: int = 20) -> dict[str, Any]:
    if snap is None:
        return _unavailable(None, "No trained model is loaded.")
    meta = snap.meta
    if input_type(meta) != "features":
        return _unavailable(snap, "The active model works on raw EEG segments (a neural network). Impurity-based feature "
                                  "importance does not apply to it, so no explanation is shown.")
    found = _forest(snap.model)
    if found is None:
        return _unavailable(snap, f"Global feature importance is only supported for the Random Forest; the active model "
                                  f"is '{meta.get('algorithm')}'.")
    forest, pipeline = found
    names = list((meta.get("feature_schema") or {}).get("names") or [])
    importances = np.asarray(forest.feature_importances_, dtype=float)
    aligned = (len(names) == importances.shape[0] == int(getattr(forest, "n_features_in_", -1))
               == int(getattr(pipeline, "n_features_in_", len(names))) == int((meta.get("feature_schema") or {}).get("count", -1)))
    if not aligned:
        return _unavailable(snap, "The model's inputs do not match its recorded feature schema, so importance values "
                                  "cannot be attributed to named features.")
    if not np.all(np.isfinite(importances)) or abs(float(importances.sum()) - 1.0) > 1e-6:
        return _unavailable(snap, "The model's importance values are not a valid distribution.")

    order = np.argsort(importances)[::-1]
    top = max(1, min(int(top), len(names)))
    features = [{"rank": i + 1, "name": names[j], "importance": float(importances[j]), **describe_feature(names[j])}
                for i, j in enumerate(order[:top])]

    def grouped(key: str, labels: dict[str, str]) -> list[dict[str, Any]]:
        totals: dict[str, float] = {}
        for name, value in zip(names, importances, strict=True):
            group = describe_feature(name)[key] or "none"
            totals[group] = totals.get(group, 0.0) + float(value)
        return [{"group": g, "label": labels.get(g, "No band (whole-signal feature)" if g == "none" else g), "importance": v}
                for g, v in sorted(totals.items(), key=lambda kv: -kv[1])]

    return {
        "available": True, "reason": None, "model_version": meta["version"], "algorithm": meta["algorithm"],
        "scope": "global", "method": METHOD, "interpretation": INTERPRETATION, "limitations": list(LIMITATIONS),
        "n_features": len(names), "n_trees": len(forest.estimators_), "top": features,
        "by_family": grouped("family", FAMILY_LABELS), "by_band": grouped("band", BAND_LABELS),
        "uniform_share": 1.0 / len(names),
    }
