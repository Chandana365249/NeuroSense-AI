"""Exact per-feature attributions computed from the trained model and the exact features it received.

Random Forest (StandardScaler + RandomForestClassifier, the active DEAP model)
    TreeSHAP (Lundberg et al. 2020, `shap.TreeExplainer`, path-dependent) on the forest's probability for
    `high_valence`. For each segment:  base value + sum of SHAP values = the model's high-valence probability.
    The scaler transforms each feature on its own, so a SHAP value of a scaled feature is the attribution of the
    named raw feature.

Logistic regression (StandardScaler + LogisticRegression, the experimental FACED model)
    Exact linear attribution on the log-odds of `high_valence`: contribution_j = coef_j * (x_j - mean_j) / scale_j,
    baseline = intercept. For each segment:  intercept + sum of contributions = the model's decision score. This is
    linear SHAP with the scaler's training mean as the reference point.

XGBoost (StandardScaler + XGBClassifier)
    TreeSHAP (exact, path-dependent) on the booster's margin, i.e. the log-odds of `high_valence`. For each segment:
    base value + sum of SHAP values = the model's log-odds.

CNN (raw EEG segments, no hand-made features)
    Integrated gradients (Sundararajan et al. 2017) on the high-vs-low log-odds, log(p_high / p_low), from an
    all-zero normalised input (each channel at its training mean). Completeness holds up to the numerical integration
    error, which is measured for every segment and must stay below CNN_TOLERANCE. Attributions are summed per channel
    and per second; the CNN has no frequency-band features, so no band breakdown is claimed for it.

Trial level: the app averages segment outputs, so the trial attribution is the mean of the segment attributions
(an exact identity for the averaged output). SHAP values and log-odds contributions are model attributions, not
probabilities, confidence or evidence about the brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.explainability import BAND_LABELS, FAMILY_LABELS, describe_feature

TOLERANCE = 1e-6
XGB_TOLERANCE = 1e-4  # XGBoost computes in float32
CNN_TOLERANCE = 0.02  # |base + sum - log-odds| for integrated gradients (numerical integration, 128 steps)
IG_STEPS = 128
_EXPLAINERS: dict[int, Any] = {}


@dataclass
class Attribution:
    method: str
    output: str  # "probability" (Random Forest) or "log_odds" (logistic regression)
    base_value: float
    values: np.ndarray  # (segments, features)
    segment_outputs: np.ndarray  # the model's own output per segment, for the high-valence class
    max_abs_error: float  # |base + sum(values) - model output|, worst segment


def split_pipeline(model: Any) -> tuple[StandardScaler, Any]:
    """The scaler and final estimator of a two-step pipeline, unwrapping the project's label-encoding wrapper."""
    pipeline = getattr(model, "estimator_", model)
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) != 2 or not isinstance(pipeline.steps[0][1], StandardScaler):
        raise TypeError("explanations need a StandardScaler + classifier pipeline")
    return pipeline.steps[0][1], pipeline.steps[1][1]


def high_class_index(model: Any) -> int:
    """Column of `high_valence` in the model's predict_proba. The DEAP model names its classes; the FACED model was
    trained on 1 = high valence, 0 = low valence."""
    classes = [str(c) for c in model.classes_]
    if "high_valence" in classes:
        return classes.index("high_valence")
    if classes == ["0", "1"]:
        return 1
    raise ValueError(f"cannot identify the high-valence class in {classes}")


def tree_shap(model: Any, X: np.ndarray) -> Attribution:
    import shap

    scaler, forest = split_pipeline(model)
    if not isinstance(forest, RandomForestClassifier):
        raise TypeError("TreeSHAP here is only set up for RandomForestClassifier")
    hi = high_class_index(model)
    explainer = _EXPLAINERS.get(id(forest))
    if explainer is None:
        explainer = _EXPLAINERS[id(forest)] = shap.TreeExplainer(forest)
    values = np.asarray(explainer.shap_values(scaler.transform(X), check_additivity=True))
    values = values[..., hi] if values.ndim == 3 else values[hi]
    base = float(np.atleast_1d(explainer.expected_value)[hi])
    output = model.predict_proba(X)[:, hi]
    return Attribution("TreeSHAP (exact, tree path-dependent) on the high-valence probability", "probability", base,
                       values, output, float(np.abs(base + values.sum(axis=1) - output).max()))


def linear_log_odds(model: Any, X: np.ndarray) -> Attribution:
    scaler, clf = split_pipeline(model)
    if not isinstance(clf, LogisticRegression) or clf.coef_.shape[0] != 1:
        raise TypeError("linear attribution here is only set up for binary LogisticRegression")
    hi = high_class_index(model)
    sign = 1.0 if hi == 1 else -1.0  # decision_function is the log-odds of classes_[1]
    values = sign * clf.coef_[0] * scaler.transform(X)
    base = sign * float(clf.intercept_[0])
    output = sign * model.decision_function(X) if hasattr(model, "decision_function") else sign * clf.decision_function(scaler.transform(X))
    return Attribution("Exact linear attribution (coefficient x standardised feature) on the high-valence log-odds", "log_odds",
                       base, values, output, float(np.abs(base + values.sum(axis=1) - output).max()))


def xgb_shap(model: Any, X: np.ndarray) -> Attribution:
    import shap
    from xgboost import XGBClassifier

    scaler, booster = split_pipeline(model)
    if not isinstance(booster, XGBClassifier) or len(booster.classes_) != 2:
        raise TypeError("TreeSHAP here is only set up for binary XGBClassifier")
    sign = 1.0 if high_class_index(model) == 1 else -1.0  # the margin is the log-odds of class 1
    explainer = _EXPLAINERS.get(id(booster))
    if explainer is None:
        explainer = _EXPLAINERS[id(booster)] = shap.TreeExplainer(booster)
    Z = scaler.transform(X)
    values = sign * np.asarray(explainer.shap_values(Z, check_additivity=True))
    base = sign * float(np.atleast_1d(explainer.expected_value)[0])
    output = sign * booster.predict(Z, output_margin=True)
    return Attribution("TreeSHAP for XGBoost (exact, tree path-dependent) on the high-valence log-odds", "log_odds", base,
                       values, output, float(np.abs(base + values.sum(axis=1) - output).max()))


def integrated_gradients(model: Any, segments: np.ndarray) -> Attribution:
    """values: (segments, channels, samples) attributions of the raw input to log(p_high / p_low)."""
    import torch

    classes = [str(c) for c in model.classes_]
    if len(classes) != 2 or "high_valence" not in classes:
        raise TypeError("integrated gradients here are only set up for the binary valence CNN")
    hi, lo = classes.index("high_valence"), 1 - classes.index("high_valence")
    net = model.model.eval()
    z = torch.from_numpy(model.normalizer.apply(np.asarray(segments, dtype=np.float32))).unsqueeze(1)

    def log_odds(x: Any) -> Any:
        out = net(x)
        return out[:, hi] - out[:, lo]  # softmax log-ratio = logit difference

    alphas = (torch.arange(IG_STEPS, dtype=torch.float32) + 0.5) / IG_STEPS  # midpoint rule
    grads = torch.zeros_like(z)
    for a in alphas:
        x = (a * z).requires_grad_(True)
        log_odds(x).sum().backward()
        grads += x.grad
    with torch.no_grad():
        values_norm = (z * grads / IG_STEPS).squeeze(1).numpy().astype(np.float64)
        output = log_odds(z).numpy().astype(np.float64)
        base = float(log_odds(torch.zeros_like(z[:1])).item())
    return Attribution(f"Integrated gradients ({IG_STEPS} steps, all-zero normalised baseline) on the high-valence log-odds",
                       "log_odds", base, values_norm, output, float(np.abs(base + values_norm.sum(axis=(1, 2)) - output).max()))


def explain(model: Any, X: np.ndarray) -> Attribution:
    if type(model).__name__ == "CNNClassifier":  # no torch import unless a CNN is actually in use
        return integrated_gradients(model, X)
    _, final = split_pipeline(model)
    if isinstance(final, RandomForestClassifier):
        return tree_shap(model, X)
    if isinstance(final, LogisticRegression):
        return linear_log_odds(model, X)
    from xgboost import XGBClassifier

    if isinstance(final, XGBClassifier):
        return xgb_shap(model, X)
    raise TypeError(f"no reliable explanation method is implemented for {type(final).__name__}")


def tolerance_for(att: Attribution) -> float:
    if att.method.startswith("Integrated gradients"):
        return CNN_TOLERANCE
    return XGB_TOLERANCE if "XGBoost" in att.method else TOLERANCE


# ---------------------------------------------------------------- grouping by EEG channel and band

def feature_info(names: list[str]) -> list[dict[str, Any]]:
    """Channel, family and band of every feature, parsed from the feature extractor's own names. Asymmetry features
    belong to a channel pair; whole-signal features (entropy, Hjorth, variance) have no band."""
    out = []
    for name in names:
        d = describe_feature(name)
        out.append({"name": name, "channel": d["channel"], "family": d["family"], "band": d["band"] or "broadband",
                    "kind": "pair" if d["family"] == "asym" else "channel"})
    return out


def grouped(info: list[dict[str, Any]], contribution: np.ndarray, key: str, only_kind: str | None = None) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    for f, c in zip(info, contribution, strict=True):
        if only_kind and f["kind"] != only_kind:
            continue
        totals[f[key]] = totals.get(f[key], 0.0) + float(c)
    return [{"group": g, "contribution": v} for g, v in sorted(totals.items(), key=lambda kv: -abs(kv[1]))]


def label_for(group: str, key: str) -> str:
    if key == "band":
        return "Broadband (4–45 Hz, no single band)" if group == "broadband" else BAND_LABELS.get(group, group)
    if key == "family":
        return FAMILY_LABELS.get(group, group)
    return group
