"""Uncertainty estimates for participant-aware EEG evaluation.

Segments of one trial share a label and are strongly correlated, and trials of one participant share
that person's EEG characteristics. Resampling or permuting individual segments would therefore
understate uncertainty. The procedures below treat the TRIAL as the exchangeable unit and keep every
trial inside its own participant (stratum):

- cluster_bootstrap: resample trials with replacement within each participant, recompute the metric.
- permutation_test: shuffle trial labels within each participant (predictions fixed). The null is
  "predictions carry no information about the label beyond each participant's label distribution".
- mean_ci: t-interval across independent units (e.g. LOPO folds = participants).

With only a handful of participants these intervals describe uncertainty over trials of THESE people;
they do not cover the larger between-person variability, which only more participants can measure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats as sps

MetricFn = Callable[[np.ndarray, np.ndarray], float | None]


def binary_metric_fns(classes: Sequence[str]) -> dict[str, MetricFn]:
    """Fast binary metrics over (y_true labels, proba with columns in `classes` order).

    Same definitions as app.ml.evaluation.compute_metrics (argmax prediction, ties -> classes[0];
    macro-F1 with zero_division=0; balanced accuracy over classes present in y_true; AUC with
    classes[1] as positive, None if y_true has one class), vectorized for thousands of resamples.
    """
    if len(classes) != 2:
        raise ValueError("binary_metric_fns needs exactly two classes")
    positive = classes[1]

    def counts(y: np.ndarray, proba: np.ndarray) -> tuple[int, int, int, int]:
        pos = y == positive
        pred_pos = proba[:, 1] > proba[:, 0]
        return int((pos & pred_pos).sum()), int((~pos & ~pred_pos).sum()), int((~pos & pred_pos).sum()), int((pos & ~pred_pos).sum())

    def accuracy(y: np.ndarray, proba: np.ndarray) -> float:
        tp, tn, _, _ = counts(y, proba)
        return (tp + tn) / len(y)

    def balanced(y: np.ndarray, proba: np.ndarray) -> float:
        tp, tn, fp, fn = counts(y, proba)
        recalls = [r for r, n in ((tp / (tp + fn) if tp + fn else 0.0, tp + fn), (tn / (tn + fp) if tn + fp else 0.0, tn + fp)) if n]
        return float(np.mean(recalls))

    def f1_macro(y: np.ndarray, proba: np.ndarray) -> float:
        tp, tn, fp, fn = counts(y, proba)
        f_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f_neg = 2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp else 0.0
        return (f_pos + f_neg) / 2

    def auc(y: np.ndarray, proba: np.ndarray) -> float | None:
        pos = y == positive
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if not n_pos or not n_neg:
            return None
        ranks = sps.rankdata(proba[:, 1])
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return {"accuracy": accuracy, "balanced_accuracy": balanced, "f1_macro": f1_macro, "roc_auc": auc}


def _cluster_index(clusters: np.ndarray, strata: np.ndarray) -> list[list[np.ndarray]]:
    """For each stratum, the list of row-index arrays of its clusters."""
    out = []
    for s in np.unique(strata):
        in_s = strata == s
        out.append([np.flatnonzero(in_s & (clusters == c)) for c in np.unique(clusters[in_s])])
    return out


def cluster_bootstrap(y: np.ndarray, proba: np.ndarray, clusters: np.ndarray, strata: np.ndarray,
                      metrics: dict[str, MetricFn], n_boot: int = 2000, seed: int = 0,
                      level: float = 0.95) -> dict[str, dict[str, float | None]]:
    """Percentile CI per metric, resampling clusters (trials) with replacement within each stratum (participant)."""
    y, proba = np.asarray(y), np.asarray(proba)
    groups = _cluster_index(np.asarray(clusters), np.asarray(strata))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for g in groups])
        for k, fn in metrics.items():
            v = fn(y[idx], proba[idx])
            if v is not None:
                draws[k].append(v)
    alpha = (1 - level) / 2
    result: dict[str, dict[str, float | None]] = {}
    for k, fn in metrics.items():
        d = np.asarray(draws[k])
        result[k] = {
            "estimate": fn(y, proba),
            "ci_low": float(np.quantile(d, alpha)) if len(d) else None,
            "ci_high": float(np.quantile(d, 1 - alpha)) if len(d) else None,
            "n_boot": int(len(d)),
        }
    return result


def permutation_test(y: np.ndarray, proba: np.ndarray, strata: np.ndarray, metrics: dict[str, MetricFn],
                     n_perm: int = 2000, seed: int = 0) -> dict[str, dict[str, float | None]]:
    """One-sided p-value (metric >= observed) with labels permuted within each stratum.

    Call it on TRIAL-level arrays (one row per trial) so the trial is the exchangeable unit.
    p = (1 + #{permuted >= observed}) / (1 + n_perm), which is never 0.
    """
    y, proba, strata = np.asarray(y), np.asarray(proba), np.asarray(strata)
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    observed = {k: fn(y, proba) for k, fn in metrics.items()}
    exceed = {k: 0 for k in metrics}
    for _ in range(n_perm):
        y_perm = y.copy()
        for b in blocks:
            y_perm[b] = y[b][rng.permutation(len(b))]
        for k, fn in metrics.items():
            v = fn(y_perm, proba)
            if observed[k] is not None and v is not None and v >= observed[k] - 1e-12:
                exceed[k] += 1
    return {k: {"observed": observed[k], "p_value": None if observed[k] is None else (1 + exceed[k]) / (1 + n_perm),
                "n_perm": n_perm} for k in metrics}


def mean_ci(values: Sequence[float], level: float = 0.95) -> dict[str, Any]:
    """Mean, population std (as the baseline reports), sample std and a t-based CI over independent units."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "sd_sample": None, "ci_low": None, "ci_high": None}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(sps.t.ppf((1 + level) / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": float(v.std()), "sd_sample": sd,
            "min": float(v.min()), "max": float(v.max()), "ci_low": mean - half, "ci_high": mean + half}


def wilcoxon(a: Sequence[float], b: Sequence[float] | float) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test of paired values (or of `a` against a constant)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.full_like(a_arr, b) if np.isscalar(b) else np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    if len(diff) < 2 or np.allclose(diff, 0):
        return {"n": int(len(diff)), "median_difference": float(np.median(diff)) if len(diff) else None,
                "statistic": None, "p_value": None}
    res = sps.wilcoxon(a_arr, b_arr)
    return {"n": int(len(diff)), "median_difference": float(np.median(diff)),
            "statistic": float(res.statistic), "p_value": float(res.pvalue)}
