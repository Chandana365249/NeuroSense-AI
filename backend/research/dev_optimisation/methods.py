"""Pure functions for the development-only optimisation experiment. Nothing here reads files or labels of held-out people.

Unlabelled adaptation modes for a held-out person (their emotion ratings are never used):
- batch:  statistics from all of that person's development-clip EEG, including the clips being predicted;
- online: statistics from the development clips that person watched *before* the clip being predicted.
          With fewer than MIN_HISTORY earlier clips, the population statistics from the training people are used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as sps

MIN_HISTORY = 2  # earlier clips needed before online statistics replace the population fallback
SHRINK = 0.05  # covariance shrinkage towards the scaled identity, as in docs/COVARIANCE_ALIGNMENT_EXPERIMENT.md
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
FS = 125


# ---------------------------------------------------------------- rows and folds

def dev_mask(p: np.ndarray, v: np.ndarray, split: dict[str, Any]) -> np.ndarray:
    """Development people on development clips: the only rows this experiment may use."""
    keep = np.isin(p, split["dev_participants"]) & np.isin(v, split["dev_videos"])
    assert not np.isin(p[keep], split["test_participants"]).any() and not np.isin(v[keep], split["test_videos"]).any()
    return keep


def fold_rows(p: np.ndarray, v: np.ndarray, split: dict[str, Any], k: int, exclude: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(train, validation) for inner fold k: validation = people group k on clip group k; training = the other
    development people on the other development clips. `exclude` removes a second group of people and clips from
    training, which early stopping uses as its own validation."""
    val_p, val_v = split["inner_participant_groups"][k], split["inner_video_groups"][k]
    dev = dev_mask(p, v, split)
    train = dev & ~np.isin(p, val_p) & ~np.isin(v, val_v)
    if exclude is not None:
        train &= ~np.isin(p, split["inner_participant_groups"][exclude]) & ~np.isin(v, split["inner_video_groups"][exclude])
    val = dev & np.isin(p, val_p) & np.isin(v, val_v)
    assert not set(p[train].tolist()) & set(p[val].tolist()), "a validation person is in training"
    assert not set(v[train].tolist()) & set(v[val].tolist()), "a validation clip is in training"
    return train, val


def earlier_clips(order: list[int], clip: int, available: set[int]) -> list[int]:
    """Available clips shown before `clip`, oldest first."""
    position = order.index(clip)
    return [c for c in order[:position] if c in available]


def reference_rows(mode: str, order: list[int], clip: int, person_rows: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Rows of one held-out person whose unlabelled EEG may inform the prediction of `clip`. None means: use the
    population fallback."""
    if mode == "batch":
        return person_rows
    assert mode == "online"
    earlier = earlier_clips(order, clip, set(v[person_rows].tolist()))
    if len(earlier) < MIN_HISTORY:
        return None
    return person_rows[np.isin(v[person_rows], earlier)]


# ---------------------------------------------------------------- B: per-person standardisation

def moments(x: np.ndarray, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=axis)
    sd = x.std(axis=axis)
    return mean, np.where(sd > 1e-12, sd, 1.0)


def standardise(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mean) / sd


# ---------------------------------------------------------------- C: per-person score threshold

def person_threshold(trial_scores: np.ndarray | None, rule: str, high_share: float) -> float:
    """Decision threshold for one person from their own unlabelled trial scores.
    median: half of the person's trials are called high.  base_rate: the training people's high share is called high."""
    if trial_scores is None or len(trial_scores) < MIN_HISTORY:
        return 0.5
    if rule == "median":
        return float(np.median(trial_scores))
    assert rule == "base_rate" and 0 < high_share < 1
    return float(np.quantile(trial_scores, 1 - high_share))


# ---------------------------------------------------------------- D: covariance alignment

def _eig_fn(c: np.ndarray, fn: Any) -> np.ndarray:
    w, vec = np.linalg.eigh(c)
    return (vec * fn(w)[..., None, :]) @ np.swapaxes(vec, -1, -2)


def logm(c: np.ndarray) -> np.ndarray:
    """For positive definite matrices only."""
    return _eig_fn(c, lambda w: np.log(np.maximum(w, 1e-12)))


def expm(c: np.ndarray) -> np.ndarray:
    """For any symmetric matrix, such as a matrix logarithm with negative eigenvalues."""
    return _eig_fn(c, np.exp)


def invsqrtm(c: np.ndarray) -> np.ndarray:
    """For positive definite matrices only."""
    return _eig_fn(c, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-12)))


def shrunk_covariance(x: np.ndarray) -> np.ndarray:
    """(..., channels, samples) -> (..., channels, channels). Shrinkage keeps it positive definite after the common
    average reference, which removes one dimension."""
    x = x - x.mean(axis=-1, keepdims=True)
    c = x @ np.swapaxes(x, -1, -2) / x.shape[-1]
    n = c.shape[-1]
    trace = np.trace(c, axis1=-2, axis2=-1)[..., None, None]
    return (1 - SHRINK) * c + SHRINK * trace / n * np.eye(n)


def band_covariances(segments: np.ndarray) -> np.ndarray:
    """(rows, channels, samples) -> (rows, bands, channels, channels). Each segment is filtered on its own."""
    out = np.empty((len(segments), len(BANDS), segments.shape[1], segments.shape[1]))
    for j, band in enumerate(BANDS.values()):
        sos = sps.butter(4, band, btype="bandpass", fs=FS, output="sos")
        out[:, j] = shrunk_covariance(sps.sosfiltfilt(sos, segments.astype(np.float64), axis=-1))
    return out


def log_euclidean_mean(log_c: np.ndarray) -> np.ndarray:
    """Mean over the first axis of matrices given as their matrix logarithms."""
    return expm(log_c.mean(axis=0))


def tangent_features(c: np.ndarray, ref_isqrt: np.ndarray) -> np.ndarray:
    """(rows, bands, n, n) re-centred at the reference, then the upper triangle of the log (off-diagonal x sqrt 2)."""
    n = c.shape[-1]
    iu = np.triu_indices(n)
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    log_c = logm(ref_isqrt @ c @ ref_isqrt)
    return (log_c[..., iu[0], iu[1]] * weight).reshape(len(c), -1)


def euclidean_alignment(segments: np.ndarray, ref_isqrt: np.ndarray) -> np.ndarray:
    """Signal-level alignment for the neural model: each segment multiplied by R^-1/2 of its person's reference."""
    return (ref_isqrt @ segments).astype(np.float32)


# ---------------------------------------------------------------- trials and metrics

def trial_scores(p: np.ndarray, v: np.ndarray, s: np.ndarray) -> dict[tuple[str, int], float]:
    """Mean segment score per (person, clip)."""
    out: dict[tuple[str, int], list[float]] = {}
    for q, c, x in zip(p.tolist(), v.tolist(), s.tolist(), strict=True):
        out.setdefault((q, int(c)), []).append(x)
    return {k: float(np.mean(x)) for k, x in out.items()}


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    return {"n": len(y), "balanced_accuracy": (sens + spec) / 2, "accuracy": (tp + tn) / max(len(y), 1),
            "f1_macro": (f1_pos + f1_neg) / 2, "sensitivity": sens, "specificity": spec,
            "high_share_true": float(np.mean(y)), "high_share_pred": float(np.mean(pred)), "confusion": [[tn, fp], [fn, tp]]}


def bootstrap_ba(person: np.ndarray, y: np.ndarray, preds: dict[str, np.ndarray], n: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
    """Participant-clustered resamples of pooled trial balanced accuracy, the same resamples for every method so that
    paired differences can be read off directly."""
    people = np.unique(person)
    by = [np.flatnonzero(person == q) for q in people]
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n) for name in preds}
    for b in range(n):
        pick = np.concatenate([by[i] for i in rng.integers(0, len(people), len(people))])
        yy = y[pick]
        for name, pred in preds.items():
            pp = pred[pick]
            out[name][b] = 0.5 * ((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean())
    return out
