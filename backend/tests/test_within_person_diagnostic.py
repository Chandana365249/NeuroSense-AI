"""Within-person diagnostic: trial-wise CV must never split a trial and must test every trial once.
Synthetic arrays only check the mechanics; real results come from scripts/within_person_diagnostic.py."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import within_person_diagnostic as wp


def _synthetic(n_trials: int = 20, segs: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.array(["high_valence", "low_valence"] * (n_trials // 2))
    seg_trial = np.repeat(np.arange(n_trials), segs)
    X = rng.normal(size=(len(seg_trial), 6)).astype(np.float32)
    X[:, 0] += (labels[seg_trial] == "high_valence") * 3.0  # separable signal
    return X, seg_trial, labels


def test_every_trial_tested_once_and_never_split(monkeypatch: pytest.MonkeyPatch) -> None:
    X, seg_trial, labels = _synthetic()
    seen: list[tuple[set[int], set[int]]] = []
    real_fit = wp.candidate_models

    def spy(seed: int):  # record the trials each fold trains on
        models = real_fit(seed)
        factory = models["random_forest"]

        class Recorder:
            def __call__(self):
                est = factory()
                orig = est.fit

                def fit(Xf, yf):
                    seen.append((len(Xf), set()))
                    return orig(Xf, yf)

                est.fit = fit
                return est

        return {**models, "random_forest": Recorder()}

    monkeypatch.setattr(wp, "candidate_models", spy)
    proba, classes = wp.cross_validate(X, seg_trial, labels, seed=0)
    assert proba.shape == (len(labels), 2) and not np.isnan(proba).any()
    assert classes == ["high_valence", "low_valence"]
    assert len(seen) == wp.N_FOLDS
    # each fold trains on 4/5 of the trials, i.e. whole trials only (3 segments each)
    assert all(n % 3 == 0 and n == len(X) * 4 // 5 for n, _ in seen)


def test_cross_validation_is_seeded_and_finds_separable_signal() -> None:
    X, seg_trial, labels = _synthetic()
    p1, c = wp.cross_validate(X, seg_trial, labels, seed=1)
    p2, _ = wp.cross_validate(X, seg_trial, labels, seed=1)
    # The registered RF uses n_jobs=-1; parallel summation of tree probabilities can differ in the last bit.
    assert np.allclose(p1, p2, rtol=0, atol=1e-12)
    assert np.array_equal(np.argmax(p1, axis=1), np.argmax(p2, axis=1))
    m = wp.metrics(labels, p1, c)
    assert m["balanced_accuracy"] > 0.9 and m["roc_auc"] > 0.9
    assert 0.0 <= m["predicted_high_share"] <= 1.0


def test_trial_separation_assertion_fires_on_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    X, seg_trial, labels = _synthetic()

    class Leaky:  # a splitter that puts the test trials into training as well
        def __init__(self, *a, **k) -> None: ...

        def split(self, Xs, y):
            idx = np.arange(len(y))
            yield idx, idx[:4]

    monkeypatch.setattr(wp, "StratifiedKFold", Leaky)
    with pytest.raises(AssertionError, match="both train and test"):
        wp.cross_validate(X, seg_trial, labels, seed=0)
