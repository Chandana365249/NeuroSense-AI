"""Within-person calibration: label-free nested subsets, reproducible permutations, single-class rule, S1 band.
Synthetic arrays only check the mechanics; real results come from scripts/within_person_calibration.py."""

from __future__ import annotations

import numpy as np

from scripts import within_person_calibration as wc


def test_learning_curve_subsets_are_nested_label_free_and_seeded() -> None:
    train = sorted(set(range(40)) - {0, 5, 10, 15, 20, 25, 30, 35})
    s = wc.lc_subsets(42, 0, train)
    assert [len(s[n]) for n in wc.LC_SIZES] == [8, 16, 24, 32] and s[32] == train
    assert set(s[8]) <= set(s[16]) <= set(s[24]) <= set(s[32])
    assert s == wc.lc_subsets(42, 0, list(reversed(train)))  # depends only on (seed, fold, video set)
    assert s[8] != wc.lc_subsets(42, 1, train)[8] or s[8] != wc.lc_subsets(43, 0, train)[8]


def test_permutations_are_reproducible_and_keep_label_counts() -> None:
    labels = np.array([wc.HIGH] * 15 + [wc.LOW] * 25)
    a, b = wc.permuted_labels(labels, 3, 7), wc.permuted_labels(labels, 3, 7)
    assert np.array_equal(a, b) and sorted(a.tolist()) == sorted(labels.tolist())
    assert not np.array_equal(a, wc.permuted_labels(labels, 3, 8))
    assert not np.array_equal(a, wc.permuted_labels(labels, 4, 7))


def test_single_class_training_returns_none_and_videos_stay_separate() -> None:
    rng = np.random.default_rng(0)
    vidx = np.repeat(np.arange(6), 4)
    X = rng.normal(size=(24, 5))
    labels = np.array([wc.HIGH, wc.HIGH, wc.HIGH, wc.LOW, wc.LOW, wc.LOW])
    assert wc.fit_predict(X, vidx, labels, [0, 1, 2], [4, 5], seed=42) is None
    proba = wc.fit_predict(X, vidx, labels, [0, 1, 3, 4], [2, 5], seed=42)
    assert proba is not None and proba.shape == (2, 2) and np.allclose(proba.sum(axis=1), 1)


def test_borderline_band_is_inclusive() -> None:
    assert wc.is_borderline(np.array([4.49, 4.5, 5.0, 5.5, 5.51])).tolist() == [False, True, True, True, False]
