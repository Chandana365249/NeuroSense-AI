"""Stimulus-confound control: leave-self-out consensus labels, shared folds and trial separation.
Synthetic arrays only check the mechanics; real results come from scripts/stimulus_confound_control.py."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import stimulus_confound_control as sc


def test_consensus_excludes_target_participant() -> None:
    # 4 participants x 2 videos (3 voters each, no ties). Video 0: only participant 0 rates it high.
    valence = np.array([[9.0, 1.0], [1.0, 9.0], [2.0, 8.0], [3.0, 7.0]])
    cons = sc.consensus_labels(valence)
    assert cons[0, 0] == "low_valence"  # participant 0's own high rating is not counted
    assert all(cons[i, 0] == "low_valence" for i in (1, 2, 3))
    assert list(cons[0]) == ["low_valence", "high_valence"]
    assert cons[1, 1] == "high_valence"  # others (1, 8, 7): 2 of 3 high


def test_consensus_tie_is_rejected() -> None:
    with pytest.raises(ValueError, match="tie"):
        sc.consensus_labels(np.array([[1.0], [9.0], [1.0]]))  # target 0: others 9 vs 1 -> 1-1 tie


def test_folds_partition_trials_and_are_shared_between_tasks() -> None:
    rng = np.random.default_rng(0)
    personal = np.array(["high_valence", "low_valence"] * 20)
    consensus = personal.copy()
    consensus[rng.choice(40, 8, replace=False)] = "low_valence"
    folds = sc.make_folds(personal, consensus, seed=42)
    assert len(folds) == sc.N_FOLDS and sorted(t for f in folds for t in f) == list(range(40))
    assert folds == sc.make_folds(personal, consensus, seed=42)


def test_cv_with_folds_rejects_overlap_and_single_class() -> None:
    X = np.random.default_rng(1).normal(size=(12, 4)).astype(np.float32)
    seg_trial = np.repeat(np.arange(6), 2)
    labels = np.array(["high_valence", "low_valence"] * 3)
    with pytest.raises(AssertionError, match="partition"):
        sc.cv_with_folds(X, seg_trial, labels, [[0, 1], [1, 2], [3, 4, 5]], seed=0)
    one_class = np.array(["high_valence"] * 4 + ["low_valence"] * 2)
    with pytest.raises(ValueError, match="one class"):
        sc.cv_with_folds(X, seg_trial, one_class, [[4, 5], [0, 1, 2, 3]], seed=0)
    proba, classes = sc.cv_with_folds(X, seg_trial, labels, [[0, 1], [2, 3], [4, 5]], seed=0)
    assert proba.shape == (6, 2) and classes == ["high_valence", "low_valence"]


def test_sign_flip_exact_and_alignment_check() -> None:
    assert sc._sign_flip_exact(np.array([0.1] * 6)) == pytest.approx(2 / 64)  # only all-+ and all-- reach it
    v = np.tile(np.linspace(1, 9, 10), (4, 1)) + np.random.default_rng(0).normal(0, 0.1, (4, 10))
    res = sc.alignment_check(v, n_null=50)
    assert res["mean_pairwise_spearman"] > 0.9 and res["p"] < 0.05
