"""Development-only optimisation experiment: row selection, unlabelled adaptation and alignment. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from research.dev_optimisation import methods as M

SPLIT = {"dev_participants": ["a", "b", "c", "d"], "test_participants": ["t"], "dev_videos": [1, 2, 3, 4], "test_videos": [9],
         "inner_participant_groups": [["a"], ["b"], ["c"]], "inner_video_groups": [[1], [2], [3]]}
P = np.array(["a", "a", "b", "b", "c", "d", "t", "a"])
V = np.array([1, 2, 1, 2, 3, 4, 1, 9])


def test_only_development_people_on_development_clips_are_kept() -> None:
    keep = M.dev_mask(P, V, SPLIT)
    assert "t" not in P[keep] and 9 not in V[keep] and keep.sum() == 6


def test_folds_hold_out_people_and_clips_and_never_touch_test_rows() -> None:
    train, val = M.fold_rows(P, V, SPLIT, 0)
    assert P[val].tolist() == ["a"] and V[val].tolist() == [1]
    assert "a" not in P[train] and 1 not in V[train] and "t" not in P[train | val] and 9 not in V[train | val]


def test_early_stopping_group_is_removed_from_training() -> None:
    train, _ = M.fold_rows(P, V, SPLIT, 0, exclude=1)
    assert "b" not in P[train] and 2 not in V[train]


def test_online_reference_uses_only_earlier_clips_and_falls_back_when_too_few() -> None:
    order = [3, 1, 4, 2]
    v = np.array([1, 1, 2, 3, 4])
    person = np.arange(5)
    assert M.reference_rows("online", order, 2, person, v).tolist() == [0, 1, 3, 4]
    assert M.reference_rows("online", order, 4, person, v).tolist() == [0, 1, 3]
    assert M.reference_rows("online", order, 1, person, v) is None  # only one earlier clip
    assert M.reference_rows("batch", order, 3, person, v).tolist() == person.tolist()


def test_person_thresholds_use_scores_only() -> None:
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert M.person_threshold(s, "median", 0.4) == 0.3
    assert np.mean(s > M.person_threshold(s, "base_rate", 0.4)) == pytest.approx(0.4)
    assert M.person_threshold(s[:1], "median", 0.4) == 0.5


def test_alignment_recentres_each_persons_covariance() -> None:
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((6, 6))
    x = mix @ rng.standard_normal((40, 6, 500))
    ref = M.shrunk_covariance(x).mean(axis=0)
    aligned = M.euclidean_alignment(x, M.invsqrtm(ref))
    after = M.shrunk_covariance(aligned.astype(np.float64)).mean(axis=0)
    # every eigenvalue of the aligned mean is pulled close to 1 (shrinkage keeps them a little below)
    w = np.linalg.eigvalsh(after)
    assert w.min() > 0.3 and w.max() < 1.1 and np.linalg.cond(after) < np.linalg.cond(ref) / 5
    c = M.shrunk_covariance(x)[:, None]
    feats = M.tangent_features(c, M.invsqrtm(M.log_euclidean_mean(M.logm(c))))
    assert feats.shape == (40, 21) and abs(feats.mean(axis=0)).max() < 0.1


def test_metrics_and_bootstrap_are_paired() -> None:
    y = np.array([1, 0, 1, 0, 1, 0])
    pred = np.array([1, 0, 0, 0, 1, 1])
    m = M.binary_metrics(y, pred)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3) and m["confusion"] == [[2, 1], [1, 2]]
    boots = M.bootstrap_ba(np.array(["a", "a", "b", "b", "c", "c"]), y, {"x": pred, "same": pred.copy()}, n=50)
    assert np.array_equal(boots["x"], boots["same"])
