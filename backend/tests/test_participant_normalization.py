"""Participant-specific normalization: statistics come only from each participant's own non-test-video rows.
Synthetic arrays only check the mechanics; real results come from scripts/participant_normalization.py."""

from __future__ import annotations

import numpy as np

from scripts import participant_normalization as pn


def test_each_participant_uses_only_its_own_stats_rows() -> None:
    rng = np.random.default_rng(0)
    pidx = np.repeat([0, 1], 6)
    X = rng.normal(size=(12, 3)) + np.where(pidx[:, None] == 0, 100.0, -50.0)
    stats_rows = np.tile([True, True, True, True, False, False], 2)
    Xn = pn.normalize_by_participant(X, pidx, stats_rows)
    for i in (0, 1):
        ref = Xn[(pidx == i) & stats_rows]
        assert np.allclose(ref.mean(axis=0), 0) and np.allclose(ref.std(axis=0), 1)
        # non-stats rows are transformed with the stats-row parameters, not their own
        mu, sd = X[(pidx == i) & stats_rows].mean(axis=0), X[(pidx == i) & stats_rows].std(axis=0)
        rows = (pidx == i) & ~stats_rows
        assert np.allclose(Xn[rows], (X[rows] - mu) / sd)


def test_test_rows_do_not_influence_statistics() -> None:
    pidx = np.zeros(6, dtype=int)
    X = np.arange(6, dtype=float)[:, None]
    stats_rows = np.array([True, True, True, False, False, False])
    changed = X.copy(); changed[3:] = 1e6  # altering non-stats (test-video) rows must not change the others
    a, b = pn.normalize_by_participant(X, pidx, stats_rows), pn.normalize_by_participant(changed, pidx, stats_rows)
    assert np.array_equal(a[:3], b[:3])


def test_constant_feature_and_missing_stats() -> None:
    pidx = np.array([0, 0, 1])
    X = np.array([[5.0, 1.0], [5.0, 3.0], [7.0, 7.0]])
    Xn = pn.normalize_by_participant(X, pidx, np.array([True, True, False]))
    assert np.allclose(Xn[:2, 0], 0)  # SD 0 replaced by 1
    assert np.isnan(Xn[2]).all()      # a participant with no stats rows is never silently normalized
