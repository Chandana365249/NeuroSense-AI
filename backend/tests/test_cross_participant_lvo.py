"""Cross-participant leave-video-out: video folds, leave-two-out training consensus, ties and Holm correction.
Synthetic arrays only check the mechanics; real results come from scripts/cross_participant_lvo.py."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import cross_participant_lvo as lvo


def test_video_folds_partition_and_are_seeded() -> None:
    folds = lvo.video_folds(42)
    assert len(folds) == 5 and all(len(f) == 8 for f in folds)
    assert sorted(v for f in folds for v in f) == list(range(40))
    assert folds == lvo.video_folds(42) and folds != lvo.video_folds(43)


def test_training_consensus_excludes_example_and_held_out_participant() -> None:
    # 5 participants x 1 video. Held-out p = 0 votes high; for q = 1 the voters are {2, 3, 4}.
    high = np.array([[True], [True], [False], [False], [True]])
    labels, tied = lvo.training_consensus(high, p=0)
    assert tied[0, 0]  # the held-out participant's row is never a training example
    assert labels[1, 0] == lvo.LOW and not tied[1, 0]   # voters 2,3,4 -> 1 of 3 high
    assert labels[2, 0] == lvo.HIGH and not tied[2, 0]  # voters 1,3,4 -> 2 of 3 high
    # changing the held-out participant's own rating must not change any training label
    flipped = high.copy(); flipped[0, 0] = False
    assert np.array_equal(lvo.training_consensus(flipped, p=0)[0][1:], labels[1:])


def test_training_ties_are_flagged_and_test_consensus_uses_other_participants() -> None:
    # 4 participants: for p = 0 and q = 1 the voters are {2, 3} -> 1 vs 1 tie
    high = np.array([[True], [True], [True], [False]])
    _, tied = lvo.training_consensus(high, p=0)
    assert tied[1, 0]
    cons = lvo.test_consensus(np.array([[False], [True], [True], [False]]), p=3)  # voters 0,1,2 -> 2 of 3 high
    assert cons[0] == lvo.HIGH
    with pytest.raises(ValueError, match="tie"):
        lvo.test_consensus(np.array([[False], [True], [False]]), p=0)  # voters 1,2 -> tie


def test_holm_adjustment() -> None:
    adj = lvo._holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
