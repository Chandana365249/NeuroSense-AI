"""Split protocol and leakage checks. Every check raises on failure; none reads a model or a score.

Zero-shot branch:
- training and selection use only faced-protocol-v1 development participants and clips;
- selection uses inner folds holding out both people and clips;
- test 1, FACED: the 25 participants x 9 clips of faced-protocol-v1. PREVIOUSLY EXAMINED, so never called untouched;
- test 2, DEAP: 12 authorised participants, all trials. External to these models, but examined in earlier DEAP work.

Personalised branch (FACED only: DEAP's release has no presentation order, so its blocks cannot be known):
- the encoder comes from the zero-shot training; participants are the same 25, previously examined;
- calibration is a person's first n clips in presentation order, so no future rating is used;
- test clips are the whole presentation blocks that contain no calibration clip, all later in the session.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CALIBRATION_SIZES = (7, 14, 21)
FROZEN_DEAP = {"s12", "s17", "s18", "s20", "s21", "s23", "s29", "s30"}


def inner_rows(p: np.ndarray, v: np.ndarray, split: dict[str, Any], k: int) -> tuple[np.ndarray, np.ndarray]:
    val_p, val_v = list(split["inner_participant_groups"][k]), list(split["inner_video_groups"][k])
    dev_p, dev_v = np.isin(p, split["dev_participants"]), np.isin(v, split["dev_videos"])
    train = dev_p & dev_v & ~np.isin(p, val_p) & ~np.isin(v, val_v)
    val = np.isin(p, val_p) & np.isin(v, val_v)
    return train, val


def final_rows(p: np.ndarray, v: np.ndarray, split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train = np.isin(p, split["dev_participants"]) & np.isin(v, split["dev_videos"])
    test = np.isin(p, split["test_participants"]) & np.isin(v, split["test_videos"])
    return train, test


def check_disjoint(p: np.ndarray, v: np.ndarray, train: np.ndarray, test: np.ndarray, split: dict[str, Any]) -> None:
    if set(p[train].tolist()) & set(p[test].tolist()):
        raise AssertionError("a test participant is in training")
    if set(v[train].tolist()) & set(v[test].tolist()):
        raise AssertionError("a test clip is in training")
    if set(p[train].tolist()) & set(split["test_participants"]):
        raise AssertionError("a final-test participant is in training or selection")
    if set(v[train].tolist()) & set(split["test_videos"]):
        raise AssertionError("a final-test clip is in training or selection")


def check_no_frozen(participants: np.ndarray) -> None:
    found = {str(q) for q in participants} & FROZEN_DEAP
    if found:
        raise AssertionError(f"frozen DEAP participants present: {sorted(found)}")


def personal_plan(order: list[int], blocks: list[list[int]], n: int) -> tuple[list[int], list[int]]:
    """(calibration clips, test clips). Calibration: the first n clips shown. Test: whole blocks with no calibration clip."""
    if sorted(c for b in blocks for c in b) != sorted(order):
        raise ValueError("blocks must partition the presentation order")
    calibration = list(order[:n])
    test = [c for b in blocks if not set(b) & set(calibration) for c in b]
    return calibration, test


def check_personal_plan(order: list[int], blocks: list[list[int]], calibration: list[int], test: list[int]) -> None:
    position = {c: i for i, c in enumerate(order)}
    if set(calibration) & set(test):
        raise AssertionError("a clip is in both calibration and test")
    if test and max(position[c] for c in calibration) >= min(position[c] for c in test):
        raise AssertionError("calibration uses a rating from after a test clip")
    for b in blocks:
        if set(b) & set(test) and not set(b) <= set(test):
            raise AssertionError("a presentation block is split between calibration and test")


def nearest_past_rating(order: list[int], calibration: list[int], clip: int) -> int:
    """The calibration clip shown most recently before `clip`. Raises if none precedes it."""
    position = {c: i for i, c in enumerate(order)}
    earlier = [c for c in calibration if position[c] < position[clip]]
    if not earlier:
        raise AssertionError("no calibration clip precedes this test clip")
    return max(earlier, key=lambda c: position[c])
