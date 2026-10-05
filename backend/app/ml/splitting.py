"""Participant-aware train/validation/test split.

All segments of a participant land in exactly one split, so the model is always evaluated on
people it has never seen (cross-subject generalization). Splitting segments at random would put
neighbouring 4 s windows of the same trial in both train and test and inflate the scores.

Procedure: sort participant IDs, shuffle them with numpy's default_rng(seed), take the first
n_test as test, the next n_validation as validation, and the rest as training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ParticipantSplit:
    train: list[str]
    validation: list[str]
    test: list[str]
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def mask(self, participants: np.ndarray, part: str) -> np.ndarray:
        return np.isin(participants, getattr(self, part))


def split_participants(participant_ids: list[str], n_test: int, n_validation: int, seed: int) -> ParticipantSplit:
    ids = sorted(set(participant_ids))
    if n_test < 1 or n_validation < 0:
        raise ValueError("Need at least one test participant and a non-negative validation count")
    if len(ids) - n_test - n_validation < 1:
        raise ValueError(f"{len(ids)} participants cannot supply {n_test} test + {n_validation} validation + >=1 train")
    shuffled = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    return ParticipantSplit(
        train=sorted(shuffled[n_test + n_validation :]),
        validation=sorted(shuffled[n_test : n_test + n_validation]),
        test=sorted(shuffled[:n_test]),
        seed=seed,
    )
