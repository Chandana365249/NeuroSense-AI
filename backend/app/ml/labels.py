"""Operational class labels derived from DEAP self-assessment ratings.

DEAP participants rated each video on 1-9 scales. A threshold turns these continuous self-reports
into project classes: rating > threshold => "high", otherwise "low" (a rating exactly at the
threshold counts as low). The default threshold of 5 is the scale midpoint, a common convention in
DEAP studies; it is an operational choice for this prototype, not a psychological ground truth,
and labels reflect what participants reported, not a measured internal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

VALENCE_COL, AROUSAL_COL = 0, 1


class LabelStrategy(StrEnum):
    VALENCE_BINARY = "valence_binary"
    AROUSAL_BINARY = "arousal_binary"
    QUADRANT = "quadrant"


@dataclass(frozen=True)
class LabelConfig:
    strategy: LabelStrategy = LabelStrategy.VALENCE_BINARY
    valence_threshold: float = 5.0
    arousal_threshold: float = 5.0

    @property
    def class_names(self) -> list[str]:
        return class_names_for(self.strategy)

    def describe(self) -> str:
        v, a = self.valence_threshold, self.arousal_threshold
        if self.strategy is LabelStrategy.VALENCE_BINARY:
            return f"Binary valence: high_valence if self-reported valence > {v} (1-9 scale), else low_valence"
        if self.strategy is LabelStrategy.AROUSAL_BINARY:
            return f"Binary arousal: high_arousal if self-reported arousal > {a} (1-9 scale), else low_arousal"
        return (
            f"Four quadrants of the valence/arousal plane (valence threshold {v}, arousal threshold {a}); "
            "positive = valence above threshold, high = arousal above threshold"
        )


def class_names_for(strategy: LabelStrategy) -> list[str]:
    if strategy is LabelStrategy.VALENCE_BINARY:
        return ["low_valence", "high_valence"]
    if strategy is LabelStrategy.AROUSAL_BINARY:
        return ["low_arousal", "high_arousal"]
    return ["negative_low_arousal", "negative_high_arousal", "positive_low_arousal", "positive_high_arousal"]


def make_labels(ratings: np.ndarray, config: LabelConfig) -> np.ndarray:
    """Map a (trials, 4) ratings array to an array of class-name strings, one per trial."""
    ratings = np.asarray(ratings, dtype=np.float64)
    if ratings.ndim != 2 or ratings.shape[1] < 2:
        raise ValueError(f"ratings must be (trials, >=2), got {ratings.shape}")
    high_valence = ratings[:, VALENCE_COL] > config.valence_threshold
    high_arousal = ratings[:, AROUSAL_COL] > config.arousal_threshold

    if config.strategy is LabelStrategy.VALENCE_BINARY:
        return np.where(high_valence, "high_valence", "low_valence")
    if config.strategy is LabelStrategy.AROUSAL_BINARY:
        return np.where(high_arousal, "high_arousal", "low_arousal")
    valence_part = np.where(high_valence, "positive", "negative")
    arousal_part = np.where(high_arousal, "high_arousal", "low_arousal")
    return np.char.add(np.char.add(valence_part, "_"), arousal_part).astype(str)
