from __future__ import annotations

import numpy as np
import pytest

from app.ml.labels import LabelConfig, LabelStrategy, class_names_for, make_labels
from app.ml.splitting import split_participants
from app.preprocessing.features import FeatureExtractor

SR = 128


# ---------- features ----------
def _segments(freq: float, amp: float, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(512) / SR
    return rng.normal(0, 0.5, size=(n, 32, 512)) + amp * np.sin(2 * np.pi * freq * t)


def test_feature_schema_is_440_unique_names() -> None:
    ext = FeatureExtractor()
    assert ext.n_features == 440
    assert len(set(ext.feature_names)) == 440
    assert "Fp1_logpow_alpha" in ext.feature_names
    assert "asym_F4-F3_alpha" in ext.feature_names


def test_features_are_finite_and_shaped() -> None:
    feats = FeatureExtractor().extract(_segments(10, 5))
    assert feats.shape == (3, 440)
    assert np.isfinite(feats).all()


def test_entropy_is_finite_for_negative_signals() -> None:
    """Regression: old code applied scipy entropy to raw (negative) amplitudes and produced NaN."""
    ext = FeatureExtractor()
    feats = ext.extract(-np.abs(_segments(10, 5)))
    col = ext.feature_names.index("Cz_spectral_entropy")
    assert np.isfinite(feats[:, col]).all()
    assert 0 < feats[0, col] <= 1


def test_alpha_power_tracks_alpha_amplitude() -> None:
    ext = FeatureExtractor()
    col = ext.feature_names.index("O1_logpow_alpha")
    beta_col = ext.feature_names.index("O1_logpow_beta")
    strong, weak = ext.extract(_segments(10, 8)), ext.extract(_segments(10, 1))
    assert strong[:, col].mean() > weak[:, col].mean() + 1
    assert abs(strong[:, beta_col].mean() - weak[:, beta_col].mean()) < 0.5


def test_extract_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError):
        FeatureExtractor().extract(np.zeros((2, 8, 512)))


# ---------- labels ----------
RATINGS = np.array([[7.0, 2.0, 5, 5], [5.0, 5.0, 5, 5], [3.0, 8.0, 5, 5], [5.01, 6.0, 5, 5]])


def test_binary_valence_threshold_is_exclusive() -> None:
    labels = make_labels(RATINGS, LabelConfig())
    assert labels.tolist() == ["high_valence", "low_valence", "low_valence", "high_valence"]


def test_threshold_is_configurable() -> None:
    labels = make_labels(RATINGS, LabelConfig(valence_threshold=6.5))
    assert labels.tolist() == ["high_valence", "low_valence", "low_valence", "low_valence"]


def test_binary_arousal() -> None:
    labels = make_labels(RATINGS, LabelConfig(strategy=LabelStrategy.AROUSAL_BINARY))
    assert labels.tolist() == ["low_arousal", "low_arousal", "high_arousal", "high_arousal"]


def test_quadrants() -> None:
    labels = make_labels(RATINGS, LabelConfig(strategy=LabelStrategy.QUADRANT))
    assert labels.tolist() == ["positive_low_arousal", "negative_low_arousal", "negative_high_arousal", "positive_high_arousal"]
    assert set(labels) <= set(class_names_for(LabelStrategy.QUADRANT))


def test_labels_reject_bad_shape() -> None:
    with pytest.raises(ValueError):
        make_labels(np.array([1.0, 2.0]), LabelConfig())


# ---------- participant split ----------
IDS = [f"s{i:02d}" for i in range(1, 13)]


def test_split_is_disjoint_and_complete() -> None:
    split = split_participants(IDS, n_test=3, n_validation=2, seed=42)
    parts = [set(split.train), set(split.validation), set(split.test)]
    assert sum(len(p) for p in parts) == 12
    assert set().union(*parts) == set(IDS)
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert (len(split.test), len(split.validation), len(split.train)) == (3, 2, 7)


def test_split_is_reproducible_and_seed_dependent() -> None:
    a = split_participants(IDS, 3, 2, seed=42)
    assert a == split_participants(list(reversed(IDS)), 3, 2, seed=42)
    assert a.test != split_participants(IDS, 3, 2, seed=7).test


def test_segment_masks_never_share_participants() -> None:
    split = split_participants(IDS, 3, 2, seed=1)
    participants = np.repeat(IDS, 15)
    train, test = split.mask(participants, "train"), split.mask(participants, "test")
    assert not (train & test).any()
    assert set(participants[test]) == set(split.test)


def test_split_rejects_impossible_sizes() -> None:
    with pytest.raises(ValueError):
        split_participants(IDS[:3], n_test=2, n_validation=1, seed=0)
