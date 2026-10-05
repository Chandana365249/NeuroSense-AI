from __future__ import annotations

import numpy as np
import pytest

from app.preprocessing.eeg_preprocessor import EEGPreprocessor, PreprocessingConfig, SignalValidationError

SR = 128


def sine(freq: float, seconds: float, amp: float = 1.0, channels: int = 32) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return np.tile(amp * np.sin(2 * np.pi * freq * t), (channels, 1))


@pytest.fixture
def pre() -> EEGPreprocessor:
    return EEGPreprocessor(PreprocessingConfig())


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros(1024), "2-D"),
        (np.random.default_rng(0).normal(size=(8, 1024)), "32 EEG channels"),
        (np.random.default_rng(0).normal(size=(32, 100)), "at least 512"),
        (np.full((32, 1024), np.nan), "NaN"),
        (np.ones((32, 1024)), "flat"),
        ([["a"]], "numeric"),
    ],
)
def test_validation_rejects_bad_input(pre: EEGPreprocessor, data: object, message: str) -> None:
    with pytest.raises(SignalValidationError, match=message):
        pre.validate(data)  # type: ignore[arg-type]


def test_baseline_removal_drops_three_seconds(pre: EEGPreprocessor) -> None:
    out = pre.remove_baseline(sine(10, 63))
    assert out.shape == (32, 60 * SR)


def test_bandpass_keeps_alpha_and_removes_out_of_band(pre: EEGPreprocessor) -> None:
    alpha = pre.bandpass_filter(sine(10, 8))[:, SR:-SR]
    slow = pre.bandpass_filter(sine(1, 8))[:, SR:-SR]
    line = pre.bandpass_filter(sine(55, 8))[:, SR:-SR]
    assert alpha.std() > 0.6
    assert slow.std() < 0.1
    assert line.std() < 0.1


def test_artifact_clipping_limits_spikes_without_zeroing(pre: EEGPreprocessor) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=(32, 1024))
    x[0, 500] = 500.0
    clipped, mask = pre.remove_artifacts(x)
    assert mask[0, 500]
    assert 3 < clipped[0, 500] < 10
    assert clipped[0, 500] != 0.0
    assert mask.mean() < 0.01


def test_segmentation_shape(pre: EEGPreprocessor) -> None:
    segs = pre.segment(sine(10, 10))
    assert segs.shape == (2, 32, 512)


def test_preprocess_full_trial(pre: EEGPreprocessor) -> None:
    rng = np.random.default_rng(2)
    trial = rng.normal(0, 5, size=(32, 63 * SR)) + sine(10, 63, 10)
    result = pre.preprocess(trial, has_baseline=True)
    assert result.n_segments_total == 15
    assert result.segments.shape == (15, 32, 512)
    assert result.segments.dtype == np.float32


def test_heavily_contaminated_segment_is_rejected(pre: EEGPreprocessor) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=(32, 3 * 512))
    x[:, 512:1024:3] = 1000.0
    result = pre.preprocess(x)
    assert 1 in result.rejected_segment_indices
    assert result.segments.shape[0] == 2


def test_normalize_is_per_channel_zscore(pre: EEGPreprocessor) -> None:
    z = pre.normalize(np.random.default_rng(4).normal(5, 3, size=(32, 1024)))
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1, atol=1e-9)
