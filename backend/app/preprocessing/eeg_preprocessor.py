"""Multi-channel EEG preprocessing for DEAP-style recordings.

Steps (each is a method so it can be inspected/plotted on its own):
1. validate            - input must be (channels, samples), finite, non-flat, long enough for one segment.
2. remove_baseline     - drop the 3 s pre-stimulus period at the start of each DEAP trial.
3. bandpass_filter     - zero-phase Butterworth 4-45 Hz. DEAP's preprocessed release is already
                         band-limited to 4-45 Hz; re-applying it makes arbitrary API input comparable.
4. remove_artifacts    - per-channel robust clipping at median +/- k*MAD-sigma, so isolated spikes
                         cannot dominate band power. Clipping (not zeroing) avoids creating steps.
5. segment             - non-overlapping fixed-length windows (default 4 s); every window inherits
                         its trial's label. Windows with too many clipped samples are rejected.
6. normalize           - optional per-channel z-score (for future deep models). The feature-based
                         baseline does not z-score raw signals because that erases absolute band
                         power; features are standardized later by a scaler fit on training data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as scipy_signal

from app.core.logging import get_logger

logger = get_logger("preprocessing")


class SignalValidationError(ValueError):
    """Raised when an EEG array cannot be processed."""


@dataclass(frozen=True)
class PreprocessingConfig:
    sampling_rate: int = 128
    n_channels: int = 32
    baseline_seconds: float = 3.0
    low_hz: float = 4.0
    high_hz: float = 45.0
    filter_order: int = 4
    artifact_clip_sigma: float = 6.0
    max_clipped_fraction: float = 0.05
    segment_seconds: float = 4.0

    @property
    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.sampling_rate))

    @property
    def baseline_samples(self) -> int:
        return int(round(self.baseline_seconds * self.sampling_rate))


@dataclass
class PreprocessResult:
    segments: np.ndarray  # (n_kept, channels, segment_samples)
    kept_segment_indices: list[int]
    n_segments_total: int
    clipped_fraction: float
    rejected_segment_indices: list[int] = field(default_factory=list)


class EEGPreprocessor:
    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        nyquist = 0.5 * self.config.sampling_rate
        if not 0 < self.config.low_hz < self.config.high_hz < nyquist:
            raise ValueError("Band-pass edges must satisfy 0 < low < high < Nyquist")
        self._sos = scipy_signal.butter(
            self.config.filter_order,
            [self.config.low_hz, self.config.high_hz],
            btype="bandpass",
            fs=self.config.sampling_rate,
            output="sos",
        )

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    def load_signal(self, data: np.ndarray | list) -> np.ndarray:
        return self.validate(data)

    def validate(self, data: np.ndarray | list) -> np.ndarray:
        try:
            arr = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SignalValidationError("EEG must be a numeric (channels, samples) array") from exc
        if arr.ndim != 2:
            raise SignalValidationError(f"EEG must be 2-D (channels, samples), got shape {arr.shape}")
        if arr.shape[0] != self.config.n_channels:
            raise SignalValidationError(
                f"Expected {self.config.n_channels} EEG channels in DEAP order, got {arr.shape[0]}"
            )
        if arr.shape[1] < self.config.segment_samples:
            raise SignalValidationError(
                f"Need at least {self.config.segment_samples} samples "
                f"({self.config.segment_seconds} s at {self.config.sampling_rate} Hz), got {arr.shape[1]}"
            )
        if not np.isfinite(arr).all():
            raise SignalValidationError("EEG contains NaN or infinite values")
        if np.all(arr.std(axis=1) == 0):
            raise SignalValidationError("EEG is flat on every channel")
        return arr

    def remove_baseline(self, eeg: np.ndarray) -> np.ndarray:
        n = self.config.baseline_samples
        if eeg.shape[-1] - n < self.config.segment_samples:
            raise SignalValidationError("Signal is too short once the pre-stimulus baseline is removed")
        return eeg[..., n:]

    def bandpass_filter(self, eeg: np.ndarray) -> np.ndarray:
        return scipy_signal.sosfiltfilt(self._sos, eeg, axis=-1)

    def remove_artifacts(self, eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clip each channel to median +/- k * robust sigma. Returns (clipped, boolean clip mask)."""
        median = np.median(eeg, axis=-1, keepdims=True)
        mad = np.median(np.abs(eeg - median), axis=-1, keepdims=True)
        sigma = 1.4826 * mad
        sigma[sigma == 0] = np.inf
        limit = self.config.artifact_clip_sigma * sigma
        lower, upper = median - limit, median + limit
        mask = (eeg < lower) | (eeg > upper)
        return np.clip(eeg, lower, upper), mask

    def normalize(self, eeg: np.ndarray) -> np.ndarray:
        mean = eeg.mean(axis=-1, keepdims=True)
        std = eeg.std(axis=-1, keepdims=True)
        std[std == 0] = 1.0
        return (eeg - mean) / std

    def segment(self, eeg: np.ndarray) -> np.ndarray:
        """(channels, samples) -> (n_segments, channels, segment_samples); trailing remainder is dropped."""
        seg = self.config.segment_samples
        n = eeg.shape[-1] // seg
        if n == 0:
            raise SignalValidationError("Signal shorter than one segment")
        trimmed = eeg[:, : n * seg]
        return trimmed.reshape(eeg.shape[0], n, seg).transpose(1, 0, 2)

    def preprocess(self, eeg: np.ndarray | list, has_baseline: bool = False) -> PreprocessResult:
        arr = self.validate(eeg)
        if has_baseline:
            arr = self.remove_baseline(arr)
        filtered = self.bandpass_filter(arr)
        cleaned, clip_mask = self.remove_artifacts(filtered)
        segments = self.segment(cleaned)
        mask_segments = self.segment(clip_mask.astype(np.float32))
        clipped_per_segment = mask_segments.mean(axis=(1, 2))

        keep = clipped_per_segment <= self.config.max_clipped_fraction
        kept = [i for i, k in enumerate(keep) if k]
        rejected = [i for i, k in enumerate(keep) if not k]
        if rejected:
            logger.debug("Rejected segments with excessive clipping", extra={"rejected": rejected})
        return PreprocessResult(
            segments=segments[keep].astype(np.float32),
            kept_segment_indices=kept,
            n_segments_total=int(segments.shape[0]),
            clipped_fraction=float(clip_mask.mean()),
            rejected_segment_indices=rejected,
        )
