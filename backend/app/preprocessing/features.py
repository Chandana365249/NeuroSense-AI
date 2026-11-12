"""Hand-crafted EEG features computed per segment.

Per channel (32 channels x 12 = 384 features):
- log10 absolute band power in theta (4-8), alpha (8-13), beta (13-30), gamma (30-45) Hz, from a
  Welch PSD (1 s Hann windows, 50% overlap) integrated with the trapezoid rule.
- relative band power (band / total 4-45 Hz power): less sensitive to between-person amplitude
  differences than absolute power.
- spectral entropy (normalized Shannon entropy of the 4-45 Hz PSD): how spread the spectrum is.
- Hjorth mobility and complexity: time-domain descriptors of dominant frequency and its variability.
- log10 variance: overall signal power.

Hemispheric asymmetry (14 left/right pairs x 4 bands = 56 features): log power(right) - log power(left).
Frontal alpha asymmetry is a frequently studied correlate of valence in the EEG literature, but the
effect is small and inconsistent across studies.

Total: 440 features. Delta (<4 Hz) is not used: DEAP's preprocessed release is high-passed at 4 Hz.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from app.ml.dataset import EEG_CHANNEL_NAMES

FEATURE_VERSION = "v1"

FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

ASYMMETRY_PAIRS: tuple[tuple[str, str], ...] = (
    ("Fp1", "Fp2"), ("AF3", "AF4"), ("F3", "F4"), ("F7", "F8"), ("FC5", "FC6"), ("FC1", "FC2"),
    ("C3", "C4"), ("T7", "T8"), ("CP5", "CP6"), ("CP1", "CP2"), ("P3", "P4"), ("P7", "P8"),
    ("PO3", "PO4"), ("O1", "O2"),
)

_EPS = 1e-12


class FeatureExtractor:
    def __init__(
        self,
        sampling_rate: int = 128,
        channel_names: tuple[str, ...] = EEG_CHANNEL_NAMES,
        asymmetry_pairs: tuple[tuple[str, str], ...] = ASYMMETRY_PAIRS,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.channel_names = channel_names
        index = {name: i for i, name in enumerate(channel_names)}
        self._pairs = [(index[l], index[r], l, r) for l, r in asymmetry_pairs if l in index and r in index]
        self._names = self._build_names()

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    @property
    def n_features(self) -> int:
        return len(self._names)

    def _build_names(self) -> list[str]:
        per_channel = (
            [f"logpow_{b}" for b in FREQUENCY_BANDS]
            + [f"relpow_{b}" for b in FREQUENCY_BANDS]
            + ["spectral_entropy", "hjorth_mobility", "hjorth_complexity", "log_variance"]
        )
        names = [f"{ch}_{f}" for ch in self.channel_names for f in per_channel]
        names += [f"asym_{r}-{l}_{b}" for (_, _, l, r) in self._pairs for b in FREQUENCY_BANDS]
        return names

    def extract(self, segments: np.ndarray) -> np.ndarray:
        """(n_segments, channels, samples) -> (n_segments, n_features) float32."""
        segments = np.asarray(segments, dtype=np.float64)
        if segments.ndim != 3 or segments.shape[1] != len(self.channel_names):
            raise ValueError(f"segments must be (n, {len(self.channel_names)}, samples), got {segments.shape}")
        n_seg, n_ch, n_samp = segments.shape
        nperseg = min(self.sampling_rate, n_samp)
        freqs, psd = scipy_signal.welch(segments, fs=self.sampling_rate, nperseg=nperseg, axis=-1)

        lo_all, hi_all = min(b[0] for b in FREQUENCY_BANDS.values()), max(b[1] for b in FREQUENCY_BANDS.values())
        in_range = (freqs >= lo_all) & (freqs <= hi_all)
        total_power = np.trapezoid(psd[..., in_range], freqs[in_range], axis=-1) + _EPS

        band_power = np.stack(
            [
                np.trapezoid(psd[..., (freqs >= lo) & (freqs <= hi)], freqs[(freqs >= lo) & (freqs <= hi)], axis=-1)
                for lo, hi in FREQUENCY_BANDS.values()
            ],
            axis=-1,
        )  # (n_seg, n_ch, n_bands)
        log_power = np.log10(band_power + _EPS)
        rel_power = band_power / total_power[..., None]

        p = psd[..., in_range]
        p = p / (p.sum(axis=-1, keepdims=True) + _EPS)
        spectral_entropy = -(p * np.log2(p + _EPS)).sum(axis=-1) / np.log2(p.shape[-1])

        d1 = np.diff(segments, axis=-1)
        d2 = np.diff(d1, axis=-1)
        var0 = segments.var(axis=-1) + _EPS
        var1 = d1.var(axis=-1) + _EPS
        var2 = d2.var(axis=-1)
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / (mobility + _EPS)
        log_var = np.log10(var0)

        per_channel = np.concatenate(
            [log_power, rel_power, spectral_entropy[..., None], mobility[..., None], complexity[..., None], log_var[..., None]],
            axis=-1,
        )  # (n_seg, n_ch, 12)
        features = [per_channel.reshape(n_seg, -1)]
        if self._pairs:
            asym = np.stack([log_power[:, r, :] - log_power[:, l, :] for (l, r, _, _) in self._pairs], axis=1)
            features.append(asym.reshape(n_seg, -1))
        out = np.concatenate(features, axis=1).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Feature extraction produced non-finite values")
        return out
