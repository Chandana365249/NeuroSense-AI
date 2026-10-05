from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy


class EEGPreprocessor:
    """A lightweight EEG preprocessing pipeline tailored to DEAP-style recordings."""

    def __init__(self, sampling_rate: int = 128, window_size: int = 128) -> None:
        self.sampling_rate = sampling_rate
        self.window_size = window_size

    def load_signal(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        nyquist = 0.5 * self.sampling_rate
        low = 0.5 / nyquist
        high = 45.0 / nyquist
        b, a = scipy_signal.butter(4, [low, high], btype="band")
        return scipy_signal.filtfilt(b, a, signal, axis=0)

    def remove_artifacts(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        std = np.std(signal)
        threshold = 5 * std
        mask = np.abs(signal) > threshold
        signal = signal.copy()
        signal[mask] = 0.0
        return signal

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.size == 0:
            return signal
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return np.zeros_like(signal, dtype=np.float32)
        return (signal - mean) / std

    def segment(self, signal: np.ndarray) -> list[np.ndarray]:
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        segments: list[np.ndarray] = []
        for start in range(0, len(signal) - self.window_size + 1, self.window_size):
            segments.append(signal[start : start + self.window_size])
        if not segments:
            segments.append(signal[: self.window_size])
        return segments

    def extract_features(self, signal: np.ndarray) -> dict[str, Any]:
        signal = self.load_signal(signal)
        filtered = self.bandpass_filter(signal)
        cleaned = self.remove_artifacts(filtered)
        normalized = self.normalize(cleaned)
        segments = self.segment(normalized.reshape(-1))

        features: dict[str, Any] = {}
        values = np.concatenate([seg for seg in segments if seg.size > 0])
        features["mean"] = float(np.mean(values))
        features["variance"] = float(np.var(values))
        features["std"] = float(np.std(values))
        features["entropy"] = float(scipy_entropy(values + 1e-8))

        band_power: dict[str, float] = {}
        freqs, psd = scipy_signal.welch(values, fs=self.sampling_rate)
        for band, (low, high) in {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power[band] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        features["band_power"] = band_power
        features["psd"] = {"freqs": freqs.tolist(), "psd": psd.tolist()}
        return features
