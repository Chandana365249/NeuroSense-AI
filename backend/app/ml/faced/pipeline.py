"""FACED signal processing and features (declared before any FACED model was trained).

Per video trial, with no information crossing trials or participants:
1. Window: the LAST 28 s of the video span (ratings were given right after the clip; every clip is at least 34 s,
   so every trial yields the same 7 x 4 s segments and contributes equally).
2. Read the 30 shared scalp channels in the canonical FACED order (dataset.FACED_CHANNELS; both recording layouts
   are mapped by name), with a 1 s margin on both sides of the window.
3. Common average reference across the 30 channels (stateless, per sample).
4. 1000 Hz recordings are resampled to 250 Hz with scipy.signal.resample_poly (polyphase FIR anti-alias filter);
   250 Hz recordings are used as recorded. The margin absorbs filter edge effects and is then removed.
5. The project's EEGPreprocessor at 250 Hz: 4-45 Hz zero-phase Butterworth (order 4), per-channel robust clipping
   (median +/- 6 MAD-sigma), 4 s non-overlapping segments, rejection of segments with >5% clipped samples.
6. The project's FeatureExtractor at 250 Hz with FACED channel names: per channel log and relative band power
   (theta, alpha, beta, gamma), spectral entropy, Hjorth mobility/complexity, log variance (30 x 12 = 360) and 13
   hemispheric asymmetry pairs x 4 bands (52): 412 features.
Learned transforms (scaling, models) are fitted later, inside training folds only.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from app.ml.faced.bdf import read_window
from app.ml.faced.dataset import FACED_ASYMMETRY_PAIRS, FACED_CHANNELS, FacedParticipant, VideoTrial
from app.preprocessing.eeg_preprocessor import EEGPreprocessor, PreprocessingConfig
from app.preprocessing.features import FeatureExtractor

PIPELINE_VERSION = "faced-v2"  # v2: 30 harmonised scalp channels across both recording layouts
TARGET_FS = 250
WINDOW_S = 28.0
MARGIN_S = 1.0
VALENCE_THRESHOLD = 3.5  # 0-7 scale midpoint: high_valence if valence > 3.5, else low_valence (declared in advance)

PREPROCESSING = PreprocessingConfig(sampling_rate=TARGET_FS, n_channels=len(FACED_CHANNELS), baseline_seconds=0.0)
_preprocessor = EEGPreprocessor(PREPROCESSING)
_extractor = FeatureExtractor(TARGET_FS, FACED_CHANNELS, FACED_ASYMMETRY_PAIRS)


def feature_names() -> list[str]:
    return _extractor.feature_names


def valence_label(valence: float) -> int:
    """1 = high_valence, 0 = low_valence."""
    return int(valence > VALENCE_THRESHOLD)


def trial_signal(participant: FacedParticipant, trial: VideoTrial) -> np.ndarray:
    """(30, WINDOW_S * 250) common-average-referenced EEG in microvolts at 250 Hz, before band-pass filtering."""
    fs = participant.sampling_rate
    end = trial.onset_s + trial.duration_s
    start = end - WINDOW_S
    if start < trial.onset_s - 1e-9:
        raise ValueError(f"{participant.participant_id} video {trial.video_index}: shorter than {WINDOW_S} s")
    a = int(round((start - MARGIN_S) * fs))
    b = a + int(round((WINDOW_S + 2 * MARGIN_S) * fs))
    a_clip, b_clip = max(a, 0), min(b, participant.header.n_samples)
    x = read_window(participant.header, participant.channel_indices, a_clip, b_clip)
    if a_clip > a or b_clip < b:  # margin outside the recording: pad by reflection (never needed in v1.1.4)
        x = np.pad(x, ((0, 0), (a_clip - a, b - b_clip)), mode="reflect")
    x = x - x.mean(axis=0, keepdims=True)
    if fs == 1000.0:
        x = scipy_signal.resample_poly(x, up=1, down=4, axis=1)
    elif fs != TARGET_FS:
        raise ValueError(f"unsupported sampling rate {fs}")
    m = int(MARGIN_S * TARGET_FS)
    out = x[:, m:m + int(WINDOW_S * TARGET_FS)]
    if out.shape != (len(FACED_CHANNELS), int(WINDOW_S * TARGET_FS)) or not np.isfinite(out).all():
        raise ValueError("trial window has an unexpected shape or non-finite values")
    return out


def trial_features(participant: FacedParticipant, trial: VideoTrial) -> tuple[np.ndarray, list[int], int]:
    """(kept segments x 436 features, kept segment indices, number of segments rejected)."""
    result = _preprocessor.preprocess(trial_signal(participant, trial), has_baseline=False)
    if result.segments.shape[0] == 0:
        return np.empty((0, len(_extractor.feature_names)), np.float32), [], len(result.rejected_segment_indices)
    return _extractor.extract(result.segments), result.kept_segment_indices, len(result.rejected_segment_indices)


def trial_segments(participant: FacedParticipant, trial: VideoTrial) -> np.ndarray:
    """Preprocessed (kept segments, 30, 1000) arrays for an EEG-specific network."""
    return _preprocessor.preprocess(trial_signal(participant, trial), has_baseline=False).segments


def amplitude_check(participant: FacedParticipant) -> float:
    """Median over channels of the robust SD (1.4826 * MAD, microvolts) of 4-45 Hz EEG in the first video's window.
    Used to confirm every recording is on the same microvolt scale (the source had a units-label defect)."""
    x = _preprocessor.bandpass_filter(trial_signal(participant, participant.trials[0]))
    mad = np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    return float(np.median(1.4826 * mad))
