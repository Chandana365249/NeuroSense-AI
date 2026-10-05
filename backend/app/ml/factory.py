from __future__ import annotations

from app.core.config import Settings
from app.ml.dataset import DEAPDataset
from app.ml.labels import LabelConfig, LabelStrategy
from app.preprocessing.eeg_preprocessor import EEGPreprocessor, PreprocessingConfig
from app.preprocessing.features import FeatureExtractor


def make_dataset(settings: Settings) -> DEAPDataset:
    return DEAPDataset(settings.dataset_dir, settings.n_eeg_channels, settings.sampling_rate, settings.baseline_seconds)


def make_preprocessing_config(settings: Settings) -> PreprocessingConfig:
    return PreprocessingConfig(
        sampling_rate=settings.sampling_rate,
        n_channels=settings.n_eeg_channels,
        baseline_seconds=settings.baseline_seconds,
        artifact_clip_sigma=settings.artifact_clip_sigma,
        max_clipped_fraction=settings.max_clipped_fraction,
        segment_seconds=settings.segment_seconds,
    )


def make_preprocessor(settings: Settings) -> EEGPreprocessor:
    return EEGPreprocessor(make_preprocessing_config(settings))


def make_feature_extractor(settings: Settings) -> FeatureExtractor:
    return FeatureExtractor(sampling_rate=settings.sampling_rate)


def make_label_config(settings: Settings) -> LabelConfig:
    return LabelConfig(
        strategy=LabelStrategy(settings.label_strategy),
        valence_threshold=settings.valence_threshold,
        arousal_threshold=settings.arousal_threshold,
    )
