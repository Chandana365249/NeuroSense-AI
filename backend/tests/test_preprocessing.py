import numpy as np

from backend.preprocessing.eeg_preprocessor import EEGPreprocessor


def test_feature_extraction_returns_expected_keys() -> None:
    preprocessor = EEGPreprocessor(sampling_rate=128, window_size=128)
    signal = np.random.randn(128 * 2).astype(np.float32)

    features = preprocessor.extract_features(signal)

    assert isinstance(features, dict)
    assert {"mean", "variance", "std", "entropy", "band_power", "psd"}.issubset(features.keys())
    assert isinstance(features["band_power"], dict)
    assert set(features["band_power"].keys()) >= {"delta", "theta", "alpha", "beta", "gamma"}
