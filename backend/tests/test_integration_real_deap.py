"""Tests against the real DEAP files configured in backend/.env. Run with: pytest -m integration"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import get_settings
from app.ml.factory import make_dataset, make_feature_extractor, make_preprocessor
from app.ml.dataset import DatasetError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset():  # type: ignore[no-untyped-def]
    ds = make_dataset(get_settings())
    try:
        ds.discover()
    except DatasetError as exc:
        pytest.skip(f"Real DEAP dataset not available: {exc}")
    return ds


def test_real_files_have_expected_structure(dataset) -> None:  # type: ignore[no-untyped-def]
    summary = dataset.summarize()
    assert summary.participants_valid == summary.participants_found >= 1
    for f in summary.files:
        assert (f.trials, f.channels, f.samples, f.label_shape) == (40, 40, 8064, [40, 4])
    assert all(1.0 <= lo and hi <= 9.0 for lo, hi in summary.rating_ranges.values())


def test_real_trial_preprocesses_into_15_segments(dataset) -> None:  # type: ignore[no-untyped-def]
    settings = get_settings()
    pid = dataset.participant_ids()[0]
    eeg, ratings = dataset.load_trial(pid, 0)
    result = make_preprocessor(settings).preprocess(eeg, has_baseline=True)
    assert result.n_segments_total == 15
    feats = make_feature_extractor(settings).extract(result.segments)
    assert feats.shape == (result.segments.shape[0], 440)
    assert np.isfinite(feats).all()
