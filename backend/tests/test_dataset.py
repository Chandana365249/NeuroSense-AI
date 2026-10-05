from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.ml.dataset import DatasetError, DEAPDataset
from tests.conftest import N_TRIALS, PARTICIPANTS


def test_discovers_participant_files_in_order(fake_deap_dir: Path) -> None:
    files = DEAPDataset(fake_deap_dir).discover()
    assert [f.stem for f in files] == PARTICIPANTS


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        DEAPDataset(tmp_path / "nope").discover()


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="No DEAP participant files"):
        DEAPDataset(tmp_path).discover()


def test_load_participant_extracts_32_eeg_channels_and_ratings(fake_deap_dir: Path) -> None:
    record = DEAPDataset(fake_deap_dir).load_participant(fake_deap_dir / "s01.dat")
    assert record.participant_id == "s01"
    assert record.eeg.shape == (N_TRIALS, 32, 3 * 128 + 2 * 512)
    assert record.eeg.dtype == np.float32
    assert record.ratings.shape == (N_TRIALS, 4)
    assert record.total_channels == 40


def test_summary_reports_counts(fake_deap_dir: Path) -> None:
    summary = DEAPDataset(fake_deap_dir).summarize()
    assert summary.participants_found == summary.participants_valid == len(PARTICIPANTS)
    assert summary.total_trials == len(PARTICIPANTS) * N_TRIALS
    assert set(summary.rating_ranges) == {"valence", "arousal", "dominance", "liking"}


def test_corrupt_file_is_reported_not_fatal(tmp_path: Path, fake_deap_dir: Path) -> None:
    (tmp_path / "s01.dat").write_bytes((fake_deap_dir / "s01.dat").read_bytes())
    (tmp_path / "s02.dat").write_bytes(b"this is not a pickle")
    summary = DEAPDataset(tmp_path).summarize()
    assert summary.participants_found == 2
    assert summary.participants_valid == 1
    bad = next(f for f in summary.files if f.file == "s02.dat")
    assert not bad.ok and "unpickled" in bad.error


def test_wrong_shapes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s01.dat"
    with path.open("wb") as fh:
        pickle.dump({"labels": np.ones((3, 4)) * 5, "data": np.zeros((3, 10, 2000))}, fh, protocol=2)
    with pytest.raises(DatasetError, match="channels"):
        DEAPDataset(tmp_path).load_participant(path)


def test_out_of_range_ratings_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s01.dat"
    with path.open("wb") as fh:
        pickle.dump({"labels": np.full((2, 4), 12.0), "data": np.ones((2, 40, 2000))}, fh, protocol=2)
    with pytest.raises(DatasetError, match="ratings"):
        DEAPDataset(tmp_path).load_participant(path)


class _Exploit:
    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (os.getenv, ("PATH",))


def test_restricted_unpickler_blocks_arbitrary_objects(tmp_path: Path) -> None:
    path = tmp_path / "s01.dat"
    path.write_bytes(pickle.dumps({"labels": _Exploit(), "data": np.zeros(1)}, protocol=2))
    with pytest.raises(DatasetError, match="Disallowed"):
        DEAPDataset(tmp_path).load_participant(path)


def test_load_trial_bounds(fake_deap_dir: Path) -> None:
    ds = DEAPDataset(fake_deap_dir)
    eeg, ratings = ds.load_trial("s01", 0)
    assert eeg.shape[0] == 32 and ratings.shape == (4,)
    with pytest.raises(DatasetError):
        ds.load_trial("s01", N_TRIALS)
    with pytest.raises(DatasetError):
        ds.load_trial("s99", 0)
