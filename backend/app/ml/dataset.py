"""Loader for the DEAP "data_preprocessed_python" release (one pickle per participant).

Each sNN.dat file is a Python-2 pickle of {"labels": (trials, 4), "data": (trials, channels, samples)}.
In the official release: 40 trials, 40 channels (32 EEG in Geneva order + 8 peripheral), 8064 samples
= 63 s at 128 Hz (3 s pre-trial baseline + 60 s video). Ratings are valence, arousal, dominance,
liking on a continuous 1-9 scale. Shapes are validated per file rather than assumed.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger

logger = get_logger("dataset")

RATING_NAMES: tuple[str, ...] = ("valence", "arousal", "dominance", "liking")
RATING_MIN, RATING_MAX = 1.0, 9.0

EEG_CHANNEL_NAMES: tuple[str, ...] = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
)

_PARTICIPANT_FILE = re.compile(r"^s(\d{2})\.dat$", re.IGNORECASE)
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),  # used by Python 3 protocol-2 pickles to store bytes; only converts str -> bytes
}


class DatasetError(Exception):
    """Raised when the dataset folder or a participant file is missing or invalid."""


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Refuses any global other than numpy array reconstruction, so a tampered file cannot run code."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise DatasetError(f"Disallowed object in pickle: {module}.{name}")
        if module == "numpy.core.multiarray":
            module = "numpy._core.multiarray"
        return super().find_class(module, name)


@dataclass(frozen=True)
class ParticipantRecord:
    participant_id: str
    path: Path
    eeg: np.ndarray  # (trials, n_eeg_channels, samples), float32, includes the pre-trial baseline
    ratings: np.ndarray  # (trials, 4) valence, arousal, dominance, liking
    total_channels: int

    @property
    def n_trials(self) -> int:
        return int(self.eeg.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.eeg.shape[2])


@dataclass
class FileReport:
    participant_id: str
    file: str
    ok: bool
    size_bytes: int
    trials: int | None = None
    channels: int | None = None
    samples: int | None = None
    label_shape: list[int] | None = None
    error: str | None = None


@dataclass
class DatasetSummary:
    dataset_dir: str
    sampling_rate_hz: int
    baseline_seconds: float
    participants_found: int
    participants_valid: int
    total_trials: int
    eeg_channels_used: int
    eeg_channel_names: list[str]
    rating_names: list[str]
    rating_ranges: dict[str, list[float]] = field(default_factory=dict)
    rating_means: dict[str, float] = field(default_factory=dict)
    files: list[FileReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DEAPDataset:
    def __init__(
        self,
        root: Path,
        n_eeg_channels: int = 32,
        sampling_rate: int = 128,
        baseline_seconds: float = 3.0,
    ) -> None:
        self.root = Path(root)
        self.n_eeg_channels = n_eeg_channels
        self.sampling_rate = sampling_rate
        self.baseline_seconds = baseline_seconds

    @property
    def baseline_samples(self) -> int:
        return int(round(self.baseline_seconds * self.sampling_rate))

    def discover(self) -> list[Path]:
        if not self.root.is_dir():
            raise DatasetError(f"Dataset directory not found: {self.root}")
        files = sorted(p for p in self.root.iterdir() if p.is_file() and _PARTICIPANT_FILE.match(p.name))
        if not files:
            raise DatasetError(f"No DEAP participant files (sNN.dat) found in {self.root}")
        logger.info("Discovered participant files", extra={"count": len(files), "dataset_dir": str(self.root)})
        return files

    @staticmethod
    def participant_id(path: Path) -> str:
        match = _PARTICIPANT_FILE.match(path.name)
        if not match:
            raise DatasetError(f"Not a DEAP participant file name: {path.name}")
        return f"s{match.group(1)}"

    def load_participant(self, path: Path) -> ParticipantRecord:
        pid = self.participant_id(path)
        try:
            with path.open("rb") as fh:
                raw = _NumpyOnlyUnpickler(fh, encoding="latin1").load()
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"{path.name}: could not be unpickled ({type(exc).__name__}: {exc})") from exc

        if not isinstance(raw, dict) or "data" not in raw or "labels" not in raw:
            raise DatasetError(f"{path.name}: expected a dict with 'data' and 'labels'")
        data = np.asarray(raw["data"])
        labels = np.asarray(raw["labels"], dtype=np.float64)

        if data.ndim != 3:
            raise DatasetError(f"{path.name}: data must be 3-D (trials, channels, samples), got {data.shape}")
        if labels.ndim != 2 or labels.shape[1] != len(RATING_NAMES):
            raise DatasetError(f"{path.name}: labels must be (trials, 4), got {labels.shape}")
        if labels.shape[0] != data.shape[0]:
            raise DatasetError(f"{path.name}: {labels.shape[0]} label rows for {data.shape[0]} trials")
        if data.shape[1] < self.n_eeg_channels:
            raise DatasetError(f"{path.name}: {data.shape[1]} channels, need at least {self.n_eeg_channels} EEG")
        if data.shape[2] <= self.baseline_samples:
            raise DatasetError(f"{path.name}: {data.shape[2]} samples is not longer than the baseline")

        eeg = data[:, : self.n_eeg_channels, :].astype(np.float32)
        if not np.isfinite(eeg).all():
            raise DatasetError(f"{path.name}: EEG contains NaN or infinite values")
        if not np.isfinite(labels).all() or labels.min() < RATING_MIN or labels.max() > RATING_MAX:
            raise DatasetError(f"{path.name}: ratings outside the {RATING_MIN}-{RATING_MAX} scale")

        logger.debug("Loaded participant", extra={"participant": pid, "shape": list(data.shape)})
        return ParticipantRecord(pid, path, eeg, labels, total_channels=int(data.shape[1]))

    def iter_participants(self, participant_ids: set[str] | None = None) -> Iterator[ParticipantRecord]:
        for path in self.discover():
            if participant_ids is not None and self.participant_id(path) not in participant_ids:
                continue
            yield self.load_participant(path)

    def participant_ids(self) -> list[str]:
        return [self.participant_id(p) for p in self.discover()]

    def load_trial(self, participant_id: str, trial_index: int) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / f"{participant_id}.dat"
        if not path.is_file():
            raise DatasetError(f"Participant {participant_id} not found in dataset")
        record = self.load_participant(path)
        if not 0 <= trial_index < record.n_trials:
            raise DatasetError(f"Trial index must be in [0, {record.n_trials - 1}]")
        return record.eeg[trial_index], record.ratings[trial_index]

    def summarize(self) -> DatasetSummary:
        files = self.discover()
        summary = DatasetSummary(
            dataset_dir=str(self.root),
            sampling_rate_hz=self.sampling_rate,
            baseline_seconds=self.baseline_seconds,
            participants_found=len(files),
            participants_valid=0,
            total_trials=0,
            eeg_channels_used=self.n_eeg_channels,
            eeg_channel_names=list(EEG_CHANNEL_NAMES[: self.n_eeg_channels]),
            rating_names=list(RATING_NAMES),
        )
        all_ratings: list[np.ndarray] = []
        for path in files:
            pid = self.participant_id(path)
            report = FileReport(participant_id=pid, file=path.name, ok=False, size_bytes=path.stat().st_size)
            try:
                record = self.load_participant(path)
            except DatasetError as exc:
                report.error = str(exc)
                logger.warning("Invalid participant file", extra={"participant": pid, "error": str(exc)})
            else:
                report.ok = True
                report.trials = record.n_trials
                report.channels = record.total_channels
                report.samples = record.n_samples
                report.label_shape = list(record.ratings.shape)
                summary.participants_valid += 1
                summary.total_trials += record.n_trials
                all_ratings.append(record.ratings)
            summary.files.append(report)

        if all_ratings:
            ratings = np.concatenate(all_ratings)
            for i, name in enumerate(RATING_NAMES):
                summary.rating_ranges[name] = [round(float(ratings[:, i].min()), 2), round(float(ratings[:, i].max()), 2)]
                summary.rating_means[name] = round(float(ratings[:, i].mean()), 3)
        return summary
