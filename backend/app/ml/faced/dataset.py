"""FACED BIDS reader: participants, verified channel order, video spans and each participant's own ratings.

Source: NEMAR nm000112 v1.1.4 (BIDS conversion of FACED, Chen et al. 2023, Sci Data 10:740; CC-BY-4.0).
Layout per participant: sub-XXX/eeg/sub-XXX_task-watchingVideoClips_{eeg.bdf,eeg.json,channels.tsv,events.tsv}.
Each events.tsv has one row per presented video (video_index 1-28) with onset/duration in seconds and the
participant's self-reported ratings on a continuous 0-7 scale (12 items, including Valence).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from app.ml.faced.bdf import BDFError, BDFHeader, read_header

DATASET_ID, DATASET_VERSION = "nm000112", "1.1.4"
TASK = "watchingVideoClips"
RATING_KEYS = ("Joy", "Tenderness", "Inspiration", "Amusement", "Anger", "Disgust", "Fear", "Sadness",
               "Arousal", "Valence", "Familiarity", "Liking")
RATING_MIN, RATING_MAX = 0.0, 7.0
N_VIDEOS = 28
# FACED v1.1.4 has two recording layouts (verified on all 123 participants: channels.tsv equals the BDF header):
#   A (sub-000..sub-060): 30 scalp channels with the older names T3/T4/T5/T6, plus the ear references A1/A2;
#   B (sub-061..sub-122): the same 30 scalp sites with the current names T7/T8/P7/P8, plus HEOR/HEOL (eye channels).
# Both are harmonised to the 30 shared scalp EEG channels in one canonical order with current 10-20 names
# (T3=T7, T4=T8, T5=P7, T6=P8). A1/A2 and HEOR/HEOL are not used.
LAYOUT_A: tuple[str, ...] = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T3", "T4",
    "A1", "A2", "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2",
)
LAYOUT_B: tuple[str, ...] = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2", "HEOR", "HEOL",
)
OLD_TO_CURRENT = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
FACED_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
)
FACED_ASYMMETRY_PAIRS: tuple[tuple[str, str], ...] = (
    ("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("FC5", "FC6"), ("FC1", "FC2"), ("C3", "C4"), ("T7", "T8"),
    ("CP5", "CP6"), ("CP1", "CP2"), ("P3", "P4"), ("P7", "P8"), ("PO3", "PO4"), ("O1", "O2"),
)


def channel_map(labels: tuple[str, ...]) -> tuple[str, list[int]]:
    """(layout name, file indices of FACED_CHANNELS) for a recording's channel list; refuses anything unknown."""
    if labels == LAYOUT_A:
        current = [OLD_TO_CURRENT.get(c, c) for c in labels]
        return "A", [current.index(c) for c in FACED_CHANNELS]
    if labels == LAYOUT_B:
        return "B", [labels.index(c) for c in FACED_CHANNELS]
    raise ValueError("unknown channel layout")


class FacedError(ValueError):
    """A FACED participant's files are missing, malformed or inconsistent."""


@dataclass(frozen=True)
class VideoTrial:
    video_index: int
    title: str
    targeted_emotion: str
    targeted_valence: str  # stimulus metadata (positive/negative/neutral); never used as a label
    onset_s: float
    duration_s: float
    ratings: dict[str, float]

    @property
    def valence(self) -> float:
        return self.ratings["Valence"]


@dataclass
class FacedParticipant:
    participant_id: str
    age: str
    sex: str
    bdf_path: Path
    header: BDFHeader
    sampling_rate: float
    layout: str
    channel_indices: list[int]
    trials: list[VideoTrial]
    problems: list[str] = field(default_factory=list)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def participant_ids(root: Path) -> list[str]:
    """IDs listed in participants.tsv (the dataset's own declaration)."""
    return [r["participant_id"] for r in _read_tsv(root / "participants.tsv")]


def _float(value: str) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_participant(root: Path, pid: str, demographics: dict[str, dict[str, str]] | None = None) -> FacedParticipant:
    """Parse and validate one participant. Raises FacedError when the recording cannot be used at all;
    recoverable issues (for example a video without a valence rating) are listed in `problems`."""
    eeg_dir = root / pid / "eeg"
    stem = f"{pid}_task-{TASK}"
    paths = {k: eeg_dir / f"{stem}_{k}" for k in ("eeg.bdf", "eeg.json", "channels.tsv", "events.tsv")}
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FacedError(f"{pid}: missing {', '.join(missing)}")
    try:
        header = read_header(paths["eeg.bdf"])
    except BDFError as exc:
        raise FacedError(str(exc)) from exc
    channels = tuple(r["name"] for r in _read_tsv(paths["channels.tsv"]))
    if channels != tuple(header.labels):
        raise FacedError(f"{pid}: channels.tsv does not match the BDF header")
    try:
        layout, indices = channel_map(channels)
    except ValueError as exc:
        raise FacedError(f"{pid}: channel names or order match neither verified FACED layout") from exc
    sidecar = json.loads(paths["eeg.json"].read_text(encoding="utf-8"))
    fs = float(sidecar.get("SamplingFrequency", 0))
    if any(abs(header.sampling_rate(i) - fs) > 1e-6 for i in indices):
        raise FacedError(f"{pid}: BDF sampling rate differs from eeg.json ({fs} Hz)")
    if fs not in (250.0, 1000.0):
        raise FacedError(f"{pid}: unsupported sampling rate {fs} Hz")
    units = {header.signals[i].physical_dimension for i in indices}
    if units != {"uV"}:
        raise FacedError(f"{pid}: unexpected physical unit(s) {sorted(units)}")

    problems: list[str] = []
    trials: list[VideoTrial] = []
    seen: set[int] = set()
    duration_total = header.n_samples / fs
    for row in _read_tsv(paths["events.tsv"]):
        vi = _float(row.get("video_index", "n/a"))
        if vi is None:
            continue
        v = int(vi)
        onset, duration = _float(row["onset"]), _float(row["duration"])
        if not 1 <= v <= N_VIDEOS or onset is None or duration is None or duration <= 0:
            raise FacedError(f"{pid}: malformed video event {row}")
        if v in seen:
            raise FacedError(f"{pid}: video {v} appears twice")
        if onset + duration > duration_total + 1e-6:
            raise FacedError(f"{pid}: video {v} ends after the recording")
        seen.add(v)
        ratings = {k: _float(row.get(k, "n/a")) for k in RATING_KEYS}
        bad = [k for k, x in ratings.items() if x is None or not RATING_MIN - 1e-9 <= x <= RATING_MAX + 1e-9]
        if "Valence" in bad:
            problems.append(f"video {v}: valence rating missing or out of range")
        trials.append(VideoTrial(v, row.get("trial_type", ""), row.get("emotion_label", ""), row.get("binary_label", ""),
                                 onset, duration, {k: x for k, x in ratings.items() if x is not None}))
    if seen != set(range(1, N_VIDEOS + 1)):
        problems.append(f"videos present: {len(seen)} of {N_VIDEOS}")
    demo = (demographics or {}).get(pid, {})
    return FacedParticipant(pid, demo.get("age", "n/a"), demo.get("sex", "n/a"), paths["eeg.bdf"], header, fs, layout,
                            indices, sorted(trials, key=lambda t: t.video_index), problems)


def load_demographics(root: Path) -> dict[str, dict[str, str]]:
    return {r["participant_id"]: r for r in _read_tsv(root / "participants.tsv")}


def stimuli(root: Path) -> dict[int, dict[str, str]]:
    return {int(r["stim_id"]): r for r in _read_tsv(root / "stimuli" / "stimuli.tsv")}
