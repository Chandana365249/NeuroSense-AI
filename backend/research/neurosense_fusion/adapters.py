"""Dataset adapters and label harmonisation.

Each adapter keeps the original participant IDs, trial identifiers, rating values and rating scale. Dataset-specific
processing stays in its adapter; only the final representation is shared:
30 channels in FACED order, 125 Hz, 4 s segments from the last 28 s of a trial, common average reference, 4-45 Hz.

Self-reported valence and the intended stimulus category are separate fields and are never substituted for each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from app.core.config import get_settings
from app.ml.dataset import EEG_CHANNEL_NAMES, RATING_NAMES
from app.ml.faced import dataset as fd
from app.ml.faced import pipeline as fpipe
from app.ml.participants import is_frozen
from app.preprocessing.eeg_preprocessor import EEGPreprocessor

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "experiments" / "neurosense_fusion"
SHARED_CHANNELS: tuple[str, ...] = fd.FACED_CHANNELS
SHARED_RATE = 125
SEGMENT_SAMPLES = 500
SEGMENTS_PER_TRIAL = 7

RATING_SCALES: dict[str, tuple[float, float]] = {"FACED": (0.0, 7.0), "DEAP": (1.0, 9.0)}

# Datasets named in the plan that are not on this computer. Nothing is implemented against files that cannot be read.
COMPATIBILITY: dict[str, dict[str, str]] = {
    "FACED": {"status": "integrated", "role": "training, selection, reused test", "participants": "123", "rate": "250/1000 Hz, 125 Hz here",
              "channels": "32 recorded, 30 shared", "labels": "own valence 0-7; intended category kept separately"},
    "DEAP": {"status": "integrated", "role": "external evaluation only", "participants": "12 authorised; 8 frozen never read", "rate": "128 Hz release, 125 Hz here",
             "channels": "32 EEG, 30 shared (AF3, AF4 unused)", "labels": "own valence 1-9"},
    "DREAMER": {"status": "not available: not on this computer, needs requesting from its authors", "role": "candidate external test", "participants": "23 (published)",
                "rate": "128 Hz (published)", "channels": "14-channel Emotiv: most of the 30 shared channels are absent",
                "labels": "own valence 1-5"},
    "AMIGOS": {"status": "not available: not on this computer, access by request", "role": "optional", "participants": "40 (published)", "rate": "128 Hz (published)",
               "channels": "14-channel Emotiv: most of the 30 shared channels are absent", "labels": "own valence 1-9"},
    "SEED": {"status": "not available: not on this computer, access by application", "role": "optional unlabelled pretraining", "participants": "15 (published)",
             "rate": "1000 Hz, 200 Hz release (published)", "channels": "62-channel: covers the shared channels",
             "labels": "intended film category only: not self-reported valence, so never merged with these labels"},
}


def normalised_rating(raw: float, dataset: str) -> float:
    low, high = RATING_SCALES[dataset]
    if not low <= raw <= high:
        raise ValueError(f"{raw} is outside the {dataset} scale {low}-{high}")
    return (raw - low) / (high - low)


def valence_label(raw: float, dataset: str) -> int:
    """Common target: 1 if the rating is above the midpoint of that dataset's own scale. The midpoint itself is low."""
    return int(normalised_rating(raw, dataset) > 0.5)


def normalise_segments(x: np.ndarray) -> np.ndarray:
    """Per segment: remove each channel's mean, divide by one scale for the whole segment. Uses that segment only, so it
    cannot carry information between participants, and it keeps the spatial pattern across channels."""
    x = x - x.mean(axis=-1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(x), axis=(-2, -1), keepdims=True)
    return (x / np.maximum(scale, 1e-6)).astype(np.float32)


@dataclass
class Bundle:
    dataset: str
    segments: np.ndarray  # (rows, 30, 500), possibly a memmap
    participant: np.ndarray
    trial: np.ndarray  # original identifier: FACED video index (1-28), DEAP trial index (0-39)
    segment: np.ndarray
    rating: np.ndarray  # original scale
    y: np.ndarray
    stimulus_category: dict[int, str] = field(default_factory=dict)  # empty when the source gives none
    native_rates: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def load_faced() -> Bundle:
    faced_out = ROOT / "outputs" / "faced"
    segments = np.load(faced_out / f"segments125_{fpipe.PIPELINE_VERSION}.npy", mmap_mode="r")
    meta = np.load(faced_out / f"segments125_{fpipe.PIPELINE_VERSION}_meta.npz")
    z = np.load(faced_out / f"features_{fpipe.PIPELINE_VERSION}.npz", allow_pickle=False)
    assert np.array_equal(z["participant"], meta["participant"]) and np.array_equal(z["video"], meta["video"])
    rating = z["ratings"][:, list(z["rating_keys"]).index("Valence")].astype(float)
    y = np.array([valence_label(r, "FACED") for r in rating], dtype=np.int64)
    assert np.array_equal(y, meta["y"].astype(np.int64)), "the shared label must equal the FACED label"
    stim = fd.stimuli(get_settings().faced_dataset_dir)
    return Bundle("FACED", segments, meta["participant"], meta["video"].astype(int), z["segment"].astype(int), rating, y,
                  {v: s.get("targeted_valence", "") for v, s in stim.items()}, info={"scale": RATING_SCALES["FACED"], "midpoint_is": "low"})


def deap_trial_segments(eeg: np.ndarray, preprocessor: EEGPreprocessor) -> tuple[np.ndarray, list[int]]:
    """One DEAP trial (32+ channels, 128 Hz, 3 s baseline + 60 s) through the FACED representation."""
    order = [EEG_CHANNEL_NAMES.index(c) for c in SHARED_CHANNELS]
    x = eeg[order].astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    x = scipy_signal.resample_poly(x, up=125, down=64, axis=1)  # 128 -> 250 Hz, the rate of the FACED preprocessor
    window = int(fpipe.WINDOW_S * fpipe.TARGET_FS)
    result = preprocessor.preprocess(x[:, -window:], has_baseline=False)
    return scipy_signal.resample_poly(result.segments, up=1, down=2, axis=-1).astype(np.float32), result.kept_segment_indices


def load_deap(cache: Path | None = None) -> Bundle:
    """Only the participants authorised in settings; a frozen ID stops the run before any file is opened."""
    from app.ml.factory import make_dataset

    settings = get_settings()
    ids = sorted(settings.development_participants)
    assert ids and not any(is_frozen(p) for p in ids), "frozen participant in the authorised list"
    cache = cache or OUT / "cache"
    path, meta_path = cache / "deap_segments125.npy", cache / "deap_segments125_meta.npz"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        data = make_dataset(settings)
        preprocessor = fpipe._preprocessor
        segs, participant, trial, segment, rating = [], [], [], [], []
        assert RATING_NAMES[0] == "valence"
        records = list(data.iter_participants(set(ids)))
        assert sorted(r.participant_id for r in records) == ids, "an authorised participant could not be read"
        for record in records:
            for t in range(record.n_trials):
                s, kept = deap_trial_segments(record.eeg[t], preprocessor)
                segs.append(s)
                participant += [record.participant_id] * len(kept)
                trial += [t] * len(kept)
                segment += kept
                rating += [float(record.ratings[t][0])] * len(kept)
        np.save(path, np.concatenate(segs))
        np.savez(meta_path, participant=np.array(participant), trial=np.array(trial), segment=np.array(segment), rating=np.array(rating))
    meta = np.load(meta_path)
    y = np.array([valence_label(r, "DEAP") for r in meta["rating"]], dtype=np.int64)
    return Bundle("DEAP", np.load(path, mmap_mode="r"), meta["participant"], meta["trial"].astype(int), meta["segment"].astype(int), meta["rating"], y,
                  info={"scale": RATING_SCALES["DEAP"], "midpoint_is": "low", "presentation_order": "not in the preprocessed release"})
