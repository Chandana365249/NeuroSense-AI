"""Write example EEG files for the upload form (frontend/public/samples/).

Usage (from backend/):  python -m scripts.make_sample_eeg

The examples are real EEG from FACED (Chen et al. 2023, Sci Data 10:740; CC-BY 4.0), converted to the upload format
of the DEAP model: 32 channels in DEAP order, 128 samples per second, microvolts. No DEAP recording is used (its
licence does not allow redistribution).

How the clips are chosen: only FACED development participants (never the final-test participants of
faced-protocol-v1), one clip each, picked by the participant's OWN valence rating (clearly pleasant, clearly
unpleasant, in between) in a fixed order. The model is never run to pick them, so they are not cherry-picked.

Conversion, stated in every file: the last 60 s of the clip; common average reference over the 30 FACED scalp
channels; 1 Hz high-pass (zero-phase) so the preview is readable; polyphase resampling to 128 Hz. FACED has no AF3 or
AF4 electrode, so those two channels are ESTIMATED as the mean of their neighbours (AF3 = (Fp1 + F3) / 2,
AF4 = (Fp2 + F4) / 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal as ss

from app.core.config import PROJECT_ROOT, get_settings
from app.ml.dataset import EEG_CHANNEL_NAMES as DEAP_CHANNELS
from app.ml.faced import dataset as fd
from app.ml.faced.bdf import read_window

OUT = PROJECT_ROOT / "frontend" / "public" / "samples"
SECONDS, FS = 60, 128
ESTIMATED = {"AF3": ("Fp1", "F3"), "AF4": ("Fp2", "F4")}
# (file name, title, rule on the participant's own 0-7 valence rating)
WANTED = (
    ("faced-sample-pleasant.json", "Pleasant clip (rated pleasant by the viewer)", lambda v: v >= 6.0),
    ("faced-sample-unpleasant.json", "Unpleasant clip (rated unpleasant by the viewer)", lambda v: v <= 1.0),
    ("faced-sample-neutral.json", "Neutral clip (rated in between by the viewer)", lambda v: 3.0 <= v <= 4.0),
)


def convert(participant: fd.FacedParticipant, trial: fd.VideoTrial) -> np.ndarray:
    fs = participant.sampling_rate
    end = trial.onset_s + trial.duration_s
    margin = 2.0
    a, b = int(round((end - SECONDS - margin) * fs)), int(round((end + margin) * fs))
    b = min(b, participant.header.n_samples)
    x = read_window(participant.header, participant.channel_indices, a, b)
    x = x - x.mean(axis=0, keepdims=True)  # common average reference
    sos = ss.butter(4, 1.0, btype="highpass", fs=fs, output="sos")
    x = ss.sosfiltfilt(sos, x, axis=1)
    up, down = {1000.0: (16, 125), 250.0: (64, 125)}[fs]
    x = ss.resample_poly(x, up, down, axis=1)
    m = int(margin * FS)
    x = x[:, m:m + SECONDS * FS]
    by_name = dict(zip(fd.FACED_CHANNELS, x, strict=True))
    for name, (c1, c2) in ESTIMATED.items():
        by_name[name] = (by_name[c1] + by_name[c2]) / 2
    out = np.stack([by_name[c] for c in DEAP_CHANNELS[:32]])
    assert out.shape == (32, SECONDS * FS) and np.isfinite(out).all()
    return out


def main() -> None:
    settings = get_settings()
    split = json.loads((settings.faced_output_dir / "protocol_split_v1.json").read_text(encoding="utf-8"))
    test = set(split["test_participants"])
    stimuli = fd.stimuli(settings.faced_dataset_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    index = []
    for file_name, title, rule in WANTED:
        for pid in fd.participant_ids(settings.faced_dataset_dir):
            if pid in test or pid in used:
                continue
            participant = fd.load_participant(settings.faced_dataset_dir, pid)
            trial = next((t for t in participant.trials if t.duration_s >= SECONDS + 2 and rule(t.valence)), None)
            if trial is None:
                continue
            used.add(pid)
            x = convert(participant, trial)
            meta = {
                "title": title,
                "source": f"FACED {pid}, video {trial.video_index} ({stimuli.get(trial.video_index, {}).get('source_title')})",
                "viewer_valence_rating_0_to_7": round(float(trial.valence), 2),
                "licence": "FACED, CC-BY 4.0. Chen, J. et al. (2023) A Large Finer-grained Affective Computing EEG "
                           "Dataset. Scientific Data 10, 740. doi:10.1038/s41597-023-02650-w",
                "conversion": "Last 60 s of the clip; common average reference; 1 Hz high-pass; resampled to 128 Hz; "
                              "DEAP channel order. AF3 and AF4 are not recorded in FACED and are estimated as "
                              "(Fp1+F3)/2 and (Fp2+F4)/2.",
                "note": "The DEAP model was trained on another headset and other people, so its result on this file "
                        "is a demonstration, not a validated estimate.",
            }
            body = {"sample": meta, "sampling_rate": FS, "includes_baseline": False,
                    "signal": np.round(x, 1).tolist()}
            (OUT / file_name).write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
            index.append({"file": file_name, "title": title, "source": meta["source"],
                          "viewer_valence_rating_0_to_7": meta["viewer_valence_rating_0_to_7"]})
            print(file_name, meta["source"], meta["viewer_valence_rating_0_to_7"], f"{(OUT / file_name).stat().st_size / 1e6:.1f} MB")
            break
    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
