"""FACED support: BDF reader, BIDS validation, rating alignment, resampling, labels, frozen split, leakage guards and
trial aggregation. Uses small synthetic BDF/BIDS files in temporary folders; never the real dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.ml.faced import dataset as fd
from app.ml.faced import pipeline as fp
from app.ml.faced import protocol as fprot
from app.ml.faced.bdf import BDFError, read_header, read_window

PHYS, DIG = 375000.0, 8388607


def write_bdf(path: Path, data_uv: np.ndarray, fs: int, labels: tuple[str, ...] = fd.LAYOUT_A, unit: str = "uV") -> None:
    """Minimal 24-bit BDF writer (1 s records) for tests."""
    ns, n = data_uv.shape
    n_rec = n // fs
    digital = np.clip(np.round((data_uv + PHYS) / (2 * PHYS) * (2 * DIG + 1) - DIG - 1), -DIG - 1, DIG).astype(np.int64)

    def f(s: object, w: int) -> bytes:
        return str(s).ljust(w)[:w].encode("ascii")

    head = b"\xffBIOSEMI" + f("", 80) + f("", 80) + f("01.01.23", 8) + f("00.00.00", 8) + f(256 * (ns + 1), 8)
    head += f("24BIT", 44) + f(n_rec, 8) + f(1, 8) + f(ns, 4)
    sig = b"".join(f(lb, 16) for lb in labels) + f("", 80) * ns + f(unit, 8) * ns + f(-PHYS, 8) * ns + f(PHYS, 8) * ns
    sig += f(-DIG - 1, 8) * ns + f(DIG, 8) * ns + f("", 80) * ns + f(fs, 8) * ns + f("", 32) * ns
    body = bytearray()
    for r in range(n_rec):
        for c in range(ns):
            v = digital[c, r * fs:(r + 1) * fs] & 0xFFFFFF
            b = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1).astype(np.uint8)
            body += b.tobytes()
    path.write_bytes(head + sig + bytes(body))


def make_participant(root: Path, pid: str, fs: int = 250, seconds: int = 1200, ratings: dict[int, float] | None = None,
                     channels: tuple[str, ...] = fd.LAYOUT_A, n_videos: int = 28, seed: int = 0) -> None:
    eeg = root / pid / "eeg"
    eeg.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    t = np.arange(fs * seconds) / fs
    data = 10 * rng.standard_normal((32, len(t)))
    data[channels.index("Oz")] += 40 * np.sin(2 * np.pi * 10 * t)  # local alpha (a common signal would be removed by the average reference)
    if "T3" in channels:
        data[channels.index("T3")] += 1000.0  # a constant offset marks the old-name temporal channel
    write_bdf(eeg / f"{pid}_task-watchingVideoClips_eeg.bdf", data, fs, labels=channels)
    (eeg / f"{pid}_task-watchingVideoClips_eeg.json").write_text(json.dumps({"SamplingFrequency": float(fs)}))
    (eeg / f"{pid}_task-watchingVideoClips_channels.tsv").write_text(
        "name\ttype\tunits\n" + "".join(f"{c}\tEEG\tuV\n" for c in channels), encoding="utf-8")
    cols = ["onset", "duration", "trial_type", "value", "sample", "emotion_label", "binary_label", "video_index", *fd.RATING_KEYS]
    rows = ["\t".join(cols), "\t".join(["5.0", "0.0", "Experiment start", "5", "1250"] + ["n/a"] * (len(cols) - 5))]
    for v in range(1, n_videos + 1):
        val = (ratings or {}).get(v, (v % 8) * 1.0)
        r = {k: "1.0" for k in fd.RATING_KEYS}
        r["Valence"] = "n/a" if val is None else str(val)
        rows.append("\t".join([str(10 + 40 * (v - 1)), "35.0", f"Film {v}", str(v), str((10 + 40 * (v - 1)) * fs),
                               "Joy", "positive", str(v)] + [r[k] for k in fd.RATING_KEYS]))
    (eeg / f"{pid}_task-watchingVideoClips_events.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def faced_root(tmp_path: Path) -> Path:
    (tmp_path / "participants.tsv").write_text("participant_id\tage\tsex\nsub-000\t21\tF\nsub-001\t22\tM\n")
    make_participant(tmp_path, "sub-000", fs=250, seed=1)
    make_participant(tmp_path, "sub-001", fs=1000, seed=2, channels=fd.LAYOUT_B)
    return tmp_path


def test_bdf_reader_roundtrip_and_calibration(tmp_path: Path) -> None:
    fs, n = 250, 250 * 4
    data = np.vstack([np.linspace(-100, 100, n)] * 32)
    write_bdf(tmp_path / "x.bdf", data, fs)
    h = read_header(tmp_path / "x.bdf")
    assert h.labels == list(fd.LAYOUT_A) and h.sampling_rate(0) == fs and h.n_samples == n
    x = read_window(h, [0, 31], 100, 900)
    assert x.shape == (2, 800) and np.allclose(x, data[[0, 31], 100:900], atol=0.1)


def test_bdf_reader_rejects_non_bdf_and_truncated(tmp_path: Path) -> None:
    (tmp_path / "bad.bdf").write_bytes(b"0       " + b" " * 248)
    with pytest.raises(BDFError):
        read_header(tmp_path / "bad.bdf")
    write_bdf(tmp_path / "t.bdf", np.zeros((32, 500)), 250)
    raw = (tmp_path / "t.bdf").read_bytes()
    (tmp_path / "t.bdf").write_bytes(raw[:-100])
    with pytest.raises(BDFError, match="truncated"):
        read_header(tmp_path / "t.bdf")


def test_participant_validation_and_rating_alignment(faced_root: Path) -> None:
    p = fd.load_participant(faced_root, "sub-000", fd.load_demographics(faced_root))
    assert p.sampling_rate == 250 and p.age == "21" and p.problems == []
    assert [t.video_index for t in p.trials] == list(range(1, 29))
    assert p.trials[2].valence == 3.0 and p.trials[2].onset_s == 90.0  # video 3: valence (3 % 8), onset 10 + 80


def test_missing_rating_is_reported_not_invented(tmp_path: Path) -> None:
    make_participant(tmp_path, "sub-005", ratings={4: None})
    p = fd.load_participant(tmp_path, "sub-005")
    assert any("video 4" in x for x in p.problems)
    assert "Valence" not in next(t for t in p.trials if t.video_index == 4).ratings


def test_wrong_channel_order_and_missing_files_are_refused(tmp_path: Path) -> None:
    swapped = (fd.LAYOUT_A[1], fd.LAYOUT_A[0], *fd.LAYOUT_A[2:])
    make_participant(tmp_path, "sub-007", channels=swapped)
    with pytest.raises(fd.FacedError, match="channel"):
        fd.load_participant(tmp_path, "sub-007")
    make_participant(tmp_path, "sub-008")
    (tmp_path / "sub-008" / "eeg" / "sub-008_task-watchingVideoClips_events.tsv").unlink()
    with pytest.raises(fd.FacedError, match="missing"):
        fd.load_participant(tmp_path, "sub-008")


def test_both_recording_layouts_map_to_the_same_30_channels_by_name(faced_root: Path) -> None:
    a = fd.load_participant(faced_root, "sub-000")
    b = fd.load_participant(faced_root, "sub-001")
    assert (a.layout, b.layout) == ("A", "B")
    assert [fd.LAYOUT_A[i] for i in a.channel_indices][14:16] == ["T3", "T4"]
    assert [fd.LAYOUT_B[i] for i in b.channel_indices] == list(fd.FACED_CHANNELS)
    assert not {"A1", "A2"} & {fd.LAYOUT_A[i] for i in a.channel_indices}
    assert not {"HEOR", "HEOL"} & {fd.LAYOUT_B[i] for i in b.channel_indices}
    # the offset written on the old-name channel T3 appears in the canonical T7 position
    t = a.trials[0]
    raw = read_window(a.header, [fd.LAYOUT_A.index("T3")], int(t.onset_s * 250), int(t.onset_s * 250) + 250)
    mapped = read_window(a.header, [a.channel_indices[fd.FACED_CHANNELS.index("T7")]], int(t.onset_s * 250), int(t.onset_s * 250) + 250)
    assert np.allclose(raw, mapped) and raw.mean() > 500


def test_incomplete_video_set_is_reported(tmp_path: Path) -> None:
    make_participant(tmp_path, "sub-009", n_videos=27)
    assert any("27 of 28" in x for x in fd.load_participant(tmp_path, "sub-009").problems)


def test_both_sampling_rates_give_the_same_250hz_window_and_features(faced_root: Path) -> None:
    p250 = fd.load_participant(faced_root, "sub-000")
    p1000 = fd.load_participant(faced_root, "sub-001")
    for p in (p250, p1000):
        x = fp.trial_signal(p, p.trials[0])
        assert x.shape == (30, int(fp.WINDOW_S * fp.TARGET_FS))
        f, kept, rejected = fp.trial_features(p, p.trials[0])
        assert f.shape == (7, len(fp.feature_names())) and kept == list(range(7)) and rejected == 0
    # the shared 10 Hz component survives resampling: alpha dominates the relative power in both
    names = fp.feature_names()
    a = names.index("Oz_relpow_alpha")
    assert fp.trial_features(p250, p250.trials[0])[0][:, a].mean() > 0.5
    assert fp.trial_features(p1000, p1000.trials[0])[0][:, a].mean() > 0.5


def test_trial_window_is_the_last_28_seconds_of_the_video(faced_root: Path) -> None:
    p = fd.load_participant(faced_root, "sub-000")
    t = p.trials[0]
    fs = p.sampling_rate
    end = int((t.onset_s + t.duration_s) * fs)
    raw = read_window(p.header, p.channel_indices, end - int(fp.WINDOW_S * fs), end)
    raw = raw - raw.mean(axis=0, keepdims=True)
    assert np.allclose(fp.trial_signal(p, t), raw, atol=1e-6)


def test_valence_threshold_is_the_declared_midpoint() -> None:
    assert fp.VALENCE_THRESHOLD == 3.5
    assert fp.valence_label(3.5) == 0 and fp.valence_label(3.5001) == 1 and fp.valence_label(0.0) == 0


def test_feature_schema_differs_from_deap() -> None:
    from app.preprocessing.features import FeatureExtractor
    names = fp.feature_names()
    assert len(names) == 30 * 12 + 13 * 4 and "T7_logpow_alpha" in names and "asym_T8-T7_alpha" in names
    assert not any(n.startswith(("A1_", "A2_", "HEOR", "HEOL", "T3_")) for n in names)
    assert names != FeatureExtractor().feature_names


def _split() -> fprot.FacedSplit:
    cats = {v: c for c, vs in {"Anger": (1, 2, 3), "Disgust": (4, 5, 6), "Fear": (7, 8, 9), "Sadness": (10, 11, 12),
                              "n/a": (13, 14, 15, 16), "Amusement": (17, 18, 19), "Inspiration": (20, 21, 22),
                              "Joy": (23, 24, 25), "Tenderness": (26, 27, 28)}.items() for v in vs}
    return fprot.make_split([f"sub-{i:03d}" for i in range(123)], cats)


def test_split_is_frozen_disjoint_and_one_test_clip_per_category() -> None:
    s = _split()
    assert s.sha256 == _split().sha256
    assert len(s.test_participants) == 25 and len(s.dev_participants) == 98
    assert not set(s.test_participants) & set(s.dev_participants)
    assert len(s.test_videos) == 9 and not set(s.test_videos) & set(s.dev_videos)
    groups = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15, 16), (17, 18, 19), (20, 21, 22), (23, 24, 25), (26, 27, 28)]
    assert all(len(set(g) & set(s.test_videos)) == 1 for g in groups)
    assert sorted(sum(s.inner_participant_groups, ())) == list(s.dev_participants)


def test_inner_folds_never_share_people_or_clips_and_exclude_the_test_set() -> None:
    from scripts import faced_experiment as fe
    s = _split()
    rows = [(p, v) for p in s.dev_participants + s.test_participants for v in range(1, 29) for _ in range(2)]
    d = {"p": np.array([r[0] for r in rows]), "v": np.array([r[1] for r in rows]),
         "split": json.loads(s.to_json())}
    for k in range(fprot.N_INNER_FOLDS):
        tr, va = fe.inner_rows(d, k)
        assert tr.any() and va.any()
        assert not set(d["p"][tr]) & set(d["p"][va]) and not set(d["v"][tr]) & set(d["v"][va])
        assert not set(d["p"][tr | va]) & set(s.test_participants) and not set(d["v"][tr | va]) & set(s.test_videos)


def test_trial_aggregation_averages_segment_probabilities() -> None:
    from scripts import faced_experiment as fe
    p = np.array(["a", "a", "a", "b", "b"])
    v = np.array([1, 1, 2, 1, 1])
    t = fe.trial_table(p, v, np.array([1, 1, 0, 0, 0]), np.array([0.9, 0.3, 0.2, 0.4, 0.7]))
    assert list(zip(t["participant"], t["video"], strict=True)) == [("a", 1), ("a", 2), ("b", 1)]
    assert np.allclose(t["score"], [0.6, 0.2, 0.55]) and list(t["pred"]) == [1, 0, 1] and list(t["y"]) == [1, 0, 0]
