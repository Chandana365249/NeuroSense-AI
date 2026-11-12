"""Controls for faced_calibration_experiment.py. Added AFTER its result was seen, to test whether the gain is an artifact.

Risk: FACED was presented in blocks of related clips, and EEG drifts during a session. If clips from one block are in
both calibration and test, a personal model can recognise the part of the session instead of any emotional response.

Blocks are inferred per person from the recording: sort clips by onset; the 6 longest gaps between clips split the 7 blocks.

Controls, all personal models (logistic regression C = 0.01), 3 seeds, 123 participants:
- random_clips : hold out 4 random clips at a time, calibrate on the other 24 (same sizes as block_held_out)
- block_held_out: hold out one whole block of 4 clips at a time, calibrate on the other 24
No-EEG baselines, using only ratings and presentation order:
- person_majority: predict the person's most common class among the 24 calibration clips
- nearest_in_time: predict the rating class of the calibration clip shown closest in time to the test clip

If block_held_out falls well below random_clips, or a no-EEG baseline matches the EEG model, the gain is not evidence
that calibrated EEG reads self-reported valence.

Usage (from backend/): python -m scripts.faced_calibration_controls
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.faced import dataset as fd  # noqa: E402
from scripts import faced_experiment as fe  # noqa: E402
from scripts import model_improvement as mi  # noqa: E402

OUT = fe.OUT / "calibration"
CFG = next(c for c in fe.CANDIDATES if c["name"] == "lr_C0.01")
SEEDS = (42, 43, 44)


def session(pid: str) -> dict[str, Any]:
    trials = sorted(fd.load_participant(get_settings().faced_dataset_dir, pid).trials, key=lambda t: t.onset_s)
    gaps = [trials[i + 1].onset_s - (trials[i].onset_s + trials[i].duration_s) for i in range(len(trials) - 1)]
    cuts = sorted(int(i) + 1 for i in np.argsort(gaps)[-6:])
    blocks = [[t.video_index for t in part] for part in np.split(np.array(trials, dtype=object), cuts)]
    return {"participant": pid, "order": [t.video_index for t in trials], "blocks": [[int(v) for v in b] for b in blocks],
            "valence_kind": {t.video_index: t.targeted_valence for t in trials}}


def person_job(X: np.ndarray, v: np.ndarray, y: np.ndarray, s: dict[str, Any], seed: int) -> dict[str, list[tuple[int, int]]]:
    """(true label, predicted label) per test trial, for each condition."""
    clips = sorted(set(v.tolist()))
    label = {c: int(y[v == c][0]) for c in clips}
    position = {c: i for i, c in enumerate(s["order"])}
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(clips).tolist()
    n = len(clips) // 7
    random_sets = [shuffled[i * n:(i + 1) * n] for i in range(7)]
    random_sets[-1] = shuffled[6 * n:]
    out: dict[str, list[tuple[int, int]]] = {"random_clips": [], "block_held_out": [], "person_majority": [], "nearest_in_time": []}
    for name, sets in (("random_clips", random_sets), ("block_held_out", s["blocks"])):
        for test in sets:
            test = [c for c in test if c in label]
            cal = [c for c in clips if c not in test]
            train = np.isin(v, cal)
            high = int(np.mean([label[c] for c in cal]) > 0.5)
            model = None if len({label[c] for c in cal}) < 2 else mi.make_model(CFG, seed).fit(X[train], y[train])
            for c in test:
                pred = high if model is None else int(mi.seg_scores(model, CFG, X[v == c]).mean() > 0.5)
                out[name].append((label[c], pred))
                if name == "block_held_out":
                    out["person_majority"].append((label[c], high))
                    nearest = min(cal, key=lambda k: abs(position[k] - position[c]))
                    out["nearest_in_time"].append((label[c], label[nearest]))
    return out


def balanced(pairs: np.ndarray) -> float:
    y, p = pairs[:, 0], pairs[:, 1]
    return float(0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()))


def main() -> None:
    d = fe.load()
    people = sorted(set(d["p"].tolist()))
    sessions = Parallel(n_jobs=6)(delayed(session)(q) for q in people)
    sizes = Counter(len(b) for s in sessions for b in s["blocks"])
    pure = [len({s["valence_kind"][c] for c in b}) == 1 for s in sessions for b in s["blocks"]]
    print("block sizes", dict(sizes), "blocks with a single valence kind", f"{np.mean(pure):.3f}")
    jobs = [(q, s, seed) for seed in SEEDS for q, s in zip(people, sessions, strict=True)]
    rows = Parallel(n_jobs=6)(delayed(person_job)(d["X"][d["p"] == q], d["v"][d["p"] == q], d["y"][d["p"] == q], s, seed) for q, s, seed in jobs)
    names = ("random_clips", "block_held_out", "person_majority", "nearest_in_time")
    by = {k: np.array([p for r in rows for p in r[k]]) for k in names}
    per = {k: np.array([balanced(np.array(r[k])) if len({a for a, _ in r[k]}) == 2 else np.nan for r in rows]).reshape(len(SEEDS), -1) for k in names}
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    groups = {k: np.array([np.array(rows[s * len(people) + i][k]) for s in range(len(SEEDS)) for i in range(len(people))], dtype=object) for k in names}
    for k in names:
        boots = []
        for _ in range(500):
            pick = rng.integers(0, len(people), len(people))
            boots.append(balanced(np.concatenate([groups[k][s * len(people) + i] for s in range(len(SEEDS)) for i in pick])))
        summary[k] = {"pooled_balanced_accuracy": balanced(by[k]), "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
                      "mean_per_participant_balanced_accuracy": float(np.nanmean(per[k])), "trials": int(len(by[k]))}
        print(k, round(summary[k]["pooled_balanced_accuracy"], 4), [round(x, 3) for x in summary[k]["ci95"]], "perP", round(summary[k]["mean_per_participant_balanced_accuracy"], 4), "n", summary[k]["trials"])
    (OUT / "controls.json").write_text(json.dumps({"created_at": datetime.now(UTC).isoformat(), "post_hoc_control": True, "block_sizes": dict(sizes),
                                                   "single_valence_kind_share": float(np.mean(pure)), "results": summary}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
