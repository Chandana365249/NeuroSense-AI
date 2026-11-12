"""EXPLORATORY FACED experiment: intended stimulus category (positive vs negative video), NOT self-reported valence.

Question: can EEG responses classify the intended positive vs negative category of a KNOWN FACED video for a
participant whose recordings were never used in training? The same 24 videos appear for training and test
participants by design; nothing here measures generalisation to unseen videos or to anyone's feelings.

Declared before any model was fitted (written to outputs/faced/stimulus_category/protocol.json by --phase protocol):
- Label: the documented targeted valence of the video (stimuli/stimuli.tsv, Chen et al. 2023 Table S1), verified against
  every participant's events.tsv. positive (videos 17-28) = 1, negative (videos 1-12) = 0. The 4 neutral videos (13-16)
  are excluded from this binary task. Self-ratings are not used as labels.
- Data: the existing faced-v2 features (last 28 s of each video, 4 s segments, 412 features); unchanged.
- Participants: the participant partition of faced-protocol-v1 (98 development / 25 test), reused unchanged rather than
  drawn anew. Videos are NOT split. No test-participant recording enters training, scaling, model selection or
  threshold selection; no per-person normalisation is used (it would fit on the test person's recordings).
- Model selection: 5 participant-grouped folds over the development participants (the protocol's inner participant
  groups); pooled trial balanced accuracy averaged over folds; ties -> earlier candidate.
- Candidates (compact, from the earlier FACED comparison): majority; logistic regression C in {0.01, 0.1, 1}; the
  project's Random Forest. Trial prediction: mean segment probability > 0.5 (fixed; no threshold tuning).
- Final: the selected model is fitted on 98 development participants x 24 videos and evaluated ONCE on 25 test
  participants x 24 videos. This is exploratory: the test participants' recordings were already examined in the
  earlier FACED valence evaluation, so this is not an independent confirmation.

Usage (from backend/):
  python -m scripts.faced_stimulus_experiment --phase protocol
  python -m scripts.faced_stimulus_experiment --phase inner
  python -m scripts.faced_stimulus_experiment --phase freeze
  python -m scripts.faced_stimulus_experiment --phase final     # once
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.faced import dataset as fd  # noqa: E402
from app.ml.faced import pipeline as fpipe  # noqa: E402
from app.ml.faced import protocol as fprot  # noqa: E402
from app.ml.provenance import git_commit  # noqa: E402
from scripts import faced_experiment as fe  # noqa: E402
from scripts import model_improvement as mi  # noqa: E402

OUT = fe.OUT / "stimulus_category"
TASK = "stimulus_category_positive_vs_negative"
POSITIVE_VIDEOS = list(range(17, 29))
NEGATIVE_VIDEOS = list(range(1, 13))
NEUTRAL_VIDEOS = [13, 14, 15, 16]
CANDIDATES: list[dict[str, Any]] = [c for c in fe.CANDIDATES if c["name"] in ("majority", "lr_C0.01", "lr_C0.1", "lr_C1", "rf")]


def sha(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- data

def load() -> dict[str, Any]:
    d = fe.load()  # asserts the faced-protocol-v1 split hash
    keep = np.isin(d["v"], POSITIVE_VIDEOS + NEGATIVE_VIDEOS)
    return {"X": d["X"][keep], "p": d["p"][keep], "v": d["v"][keep],
            "y": np.isin(d["v"][keep], POSITIVE_VIDEOS).astype(int),          # stimulus category, not self-report
            "y_self": d["y"][keep], "split": d["split"]}


def verify_stimulus_labels() -> dict[str, Any]:
    """Cross-check stimuli.tsv against each participant's events.tsv; also describe presentation order."""
    root = get_settings().faced_dataset_dir
    stim = fd.stimuli(root)
    declared = {v: stim[v]["targeted_valence"] for v in range(1, 29)}
    assert [v for v, c in declared.items() if c == "Positive"] == POSITIVE_VIDEOS
    assert [v for v, c in declared.items() if c == "Negative"] == NEGATIVE_VIDEOS
    assert [v for v, c in declared.items() if c == "Neutral"] == NEUTRAL_VIDEOS
    mismatches, order_stats = [], []
    for pid in fd.participant_ids(root):
        path = next((root / pid / "eeg").glob("*_events.tsv"))
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig"), delimiter="\t"))
        onsets: dict[int, float] = {}
        for r in rows:
            if r["video_index"] in ("", "n/a"):
                continue
            v = int(float(r["video_index"]))
            if r["binary_label"].strip().lower() != declared[v].lower() or (
                    declared[v] != "Neutral" and r["emotion_label"] != stim[v]["targeted_emotion"]):
                mismatches.append([pid, v, r["emotion_label"], r["binary_label"]])
            onsets[v] = min(onsets.get(v, np.inf), float(r["onset"]))
        rank = {v: i for i, v in enumerate(sorted(onsets, key=onsets.get))}
        pos = np.mean([rank[v] for v in POSITIVE_VIDEOS])
        neg = np.mean([rank[v] for v in NEGATIVE_VIDEOS])
        order_stats.append({"participant": pid, "mean_rank_positive": float(pos), "mean_rank_negative": float(neg),
                            "first_video": min(onsets, key=onsets.get)})
    first_cat = [declared[o["first_video"]] for o in order_stats]
    return {"declared": declared, "participants_checked": len(order_stats), "mismatches": mismatches,
            "presentation_order": {
                "share_positive_shown_earlier_on_average": float(np.mean([o["mean_rank_positive"] < o["mean_rank_negative"]
                                                                          for o in order_stats])),
                "first_block_category_counts": {c: first_cat.count(c) for c in sorted(set(first_cat))}}}


# ---------------------------------------------------------------- phases

def phase_protocol(_: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        raise SystemExit("protocol.json exists; the protocol is declared once")
    check = verify_stimulus_labels()
    assert not check["mismatches"], check["mismatches"][:5]
    d = load()
    sp = d["split"]
    protocol = {
        "task": TASK, "exploratory": True, "declared_at": datetime.now(UTC).isoformat(),
        "label": {"source": "stimuli/stimuli.tsv targeted_valence (Chen et al. 2023 Table S1), verified against events.tsv",
                  "positive_1": POSITIVE_VIDEOS, "negative_0": NEGATIVE_VIDEOS, "excluded_neutral": NEUTRAL_VIDEOS,
                  "not_self_report": True},
        "stimulus_label_verification": check,
        "features": {"file": f"features_{fpipe.PIPELINE_VERSION}.npz", "sha256": sha(fe.OUT / f"features_{fpipe.PIPELINE_VERSION}.npz"),
                     "pipeline": fpipe.PIPELINE_VERSION},
        "participants": {"source": "faced-protocol-v1 partition, reused unchanged", "split_sha256": fe.FROZEN_SPLIT_SHA,
                         "train": sp["dev_participants"], "test": sp["test_participants"],
                         "inner_groups": sp["inner_participant_groups"]},
        "videos": "the same 24 non-neutral videos for training and test participants (known stimuli)",
        "candidates": CANDIDATES, "seed": fe.SEED,
        "selection_metric": "pooled trial balanced accuracy, mean over 5 participant-grouped development folds",
        "trial_rule": "mean segment probability > 0.5 (fixed)",
        "primary_metric": "trial-level balanced accuracy on 25 test participants x 24 videos; target 0.80",
        "no_per_person_normalisation": True,
        "caveat": "Test participants' recordings were examined in the earlier FACED valence evaluation; exploratory only.",
    }
    path.write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    print(json.dumps({k: protocol[k] for k in ("task", "label", "stimulus_label_verification")}, indent=1))
    print("protocol sha256", sha(path))


def fold_rows(d: dict[str, Any], k: int) -> tuple[np.ndarray, np.ndarray]:
    sp = d["split"]
    val_p = set(sp["inner_participant_groups"][k])
    tr = np.isin(d["p"], sorted(set(sp["dev_participants"]) - val_p))
    va = np.isin(d["p"], sorted(val_p))
    assert not set(d["p"][tr]) & set(d["p"][va]), "participant leakage"
    assert not np.isin(d["p"][tr | va], sp["test_participants"]).any(), "test participant in development folds"
    return tr, va


def inner_job(d: dict[str, Any], cfg: dict[str, Any], k: int) -> dict[str, Any]:
    tr, va = fold_rows(d, k)
    s = fe.fit_predict(cfg, d["X"][tr], d["y"][tr], d["X"][va])
    t = fe.trial_table(d["p"][va], d["v"][va], d["y"][va], s)
    return {"trial": fe.metrics(t["y"], t["score"], t["pred"])}


def phase_inner(_: argparse.Namespace) -> None:
    if not (OUT / "protocol.json").exists():
        raise SystemExit("run --phase protocol first")
    d = load()
    jobs = [(cfg, k) for cfg in CANDIDATES for k in range(fprot.N_INNER_FOLDS)]
    res = Parallel(n_jobs=4)(delayed(inner_job)(d, cfg, k) for cfg, k in jobs)
    out: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(), "results": {}}
    for (cfg, k), r in zip(jobs, res, strict=True):
        out["results"].setdefault(cfg["name"], {})[str(k)] = r
    (OUT / "inner.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summarize(out["results"]), indent=1))


def summarize(results: dict[str, Any]) -> dict[str, float]:
    return {n: float(np.mean([f["trial"]["balanced_accuracy"] for f in folds.values()])) for n, folds in results.items()}


def select() -> tuple[dict[str, Any], dict[str, float]]:
    scores = summarize(json.loads((OUT / "inner.json").read_text(encoding="utf-8"))["results"])
    order = [c["name"] for c in CANDIDATES]
    best = max(order, key=lambda n: (round(scores[n], 12), -order.index(n)))
    return next(c for c in CANDIDATES if c["name"] == best), scores


def phase_freeze(_: argparse.Namespace) -> None:
    if (OUT / "final_results.json").exists():
        raise SystemExit("final results already exist")
    d = load()
    sp = d["split"]
    tr, te = np.isin(d["p"], sp["dev_participants"]), np.isin(d["p"], sp["test_participants"])
    seen = np.zeros(len(d["y"]), bool)
    for k in range(fprot.N_INNER_FOLDS):
        a, b = fold_rows(d, k)
        seen |= a | b
    checks = {"inner_folds_use_no_test_participant": not np.isin(d["p"][seen], sp["test_participants"]).any(),
              "final_training_has_no_test_participant": not set(d["p"][tr]) & set(d["p"][te]),
              "neutral_videos_excluded": not np.isin(d["v"], NEUTRAL_VIDEOS).any(),
              "labels_are_stimulus_category": bool(np.array_equal(d["y"], np.isin(d["v"], POSITIVE_VIDEOS).astype(int))),
              "test_trials": int(len(set(zip(d["p"][te].tolist(), d["v"][te].tolist(), strict=True))))}
    assert all(v for k, v in checks.items() if k != "test_trials"), checks
    cfg, scores = select()
    freeze = {"frozen_at": datetime.now(UTC).isoformat(), "task": TASK, "protocol_sha256": sha(OUT / "protocol.json"),
              "selected": cfg, "inner_scores": scores, "checks": checks, "trial_rule": "mean segment probability > 0.5"}
    (OUT / "freeze.json").write_text(json.dumps(freeze, indent=1), encoding="utf-8")
    print(json.dumps(freeze, indent=1))


def phase_final(_: argparse.Namespace) -> None:
    import joblib

    path = OUT / "final_results.json"
    if path.exists():
        raise SystemExit("final_results.json exists: this test is run once")
    if not (OUT / "freeze.json").exists():
        raise SystemExit("run --phase freeze first")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert frozen["protocol_sha256"] == sha(OUT / "protocol.json"), "protocol changed after the freeze"
    cfg, scores = select()
    assert cfg == frozen["selected"], "selection changed after the freeze"
    d = load()
    sp = d["split"]
    tr, te = np.isin(d["p"], sp["dev_participants"]), np.isin(d["p"], sp["test_participants"])
    assert not set(d["p"][tr]) & set(d["p"][te])
    if cfg["family"] == "majority":
        s = fe.fit_predict(cfg, d["X"][tr], d["y"][tr], d["X"][te])
    else:
        model = mi.make_model(cfg, fe.SEED).fit(d["X"][tr], d["y"][tr])
        s = mi.seg_scores(model, cfg, d["X"][te])
        joblib.dump(model, OUT / f"stimulus_category_{cfg['name']}.joblib")  # research artifact only; never registered
    t = fe.trial_table(d["p"][te], d["v"][te], d["y"][te], s)
    ev = fe.evaluate(t, d["y"][te], s)
    tt = fe.trial_table(d["p"][tr], d["v"][tr], d["y"][tr], np.zeros(tr.sum()))
    maj = int(tt["y"].mean() > 0.5)
    tm = fe.trial_table(d["p"][te], d["v"][te], d["y"][te], np.full(te.sum(), float(maj)))
    per_video = {}
    for v in sorted(set(t["video"].tolist())):
        m = t["video"] == v
        per_video[str(v)] = {"category": "positive" if v in POSITIVE_VIDEOS else "negative",
                             "accuracy": float((t["y"][m] == t["pred"][m]).mean()), "mean_score": float(t["score"][m].mean())}
    # descriptive only: how often the participant's own valence label agrees with the stimulus category
    ts = fe.trial_table(d["p"][te], d["v"][te], d["y_self"][te], s)
    results = {"task": TASK, "exploratory": True, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
               "selected": cfg, "selection_scores_inner": scores, "train_participants": len(sp["dev_participants"]),
               "test_participants": len(sp["test_participants"]), "train_trials": int(len(tt["y"])),
               "test_trials": int(len(t["y"])), "majority_class": maj,
               "majority_reference": fe.metrics(tm["y"], tm["score"], tm["pred"]), **ev,
               "per_video": per_video,
               "descriptive_self_report_agreement": {
                   "self_label_equals_stimulus_category": float((ts["y"] == t["y"]).mean()),
                   "model_vs_self_report_balanced_accuracy": fe.metrics(ts["y"], ts["score"], ts["pred"])["balanced_accuracy"]},
               "target_80_reached": ev["trial_pooled"]["balanced_accuracy"] >= 0.80}
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(cfg["name"], json.dumps(ev["trial_pooled"]), json.dumps(ev["trial_pooled_ci95_participant_bootstrap"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["protocol", "inner", "freeze", "final"], required=True)
    args = ap.parse_args()
    {"protocol": phase_protocol, "inner": phase_inner, "freeze": phase_freeze, "final": phase_final}[args.phase](args)


if __name__ == "__main__":
    main()
