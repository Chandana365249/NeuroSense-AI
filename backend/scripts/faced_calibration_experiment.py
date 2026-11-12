"""EXPLORATORY FACED experiment: does per-person calibration raise self-reported valence accuracy on unseen clips?

Declared before any model was fitted (protocol.json is written first). All 123 participants x 28 clips, faced-v2
features, valence > 3.5, logistic regression C = 0.01 for every model, trial = mean segment probability > 0.5.

Folds: 3 seeds. In each, the clips are split into 4 folds of 7, dealt evenly across emotion categories, and the
participants into 5 groups. A person's test clips are never used for their calibration, normalisation or labels, and
no population model sees the tested person or the tested clips.

Calibration sizes n: 7, 14, 21 rated clips, taken from that person's non-test clips, in the same random order for everyone.

Conditions, on identical folds:
A population : other people, other clips, no calibration (baseline)
B normalised : A, with each person's features z-scored using their own calibration clips (no labels used)
C personal   : a model fitted only on that person's n rated clips
D combined   : mean of the B and C probabilities. PRIMARY condition, named in advance.
If a person's calibration clips hold only one class, no personal model exists: C and D fall back to A and B. Counted.

Primary comparison: D - A at n = 21, pooled trial balanced accuracy. Secondary: B and C at n = 21, D at n = 7 and 14.
An improvement counts only if the 95% participant-cluster bootstrap interval of the difference lies above 0 AND its
Wilcoxon p (per-participant balanced accuracy) stays below 0.05 after Holm correction over the 5 comparisons.

This is exploratory, because every participant has been used or examined before. Calibrated results apply only to a
person who has first rated clips. No model is saved, and nothing used by the application changes.

Usage (from backend/): python -m scripts.faced_calibration_experiment
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ml.provenance import git_commit  # noqa: E402
from scripts import faced_experiment as fe  # noqa: E402
from scripts import model_improvement as mi  # noqa: E402

OUT = fe.OUT / "calibration"
CFG = next(c for c in fe.CANDIDATES if c["name"] == "lr_C0.01")
SEEDS = (42, 43, 44)
N_CLIP_FOLDS, N_GROUPS = 4, 5
SIZES = (7, 14, 21)
COMPARISONS = (("D", 21), ("B", 21), ("C", 21), ("D", 7), ("D", 14))


def clip_folds(categories: dict[int, str], seed: int) -> list[list[int]]:
    """N_CLIP_FOLDS folds of 7 clips, dealt round-robin through shuffled emotion categories."""
    rng = np.random.default_rng(seed)
    order = []
    for category in sorted(set(categories.values())):
        clips = [v for v, c in categories.items() if c == category]
        order += rng.permutation(clips).tolist()
    return [sorted(int(v) for v in order[k::N_CLIP_FOLDS]) for k in range(N_CLIP_FOLDS)]


def job(d: dict[str, Any], seed: int, fold: int, test_clips: list[int], group: list[str]) -> list[dict[str, Any]]:
    X, p, v, y = d["X"], d["p"], d["v"], d["y"]
    rng = np.random.default_rng([seed, fold])
    other_clips = rng.permutation(sorted(set(v.tolist()) - set(test_clips))).tolist()
    is_test_clip, in_group = np.isin(v, test_clips), np.isin(p, group)
    train = ~in_group & ~is_test_clip
    assert not set(p[train]) & set(group) and not set(v[train].tolist()) & set(test_clips)

    population = mi.make_model(CFG, seed).fit(X[train], y[train])
    z_train = mi.per_participant_z(X[train], p[train], np.ones(train.sum(), bool))  # others' own non-test clips
    normalised = mi.make_model(CFG, seed).fit(z_train, y[train])

    rows = []
    for person in group:
        mine = np.flatnonzero(p == person)
        test = mine[is_test_clip[mine]]
        if not len(test):
            continue
        base = mi.seg_scores(population, CFG, X[test])
        out: dict[str, Any] = {"A": base}
        fallback = {}
        for n in SIZES:
            cal_clips = other_clips[:n]
            cal = mine[np.isin(v[mine], cal_clips)]
            assert not set(v[cal].tolist()) & set(test_clips), "calibration uses a test clip"
            mean, sd = X[cal].mean(axis=0), X[cal].std(axis=0)
            xz = (X[test] - mean) / np.where(sd > 1e-12, sd, 1.0)
            b = mi.seg_scores(normalised, CFG, xz)
            one_class = len(set(y[cal].tolist())) < 2
            fallback[n] = bool(one_class)
            c = base if one_class else mi.seg_scores(mi.make_model(CFG, seed).fit(X[cal], y[cal]), CFG, X[test])
            d_score = b if one_class else (b + c) / 2
            out[f"B{n}"], out[f"C{n}"], out[f"D{n}"] = b, c, d_score
        for key, s in out.items():
            t = fe.trial_table(p[test], v[test], y[test], s)
            rows.append({"seed": seed, "person": person, "key": key, "y": t["y"], "score": t["score"], "pred": t["pred"]})
        rows.append({"seed": seed, "person": person, "key": "fallback", "fallback": fallback})
    return rows


def counts(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.array([((y == 1) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum(),
                     ((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()], float)


def ba(c: np.ndarray) -> np.ndarray:
    return 0.5 * (c[..., 0] / (c[..., 0] + c[..., 1]) + c[..., 2] / (c[..., 2] + c[..., 3]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise SystemExit("results.json exists: this exploratory experiment is run once")
    d = fe.load()
    root = get_settings().faced_dataset_dir
    with (root / "stimuli" / "stimuli.tsv").open(encoding="utf-8-sig") as f:
        categories = {int(r["stim_id"]): r["targeted_emotion"] for r in csv.DictReader(f, delimiter="\t")}
    people = sorted(set(d["p"].tolist()))
    assert len(people) == 123 and len(categories) == 28
    plan = [(s, k, clips, [str(q) for q in g]) for s in SEEDS
            for k, clips in enumerate(clip_folds(categories, s))
            for g in np.array_split(np.random.default_rng(s).permutation(people), N_GROUPS)]
    protocol = {"declared_at": datetime.now(UTC).isoformat(), "exploratory": True, "participants": len(people), "classifier": CFG,
                "seeds": SEEDS, "calibration_clips": SIZES, "clip_folds": {str(s): clip_folds(categories, s) for s in SEEDS},
                "primary": "D at n=21 minus A, pooled trial balanced accuracy", "comparisons": [f"{c}{n}" for c, n in COMPARISONS],
                "rule": "bootstrap 95% CI above 0 and Holm-corrected Wilcoxon p < 0.05"}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")

    done = Parallel(n_jobs=6, verbose=5)(delayed(job)(d, *item) for item in plan)
    rows = [r for batch in done for r in batch]
    keys = ["A"] + [f"{c}{n}" for c in "BCD" for n in SIZES]
    seed_index = {s: i for i, s in enumerate(SEEDS)}
    person_index = {q: i for i, q in enumerate(people)}
    table = {k: np.zeros((len(SEEDS), len(people), 4)) for k in keys}
    pooled: dict[str, dict[int, tuple[list[np.ndarray], list[np.ndarray]]]] = {k: {s: ([], []) for s in SEEDS} for k in keys}
    fallbacks = {n: 0 for n in SIZES}
    calibrations = 0
    for r in rows:
        if r["key"] == "fallback":
            calibrations += 1
            for n, used in r["fallback"].items():
                fallbacks[n] += int(used)
            continue
        table[r["key"]][seed_index[r["seed"]], person_index[r["person"]]] += counts(r["y"], r["pred"])
        pooled[r["key"]][r["seed"]][0].append(r["y"])
        pooled[r["key"]][r["seed"]][1].append(r["score"])

    per_person = {k: np.nanmean(ba(table[k]), axis=0) for k in keys}  # mean over seeds; NaN only if a class is absent
    summary = {}
    for k in keys:
        c = table[k].sum(axis=(0, 1))
        accuracy = (c[0] + c[2]) / c.sum()
        f1_high, f1_low = 2 * c[0] / (2 * c[0] + c[1] + c[3]), 2 * c[2] / (2 * c[2] + c[1] + c[3])
        auc = float(np.mean([roc_auc_score(np.concatenate(pooled[k][s][0]), np.concatenate(pooled[k][s][1])) for s in SEEDS]))
        summary[k] = {"balanced_accuracy": float(ba(c)), "accuracy": float(accuracy), "f1_macro": float((f1_high + f1_low) / 2), "roc_auc": auc,
                      "mean_per_participant_balanced_accuracy": float(np.nanmean(per_person[k])), "trials": int(c.sum())}

    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(people), (2000, len(people)))
    sums = {k: table[k].sum(axis=0)[picks].sum(axis=1) for k in keys}  # (resamples, 4)
    for k in keys:
        summary[k]["ci95"] = np.quantile(ba(sums[k]), [0.025, 0.975]).tolist()

    tests = []
    for c, n in COMPARISONS:
        key = f"{c}{n}"
        keep = ~np.isnan(per_person[key]) & ~np.isnan(per_person["A"])
        tests.append({"comparison": f"{key} - A", "difference": summary[key]["balanced_accuracy"] - summary["A"]["balanced_accuracy"],
                      "ci95": np.quantile(ba(sums[key]) - ba(sums["A"]), [0.025, 0.975]).tolist(),
                      "per_participant_mean_difference": float(np.mean(per_person[key][keep] - per_person["A"][keep])),
                      "participants_improved": int(np.sum(per_person[key][keep] > per_person["A"][keep])),
                      "participants_worse": int(np.sum(per_person[key][keep] < per_person["A"][keep])),
                      "wilcoxon_p": float(wilcoxon(per_person[key][keep], per_person["A"][keep]).pvalue)})
    running = 0.0
    for rank, i in enumerate(np.argsort([t["wilcoxon_p"] for t in tests])):
        running = max(running, min(1.0, tests[i]["wilcoxon_p"] * (len(tests) - rank)))
        tests[i]["holm_p"] = running
    for t in tests:
        t["improvement"] = bool(t["ci95"][0] > 0 and t["holm_p"] < 0.05)

    results = {"created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(), "exploratory": True, "conditions": summary, "comparisons": tests,
               "one_class_calibration_fallbacks": {str(n): fallbacks[n] for n in SIZES}, "person_fold_calibrations": calibrations}
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for k in keys:
        s = summary[k]
        print(k, round(s["balanced_accuracy"], 4), [round(x, 3) for x in s["ci95"]], "acc", round(s["accuracy"], 4), "f1", round(s["f1_macro"], 4),
              "auc", round(s["roc_auc"], 4), "perP", round(s["mean_per_participant_balanced_accuracy"], 4), "n", s["trials"])
    print(json.dumps(tests, indent=0))
    print("fallbacks", results["one_class_calibration_fallbacks"], "of", calibrations)


if __name__ == "__main__":
    main()
