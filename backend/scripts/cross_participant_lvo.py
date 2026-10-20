"""Cross-participant, leave-video-out valence decoding (Protocol v1.1, exploratory, 12 participants).

Design: docs/EXPLORATORY_CROSS_PARTICIPANT_LEAVE_VIDEO_OUT.md section 2 (declared before running).
For each seed (42-46): the 40 videos are permuted with default_rng(seed) and split into 5 folds of 8. For each
held-out participant p and video fold k, the registered Random Forest is trained on participants != p and
videos not in k, and tested on participant p's trials for the videos in k.
Task A labels = each training participant's own rating (valence > 5.0).
Task B labels = majority (valence > 5.0) of the 10 participants other than the example's participant and p;
tied training pairs (5 vs 5) are excluded from training in BOTH tasks. Test consensus = majority of the other 11.

Usage (from backend/):
  python -m scripts.cross_participant_lvo --phase main     # ~45 min plugged in, ~1.7 h on battery; resumable
  python -m scripts.cross_participant_lvo --phase report
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.ml.deep.robust_evaluation import git_commit
from app.ml.models import candidate_models
from app.ml.stats import mean_ci, wilcoxon
from scripts import within_person_diagnostic as wp
from scripts.stimulus_confound_control import _participant_bootstrap, _sign_flip_exact

PARTICIPANTS, SEEDS, LABEL = wp.PARTICIPANTS, wp.SEEDS, wp.LABEL
N_VIDEOS, N_VIDEO_FOLDS = 40, 5
TASKS = ("personal", "consensus")
HIGH, LOW = "high_valence", "low_valence"


def video_folds(seed: int) -> list[list[int]]:
    order = np.random.default_rng(seed).permutation(N_VIDEOS)
    return [sorted(order[k::N_VIDEO_FOLDS].tolist()) for k in range(N_VIDEO_FOLDS)]


def training_consensus(high: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """For outer held-out participant index p: (labels, tied) arrays of shape (participants, videos).
    Entry (q, v) uses the votes of all participants except q and p. Row p is unused (masked as tied)."""
    n = high.shape[0]
    labels = np.full(high.shape, LOW, dtype=object)
    tied = np.zeros(high.shape, dtype=bool)
    for q in range(n):
        if q == p:
            tied[q] = True
            continue
        voters = np.setdiff1d(np.arange(n), [q, p])
        votes = high[voters].sum(axis=0)
        tied[q] = votes * 2 == len(voters)
        labels[q] = np.where(votes * 2 > len(voters), HIGH, LOW)
    return labels.astype(str), tied


def test_consensus(high: np.ndarray, p: int) -> np.ndarray:
    voters = np.setdiff1d(np.arange(high.shape[0]), [p])
    votes = high[voters].sum(axis=0)
    if np.any(votes * 2 == len(voters)):
        raise ValueError("tie in test consensus")
    return np.where(votes * 2 > len(voters), HIGH, LOW).astype(str)


def load_arrays() -> dict[str, Any]:
    table, y = wp.load()
    index = {pid: i for i, pid in enumerate(PARTICIPANTS)}
    pidx = np.array([index[p] for p in table.participant])
    vidx = table.trial.astype(int)
    valence = np.full((len(PARTICIPANTS), N_VIDEOS), np.nan)
    for i, pid in enumerate(PARTICIPANTS):
        _, seg_trial, trial_labels, val = wp.participant_data(table, y, pid)
        assert len(val) == N_VIDEOS and seg_trial.max() == N_VIDEOS - 1, f"{pid}: expected 40 trials"
        valence[i] = val
        assert np.array_equal(np.where(val > 5.0, HIGH, LOW), trial_labels)
    assert np.array_equal(np.where(valence[pidx, vidx] > 5.0, HIGH, LOW), y), "row labels must equal the ratings rule"
    return {"X": table.X, "y": y, "pidx": pidx, "vidx": vidx, "valence": valence, "fingerprint": table.fingerprint}


def run_participant(d: dict[str, Any], p: int) -> dict[str, Any]:
    X, y, pidx, vidx = d["X"], d["y"], d["pidx"], d["vidx"]
    high = d["valence"] > 5.0
    cons_train, tied = training_consensus(high, p)
    cons_test = test_consensus(high, p)
    personal_test = np.where(high[p], HIGH, LOW)
    row_cons = cons_train[pidx, vidx]
    out: dict[str, Any] = {"targets": {"personal": personal_test.tolist(), "consensus": cons_test.tolist()},
                           "tied_pairs_excluded": int(tied[np.arange(len(PARTICIPANTS)) != p].sum()), "seeds": {}}
    for seed in SEEDS:
        folds = video_folds(seed)
        seed_out: dict[str, Any] = {"video_folds": folds, "fits": [], "tasks": {}}
        proba = {t: np.full((N_VIDEOS, 2), np.nan) for t in TASKS}
        classes: dict[str, list[str]] = {}
        for k, fold in enumerate(folds):
            in_fold = np.isin(vidx, fold)
            train = (pidx != p) & ~in_fold & ~tied[pidx, vidx]
            test = (pidx == p) & in_fold
            assert not np.any(pidx[train] == p) and not np.any(np.isin(vidx[train], fold)), "held-out data in training"
            assert set(vidx[test].tolist()) == set(fold), "every held-out video must have test segments"
            seed_out["fits"].append({"fold": k, "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                                     "train_pairs": int(len(set(zip(pidx[train].tolist(), vidx[train].tolist())))),
                                     "tied_pairs_in_training_videos": int(tied[np.ix_(np.arange(len(PARTICIPANTS)) != p,
                                                                                       np.setdiff1d(np.arange(N_VIDEOS), fold))].sum())})
            for task, labels in (("personal", y), ("consensus", row_cons)):
                model = candidate_models(seed)["random_forest"]().fit(X[train], labels[train])
                classes[task] = [str(c) for c in model.classes_]
                seg_p = model.predict_proba(X[test])
                for v in fold:
                    assert np.isnan(proba[task][v]).all(), "a test trial was predicted twice"
                    proba[task][v] = seg_p[vidx[test] == v].mean(axis=0)
        for task in TASKS:
            assert not np.isnan(proba[task]).any(), "every test trial must be predicted exactly once"
            target = personal_test if task == "personal" else cons_test
            seed_out["tasks"][task] = {"classes": classes[task], "proba": proba[task].round(8).tolist(),
                                       **wp.metrics(target, proba[task], classes[task])}
        out["seeds"][str(seed)] = seed_out
    return out


def phase_main(out: Path) -> None:
    d = load_arrays()
    path = out / "main.json"
    res: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "label": LABEL, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
        "fingerprint": d["fingerprint"], "seeds": list(SEEDS),
        "video_folds": {str(s): video_folds(s) for s in SEEDS},
        "design": "docs/EXPLORATORY_CROSS_PARTICIPANT_LEAVE_VIDEO_OUT.md section 2", "participants": {}}
    for p, pid in enumerate(PARTICIPANTS):
        if pid in res["participants"]:
            print(f"{pid}: already done", flush=True)
            continue
        r = run_participant(d, p)
        res["participants"][pid] = r
        path.write_text(json.dumps(res), encoding="utf-8")
        a = np.mean([r["seeds"][str(s)]["tasks"]["personal"]["balanced_accuracy"] for s in SEEDS])
        b = np.mean([r["seeds"][str(s)]["tasks"]["consensus"]["balanced_accuracy"] for s in SEEDS])
        print(f"{pid}: personal BA {a:.3f} | consensus BA {b:.3f} | tied pairs excluded {r['tied_pairs_excluded']}", flush=True)


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def phase_report(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((out / "main.json").read_text(encoding="utf-8"))
    rows = []
    for pid in PARTICIPANTS:
        r = res["participants"][pid]
        row: dict[str, Any] = {"participant": pid, "tied_pairs_excluded": r["tied_pairs_excluded"]}
        for task in TASKS:
            per_seed = [r["seeds"][str(s)]["tasks"][task] for s in SEEDS]
            target = np.array(r["targets"][task])
            row[task] = {k: float(np.mean([m[k] for m in per_seed])) for k in ("balanced_accuracy", "f1_macro", "roc_auc", "predicted_high_share")}
            row[task]["ba_sd"] = float(np.std([m["balanced_accuracy"] for m in per_seed], ddof=1))
            row[task]["true_high_share"] = float(np.mean(target == HIGH))
            row[task]["single_label_seeds"] = int(sum(m["predicted_high_share"] < 0.05 or m["predicted_high_share"] > 0.95 for m in per_seed))
        row["target_agreement"] = float(np.mean(np.array(r["targets"]["personal"]) == np.array(r["targets"]["consensus"])))
        rows.append(row)

    group: dict[str, Any] = {}
    for metric in ("balanced_accuracy", "f1_macro", "roc_auc"):
        a = np.array([r["personal"][metric] for r in rows]); b = np.array([r["consensus"][metric] for r in rows])
        group[metric] = {"personal": mean_ci(a.tolist()) | {"bootstrap_ci": _participant_bootstrap(a)},
                         "consensus": mean_ci(b.tolist()) | {"bootstrap_ci": _participant_bootstrap(b)},
                         "difference": mean_ci((a - b).tolist()) | {"bootstrap_ci": _participant_bootstrap(a - b)},
                         "n_personal_better": int((a > b).sum())}
    a = np.array([r["personal"]["balanced_accuracy"] for r in rows]); b = np.array([r["consensus"]["balanced_accuracy"] for r in rows])
    raw = {"personal_vs_0.5": _sign_flip_exact(a - 0.5), "consensus_vs_0.5": _sign_flip_exact(b - 0.5),
           "personal_minus_consensus": _sign_flip_exact(a - b)}
    group["tests_balanced_accuracy"] = {
        "method": "exact sign-flip over participants (4096 patterns), two-sided; Holm across the three tests",
        "p_raw": raw, "p_holm": _holm(raw),
        "wilcoxon": {"personal_vs_0.5": wilcoxon(a, 0.5), "consensus_vs_0.5": wilcoxon(b, 0.5),
                     "personal_minus_consensus": wilcoxon(a, b)}}
    group["collapse"] = {task: {"participants_with_any_single_label_seed": int(sum(r[task]["single_label_seeds"] > 0 for r in rows)),
                                "mean_abs_predicted_minus_true_share": float(np.mean([abs(r[task]["predicted_high_share"] - r[task]["true_high_share"]) for r in rows])),
                                "predicted_share_range": [float(min(r[task]["predicted_high_share"] for r in rows)),
                                                          float(max(r[task]["predicted_high_share"] for r in rows))]}
                         for task in TASKS}
    group["tied_pairs_excluded_total"] = int(sum(r["tied_pairs_excluded"] for r in rows))
    summary = {"label": LABEL, "git_commit": res["git_commit"], "fingerprint": res["fingerprint"], "seeds": res["seeds"],
               "video_folds": res["video_folds"], "participants": rows, "group": group}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(rows))
    for off, task, name in ((-0.2, "personal", "A: own rating"), (0.2, "consensus", "B: other participants' consensus")):
        ax.bar(x + off, [r[task]["balanced_accuracy"] for r in rows], 0.4, yerr=[r[task]["ba_sd"] for r in rows], capsize=2, label=name)
    ax.axhline(0.5, ls="--", color="0.5", lw=0.8); ax.set_ylim(0, 1)
    ax.set_xticks(x, [r["participant"] for r in rows]); ax.set_ylabel("trial balanced accuracy (mean ± SD, 5 seeds)")
    ax.set_title(f"Cross-participant, leave-video-out decoding (held-out person AND held-out videos). {LABEL}", fontsize=8)
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(out / "cross_participant_lvo.png", dpi=130); plt.close(fig)
    print(json.dumps(group, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["main", "report"], default="main")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or get_settings().output_dir / "experiments" / "cross_participant_lvo_rf_20261011"
    out.mkdir(parents=True, exist_ok=True)
    {"main": phase_main, "report": phase_report}[args.phase](out)


if __name__ == "__main__":
    main()
