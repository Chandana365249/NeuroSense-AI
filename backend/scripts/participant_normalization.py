"""Participant-specific feature normalization, cross-participant leave-video-out (Protocol v1.1, exploratory).

Design: docs/EXPLORATORY_PARTICIPANT_NORMALIZATION.md section 2 (declared before running).
Condition A = the verified cross-participant LVO results (reused, no refit). Condition B = the same folds, seeds,
training rows and Random Forest, but every participant's features are z-scored with statistics from that
participant's unlabeled segments on the videos NOT in the test fold (training participants and the held-out
participant alike). No labels enter the statistics. Condition B is unsupervised calibration, not zero-shot.

Usage (from backend/):
  python -m scripts.participant_normalization --phase main     # ~45 min plugged in; resumable
  python -m scripts.participant_normalization --phase report
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.ml.provenance import git_commit
from app.ml.models import candidate_models
from app.ml.stats import mean_ci, wilcoxon
from scripts import cross_participant_lvo as lvo
from scripts import within_person_diagnostic as wp
from scripts.stimulus_confound_control import _participant_bootstrap, _sign_flip_exact

PARTICIPANTS, SEEDS, LABEL, TASKS, HIGH = lvo.PARTICIPANTS, lvo.SEEDS, lvo.LABEL, lvo.TASKS, lvo.HIGH
CONDITIONS = ("A", "B")


def lvo_results_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "results" / "exploratory" / "cross_participant_lvo_rf" / "main.json"


def normalize_by_participant(X: np.ndarray, pidx: np.ndarray, stats_rows: np.ndarray) -> np.ndarray:
    """z-score each participant's rows with mean/SD (ddof 0) over that participant's rows in stats_rows only.
    Uses no labels. Rows of a participant with no stats rows are left as NaN (callers must not use them)."""
    out = np.full(X.shape, np.nan)
    for i in np.unique(pidx):
        own = pidx == i
        ref = X[own & stats_rows]
        if len(ref) == 0:
            continue
        mu, sd = ref.mean(axis=0), ref.std(axis=0)
        sd[sd == 0] = 1.0
        out[own] = (X[own] - mu) / sd
    return out


def run_participant(d: dict[str, Any], p: int, lvo_ref: dict[str, Any]) -> dict[str, Any]:
    X, y, pidx, vidx = d["X"], d["y"], d["pidx"], d["vidx"]
    high = d["valence"] > 5.0
    cons_train, tied = lvo.training_consensus(high, p)
    cons_test = lvo.test_consensus(high, p)
    personal_test = np.where(high[p], HIGH, lvo.LOW)
    row_cons = cons_train[pidx, vidx]
    assert personal_test.tolist() == lvo_ref["targets"]["personal"] and cons_test.tolist() == lvo_ref["targets"]["consensus"]
    out: dict[str, Any] = {"targets": {"personal": personal_test.tolist(), "consensus": cons_test.tolist()},
                           "tied_pairs_excluded": int(tied[np.arange(len(PARTICIPANTS)) != p].sum()), "seeds": {}}
    for seed in SEEDS:
        folds = lvo.video_folds(seed)
        ref_seed = lvo_ref["seeds"][str(seed)]
        assert folds == ref_seed["video_folds"], "video folds must equal the LVO experiment's"
        seed_out: dict[str, Any] = {"video_folds": folds, "fits": [], "tasks": {}}
        proba = {t: np.full((lvo.N_VIDEOS, 2), np.nan) for t in TASKS}
        classes: dict[str, list[str]] = {}
        for k, fold in enumerate(folds):
            in_fold = np.isin(vidx, fold)
            train = (pidx != p) & ~in_fold & ~tied[pidx, vidx]
            test = (pidx == p) & in_fold
            assert not np.any(pidx[train] == p) and not np.any(np.isin(vidx[train], fold)), "held-out data in training"
            assert set(vidx[test].tolist()) == set(fold), "every held-out video must have test segments"
            assert int(train.sum()) == ref_seed["fits"][k]["train_rows"], "training rows must equal the LVO experiment's"
            stats_rows = ~in_fold  # every participant, non-test videos only; label-free
            assert not np.any(np.isin(vidx[stats_rows], fold))
            Xn = normalize_by_participant(X, pidx, stats_rows)
            assert np.isfinite(Xn[train]).all() and np.isfinite(Xn[test]).all()
            seed_out["fits"].append({"fold": k, "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                                     "stats_rows_held_out_participant": int((stats_rows & (pidx == p)).sum())})
            for task, labels in (("personal", y), ("consensus", row_cons)):
                model = candidate_models(seed)["random_forest"]().fit(Xn[train], labels[train])
                classes[task] = [str(c) for c in model.classes_]
                seg_p = model.predict_proba(Xn[test])
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
    d = lvo.load_arrays()
    ref = json.loads(lvo_results_path().read_text(encoding="utf-8"))
    assert ref["fingerprint"] == d["fingerprint"], "feature cache differs from the LVO experiment"
    path = out / "main.json"
    res: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "label": LABEL, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
        "fingerprint": d["fingerprint"], "seeds": list(SEEDS),
        "video_folds": {str(s): lvo.video_folds(s) for s in SEEDS},
        "condition_A_source": "docs/results/exploratory/cross_participant_lvo_rf/main.json",
        "design": "docs/EXPLORATORY_PARTICIPANT_NORMALIZATION.md section 2", "participants": {}}
    for p, pid in enumerate(PARTICIPANTS):
        if pid in res["participants"]:
            print(f"{pid}: already done", flush=True)
            continue
        r = run_participant(d, p, ref["participants"][pid])
        res["participants"][pid] = r
        path.write_text(json.dumps(res), encoding="utf-8")
        ba = {(c, t): np.mean([src["seeds"][str(s)]["tasks"][t]["balanced_accuracy"] for s in SEEDS])
              for c, src in (("A", ref["participants"][pid]), ("B", r)) for t in TASKS}
        print(f"{pid}: personal BA A {ba['A', 'personal']:.3f} -> B {ba['B', 'personal']:.3f} | "
              f"consensus BA A {ba['A', 'consensus']:.3f} -> B {ba['B', 'consensus']:.3f}", flush=True)


def _participant_row(src: dict[str, Any], task: str) -> dict[str, Any]:
    per_seed = [src["seeds"][str(s)]["tasks"][task] for s in SEEDS]
    row = {k: float(np.mean([m[k] for m in per_seed])) for k in ("balanced_accuracy", "f1_macro", "roc_auc", "predicted_high_share")}
    row["ba_sd"] = float(np.std([m["balanced_accuracy"] for m in per_seed], ddof=1))
    row["true_high_share"] = float(np.mean(np.array(src["targets"][task]) == HIGH))
    row["single_label_seeds"] = int(sum(m["predicted_high_share"] < 0.05 or m["predicted_high_share"] > 0.95 for m in per_seed))
    return row


def phase_report(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((out / "main.json").read_text(encoding="utf-8"))
    ref = json.loads(lvo_results_path().read_text(encoding="utf-8"))
    rows = []
    for pid in PARTICIPANTS:
        src = {"A": ref["participants"][pid], "B": res["participants"][pid]}
        assert src["A"]["targets"] == src["B"]["targets"]
        rows.append({"participant": pid, **{f"{c}_{t}": _participant_row(src[c], t) for c in CONDITIONS for t in TASKS}})

    group: dict[str, Any] = {}
    for task in TASKS:
        g: dict[str, Any] = {}
        for metric in ("balanced_accuracy", "f1_macro", "roc_auc"):
            a = np.array([r[f"A_{task}"][metric] for r in rows]); b = np.array([r[f"B_{task}"][metric] for r in rows])
            g[metric] = {"A": mean_ci(a.tolist()) | {"bootstrap_ci": _participant_bootstrap(a)},
                         "B": mean_ci(b.tolist()) | {"bootstrap_ci": _participant_bootstrap(b)},
                         "B_minus_A": mean_ci((b - a).tolist()) | {"bootstrap_ci": _participant_bootstrap(b - a)},
                         "n_B_better": int((b > a).sum()), "n_B_worse": int((b < a).sum())}
        g["collapse"] = {c: {"participants_mean_share_outside_0.05_0.95": [r["participant"] for r in rows
                                                                          if not 0.05 <= r[f"{c}_{task}"]["predicted_high_share"] <= 0.95],
                             "participants_with_any_single_label_seed": int(sum(r[f"{c}_{task}"]["single_label_seeds"] > 0 for r in rows)),
                             "single_label_seeds_total": int(sum(r[f"{c}_{task}"]["single_label_seeds"] for r in rows)),
                             "mean_abs_predicted_minus_true_share": float(np.mean([abs(r[f"{c}_{task}"]["predicted_high_share"]
                                                                                       - r[f"{c}_{task}"]["true_high_share"]) for r in rows]))}
                         for c in CONDITIONS}
        group[task] = g

    ba = {(c, t): np.array([r[f"{c}_{t}"]["balanced_accuracy"] for r in rows]) for c in CONDITIONS for t in TASKS}
    primary = _sign_flip_exact(ba["B", "personal"] - ba["A", "personal"])
    secondary = {"personal_B_vs_0.5": _sign_flip_exact(ba["B", "personal"] - 0.5),
                 "consensus_B_minus_A": _sign_flip_exact(ba["B", "consensus"] - ba["A", "consensus"]),
                 "consensus_B_vs_0.5": _sign_flip_exact(ba["B", "consensus"] - 0.5)}
    group["tests_balanced_accuracy"] = {
        "method": "exact sign-flip over participants (4096 patterns), two-sided",
        "primary_personal_B_minus_A": {"p": primary, "wilcoxon": wilcoxon(ba["B", "personal"], ba["A", "personal"])},
        "secondary": {"p_raw": secondary, "p_holm": lvo._holm(secondary),
                      "wilcoxon": {"personal_B_vs_0.5": wilcoxon(ba["B", "personal"], 0.5),
                                   "consensus_B_minus_A": wilcoxon(ba["B", "consensus"], ba["A", "consensus"]),
                                   "consensus_B_vs_0.5": wilcoxon(ba["B", "consensus"], 0.5)}}}
    summary = {"label": LABEL, "git_commit": res["git_commit"], "fingerprint": res["fingerprint"], "seeds": res["seeds"],
               "condition_A_source": res["condition_A_source"], "participants": rows, "group": group}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(rows))
    for ax, task, title in ((axes[0], "personal", "Primary: own rating"), (axes[1], "consensus", "Secondary control: video consensus")):
        for off, c, name in ((-0.2, "A", "A: no participant normalization"), (0.2, "B", "B: participant-standardized (unsupervised calibration)")):
            ax.bar(x + off, [r[f"{c}_{task}"]["balanced_accuracy"] for r in rows], 0.4,
                   yerr=[r[f"{c}_{task}"]["ba_sd"] for r in rows], capsize=2, label=name)
        ax.axhline(0.5, ls="--", color="0.5", lw=0.8); ax.set_ylim(0, 1); ax.set_title(title, fontsize=9)
        ax.set_xticks(x, [r["participant"] for r in rows], fontsize=7)
    axes[0].set_ylabel("trial balanced accuracy (mean ± SD, 5 seeds)"); axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Participant-specific normalization, held-out person AND held-out videos. {LABEL}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "participant_normalization.png", dpi=130); plt.close(fig)
    print(json.dumps(group["tests_balanced_accuracy"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["main", "report"], default="main")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or get_settings().output_dir / "experiments" / "participant_normalization_rf_20261011"
    out.mkdir(parents=True, exist_ok=True)
    {"main": phase_main, "report": phase_report}[args.phase](out)


if __name__ == "__main__":
    main()
