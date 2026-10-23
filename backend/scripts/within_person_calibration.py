"""Supervised within-person calibration on held-out videos (Protocol v1.1, exploratory, 12 participants).

Design: docs/EXPLORATORY_WITHIN_PERSON_CALIBRATION.md section 2 (declared before running).
Per participant, seed (42-46) and label-free video fold (identical to the cross-person LVO folds): the registered
Random Forest is trained on that participant's 32 training videos and tested on the 8 held-out videos.
Tasks: P = own rating; C = consensus of the other 11 participants. Also a nested calibration learning curve
(8/16/24/32 training videos, personal task) and Protocol S1 (borderline 4.5-5.5 removed from training and scoring).
Permutation phase: seed-42 folds fixed, 500 permutations of the participant's own video labels.

Usage (from backend/):
  python -m scripts.within_person_calibration --phase main          # 1,800 fits
  python -m scripts.within_person_calibration --phase permutation   # 30,000 fits; resumable
  python -m scripts.within_person_calibration --phase report
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

PARTICIPANTS, SEEDS, LABEL, HIGH, LOW = lvo.PARTICIPANTS, lvo.SEEDS, lvo.LABEL, lvo.HIGH, lvo.LOW
CLASSES = [HIGH, LOW]  # sorted, as the Random Forest orders them
LC_SIZES = (8, 16, 24, 32)
BORDERLINE = (4.5, 5.5)
PERM_SEED, N_PERM, PERM_SAVE_EVERY = 42, 500, 25
DOCS_RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results" / "exploratory"


def lc_subsets(seed: int, k: int, train_videos: list[int]) -> dict[int, list[int]]:
    """Nested, label-free calibration subsets of the training videos (shared by all participants)."""
    order = np.random.default_rng([seed, k]).permutation(sorted(train_videos))
    return {n: sorted(order[:n].tolist()) for n in LC_SIZES}


def permuted_labels(video_labels: np.ndarray, p_index: int, i: int) -> np.ndarray:
    return video_labels[np.random.default_rng([PERM_SEED, p_index, i]).permutation(len(video_labels))]


def is_borderline(valence: np.ndarray) -> np.ndarray:
    return (valence >= BORDERLINE[0]) & (valence <= BORDERLINE[1])


def fit_predict(X: np.ndarray, vidx: np.ndarray, video_labels: np.ndarray, train_videos: list[int],
                test_videos: list[int], seed: int) -> np.ndarray | None:
    """Trial probabilities (len(test_videos), 2) in CLASSES order, or None if training labels are single-class.
    X/vidx are one participant's rows; segment labels follow their video's label."""
    train, test = np.isin(vidx, train_videos), np.isin(vidx, test_videos)
    assert not set(vidx[train].tolist()) & set(vidx[test].tolist()), "a video is in both train and test"
    assert set(vidx[test].tolist()) == set(test_videos), "every test video must have segments"
    labels = video_labels[vidx[train]]
    if len(set(labels.tolist())) < 2:
        return None
    model = candidate_models(seed)["random_forest"]().fit(X[train], labels)
    assert [str(c) for c in model.classes_] == CLASSES
    seg_p = model.predict_proba(X[test])
    return np.array([seg_p[vidx[test] == v].mean(axis=0) for v in test_videos])


def participant_arrays(d: dict[str, Any], p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = d["pidx"] == p
    X, vidx = d["X"][rows], d["vidx"][rows]
    high = d["valence"] > 5.0
    personal = np.where(high[p], HIGH, LOW)
    consensus = lvo.test_consensus(high, p)  # majority of the other 11; raises on a tie
    assert np.array_equal(personal[vidx], d["y"][rows]), "personal labels must equal the trusted ratings rule"
    return X, vidx, personal, consensus, d["valence"][p]


def _metrics(target: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    return wp.metrics(target, proba, CLASSES)


def run_participant(d: dict[str, Any], p: int, ref_folds: dict[str, list[list[int]]]) -> dict[str, Any]:
    X, vidx, personal, consensus, valence = participant_arrays(d, p)
    border = is_borderline(valence)
    out: dict[str, Any] = {"targets": {"personal": personal.tolist(), "consensus": consensus.tolist()},
                           "valence": valence.tolist(), "borderline_videos": np.flatnonzero(border).tolist(), "seeds": {}}
    for seed in SEEDS:
        folds = lvo.video_folds(seed)
        assert folds == ref_folds[str(seed)], "folds must equal the cross-person experiments' folds"
        proba = {t: np.full((40, 2), np.nan) for t in ("personal", "consensus")}
        lc = {n: np.full((40, 2), np.nan) for n in LC_SIZES[:-1]}
        lc_invalid: dict[int, list[int]] = {n: [] for n in LC_SIZES[:-1]}
        s1, s1_invalid = np.full((40, 2), np.nan), []
        subsets_used = []
        for k, fold in enumerate(folds):
            train_videos = sorted(set(range(40)) - set(fold))
            for task, labels in (("personal", personal), ("consensus", consensus)):
                r = fit_predict(X, vidx, labels, train_videos, fold, seed)
                assert r is not None, f"{task}: 32 training videos must contain both classes"
                assert np.isnan(proba[task][fold]).all(), "a test trial was predicted twice"
                proba[task][fold] = r
            subsets = lc_subsets(seed, k, train_videos)
            subsets_used.append({str(n): v for n, v in subsets.items()})
            assert subsets[32] == train_videos
            for n in LC_SIZES[:-1]:
                r = fit_predict(X, vidx, personal, subsets[n], fold, seed)
                if r is None:
                    lc_invalid[n].append(k)
                else:
                    lc[n][fold] = r
            s1_train = [v for v in train_videos if not border[v]]
            s1_test = [v for v in fold if not border[v]]
            if s1_test:
                r = fit_predict(X, vidx, personal, s1_train, s1_test, seed)
                if r is None:
                    s1_invalid.append(k)
                else:
                    s1[s1_test] = r
        seed_out: dict[str, Any] = {"video_folds": folds, "lc_subsets": subsets_used, "tasks": {}}
        for task, target in (("personal", personal), ("consensus", consensus)):
            assert not np.isnan(proba[task]).any(), "every trial must be predicted exactly once"
            seed_out["tasks"][task] = {"proba": proba[task].round(8).tolist(), **_metrics(target, proba[task])}
        seed_out["learning_curve"] = {str(n): ({"valid": False, "single_class_folds": lc_invalid[n]} if lc_invalid[n] else
                                               {"valid": True, "proba": lc[n].round(8).tolist(), **_metrics(personal, lc[n])})
                                      for n in LC_SIZES[:-1]}
        keep = ~border
        seed_out["s1_borderline_excluded"] = ({"valid": False, "single_class_folds": s1_invalid} if s1_invalid else
                                              {"valid": True, "n_trials": int(keep.sum()), "proba": s1[keep].round(8).tolist(),
                                               **_metrics(personal[keep], s1[keep])})
        out["seeds"][str(seed)] = seed_out
    return out


def phase_main(out: Path) -> None:
    d = lvo.load_arrays()
    ref = json.loads((DOCS_RESULTS / "cross_participant_lvo_rf" / "main.json").read_text(encoding="utf-8"))
    assert ref["fingerprint"] == d["fingerprint"], "feature cache differs from the cross-person experiments"
    path = out / "main.json"
    res: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "label": LABEL, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
        "fingerprint": d["fingerprint"], "seeds": list(SEEDS), "video_folds": ref["video_folds"],
        "design": "docs/EXPLORATORY_WITHIN_PERSON_CALIBRATION.md section 2", "participants": {}}
    for p, pid in enumerate(PARTICIPANTS):
        if pid in res["participants"]:
            print(f"{pid}: already done", flush=True)
            continue
        r = run_participant(d, p, ref["video_folds"])
        res["participants"][pid] = r
        path.write_text(json.dumps(res), encoding="utf-8")
        print(f"{pid}: main done", flush=True)  # outcome metrics are inspected only in the report phase


def phase_permutation(out: Path) -> None:
    d = lvo.load_arrays()
    path = out / "permutation.json"
    res: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "label": LABEL, "created_at": datetime.now(UTC).isoformat(), "seed": PERM_SEED, "n_permutations": N_PERM,
        "statistic": "trial balanced accuracy, personal task, seed-42 folds", "participants": {}}
    folds = lvo.video_folds(PERM_SEED)
    for p, pid in enumerate(PARTICIPANTS):
        X, vidx, personal, _, _ = participant_arrays(d, p)
        null: list[float] = res["participants"].setdefault(pid, {"null": []})["null"]
        if len(null) >= N_PERM:
            print(f"{pid}: already done", flush=True)
            continue
        for i in range(len(null), N_PERM):
            labels = permuted_labels(personal, p, i)
            proba = np.full((40, 2), np.nan)
            for fold in folds:
                proba[fold] = fit_predict(X, vidx, labels, sorted(set(range(40)) - set(fold)), fold, PERM_SEED)
            assert not np.isnan(proba).any()
            null.append(float(_metrics(labels, proba)["balanced_accuracy"]))
            if len(null) % PERM_SAVE_EVERY == 0 or len(null) == N_PERM:
                path.write_text(json.dumps(res), encoding="utf-8")
                print(f"{pid}: {len(null)}/{N_PERM}", flush=True)


def _bh(pvals: list[float]) -> list[float]:
    return wp._bh(pvals)


def _seed_mean(src: dict[str, Any], getter: Any) -> float:
    return float(np.mean([getter(src["seeds"][str(s)]) for s in SEEDS]))


def phase_report(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((out / "main.json").read_text(encoding="utf-8"))
    perm = json.loads((out / "permutation.json").read_text(encoding="utf-8")) if (out / "permutation.json").exists() else None
    cross = {c: json.loads((DOCS_RESULTS / f / "main.json").read_text(encoding="utf-8"))["participants"]
             for c, f in (("A", "cross_participant_lvo_rf"), ("B", "participant_normalization_rf"))}
    rows: list[dict[str, Any]] = []
    for p, pid in enumerate(PARTICIPANTS):
        r = res["participants"][pid]
        personal, consensus = np.array(r["targets"]["personal"]), np.array(r["targets"]["consensus"])
        for c in "AB":
            assert cross[c][pid]["targets"]["personal"] == r["targets"]["personal"], "cross-person targets differ"
        row: dict[str, Any] = {"participant": pid, "true_high_share": float(np.mean(personal == HIGH)),
                               "agreement": float(np.mean(personal == consensus)), "n_borderline": len(r["borderline_videos"])}
        for task in ("personal", "consensus"):
            per = [r["seeds"][str(s)]["tasks"][task] for s in SEEDS]
            row[task] = {k: float(np.mean([m[k] for m in per])) for k in ("balanced_accuracy", "f1_macro", "roc_auc", "predicted_high_share")}
            row[task]["ba_sd"] = float(np.std([m["balanced_accuracy"] for m in per], ddof=1))
            row[task]["single_label_seeds"] = int(sum(not 0.05 <= m["predicted_high_share"] <= 0.95 for m in per))
        for c in "AB":
            row[f"cross_{c}"] = _seed_mean(cross[c][pid], lambda s: s["tasks"]["personal"]["balanced_accuracy"])
        # disagreement trials: seed-averaged P (and C) predictions
        dis = personal != consensus
        for task in ("personal", "consensus"):
            p_high = np.mean([np.array(r["seeds"][str(s)]["tasks"][task]["proba"])[:, 0] for s in SEEDS], axis=0)
            pred = np.where(p_high >= 1 - p_high, HIGH, LOW)
            row[f"disagreement_{task}_model"] = {"n": int(dis.sum()), "follows_own": int((pred[dis] == personal[dis]).sum()),
                                                 "follows_consensus": int((pred[dis] == consensus[dis]).sum())}
        lc = {}
        for n in LC_SIZES:
            if n == 32:
                lc["32"] = {"valid_seeds": 5, "balanced_accuracy": row["personal"]["balanced_accuracy"],
                            "roc_auc": row["personal"]["roc_auc"], "f1_macro": row["personal"]["f1_macro"]}
                continue
            valid = [r["seeds"][str(s)]["learning_curve"][str(n)] for s in SEEDS if r["seeds"][str(s)]["learning_curve"][str(n)]["valid"]]
            lc[str(n)] = {"valid_seeds": len(valid), **({k: float(np.mean([m[k] for m in valid if m[k] is not None]))
                                                         for k in ("balanced_accuracy", "roc_auc", "f1_macro")} if valid else {})}
        row["learning_curve"] = lc
        s1 = [r["seeds"][str(s)]["s1_borderline_excluded"] for s in SEEDS]
        s1v = [m for m in s1 if m["valid"]]
        row["s1"] = {"valid_seeds": len(s1v), "n_trials": s1v[0]["n_trials"] if s1v else None,
                     **({k: float(np.mean([m[k] for m in s1v if m[k] is not None])) for k in ("balanced_accuracy", "roc_auc", "f1_macro")} if s1v else {})}
        if perm and len(perm["participants"].get(pid, {}).get("null", [])) == N_PERM:
            null = np.array(perm["participants"][pid]["null"])
            obs = r["seeds"][str(PERM_SEED)]["tasks"]["personal"]["balanced_accuracy"]
            row["permutation"] = {"observed_seed42": obs, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
                                  "p": float((1 + np.sum(null >= obs - 1e-12)) / (N_PERM + 1))}
        rows.append(row)

    def stats(x: np.ndarray) -> dict[str, Any]:
        return mean_ci(x.tolist()) | {"bootstrap_ci": _participant_bootstrap(x)}

    P = np.array([r["personal"]["balanced_accuracy"] for r in rows]); C = np.array([r["consensus"]["balanced_accuracy"] for r in rows])
    A = np.array([r["cross_A"] for r in rows]); B = np.array([r["cross_B"] for r in rows])
    share_own = np.array([r["disagreement_personal_model"]["follows_own"] / r["disagreement_personal_model"]["n"] for r in rows])
    group: dict[str, Any] = {
        "personal": {m: stats(np.array([r["personal"][m] for r in rows])) for m in ("balanced_accuracy", "f1_macro", "roc_auc")},
        "consensus": {m: stats(np.array([r["consensus"][m] for r in rows])) for m in ("balanced_accuracy", "f1_macro", "roc_auc")},
        "P_minus_crossA": stats(P - A) | {"n_P_better": int((P > A).sum())},
        "P_minus_crossB": stats(P - B) | {"n_P_better": int((P > B).sum())},
        "P_minus_C": stats(P - C) | {"n_P_better": int((P > C).sum())},
        "disagreement_share_own": stats(share_own) | {
            "pooled_follows_own": int(sum(r["disagreement_personal_model"]["follows_own"] for r in rows)),
            "pooled_n": int(sum(r["disagreement_personal_model"]["n"] for r in rows))},
    }
    secondary = {"P_minus_crossA": _sign_flip_exact(P - A), "P_minus_C": _sign_flip_exact(P - C),
                 "disagreement_share_own_vs_0.5": _sign_flip_exact(share_own - 0.5)}
    group["tests"] = {
        "method": "exact sign-flip over 12 participants (4096 patterns), two-sided",
        "primary_P_vs_0.5": {"p": _sign_flip_exact(P - 0.5), "wilcoxon": wilcoxon(P, 0.5)},
        "secondary": {"p_raw": secondary, "p_holm": lvo._holm(secondary),
                      "wilcoxon": {"P_minus_crossA": wilcoxon(P, A), "P_minus_C": wilcoxon(P, C),
                                   "disagreement_share_own_vs_0.5": wilcoxon(share_own, 0.5)}}}
    group["learning_curve"] = {}
    for n in LC_SIZES:
        vals = [r["learning_curve"][str(n)] for r in rows]
        ok = [v for v in vals if v["valid_seeds"] == 5]
        group["learning_curve"][str(n)] = {
            "participants_all_seeds_valid": len(ok), "invalid_participant_seeds": int(sum(5 - v["valid_seeds"] for v in vals)),
            **({m: mean_ci([v[m] for v in ok]) for m in ("balanced_accuracy", "roc_auc")} if len(ok) > 1 else {})}
    s1ok = [r["s1"] for r in rows if r["s1"]["valid_seeds"] == 5]
    group["s1_borderline_excluded"] = {"participants_all_seeds_valid": len(s1ok),
                                       **({m: mean_ci([v[m] for v in s1ok]) for m in ("balanced_accuracy", "roc_auc")} if len(s1ok) > 1 else {})}
    group["collapse"] = {t: {"single_label_seeds_total": int(sum(r[t]["single_label_seeds"] for r in rows)),
                             "mean_abs_predicted_minus_true_share": float(np.mean([abs(r[t]["predicted_high_share"]
                                 - (r["true_high_share"] if t == "personal" else float(np.mean(np.array(res["participants"][r["participant"]]["targets"]["consensus"]) == HIGH))))
                                 for r in rows]))} for t in ("personal", "consensus")}
    if all("permutation" in r for r in rows):
        q = _bh([r["permutation"]["p"] for r in rows])
        for r, qq in zip(rows, q):
            r["permutation"]["q_bh"] = qq
        group["permutation"] = {"n_permutations": N_PERM, "n_significant_bh_0.05": int(sum(qq < 0.05 for qq in q))}
    summary = {"label": LABEL, "git_commit": res["git_commit"], "fingerprint": res["fingerprint"], "seeds": res["seeds"],
               "participants": rows, "group": group}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(rows)); w = 0.2
    for off, vals, name in ((-1.5 * w, A, "cross-person A (zero-shot)"), (-0.5 * w, B, "cross-person B (unsupervised calibration)"),
                            (0.5 * w, P, "within-person P (supervised calibration, own rating)"), (1.5 * w, C, "within-person C (consensus control)")):
        axes[0].bar(x + off, vals, w, label=name)
    axes[0].axhline(0.5, ls="--", color="0.5", lw=0.8); axes[0].set_ylim(0, 1); axes[0].set_xticks(x, [r["participant"] for r in rows], fontsize=7)
    axes[0].set_ylabel("trial balanced accuracy (mean of 5 seeds)"); axes[0].legend(fontsize=6, loc="lower right")
    lc_means = [group["learning_curve"][str(n)].get("balanced_accuracy", {}) for n in LC_SIZES]
    axes[1].errorbar(LC_SIZES, [m.get("mean", np.nan) for m in lc_means],
                     yerr=[[m.get("mean", np.nan) - m.get("ci_low", np.nan) for m in lc_means], [m.get("ci_high", np.nan) - m.get("mean", np.nan) for m in lc_means]],
                     marker="o", capsize=3)
    axes[1].axhline(0.5, ls="--", color="0.5", lw=0.8); axes[1].set_ylim(0.3, 0.8); axes[1].set_xticks(LC_SIZES)
    axes[1].set_xlabel("labelled calibration videos"); axes[1].set_title("Calibration learning curve (P, 95% t-CI)", fontsize=8)
    fig.suptitle(f"Within-person calibration on held-out videos. {LABEL}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "within_person_calibration.png", dpi=130); plt.close(fig)
    print(json.dumps(group["tests"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["main", "permutation", "report"], default="main")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or get_settings().output_dir / "experiments" / "within_person_calibration_rf_20261012"
    out.mkdir(parents=True, exist_ok=True)
    {"main": phase_main, "permutation": phase_permutation, "report": phase_report}[args.phase](out)


if __name__ == "__main__":
    main()
