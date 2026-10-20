"""Within-person valence diagnostic (Protocol v1.1, section 9 D2). Exploratory: 12 participants.

Question: does the 440-feature pipeline carry trial-specific valence information when a model is trained
and tested inside ONE participant? This is a different question from cross-participant generalization
and is never reported as such.

Design (declared before running):
- Only the 12 trusted DEAP participants in the configured dataset (s01..s25). No other data source.
- Per participant: stratified 5-fold CV over the 40 TRIALS; all segments of a trial share its fold.
- Seeds 42-46: each seed sets the fold assignment and the Random Forest random_state.
- Model: the registered Random Forest configuration (candidate_models(seed)["random_forest"]), unchanged;
  its scaler is fitted on training trials only. No tuning.
- Trial prediction: mean of segment probabilities, then argmax. Metrics on 40 out-of-fold trials.
- Permutation (phase "permutation"): seed 42, trial labels permuted within the participant, the full CV
  re-run for every permutation (labels and fold stratification both follow the permuted labels).

Usage (from backend/):
  python -m scripts.within_person_diagnostic --phase main         # ~4 min
  python -m scripts.within_person_diagnostic --phase permutation  # ~2.7 h plugged in (250 permutations)
  python -m scripts.within_person_diagnostic --phase report       # summary.json + figure
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from app.core.config import get_settings
from app.ml.deep.robust_evaluation import git_commit
from app.ml.factory import make_dataset, make_feature_extractor, make_label_config, make_preprocessor
from app.ml.feature_table import FeatureTable, build_feature_table
from app.ml.models import candidate_models
from app.ml.stats import binary_metric_fns, mean_ci, wilcoxon

PARTICIPANTS = ("s01", "s02", "s03", "s05", "s07", "s08", "s09", "s11", "s15", "s16", "s22", "s25")
SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 5
PERM_SEED, N_PERM = 42, 250
LABEL = "exploratory (12 participants, Protocol v1.1)"


def trial_index(table: FeatureTable, pid: str) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of a participant and, per row, the position of its trial in the sorted trial list."""
    rows = np.flatnonzero(table.participant == pid)
    trials = np.unique(table.trial[rows])
    return rows, np.searchsorted(trials, table.trial[rows])


def cross_validate(X: np.ndarray, seg_trial: np.ndarray, trial_labels: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold trial probabilities (n_trials, 2) for one participant; columns follow the returned classes."""
    n_trials = len(trial_labels)
    proba = np.full((n_trials, 2), np.nan)
    classes: list[str] | None = None
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(n_trials), trial_labels)
    for train_t, test_t in folds:
        train_rows, test_rows = np.isin(seg_trial, train_t), np.isin(seg_trial, test_t)
        assert not set(seg_trial[train_rows]) & set(seg_trial[test_rows]), "a trial is in both train and test"
        assert set(seg_trial[test_rows]) == set(test_t.tolist())
        model = candidate_models(seed)["random_forest"]().fit(X[train_rows], trial_labels[seg_trial[train_rows]])
        classes = [str(c) for c in model.classes_]
        seg_p = model.predict_proba(X[test_rows])
        for t in test_t:  # mean segment probability within the trial
            proba[t] = seg_p[seg_trial[test_rows] == t].mean(axis=0)
    assert not np.isnan(proba).any(), "every trial must be tested exactly once"
    return proba, classes  # type: ignore[return-value]


def load() -> tuple[FeatureTable, np.ndarray]:
    s = get_settings()
    table = build_feature_table(make_dataset(s), make_preprocessor(s), make_feature_extractor(s), s.cache_dir)
    if sorted(set(table.participant.tolist())) != list(PARTICIPANTS):
        raise SystemExit(f"Unexpected participant set: {sorted(set(table.participant.tolist()))}")
    return table, table.labels(make_label_config(s))


def participant_data(table: FeatureTable, y: np.ndarray, pid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, seg_trial = trial_index(table, pid)
    trial_labels = np.empty(seg_trial.max() + 1, dtype=object)
    for t in range(len(trial_labels)):
        labels_t = set(y[rows][seg_trial == t].tolist())
        assert len(labels_t) == 1, "segments of one trial must share a label"
        trial_labels[t] = labels_t.pop()
    valence = np.array([table.ratings[rows][seg_trial == t][0, 0] for t in range(len(trial_labels))])
    return table.X[rows], seg_trial, trial_labels.astype(str), valence


def metrics(trial_labels: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    fns = binary_metric_fns(classes)
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    return {k: fns[k](trial_labels, proba) for k in ("balanced_accuracy", "f1_macro", "roc_auc", "accuracy")} | {
        "predicted_high_share": float(np.mean(pred == "high_valence"))}


def phase_main(out: Path) -> None:
    table, y = load()
    results: dict[str, Any] = {"label": LABEL, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
                               "design": {"folds": f"stratified {N_FOLDS}-fold over trials, per participant",
                                          "seeds": list(SEEDS), "model": "registered random_forest configuration",
                                          "trial_rule": "mean segment probability, argmax", "label_rule": "valence > 5.0"},
                               "fingerprint": table.fingerprint, "participants": {}}
    for pid in PARTICIPANTS:
        X, seg_trial, trial_labels, valence = participant_data(table, y, pid)
        per_seed, probas = [], []
        for seed in SEEDS:
            proba, classes = cross_validate(X, seg_trial, trial_labels, seed)
            per_seed.append({"seed": seed, **metrics(trial_labels, proba, classes)})
            probas.append(proba)
        avg = np.mean(probas, axis=0)
        results["participants"][pid] = {
            "n_trials": int(len(trial_labels)), "n_segments": int(len(seg_trial)),
            "n_high": int((trial_labels == "high_valence").sum()), "n_low": int((trial_labels == "low_valence").sum()),
            "n_borderline_4.5_5.5": int(((valence >= 4.5) & (valence <= 5.5)).sum()),
            "per_seed": per_seed,
            "seed_mean": {k: float(np.mean([r[k] for r in per_seed if r[k] is not None])) for k in per_seed[0] if k != "seed"},
            "seed_sd": {k: float(np.std([r[k] for r in per_seed if r[k] is not None], ddof=1)) for k in per_seed[0] if k != "seed"},
            "seed_averaged_proba_metrics": metrics(trial_labels, avg, classes),
            "classes": classes,
        }
        r = results["participants"][pid]["seed_mean"]
        print(f"{pid}: BA {r['balanced_accuracy']:.3f}  F1 {r['f1_macro']:.3f}  AUC {r['roc_auc']:.3f}  "
              f"pred-high {r['predicted_high_share']:.2f}", flush=True)
    (out / "main.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def phase_permutation(out: Path) -> None:
    table, y = load()
    path = out / "permutation.json"
    done: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for pid in PARTICIPANTS:
        if pid in done:
            print(f"{pid}: already done", flush=True)
            continue
        X, seg_trial, trial_labels, _ = participant_data(table, y, pid)
        proba, classes = cross_validate(X, seg_trial, trial_labels, PERM_SEED)
        observed = metrics(trial_labels, proba, classes)
        p_rng = np.random.default_rng([PERM_SEED, PARTICIPANTS.index(pid)])  # per-participant stream: resumable
        null = []
        for _ in range(N_PERM):
            perm_labels = trial_labels[p_rng.permutation(len(trial_labels))]
            pp, cc = cross_validate(X, seg_trial, perm_labels, PERM_SEED)
            m = metrics(perm_labels, pp, cc)
            null.append({"balanced_accuracy": m["balanced_accuracy"], "roc_auc": m["roc_auc"]})
        done[pid] = {"observed": observed, "null": null}
        path.write_text(json.dumps(done), encoding="utf-8")
        p = (1 + sum(n["balanced_accuracy"] >= observed["balanced_accuracy"] - 1e-12 for n in null)) / (1 + N_PERM)
        print(f"{pid}: observed BA {observed['balanced_accuracy']:.3f}, permutation p {p:.4f}", flush=True)


def _bootstrap_ba(labels: np.ndarray, proba: np.ndarray, classes: list[str], n: int = 2000, seed: int = 0) -> list[float]:
    """95% percentile CI of trial balanced accuracy, resampling trials within each class (keeps both classes)."""
    rng, fn = np.random.default_rng(seed), binary_metric_fns(classes)["balanced_accuracy"]
    groups = [np.flatnonzero(labels == c) for c in classes]
    draws = [fn(labels[idx], proba[idx]) for idx in
             (np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(n))]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals); order = np.argsort(p); m = len(p)
    q = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(q, 1.0)
    return out.tolist()


def phase_report(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from app.ml.registry import ModelRegistry

    main = json.loads((out / "main.json").read_text(encoding="utf-8"))
    perm_path = out / "permutation.json"
    perm = json.loads(perm_path.read_text(encoding="utf-8")) if perm_path.exists() else {}
    table, y = load()
    rows, pvals = [], []
    for pid in PARTICIPANTS:
        r = main["participants"][pid]
        X, seg_trial, trial_labels, _ = participant_data(table, y, pid)
        # trial-bootstrap CI on the seed-averaged probabilities requires them: recompute (deterministic up to 1 ulp)
        probas = [cross_validate(X, seg_trial, trial_labels, sd)[0] for sd in SEEDS]
        ci = _bootstrap_ba(trial_labels, np.mean(probas, axis=0), r["classes"])
        p_val = None
        if pid in perm:
            obs, null = perm[pid]["observed"]["balanced_accuracy"], [n["balanced_accuracy"] for n in perm[pid]["null"]]
            p_val = (1 + sum(v >= obs - 1e-12 for v in null)) / (1 + len(null))
            pvals.append(p_val)
        rows.append({"participant": pid, **{k: r[k] for k in ("n_high", "n_low", "n_borderline_4.5_5.5", "n_segments")},
                     "ba_mean": r["seed_mean"]["balanced_accuracy"], "ba_sd": r["seed_sd"]["balanced_accuracy"],
                     "f1_mean": r["seed_mean"]["f1_macro"], "auc_mean": r["seed_mean"]["roc_auc"],
                     "pred_high_share": r["seed_mean"]["predicted_high_share"], "true_high_share": r["n_high"] / r["n_trials"],
                     "ba_ci_seed_avg": ci, "ba_seed_avg": r["seed_averaged_proba_metrics"]["balanced_accuracy"],
                     "perm_seed42_ba": perm[pid]["observed"]["balanced_accuracy"] if pid in perm else None, "perm_p": p_val})
    if len(pvals) == len(PARTICIPANTS):
        for row, q in zip(rows, _bh(pvals), strict=True):
            row["perm_q_bh"] = q
    ba = [r["ba_mean"] for r in rows]
    group: dict[str, Any] = {"ba": mean_ci(ba), "f1": mean_ci([r["f1_mean"] for r in rows]),
                             "auc": mean_ci([r["auc_mean"] for r in rows]), "ba_vs_0.5_wilcoxon": wilcoxon(ba, 0.5),
                             "n_participants_ba_above_0.5": int(sum(b > 0.5 for b in ba))}
    if len(perm) == len(PARTICIPANTS):
        obs_mean = np.mean([perm[p]["observed"]["balanced_accuracy"] for p in PARTICIPANTS])
        null_mean = np.mean([[n["balanced_accuracy"] for n in perm[p]["null"]] for p in PARTICIPANTS], axis=0)
        group["group_permutation"] = {"observed_mean_ba_seed42": float(obs_mean),
                                      "p": float((1 + (null_mean >= obs_mean - 1e-12).sum()) / (1 + len(null_mean))),
                                      "null_mean_ba_95th_percentile": float(np.quantile(null_mean, 0.95))}
        group["significant_after_bh_q_0.05"] = [r["participant"] for r in rows if r.get("perm_q_bh", 1) < 0.05]

    s = get_settings()
    reg = ModelRegistry(s.model_dir)
    rf = reg.metadata(reg.active_version())
    cnn_summary = s.output_dir.parent / "docs/results/review3/robust_cnn/summary.json"
    cross = {"random_forest_lopo": {"version": rf["version"], "trial_accuracy_mean": rf["cross_validation"]["folds"] and
                                    float(np.mean([f["trial_accuracy"] for f in rf["cross_validation"]["folds"]])),
                                    "segment_accuracy_mean": rf["cross_validation"]["segment_accuracy_mean"],
                                    "note": "9 non-test participants; trial balanced accuracy was not recorded"}}
    if cnn_summary.exists():
        cs = json.loads(cnn_summary.read_text(encoding="utf-8"))["lopo"]["train_stats"]["aggregate"]
        cross["cnn_lopo_train_stats"] = {"trial_balanced_accuracy_mean": cs["trial"]["balanced_accuracy"]["mean"],
                                         "segment_balanced_accuracy_mean": cs["segment"]["balanced_accuracy"]["mean"]}
    summary = {"label": LABEL, "design": main["design"], "fingerprint": main["fingerprint"], "git_commit": main["git_commit"],
               "permutation": {"seed": PERM_SEED, "n_perm": N_PERM, "complete": len(perm) == len(PARTICIPANTS),
                               "method": "trial labels permuted within participant; full 5-fold CV re-run per permutation"},
               "participants": rows, "group": group, "cross_participant_reference": cross}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(rows))
    ax.bar(x, ba, yerr=[r["ba_sd"] for r in rows], capsize=3, color="tab:blue", label="trial balanced accuracy (mean ± SD over 5 seeds)")
    ax.errorbar(x + 0.25, [r["ba_seed_avg"] for r in rows],
                yerr=np.abs(np.array([r["ba_ci_seed_avg"] for r in rows]).T - [r["ba_seed_avg"] for r in rows]),
                fmt="o", color="k", ms=3, capsize=2, label="seed-averaged prediction, 95% trial-bootstrap CI")
    if len(perm) == len(PARTICIPANTS):
        ax.scatter(x, [np.quantile([n["balanced_accuracy"] for n in perm[p]["null"]], 0.95) for p in PARTICIPANTS],
                   marker="_", s=300, color="tab:red", label="permutation null, 95th percentile (seed 42)")
    ax.axhline(0.5, ls="--", color="0.5", lw=0.8)
    ax.set_xticks(x, [r["participant"] for r in rows]); ax.set_ylim(0, 1); ax.set_ylabel("trial balanced accuracy")
    ax.set_title(f"Within-person valence diagnostic, Random Forest, 5-fold trial-wise CV. {LABEL}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(out / "within_person_ba.png", dpi=130); plt.close(fig)
    print(json.dumps(group, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["main", "permutation", "report"], required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or get_settings().output_dir / "experiments" / "within_person_rf_20261010"
    out.mkdir(parents=True, exist_ok=True)
    {"main": phase_main, "permutation": phase_permutation, "report": phase_report}[args.phase](out)


if __name__ == "__main__":
    main()
