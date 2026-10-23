"""Stimulus-confound control for the within-person valence diagnostic (Protocol v1.1, exploratory).

Question: does the 440-feature EEG pipeline predict a participant's OWN valence rating better than the
CONSENSUS valence of the same video among the other participants?

Design (declared before running):
- Only the 12 trusted DEAP participants; trial index t is the same video for every participant (verified
  from the trusted ratings themselves: see `alignment_check` in the output).
- Personal label: own rating, valence > 5.0.
- Consensus label for participant p, video t: majority of the OTHER 11 participants' (valence > 5.0) labels
  (11 voters, so no ties). The target participant's own rating is never used for their consensus label.
- Folds: per participant and seed, stratified 5-fold over trials on the joint (personal, consensus) label;
  saved to folds.json and used identically for both tasks. Seeds 42-46 set the folds and the RF random_state.
- Model: the registered Random Forest configuration, unchanged. Trial prediction = mean segment probability.
- Statistics: paired per-participant differences (t-CI, participant bootstrap, Wilcoxon, exact sign-flip
  test), personal/consensus agreement and kappa with trial-bootstrap CIs, and a disagreement-trial analysis.

Usage (from backend/):  python -m scripts.stimulus_confound_control [--phase main|report]
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from app.core.config import get_settings
from app.ml.provenance import git_commit
from app.ml.models import candidate_models
from app.ml.stats import mean_ci, wilcoxon
from scripts import within_person_diagnostic as wp

PARTICIPANTS, SEEDS, N_FOLDS, LABEL = wp.PARTICIPANTS, wp.SEEDS, wp.N_FOLDS, wp.LABEL
TASKS = ("personal", "consensus")


def consensus_labels(valence: np.ndarray) -> np.ndarray:
    """valence: (participants, videos). Row i = majority label of all OTHER participants for each video."""
    high = valence > 5.0
    n = len(valence)
    out = np.empty(high.shape, dtype=object)
    for i in range(n):
        votes = high[np.arange(n) != i].sum(axis=0)
        if (n - 1) % 2 == 0 and np.any(votes * 2 == n - 1):
            raise ValueError("tie in consensus vote")
        out[i] = np.where(votes * 2 > n - 1, "high_valence", "low_valence")
    return out.astype(str)


def alignment_check(valence: np.ndarray, n_null: int = 1000, seed: int = 0) -> dict[str, float]:
    def mean_pairwise(m: np.ndarray) -> float:
        return float(np.mean([spearmanr(m[i], m[j])[0] for i, j in itertools.combinations(range(len(m)), 2)]))

    obs = mean_pairwise(valence)
    rng = np.random.default_rng(seed)
    null = [mean_pairwise(np.array([rng.permutation(r) for r in valence])) for _ in range(n_null)]
    return {"mean_pairwise_spearman": obs, "null_mean": float(np.mean(null)), "null_max": float(np.max(null)),
            "p": (1 + sum(n >= obs for n in null)) / (1 + n_null)}


def make_folds(personal: np.ndarray, consensus: np.ndarray, seed: int) -> list[list[int]]:
    """Test-trial indices per fold, stratified on the joint label."""
    joint = np.char.add(np.char.add(personal.astype(str), "|"), consensus.astype(str))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return [sorted(test.tolist()) for _, test in skf.split(np.zeros(len(joint)), joint)]


def cv_with_folds(X: np.ndarray, seg_trial: np.ndarray, labels: np.ndarray, folds: list[list[int]],
                  seed: int) -> tuple[np.ndarray, list[str]]:
    n = len(labels)
    assert sorted(t for f in folds for t in f) == list(range(n)), "folds must partition the trials"
    proba = np.full((n, 2), np.nan)
    classes: list[str] | None = None
    for test_t in folds:
        test_rows = np.isin(seg_trial, test_t)
        train_rows = ~test_rows
        assert not set(seg_trial[train_rows]) & set(seg_trial[test_rows]), "a trial is in both train and test"
        if len(set(labels[seg_trial[train_rows]])) < 2:
            raise ValueError("a training fold has only one class")
        model = candidate_models(seed)["random_forest"]().fit(X[train_rows], labels[seg_trial[train_rows]])
        classes = [str(c) for c in model.classes_]
        seg_p = model.predict_proba(X[test_rows])
        for t in test_t:
            proba[t] = seg_p[seg_trial[test_rows] == t].mean(axis=0)
    assert not np.isnan(proba).any()
    return proba, classes  # type: ignore[return-value]


def _agreement(personal: np.ndarray, consensus: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        po = float(np.mean(a == b))
        pa, pb = np.mean(a == "high_valence"), np.mean(b == "high_valence")
        pe = pa * pb + (1 - pa) * (1 - pb)
        return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")

    po, kappa = stats(personal, consensus)
    rng = np.random.default_rng(seed)
    draws = [stats(personal[i], consensus[i]) for i in (rng.integers(0, len(personal), len(personal)) for _ in range(n_boot))]
    a, k = np.array([d[0] for d in draws]), np.array([d[1] for d in draws])
    return {"agreement": po, "agreement_ci": [float(np.quantile(a, .025)), float(np.quantile(a, .975))],
            "kappa": kappa, "kappa_ci": [float(np.nanquantile(k, .025)), float(np.nanquantile(k, .975))],
            "n_disagree": int((personal != consensus).sum())}


def phase_main(out: Path) -> None:
    table, y = wp.load()
    data = {pid: wp.participant_data(table, y, pid) for pid in PARTICIPANTS}
    for pid, (_, seg_trial, _, valence) in data.items():
        assert len(valence) == 40 and seg_trial.max() == 39, f"{pid}: expected 40 trials"
    valence = np.array([data[p][3] for p in PARTICIPANTS])
    personal_all = np.where(valence > 5.0, "high_valence", "low_valence")
    for i, pid in enumerate(PARTICIPANTS):
        assert np.array_equal(personal_all[i], data[pid][2]), "personal labels must match the pipeline labels"
    consensus_all = consensus_labels(valence)

    path = out / "main.json"
    res: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "label": LABEL, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
        "fingerprint": table.fingerprint, "alignment_check": alignment_check(valence), "participants": {}}
    folds_path = out / "folds.json"
    folds_all = json.loads(folds_path.read_text(encoding="utf-8")) if folds_path.exists() else {}
    for i, pid in enumerate(PARTICIPANTS):
        if pid in res["participants"]:
            print(f"{pid}: already done", flush=True)
            continue
        X, seg_trial, personal, _ = data[pid]
        consensus = consensus_all[i]
        folds_all[pid] = {str(s): make_folds(personal, consensus, s) for s in SEEDS}
        folds_path.write_text(json.dumps(folds_all), encoding="utf-8")
        entry: dict[str, Any] = {"labels": {"personal": personal.tolist(), "consensus": consensus.tolist()},
                                 "agreement": _agreement(personal, consensus), "tasks": {}}
        for task, labels in (("personal", personal), ("consensus", consensus)):
            per_seed, probas = [], []
            for s in SEEDS:
                proba, classes = cv_with_folds(X, seg_trial, labels, folds_all[pid][str(s)], s)
                per_seed.append({"seed": s, **wp.metrics(labels, proba, classes)})
                probas.append(proba)
            entry["tasks"][task] = {
                "classes": classes, "per_seed": per_seed,
                "seed_mean": {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0] if k != "seed"},
                "seed_sd": {k: float(np.std([r[k] for r in per_seed], ddof=1)) for k in per_seed[0] if k != "seed"},
                "seed_averaged_proba": np.mean(probas, axis=0).round(8).tolist()}
        res["participants"][pid] = entry
        path.write_text(json.dumps(res), encoding="utf-8")
        pm, cm = entry["tasks"]["personal"]["seed_mean"], entry["tasks"]["consensus"]["seed_mean"]
        print(f"{pid}: personal BA {pm['balanced_accuracy']:.3f} | consensus BA {cm['balanced_accuracy']:.3f} | "
              f"agreement {entry['agreement']['agreement']:.3f}", flush=True)


def _sign_flip_exact(diffs: np.ndarray) -> float:
    """Two-sided exact sign-flip p-value for the mean of paired differences."""
    obs = abs(diffs.mean())
    stats = [abs((diffs * np.array(signs)).mean()) for signs in itertools.product((1, -1), repeat=len(diffs))]
    return float(np.mean([s >= obs - 1e-12 for s in stats]))


def _participant_bootstrap(diffs: np.ndarray, n: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n)]
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def phase_report(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((out / "main.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for pid in PARTICIPANTS:
        e = res["participants"][pid]
        personal, consensus = np.array(e["labels"]["personal"]), np.array(e["labels"]["consensus"])
        dis = personal != consensus
        follow = {}
        for task in TASKS:
            t = e["tasks"][task]
            pred = np.asarray(t["classes"])[np.argmax(np.asarray(t["seed_averaged_proba"]), axis=1)]
            follow[task] = {"n_disagree": int(dis.sum()),
                            "matches_personal": int((pred[dis] == personal[dis]).sum()),
                            "matches_consensus": int((pred[dis] == consensus[dis]).sum())}
        rows.append({"participant": pid, "agreement": e["agreement"],
                     "personal_high": int((personal == "high_valence").sum()),
                     "consensus_high": int((consensus == "high_valence").sum()),
                     **{f"{task}_{k}": e["tasks"][task]["seed_mean"][k] for task in TASKS
                        for k in ("balanced_accuracy", "f1_macro", "roc_auc", "predicted_high_share")},
                     **{f"{task}_ba_sd": e["tasks"][task]["seed_sd"]["balanced_accuracy"] for task in TASKS},
                     "disagreement_trials": follow})
    group: dict[str, Any] = {}
    for metric in ("balanced_accuracy", "f1_macro", "roc_auc"):
        p = np.array([r[f"personal_{metric}"] for r in rows]); c = np.array([r[f"consensus_{metric}"] for r in rows])
        d = p - c
        group[metric] = {"personal": mean_ci(p), "consensus": mean_ci(c), "difference": mean_ci(d),
                         "difference_participant_bootstrap_ci": _participant_bootstrap(d),
                         "difference_wilcoxon": wilcoxon(p, c), "difference_sign_flip_exact_p": _sign_flip_exact(d),
                         "n_personal_better": int((d > 0).sum()),
                         "consensus_vs_0.5_wilcoxon": wilcoxon(c, 0.5) if metric != "roc_auc" else wilcoxon(c, 0.5)}
    group["agreement"] = mean_ci([r["agreement"]["agreement"] for r in rows])
    group["kappa"] = mean_ci([r["agreement"]["kappa"] for r in rows])
    # disagreement trials: does the PERSONAL-task model follow the personal or the consensus label?
    prop = np.array([r["disagreement_trials"]["personal"]["matches_personal"] / r["disagreement_trials"]["personal"]["n_disagree"]
                     for r in rows])
    pooled_n = sum(r["disagreement_trials"]["personal"]["n_disagree"] for r in rows)
    pooled_k = sum(r["disagreement_trials"]["personal"]["matches_personal"] for r in rows)
    group["disagreement_trials_personal_model"] = {
        "pooled_trials": pooled_n, "pooled_follow_personal": pooled_k, "pooled_share_personal": pooled_k / pooled_n,
        "per_participant_share_personal": mean_ci(prop.tolist()),
        "sign_flip_exact_p_vs_0.5": _sign_flip_exact(prop - 0.5)}
    agree = np.array([r["agreement"]["agreement"] for r in rows])
    pba = np.array([r["personal_balanced_accuracy"] for r in rows])
    rho = spearmanr(agree, pba)
    group["agreement_vs_personal_ba_spearman"] = {"rho": float(rho[0]), "p": float(rho[1]), "n": len(rows)}

    summary = {"label": LABEL, "git_commit": res["git_commit"], "fingerprint": res["fingerprint"],
               "alignment_check": res["alignment_check"], "participants": rows, "group": group}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, pba, 0.4, yerr=[r["personal_ba_sd"] for r in rows], capsize=2, label="own rating (personal task)")
    axes[0].bar(x + 0.2, [r["consensus_balanced_accuracy"] for r in rows], 0.4,
                yerr=[r["consensus_ba_sd"] for r in rows], capsize=2, label="other participants' consensus")
    axes[0].axhline(0.5, ls="--", color="0.5", lw=0.8); axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x, [r["participant"] for r in rows]); axes[0].set_ylabel("trial balanced accuracy (mean ± SD, 5 seeds)")
    axes[0].legend(fontsize=7, loc="lower right"); axes[0].set_title("Within-person decoding: own rating vs video consensus", fontsize=9)
    axes[1].scatter(agree, pba)
    for r in rows:
        axes[1].annotate(r["participant"], (r["agreement"]["agreement"], r["personal_balanced_accuracy"]), fontsize=7)
    axes[1].axhline(0.5, ls="--", color="0.5", lw=0.8)
    axes[1].set_xlabel("agreement of own labels with consensus"); axes[1].set_ylabel("personal-task balanced accuracy")
    axes[1].set_title(f"Spearman rho = {rho[0]:.2f} (n = 12)", fontsize=9)
    fig.suptitle(LABEL, fontsize=8); fig.tight_layout(); fig.savefig(out / "stimulus_control.png", dpi=130); plt.close(fig)
    print(json.dumps(group, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["main", "report"], default="main")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or get_settings().output_dir / "experiments" / "stimulus_control_rf_20261011"
    out.mkdir(parents=True, exist_ok=True)
    {"main": phase_main, "report": phase_report}[args.phase](out)


if __name__ == "__main__":
    main()
