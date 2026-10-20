"""Summary, statistics, figures and a Markdown report for a robust CNN evaluation experiment.

Reads the run records written by scripts/evaluate_cnn.py and compares them with the active Random
Forest's REGISTERED results (its held-out test predictions and its leave-one-participant-out folds),
after checking that both used the same split, dataset fingerprint and LOPO participants. Nothing here
trains or selects a model; the numbers are only re-aggregated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.config import Settings
from app.ml.evaluation import aggregate_by_trial, compute_metrics
from app.ml.factory import make_dataset, make_feature_extractor, make_label_config, make_preprocessor
from app.ml.feature_table import build_feature_table
from app.ml.registry import ModelRegistry
from app.ml.stats import binary_metric_fns, cluster_bootstrap, mean_ci, permutation_test, wilcoxon

METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")
LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "f1_macro": "Macro-F1", "roc_auc": "ROC-AUC"}
NORM_LABELS = {"train_stats": "CNN, training-set normalization", "per_participant": "CNN, per-participant normalization"}
N_BOOT, N_PERM = 2000, 2000


def load_runs(exp_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((exp_dir / "runs").glob("*.json"))]


def _pick(m: dict[str, Any]) -> dict[str, float | None]:
    return {k: m.get(k) for k in METRICS}


def prediction_statistics(participant: np.ndarray, trial: np.ndarray, y: np.ndarray, proba: np.ndarray,
                          classes: list[str], seed: int = 0) -> dict[str, Any]:
    """Trial-cluster bootstrap CIs (segment and trial level) and a within-participant trial-label permutation test."""
    fns = binary_metric_fns(classes)
    groups = np.char.add(np.char.add(participant.astype(str), ":"), trial.astype(str))
    keys, y_trial, p_trial = aggregate_by_trial(groups, y, proba)
    trial_participant = np.array([k.split(":")[0] for k in keys])
    return {
        "bootstrap_segment": cluster_bootstrap(y, proba, groups, participant, fns, N_BOOT, seed),
        "bootstrap_trial": cluster_bootstrap(y_trial, p_trial, keys, trial_participant, fns, N_BOOT, seed),
        "permutation_trial": permutation_test(y_trial, p_trial, trial_participant, fns, N_PERM, seed),
        "method": (f"{N_BOOT} bootstrap resamples of whole trials within each participant (95% percentile CI); "
                   f"{N_PERM} permutations of trial labels within each participant (one-sided p, metric >= observed)"),
    }


def _record_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    p = record["predictions"]
    return (np.asarray(p["participant"]), np.asarray(p["trial"]), np.asarray(p["true_label"]),
            np.asarray(p["proba"], dtype=float), list(p["classes"]))


def _random_forest(settings: Settings) -> dict[str, Any] | None:
    registry = ModelRegistry(settings.model_dir)
    version = registry.active_version()
    if not version:
        return None
    meta = registry.metadata(version)
    if ModelRegistry.framework_of(meta) != "sklearn":
        return None
    out: dict[str, Any] = {"version": version, "algorithm": meta["algorithm"], "split": meta["split"],
                           "fingerprint": meta["dataset"]["fingerprint"],
                           "test": {"segment": _pick(meta["test"]["segment"]), "trial": _pick(meta["test"]["trial"])},
                           "majority": {"segment": _pick(meta["test_reference_majority"]["segment"]),
                                        "trial": _pick(meta["test_reference_majority"]["trial"] or {})},
                           "lopo": meta.get("cross_validation")}
    # Recompute the registered model's test probabilities (the saved CSV is rounded to 4 decimals).
    model, _ = registry.load(version)
    table = build_feature_table(make_dataset(settings), make_preprocessor(settings), make_feature_extractor(settings),
                                settings.cache_dir)
    mask = np.isin(table.participant, meta["split"]["test"])
    y = table.labels(make_label_config(settings))[mask]
    classes = [str(c) for c in model.classes_]
    proba = model.predict_proba(table.X[mask])
    out["predictions_reproduce_registered_metrics"] = all(
        compute_metrics(y, proba, classes)[k] == meta["test"]["segment"][k] for k in (*METRICS, "confusion_matrix"))
    participant, trial = table.participant[mask], table.trial[mask]
    out["per_participant"] = _per_participant(participant, trial, y, proba, classes)
    out["statistics"] = prediction_statistics(participant, trial, y, proba, classes)
    return out


def _per_participant(participant: np.ndarray, trial: np.ndarray, y: np.ndarray, proba: np.ndarray,
                     classes: list[str]) -> dict[str, Any]:
    out = {}
    for pid in sorted(set(participant.tolist())):
        m = participant == pid
        groups = trial[m].astype(str)
        _, yt, pt = aggregate_by_trial(groups, y[m], proba[m])
        out[pid] = {"segment": _pick(compute_metrics(y[m], proba[m], classes)),
                    "trial": _pick(compute_metrics(yt, pt, classes)),
                    "predicted_fraction": {c: float(np.mean(np.argmax(proba[m], 1) == i)) for i, c in enumerate(classes)}}
    return out


def _agg(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {k: mean_ci([v[k] for v in values if v.get(k) is not None]) for k in METRICS}


def summarize(runs: list[dict[str, Any]], manifest: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rf = _random_forest(settings)
    summary: dict[str, Any] = {"experiment": {k: manifest.get(k) for k in (
        "created_at", "git_commit", "label", "class_labels", "architecture", "training_config", "preprocessing", "input", "dataset", "split",
        "normalizations", "framework", "seeds")}, "n_runs": len(runs)}

    checks: dict[str, Any] = {}
    if rf:
        checks["same_split_as_random_forest"] = all(rf["split"][p] == manifest["split"][p] for p in ("train", "validation", "test"))
        checks["same_dataset_fingerprint"] = rf["fingerprint"] == manifest["dataset"]["fingerprint"]
        if rf.get("lopo"):
            rf_folds = sorted(f["participant"] for f in rf["lopo"]["folds"])
            checks["same_lopo_participants"] = rf_folds == sorted(manifest["split"]["train"] + manifest["split"]["validation"])
        checks["random_forest_predictions_reproduce_registered_metrics"] = rf["predictions_reproduce_registered_metrics"]
    summary["comparability_checks"] = checks

    # ---- reproduction of the registered CNN (seed = settings seed, train_stats normalization)
    registry = ModelRegistry(settings.model_dir)
    cnn_meta = registry.latest("cnn")
    keys = ("max_epochs", "patience", "batch_size", "learning_rate", "weight_decay", "class_weighting", "final_fit")
    ref_run = None
    if cnn_meta and all(cnn_meta["training_config"].get(k) == (manifest.get("training_config") or {}).get(k) for k in keys):
        ref_run = next((r for r in runs if r["protocol"] == "repeated_seeds" and r["normalization"] == "train_stats"
                        and r["seed"] == cnn_meta["training_config"]["seed"]), None)
    if cnn_meta and ref_run:
        summary["reproduction_check"] = {
            "registered_version": cnn_meta["version"], "seed": ref_run["seed"],
            "identical_test_metrics": all(cnn_meta["test"][lvl][k] == ref_run["evaluated"][lvl][k]
                                          for lvl in ("segment", "trial") for k in (*METRICS, "confusion_matrix")),
            "registered_best_epoch": cnn_meta["training_config"]["best_epoch"], "rerun_best_epoch": ref_run["best_epoch"],
        }

    # ---- repeated seeds on the fixed split
    seeds_out: dict[str, Any] = {}
    for norm in NORM_LABELS:
        rs = sorted((r for r in runs if r["protocol"] == "repeated_seeds" and r["normalization"] == norm), key=lambda r: r["seed"])
        if not rs:
            continue
        per_run = []
        for r in rs:
            participant, trial, y, proba, classes = _record_arrays(r)
            sel = r["selected_checkpoint"]
            per_run.append({
                "seed": r["seed"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "seconds": r["seconds"],
                "validation_selection_score": r["best_validation"]["score"],
                "train_eval_mode": _pick(sel["train"]["segment"]), "validation": _pick(sel["validation"]["segment"]),
                "test": {"segment": _pick(r["evaluated"]["segment"]), "trial": _pick(r["evaluated"]["trial"])},
                "confusion_matrix_segment": r["evaluated"]["segment"]["confusion_matrix"],
                "per_participant": {p: {"segment": _pick(v["segment"]), "trial": _pick(v["trial"]),
                                        "predicted_fraction": v["predicted_fraction"], "true_fraction": v["true_fraction"]}
                                    for p, v in r["per_participant"].items()},
                "statistics": prediction_statistics(participant, trial, y, proba, classes, seed=r["seed"]),
            })
        pids = sorted(per_run[0]["per_participant"])
        seeds_out[norm] = {
            "runs": per_run,
            "aggregate": {lvl: _agg([p["test"][lvl] for p in per_run]) for lvl in ("segment", "trial")},
            "validation_aggregate": _agg([p["validation"] for p in per_run]),
            "train_eval_mode_aggregate": _agg([p["train_eval_mode"] for p in per_run]),
            "best_epoch": mean_ci([p["best_epoch"] for p in per_run]),
            "confusion_matrix_segment_sum": np.sum([p["confusion_matrix_segment"] for p in per_run], axis=0).tolist(),
            "per_participant": {pid: {**{lvl: _agg([p["per_participant"][pid][lvl] for p in per_run]) for lvl in ("segment", "trial")},
                                      "predicted_fraction": {c: mean_ci([p["per_participant"][pid]["predicted_fraction"][c] for p in per_run])
                                                             for c in classes},
                                      "true_fraction": per_run[0]["per_participant"][pid]["true_fraction"]}
                                for pid in pids},
            "runs_significant_trial_auc_p05": sum(1 for p in per_run
                                                  if (p["statistics"]["permutation_trial"]["roc_auc"]["p_value"] or 1) < 0.05),
            "aggregate_note": ("mean/std/CI across training seeds on the SAME 3 test participants: this measures seed "
                               "variability, not between-person variability"),
        }
    summary["repeated_seeds"] = seeds_out

    # ---- nested LOPO
    lopo_out: dict[str, Any] = {}
    rf_folds = {f["participant"]: f for f in (rf or {}).get("lopo", {}).get("folds", [])} if rf and rf.get("lopo") else {}
    for norm in NORM_LABELS:
        rs = sorted((r for r in runs if r["protocol"] == "lopo" and r["normalization"] == norm), key=lambda r: r["fold"])
        if not rs:
            continue
        folds = [{"participant": r["fold"], "inner_train": r["participants"]["train"], "inner_validation": r["participants"]["validation"],
                  "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                  "segment": _pick(r["evaluated"]["segment"]), "trial": _pick(r["evaluated"]["trial"]),
                  "majority_segment": _pick(r["majority_reference"]["segment"]),
                  "train_eval_mode": _pick(r["selected_checkpoint"]["train"]["segment"]),
                  "inner_validation_score": _pick(r["selected_checkpoint"]["validation"]["segment"])} for r in rs]
        entry: dict[str, Any] = {
            "folds": folds,
            "aggregate": {lvl: _agg([f[lvl] for f in folds]) for lvl in ("segment", "trial")},
            "majority_aggregate": _agg([f["majority_segment"] for f in folds]),
            "balanced_accuracy_vs_0.5": wilcoxon([f["segment"]["balanced_accuracy"] for f in folds], 0.5),
            "accuracy_vs_majority": wilcoxon([f["segment"]["accuracy"] for f in folds],
                                             [f["majority_segment"]["accuracy"] for f in folds]),
        }
        if rf_folds and all(f["participant"] in rf_folds for f in folds):
            entry["paired_vs_random_forest"] = {
                "segment_accuracy": wilcoxon([f["segment"]["accuracy"] for f in folds],
                                             [rf_folds[f["participant"]]["segment_accuracy"] for f in folds]),
                "segment_f1_macro": wilcoxon([f["segment"]["f1_macro"] for f in folds],
                                             [rf_folds[f["participant"]]["segment_f1_macro"] for f in folds]),
            }
        lopo_out[norm] = entry
    summary["lopo"] = lopo_out

    if rf:
        rf_summary = {k: rf[k] for k in ("version", "algorithm", "test", "majority") if k in rf}
        rf_summary["per_participant"] = rf.get("per_participant")
        rf_summary["statistics"] = rf.get("statistics")
        if rf.get("lopo"):
            folds = rf["lopo"]["folds"]
            rf_summary["lopo"] = {"scheme": rf["lopo"]["scheme"], "folds": folds,
                                  "segment_accuracy": mean_ci([f["segment_accuracy"] for f in folds]),
                                  "segment_f1_macro": mean_ci([f["segment_f1_macro"] for f in folds]),
                                  "trial_accuracy": mean_ci([f["trial_accuracy"] for f in folds]),
                                  "roc_auc": mean_ci([f["roc_auc"] for f in folds])}
        summary["random_forest"] = rf_summary
    return summary


# ---------------------------------------------------------------- figures
def _figures(summary: dict[str, Any], runs: list[dict[str, Any]], fig_dir: Path) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    norms = [n for n in NORM_LABELS if n in summary["repeated_seeds"]]

    if norms:
        fig, axes = plt.subplots(len(norms), 2, figsize=(12, 3.6 * len(norms)), squeeze=False)
        for i, norm in enumerate(norms):
            rs = [r for r in runs if r["protocol"] == "repeated_seeds" and r["normalization"] == norm]
            for r in sorted(rs, key=lambda r: r["seed"]):
                h = r["history"]["selection"]
                ep = [x["epoch"] for x in h]
                line = axes[i][0].plot(ep, [x["val_loss"] for x in h], lw=1.2, label=f"seed {r['seed']} val")[0]
                axes[i][0].plot(ep, [x["train_loss"] for x in h], lw=0.8, ls="--", color=line.get_color())
                axes[i][1].plot(ep, [x["val_f1_macro"] for x in h], lw=1.2, color=line.get_color(), label=f"seed {r['seed']}")
                axes[i][1].plot(r["best_epoch"], h[r["best_epoch"] - 1]["val_f1_macro"], "o", color=line.get_color())
            axes[i][0].set_title(f"{NORM_LABELS[norm]}: loss (solid = validation, dashed = train)", fontsize=9)
            axes[i][1].set_title(f"{NORM_LABELS[norm]}: validation macro-F1 per epoch (dot = selected epoch)", fontsize=9)
            axes[i][1].axhline(0.5, color="0.6", lw=0.8, ls=":")
            axes[i][0].set_xlabel("epoch"); axes[i][1].set_xlabel("epoch")
            axes[i][1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        paths["learning_curves"] = str(fig_dir / "learning_curves.png")
        fig.savefig(paths["learning_curves"], dpi=120); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ("f1_macro", "roc_auc"), strict=True):
            for j, norm in enumerate(norms):
                vals = [r["test"]["segment"][metric] for r in summary["repeated_seeds"][norm]["runs"]]
                ax.scatter(np.full(len(vals), j) + np.linspace(-0.08, 0.08, len(vals)), vals, s=30, zorder=3)
                ax.errorbar(j + 0.22, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0, fmt="s", color="k", capsize=4)
            rf = summary.get("random_forest")
            if rf:
                ax.axhline(rf["test"]["segment"][metric], color="tab:green", ls="--", lw=1, label="Random Forest (registered)")
                maj = rf["majority"]["segment"][metric]
                if maj is not None:
                    ax.axhline(maj, color="0.6", ls=":", lw=1, label="majority reference")
            ax.set_xticks(range(len(norms)), [NORM_LABELS[n].replace("CNN, ", "") for n in norms], fontsize=8)
            ax.set_title(f"Held-out test participants, segment {LABELS[metric]} per seed", fontsize=9)
            ax.legend(fontsize=7)
        fig.tight_layout()
        paths["seed_distribution"] = str(fig_dir / "seed_distribution.png")
        fig.savefig(paths["seed_distribution"], dpi=120); plt.close(fig)

        pids = sorted(summary["repeated_seeds"][norms[0]]["per_participant"])
        fig, ax = plt.subplots(figsize=(8, 4))
        series: list[tuple[str, list[float], list[float]]] = []
        rf_pp = (summary.get("random_forest") or {}).get("per_participant")
        if rf_pp:
            series.append(("Random Forest", [rf_pp[p]["segment"]["balanced_accuracy"] for p in pids], [0] * len(pids)))
        for norm in norms:
            pp = summary["repeated_seeds"][norm]["per_participant"]
            series.append((NORM_LABELS[norm], [pp[p]["segment"]["balanced_accuracy"]["mean"] for p in pids],
                           [pp[p]["segment"]["balanced_accuracy"]["sd_sample"] for p in pids]))
        w = 0.8 / len(series)
        for k, (name, vals, err) in enumerate(series):
            ax.bar(np.arange(len(pids)) + (k - (len(series) - 1) / 2) * w, vals, w, yerr=err, capsize=3, label=name)
        ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
        ax.set_xticks(range(len(pids)), pids); ax.set_ylim(0, 1); ax.set_ylabel("segment balanced accuracy")
        ax.set_title("Per test participant (CNN: mean +/- SD over seeds)", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout()
        paths["per_participant"] = str(fig_dir / "per_participant.png")
        fig.savefig(paths["per_participant"], dpi=120); plt.close(fig)

    lopo_norms = [n for n in NORM_LABELS if n in summary["lopo"]]
    if lopo_norms:
        pids = [f["participant"] for f in summary["lopo"][lopo_norms[0]]["folds"]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        rf_folds = {f["participant"]: f for f in ((summary.get("random_forest") or {}).get("lopo") or {}).get("folds", [])}
        for ax, (metric, rf_key) in zip(axes, (("accuracy", "segment_accuracy"), ("f1_macro", "segment_f1_macro")), strict=True):
            series = []
            if rf_folds:
                series.append(("Random Forest", [rf_folds[p][rf_key] for p in pids]))
            for norm in lopo_norms:
                series.append((NORM_LABELS[norm], [f["segment"][metric] for f in summary["lopo"][norm]["folds"]]))
            series.append(("majority reference", [f["majority_segment"][metric] for f in summary["lopo"][lopo_norms[0]]["folds"]]))
            w = 0.8 / len(series)
            for k, (name, vals) in enumerate(series):
                ax.bar(np.arange(len(pids)) + (k - (len(series) - 1) / 2) * w, vals, w, label=name,
                       color="0.7" if name == "majority reference" else None)
            ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
            ax.set_xticks(range(len(pids)), pids); ax.set_ylim(0, 1)
            ax.set_title(f"Leave-one-participant-out, segment {LABELS[metric]} per held-out participant", fontsize=9)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        paths["lopo_folds"] = str(fig_dir / "lopo_folds.png")
        fig.savefig(paths["lopo_folds"], dpi=120); plt.close(fig)
    return paths


# ---------------------------------------------------------------- markdown
def _f(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _p(v: float | None) -> str:
    return "n/a" if v is None else ("< 0.001" if v < 0.001 else f"{v:.3f}")


def _ms(a: dict[str, Any]) -> str:
    return "n/a" if a.get("mean") is None else f"{a['mean']:.3f} ± {a['sd_sample']:.3f}"


def _ci(a: dict[str, Any]) -> str:
    return "n/a" if a.get("ci_low") is None else f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]"


def render_markdown(summary: dict[str, Any], figures: dict[str, str]) -> str:
    e = summary["experiment"]
    L = [f"# Robust CNN evaluation ({(e.get('created_at') or '')[:10]})", "",
         f"Task: {e['label']['description']}  ",
         f"Model: {e['architecture']['name']}, {e['architecture']['parameters']} parameters  ",
         f"Dataset: {len(e['dataset']['participants'])} DEAP participants, {e['dataset']['segments']} segments "
         f"(fingerprint {e['dataset']['fingerprint']})  ",
         f"Split: train {', '.join(e['split']['train'])} · validation {', '.join(e['split']['validation'])} · "
         f"test {', '.join(e['split']['test'])}  ",
         f"Seeds: {e.get('seeds')} · code commit {e.get('git_commit')} · {e.get('framework')}", "",
         "All numbers are generated from the run records; nothing was selected using test or held-out participants.", ""]

    L += ["## Comparability and reproduction checks", ""]
    L += [f"- {k.replace('_', ' ')}: **{v}**" for k, v in summary["comparability_checks"].items()]
    rc = summary.get("reproduction_check")
    if rc:
        L.append(f"- seed {rc['seed']} rerun reproduces registered CNN `{rc['registered_version']}` exactly: "
                 f"**{rc['identical_test_metrics']}** (best epoch {rc['rerun_best_epoch']} vs {rc['registered_best_epoch']})")
    L.append("")

    seeds = summary["repeated_seeds"]
    rf = summary.get("random_forest")
    if seeds:
        L += ["## 1. Repeated seeds, fixed split (held-out test participants)", "",
              "Mean ± sample SD over seeds with a t-based 95% CI. All seeds are scored on the same 3 test participants, "
              "so this shows sensitivity to the training seed, not variability across people.", ""]
        for lvl, title in (("segment", "per 4 s segment"), ("trial", "per trial")):
            L += [f"### Test, {title}", "", "| Model | " + " | ".join(LABELS[m] for m in METRICS) + " | 95% CI of macro-F1 |",
                  "|---" * (len(METRICS) + 2) + "|"]
            for norm, s in seeds.items():
                a = s["aggregate"][lvl]
                L.append(f"| {NORM_LABELS[norm]} ({len(s['runs'])} seeds) | " + " | ".join(_ms(a[m]) for m in METRICS)
                         + f" | {_ci(a['f1_macro'])} |")
            if rf:
                L.append("| Random Forest (registered, deterministic) | " + " | ".join(_f(rf["test"][lvl][m]) for m in METRICS) + " | see §3 |")
                if rf["majority"][lvl].get("accuracy") is not None:
                    L.append("| Majority-class reference | " + " | ".join(_f(rf["majority"][lvl][m]) for m in METRICS) + " | |")
            L.append("")
        L += ["### Each seed", "", "| Normalization | Seed | Best epoch / run | Val. macro-F1 (selection) | Test seg. acc. | "
              "Test seg. macro-F1 | Test seg. AUC | Test trial acc. | Test trial AUC | Trial AUC perm. p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for norm, s in seeds.items():
            for r in s["runs"]:
                L.append(f"| {norm} | {r['seed']} | {r['best_epoch']} / {r['epochs_run']} | {_f(r['validation_selection_score'])} | "
                         f"{_f(r['test']['segment']['accuracy'])} | {_f(r['test']['segment']['f1_macro'])} | "
                         f"{_f(r['test']['segment']['roc_auc'])} | {_f(r['test']['trial']['accuracy'])} | "
                         f"{_f(r['test']['trial']['roc_auc'])} | {_p(r['statistics']['permutation_trial']['roc_auc']['p_value'])} |")
        L.append("")
        L += ["### Train / validation / test consistency (segment macro-F1, selected checkpoint in eval mode)", "",
              "| Normalization | Train | Validation | Test (final refit) | Best epoch |", "|---|---|---|---|---|"]
        for norm, s in seeds.items():
            L.append(f"| {norm} | {_ms(s['train_eval_mode_aggregate']['f1_macro'])} | {_ms(s['validation_aggregate']['f1_macro'])} | "
                     f"{_ms(s['aggregate']['segment']['f1_macro'])} | {_ms(s['best_epoch'])} |")
        L.append("")
        first_class = e["class_labels"][0]
        L += ["### Per test participant (segment level, mean ± SD over seeds)", "",
              f"'Predicted {first_class}' is the share of the participant's segments assigned to {first_class}; a value of 0 or 1 "
              "means the model gave every segment of that person the same label.", "",
              f"| Participant | Model | Accuracy | Balanced acc. | Macro-F1 | ROC-AUC | Predicted {first_class} | Self-reported {first_class} |",
              "|---|---|---|---|---|---|---|---|"]
        for pid in sorted(next(iter(seeds.values()))["per_participant"]):
            true_frac = next(iter(seeds.values()))["per_participant"][pid]["true_fraction"][first_class]
            if rf and rf.get("per_participant"):
                rp = rf["per_participant"][pid]
                L.append(f"| {pid} | Random Forest | " + " | ".join(_f(rp["segment"][m]) for m in METRICS)
                         + f" | {_f(rp['predicted_fraction'][first_class])} | {_f(true_frac)} |")
            for norm, s in seeds.items():
                a = s["per_participant"][pid]
                L.append(f"| {pid} | {NORM_LABELS[norm]} | " + " | ".join(_ms(a["segment"][m]) for m in METRICS)
                         + f" | {_ms(a['predicted_fraction'][first_class])} | {_f(true_frac)} |")
        L.append("")
        L += ["### Summed segment confusion matrices over seeds", "",
              f"Rows = self-reported, columns = predicted, order {', '.join(e['class_labels'])}.", ""]
        for norm, s in seeds.items():
            L.append(f"- {norm}: `{s['confusion_matrix_segment_sum']}`")
        L.append("")

    lopo = summary["lopo"]
    if lopo:
        L += ["## 2. Nested leave-one-participant-out (9 non-test participants)", "",
              "For each held-out participant, early stopping uses 2 inner validation participants from the other 8; the model "
              "is refit on all 8 for the selected epoch count and scored once on the held-out person. The 3 test participants "
              "are never used. Mean ± sample SD over the 9 folds, t-based 95% CI.", "",
              "| Model | Seg. accuracy | Seg. balanced acc. | Seg. macro-F1 | Seg. ROC-AUC | Trial accuracy | 95% CI seg. accuracy |",
              "|---|---|---|---|---|---|---|"]
        for norm, s in lopo.items():
            a, t = s["aggregate"]["segment"], s["aggregate"]["trial"]
            L.append(f"| {NORM_LABELS[norm]} | {_ms(a['accuracy'])} | {_ms(a['balanced_accuracy'])} | {_ms(a['f1_macro'])} | "
                     f"{_ms(a['roc_auc'])} | {_ms(t['accuracy'])} | {_ci(a['accuracy'])} |")
        if rf and rf.get("lopo"):
            r = rf["lopo"]
            L.append(f"| Random Forest (registered CV, no inner validation needed) | {_ms(r['segment_accuracy'])} | n/a | "
                     f"{_ms(r['segment_f1_macro'])} | {_ms(r['roc_auc'])} | {_ms(r['trial_accuracy'])} | {_ci(r['segment_accuracy'])} |")
        maj = next(iter(lopo.values()))["majority_aggregate"]
        L.append(f"| Majority-class reference | {_ms(maj['accuracy'])} | {_ms(maj['balanced_accuracy'])} | {_ms(maj['f1_macro'])} | "
                 f"{_ms(maj['roc_auc'])} | | |")
        L.append("")
        L += ["### Per fold (segment level)", "", "| Held-out | Inner validation | " + " | ".join(
            f"{n} acc. / F1 / AUC / best ep." for n in lopo) + (" | RF acc. / F1 |" if rf and rf.get("lopo") else " |") + " Majority acc. |",
            "|---|---|" + "---|" * len(lopo) + ("---|" if rf and rf.get("lopo") else "") + "---|"]
        rf_folds = {f["participant"]: f for f in (rf or {}).get("lopo", {}).get("folds", [])} if rf and rf.get("lopo") else {}
        first = next(iter(lopo.values()))["folds"]
        for i, f0 in enumerate(first):
            cells = []
            for s in lopo.values():
                f = s["folds"][i]
                cells.append(f"{_f(f['segment']['accuracy'])} / {_f(f['segment']['f1_macro'])} / {_f(f['segment']['roc_auc'])} / {f['best_epoch']}")
            rfc = f" | {_f(rf_folds[f0['participant']]['segment_accuracy'])} / {_f(rf_folds[f0['participant']]['segment_f1_macro'])}" if rf_folds else ""
            L.append(f"| {f0['participant']} | {', '.join(f0['inner_validation'])} | " + " | ".join(cells) + rfc
                     + f" | {_f(f0['majority_segment']['accuracy'])} |")
        L.append("")
        L += [f"### Tests across folds (Wilcoxon signed-rank, n = {len(first)}, two-sided; low power)", "",
              "| Comparison | Median difference | p |", "|---|---|---|"]
        for norm, s in lopo.items():
            tests = [("balanced accuracy vs 0.5", s["balanced_accuracy_vs_0.5"]),
                     ("accuracy vs majority reference", s["accuracy_vs_majority"])]
            if "paired_vs_random_forest" in s:
                tests += [("accuracy vs Random Forest", s["paired_vs_random_forest"]["segment_accuracy"]),
                          ("macro-F1 vs Random Forest", s["paired_vs_random_forest"]["segment_f1_macro"])]
            for name, w in tests:
                L.append(f"| {norm}: {name} | {_f(w['median_difference'])} | {_p(w['p_value'])} |")
        L.append("")

    if rf and rf.get("statistics") and seeds:
        L += ["## 3. Uncertainty on the held-out test participants", "",
              "95% CIs from resampling whole trials within each participant; p-values from permuting trial labels within each "
              "participant (trial level). With 3 test people these describe uncertainty over their trials only.", "",
              "| Model | Seg. macro-F1 [95% CI] | Trial accuracy [95% CI] | Trial ROC-AUC [95% CI] | p (trial macro-F1) | p (trial AUC) |",
              "|---|---|---|---|---|---|"]

        def row(name: str, st: dict[str, Any]) -> str:
            bs, bt, pt = st["bootstrap_segment"], st["bootstrap_trial"], st["permutation_trial"]
            return (f"| {name} | {_f(bs['f1_macro']['estimate'])} {_ci(bs['f1_macro'])} | {_f(bt['accuracy']['estimate'])} "
                    f"{_ci(bt['accuracy'])} | {_f(bt['roc_auc']['estimate'])} {_ci(bt['roc_auc'])} | "
                    f"{_p(pt['f1_macro']['p_value'])} | {_p(pt['roc_auc']['p_value'])} |")

        L.append(row(f"Random Forest `{rf['version']}`", rf["statistics"]))
        for norm, s in seeds.items():
            for r in s["runs"]:
                L.append(row(f"{NORM_LABELS[norm]}, seed {r['seed']}", r["statistics"]))
        L.append("")

    if figures:
        L += ["## Figures", ""] + [f"- `{Path(p).name}`: {k.replace('_', ' ')}" for k, p in figures.items()] + [""]
    return "\n".join(L)


def write_report(exp_dir: Path, settings: Settings) -> Path:
    exp_dir = Path(exp_dir)
    runs = load_runs(exp_dir)
    if not runs:
        raise FileNotFoundError(f"No run records in {exp_dir / 'runs'}")
    manifest = json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))
    summary = summarize(runs, manifest, settings)
    figures = _figures(summary, runs, exp_dir / "figures")
    summary["figures"] = figures
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out = exp_dir / "REPORT.md"
    out.write_text(render_markdown(summary, figures), encoding="utf-8")
    return out
