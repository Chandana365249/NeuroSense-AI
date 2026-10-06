"""Fair comparison of Random Forest, SVM, XGBoost and the CNN on the same participant-held-out test set.

- Random Forest and CNN metrics are read from their registered artifacts (already evaluated once).
- SVM and XGBoost only had validation scores in Review 2 (only the selected model was tested). Here they
  are refit with their unchanged Review 2 hyperparameters on train+validation participants and scored
  once on the same test participants. No hyperparameter or model choice is made from these numbers.
- Refuses to compare if split, label strategy, dataset fingerprint or test-set size differ.

Usage (from backend/):  python -m scripts.compare_models [--rf VERSION] [--cnn VERSION]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.evaluation import aggregate_by_trial, compute_metrics
from app.ml.factory import make_dataset, make_feature_extractor, make_label_config, make_preprocessor
from app.ml.feature_table import build_feature_table
from app.ml.models import candidate_models
from app.ml.registry import ModelRegistry
from app.ml.splitting import split_participants

METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")
LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "f1_macro": "Macro F1", "roc_auc": "ROC-AUC"}


def _pick(m: dict[str, Any] | None) -> dict[str, float | None]:
    return {k: (None if not m or m.get(k) is None else float(m[k])) for k in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf")
    parser.add_argument("--cnn")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging("WARNING", settings.log_dir)
    registry = ModelRegistry(settings.model_dir)

    rf = registry.metadata(args.rf) if args.rf else registry.latest("random_forest")
    cnn = registry.metadata(args.cnn) if args.cnn else registry.latest("cnn")
    if rf is None or cnn is None:
        raise SystemExit("Need one registered random_forest model and one registered cnn model")

    checks = {
        "same participant split": all(rf["split"][p] == cnn["split"][p] for p in ("train", "validation", "test")),
        "same label strategy/threshold": rf["label"] == cnn["label"],
        "same dataset fingerprint": rf["dataset"]["fingerprint"] == cnn["dataset"]["fingerprint"],
        "same test segment count": rf["test"]["segment"]["n_samples"] == cnn["test"]["segment"]["n_samples"],
    }
    if not all(checks.values()):
        raise SystemExit(f"Models are not comparable: {checks}")

    table = build_feature_table(make_dataset(settings), make_preprocessor(settings), make_feature_extractor(settings), settings.cache_dir)
    y = table.labels(make_label_config(settings))
    split = split_participants(sorted(set(table.participant.tolist())), settings.test_participants,
                               settings.validation_participants, settings.random_seed)
    if [split.train, split.validation, split.test] != [rf["split"][p] for p in ("train", "validation", "test")]:
        raise SystemExit("Current settings produce a different split than the registered models")
    fit = split.mask(table.participant, "train") | split.mask(table.participant, "validation")
    test = split.mask(table.participant, "test")
    groups = np.char.add(np.char.add(table.participant[test].astype(str), ":"), table.trial[test].astype(str))

    rows: dict[str, dict[str, Any]] = {}
    rows["Random Forest"] = {"source": rf["version"], "test_segment": _pick(rf["test"]["segment"]), "test_trial": _pick(rf["test"]["trial"]),
                             "validation_segment": _pick(rf["validation"]["random_forest"]["segment"])}
    for name, key in (("SVM (RBF)", "svm_rbf"), ("XGBoost", "xgboost")):
        model = candidate_models(settings.random_seed)[key]().fit(table.X[fit], y[fit])
        classes = [str(c) for c in model.classes_]
        proba = model.predict_proba(table.X[test])
        _, y_trial, p_trial = aggregate_by_trial(groups, y[test], proba)
        rows[name] = {"source": f"Review 2 config refit on train+validation ({key})",
                      "test_segment": _pick(compute_metrics(y[test], proba, classes)),
                      "test_trial": _pick(compute_metrics(y_trial, p_trial, classes)),
                      "validation_segment": _pick(rf["validation"][key]["segment"])}
    rows["CNN (EEGNet-style)"] = {"source": cnn["version"], "test_segment": _pick(cnn["test"]["segment"]),
                                  "test_trial": _pick(cnn["test"]["trial"]),
                                  "validation_segment": _pick(cnn["validation"]["cnn"]["segment"])}
    rows["Majority-class reference"] = {"source": "most frequent training class", "test_segment": _pick(rf["test_reference_majority"]["segment"]),
                                        "test_trial": _pick(rf["test_reference_majority"]["trial"]),
                                        "validation_segment": _pick(rf["validation"]["majority_baseline"]["segment"])}

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "task": rf["label"]["description"],
        "split": {p: rf["split"][p] for p in ("train", "validation", "test")},
        "test_segments": rf["test"]["segment"]["n_samples"],
        "test_trials": rf["test"]["trial"]["n_samples"],
        "comparability_checks": checks,
        "protocol": ("All models: selection on validation participants only; final model fit on train+validation; "
                     "one evaluation on the held-out test participants. SVM/XGBoost test scores are reported for "
                     "comparison only and were not used to choose anything."),
        "models": rows,
    }
    out_json = settings.metrics_dir / f"model_comparison_{stamp}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    lines = [f"# Model comparison ({stamp})", "", f"Task: {result['task']}  ",
             f"Test participants: {', '.join(result['split']['test'])} ({result['test_segments']} segments, {result['test_trials']} trials)", ""]
    for level, title in (("test_segment", "Test, per 4 s segment"), ("test_trial", "Test, per trial"),
                         ("validation_segment", "Validation participants, per segment (models fit on train only)")):
        lines += [f"## {title}", "", "| Model | " + " | ".join(LABELS[m] for m in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        lines += [f"| {name} | " + " | ".join(fmt(r[level][m]) for m in METRICS) + " |" for name, r in rows.items()]
        lines.append("")
    out_md = settings.metrics_dir / f"model_comparison_{stamp}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    names = [n for n in rows if n != "Majority-class reference"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    width = 0.2
    for ax, level, title in ((axes[0], "test_segment", "Held-out test participants, per segment"),
                             (axes[1], "test_trial", "Held-out test participants, per trial")):
        for i, name in enumerate(names):
            vals = [rows[name][level][m] or 0 for m in METRICS]
            bars = ax.bar(np.arange(len(METRICS)) + (i - 1.5) * width, vals, width, label=name)
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.axhline(0.5, ls="--", color="0.5", lw=0.8)
        ax.set_xticks(np.arange(len(METRICS)), [LABELS[m] for m in METRICS])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_png = settings.graphs_dir / f"model_comparison_{stamp}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    print("\n".join(lines))
    print(f"Saved {out_json}\n      {out_md}\n      {out_png}")


if __name__ == "__main__":
    main()
