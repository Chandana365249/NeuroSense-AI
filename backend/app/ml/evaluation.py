from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Classification metrics from probabilities whose columns follow `classes`.

    ROC-AUC: binary uses the probability of classes[1]; multiclass uses one-vs-rest macro averaging.
    It is reported as None when y_true contains a single class (AUC is undefined).
    """
    y_true = np.asarray(y_true)
    classes_arr = np.asarray(classes)
    y_pred = classes_arr[np.argmax(proba, axis=1)]
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "class_order": list(classes),
        "support": {c: int((y_true == c).sum()) for c in classes},
        "roc_auc": None,
    }
    if len(np.unique(y_true)) >= 2:
        if len(classes) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true == classes[1], proba[:, 1]))
        elif len(np.unique(y_true)) == len(classes):
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
    return metrics


def aggregate_by_trial(
    groups: np.ndarray, y_true: np.ndarray, proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average segment probabilities within each (participant, trial) group. Returns (keys, y_true, proba)."""
    keys, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    agg = np.zeros((len(keys), proba.shape[1]))
    np.add.at(agg, inverse, proba)
    agg /= np.bincount(inverse)[:, None]
    return keys, np.asarray(y_true)[first], agg


class EvaluationReporter:
    """Writes evaluation figures and metric files."""

    def __init__(self, graphs_dir: Path, metrics_dir: Path) -> None:
        self.graphs_dir = Path(graphs_dir)
        self.metrics_dir = Path(metrics_dir)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def save_confusion_matrix(self, cm: np.ndarray | list, labels: list[str], filename: str, title: str) -> str:
        cm = np.asarray(cm)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Self-reported (true)")
        threshold = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        path = self.graphs_dir / filename
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return str(path)

    def save_roc_curve(self, y_true: np.ndarray, proba: np.ndarray, classes: list[str], filename: str, title: str) -> str | None:
        if len(classes) != 2 or len(np.unique(y_true)) < 2:
            return None
        fpr, tpr, _ = roc_curve(np.asarray(y_true) == classes[1], proba[:, 1])
        auc = roc_auc_score(np.asarray(y_true) == classes[1], proba[:, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="0.6", label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = self.graphs_dir / filename
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return str(path)

    def save_model_comparison(self, scores: dict[str, dict[str, float]], metric: str, filename: str, title: str) -> str:
        names = list(scores)
        values = [scores[n][metric] for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, values, color=["0.65" if n == "majority_baseline" else "tab:blue" for n in names])
        ax.bar_label(bars, fmt="%.3f")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, ls="--", color="0.5", lw=0.8)
        ax.set_ylabel(metric)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        path = self.graphs_dir / filename
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return str(path)

    def save_metrics(self, metrics: dict[str, Any], filename: str = "metrics.json") -> str:
        path = self.metrics_dir / filename
        path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        return str(path)
