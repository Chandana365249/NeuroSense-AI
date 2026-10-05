from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


class EvaluationReporter:
    """Generate visual artifacts for model evaluation and persistence."""

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = Path(output_dir or "outputs/graphs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_confusion_matrix(self, cm: np.ndarray, labels: list[str]) -> str:
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        path = self.output_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)

    def save_metrics(self, metrics: dict[str, Any]) -> str:
        path = self.output_dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return str(path)
