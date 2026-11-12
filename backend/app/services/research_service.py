"""Headline research results, read from the saved result files; an unreadable file is reported, never replaced."""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings

DEAP_ZERO_SHOT = "exploratory/cross_participant_lvo_rf/summary.json"


def deap_summary(settings: Settings) -> dict[str, Any]:
    """The DEAP result for a new person on new videos, predicting the person's own rating (protocol v1.1, exploratory).

    Metric: trial-level balanced accuracy, one value per participant (mean over 5 seeds), averaged over participants;
    the interval is the saved 95% t-interval across participants."""
    try:
        saved = json.loads((settings.research_results_dir / DEAP_ZERO_SHOT).read_text(encoding="utf-8"))
        stats = saved["group"]["balanced_accuracy"]["personal"]
        value, low, high, n = float(stats["mean"]), float(stats["ci_low"]), float(stats["ci_high"]), int(stats["n"])
        status, seeds = str(saved["label"]), len(saved["seeds"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"available": False, "message": "The saved DEAP result file could not be read."}
    return {"available": True, "dataset": "DEAP", "metric": "trial_balanced_accuracy", "level": "trial", "value": value,
            "ci95": [low, high], "ci_method": "t-interval across participants", "n_participants": n, "n_seeds": seeds,
            "setting": "New person and new videos", "target": "the person's own rating", "status": status,
            "independent_test": False, "source": f"docs/results/{DEAP_ZERO_SHOT}"}
