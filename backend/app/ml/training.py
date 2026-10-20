"""Baseline training on real DEAP features with a participant-aware split.

1. Build (or load cached) segment features for every participant.
2. Derive labels from self-reported ratings (configurable strategy/threshold).
3. Split by participant: train / validation / test (no participant in more than one split).
4. Fit each candidate on train, score on validation; pick the best by validation macro-F1
   (the majority-class reference is never selected).
5. Refit the chosen algorithm on train + validation, evaluate once on the unseen test participants.
6. Optional leave-one-participant-out CV over the non-test participants, to show subject variability.
7. Save model, metadata (feature schema, class mapping, config, metrics), predictions and figures.
   The new version is registered as an inactive candidate unless `activate=True` is passed explicitly.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.ml.dataset import DEAPDataset
from app.ml.evaluation import EvaluationReporter, aggregate_by_trial, compute_metrics
from app.ml.factory import make_dataset, make_feature_extractor, make_label_config, make_preprocessor
from app.ml.feature_table import FeatureTable, build_feature_table
from app.ml.models import ALGORITHM_DESCRIPTIONS, candidate_models
from app.ml.registry import ModelRegistry
from app.ml.splitting import split_participants
from app.preprocessing.features import FEATURE_VERSION

logger = get_logger("training")

ProgressCallback = Callable[[str, float], None]

DISCLAIMER = (
    "Research prototype. Predictions estimate a participant's self-reported affect category from EEG "
    "features; they are not a diagnosis of any medical or psychological condition."
)


@dataclass
class TrainingResult:
    version: str
    algorithm: str
    metadata: dict[str, Any]


def _evaluate(model: Any, table: FeatureTable, y: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    classes = [str(c) for c in model.classes_]
    proba = model.predict_proba(table.X)
    segment = compute_metrics(y, proba, classes)
    groups = np.char.add(np.char.add(table.participant.astype(str), ":"), table.trial.astype(str))
    _, y_trial, proba_trial = aggregate_by_trial(groups, y, proba)
    trial = compute_metrics(y_trial, proba_trial, classes)
    return segment, trial, proba


class BaselineTrainer:
    def __init__(self, settings: Settings, dataset: DEAPDataset | None = None, run_cross_validation: bool = True,
                 activate: bool = False) -> None:
        self.settings = settings
        self.activate = activate
        self.dataset = dataset or make_dataset(settings)
        self.preprocessor = make_preprocessor(settings)
        self.extractor = make_feature_extractor(settings)
        self.label_config = make_label_config(settings)
        self.run_cross_validation = run_cross_validation
        self.registry = ModelRegistry(settings.model_dir)
        self.reporter = EvaluationReporter(settings.graphs_dir, settings.metrics_dir)

    def run(self, progress: ProgressCallback | None = None) -> TrainingResult:
        def report(stage: str, fraction: float) -> None:
            logger.info("Training progress", extra={"stage": stage, "progress": round(fraction, 2)})
            if progress:
                progress(stage, fraction)

        s = self.settings
        s.ensure_output_dirs()
        started = datetime.now(UTC)
        report("Extracting features from DEAP recordings", 0.05)
        table = build_feature_table(self.dataset, self.preprocessor, self.extractor, s.cache_dir)
        y_all = table.labels(self.label_config)

        split = split_participants(
            sorted(set(table.participant.tolist())), s.test_participants, s.validation_participants, s.random_seed
        )
        masks = {part: split.mask(table.participant, part) for part in ("train", "validation", "test")}
        logger.info("Participant split", extra=split.to_dict())

        candidates = candidate_models(s.random_seed)
        validation: dict[str, dict[str, Any]] = {}
        for i, (name, factory) in enumerate(candidates.items()):
            report(f"Fitting {name} on training participants", 0.15 + 0.4 * i / len(candidates))
            model = factory().fit(table.X[masks["train"]], y_all[masks["train"]])
            seg, trial, _ = _evaluate(model, table.subset(masks["validation"]), y_all[masks["validation"]])
            validation[name] = {"segment": seg, "trial": trial}
            logger.info("Validation score", extra={"model": name, "f1_macro": seg["f1_macro"], "accuracy": seg["accuracy"]})

        selectable = {k: v for k, v in validation.items() if k != "majority_baseline"}
        best = max(selectable, key=lambda k: selectable[k]["segment"]["f1_macro"])
        report(f"Refitting {best} on train + validation participants", 0.6)
        fit_mask = masks["train"] | masks["validation"]
        final_model = candidates[best]().fit(table.X[fit_mask], y_all[fit_mask])
        reference = candidates["majority_baseline"]().fit(table.X[fit_mask], y_all[fit_mask])

        report("Evaluating on held-out test participants", 0.7)
        test_table = table.subset(masks["test"])
        y_test = y_all[masks["test"]]
        test_seg, test_trial, test_proba = _evaluate(final_model, test_table, y_test)
        ref_seg, ref_trial, _ = _evaluate(reference, test_table, y_test)

        cross_validation = None
        if self.run_cross_validation:
            report(f"Leave-one-participant-out CV for {best}", 0.78)
            cross_validation = self._cross_validate(best, table.subset(fit_mask), y_all[fit_mask])

        classes = [str(c) for c in final_model.classes_]
        trained_at = datetime.now(UTC)
        version = f"{self.label_config.strategy.value}-{best}-{trained_at:%Y%m%d-%H%M%S}"

        report("Saving model, metrics and figures", 0.9)
        graphs = self._save_figures(version, best, classes, validation, test_seg, test_trial, y_test, test_proba)
        predictions_path = self._save_predictions(version, test_table, y_test, test_proba, classes)

        counts = {part: int(m.sum()) for part, m in masks.items()}
        metadata: dict[str, Any] = {
            "name": f"NeuroSense {best} ({self.label_config.strategy.value})",
            "algorithm": best,
            "algorithm_description": ALGORITHM_DESCRIPTIONS[best],
            "trained_at": trained_at.isoformat(),
            "training_seconds": round((trained_at - started).total_seconds(), 1),
            "label": {
                "strategy": self.label_config.strategy.value,
                "description": self.label_config.describe(),
                "valence_threshold": self.label_config.valence_threshold,
                "arousal_threshold": self.label_config.arousal_threshold,
            },
            "class_labels": classes,
            "feature_schema": {"version": FEATURE_VERSION, "count": len(table.feature_names), "names": table.feature_names},
            "preprocessing": asdict(self.preprocessor.config),
            "split": {**split.to_dict(), "segments": counts, "strategy": "participant-aware (whole participants per split)"},
            "dataset": {
                "dir": str(self.dataset.root),
                "fingerprint": table.fingerprint,
                "participants": sorted(set(table.participant.tolist())),
                "segments": int(table.X.shape[0]),
                "rejected_segments": table.rejected_segments,
                "class_balance": {c: int((y_all == c).sum()) for c in classes},
            },
            "training_config": {"random_seed": s.random_seed, "selection_metric": "validation segment f1_macro",
                                "candidates": list(candidates), "final_fit": "train + validation participants"},
            "validation": validation,
            "test": {"segment": test_seg, "trial": test_trial},
            "test_reference_majority": {"segment": ref_seg, "trial": ref_trial},
            "cross_validation": cross_validation,
            "artifacts": {"graphs": graphs, "test_predictions": predictions_path},
            "disclaimer": DISCLAIMER,
        }
        self.registry.save(version, final_model, metadata, activate=self.activate)
        saved = self.registry.metadata(version)
        self.reporter.save_metrics(saved | {"feature_schema": {"version": FEATURE_VERSION, "count": len(table.feature_names)}},
                                   f"{version}.json")
        report("Done", 1.0)
        logger.info("Training finished", extra={"version": version, "test_f1_macro": test_seg["f1_macro"]})
        return TrainingResult(version=version, algorithm=best, metadata=saved)

    def _cross_validate(self, algorithm: str, table: FeatureTable, y: np.ndarray) -> dict[str, Any]:
        factory = candidate_models(self.settings.random_seed)[algorithm]
        folds = []
        for pid in sorted(set(table.participant.tolist())):
            held = table.participant == pid
            model = factory().fit(table.X[~held], y[~held])
            seg, trial, _ = _evaluate(model, table.subset(held), y[held])
            folds.append({"participant": pid, "segment_accuracy": seg["accuracy"], "segment_f1_macro": seg["f1_macro"],
                          "trial_accuracy": trial["accuracy"], "roc_auc": seg["roc_auc"]})
        acc = np.array([f["segment_accuracy"] for f in folds])
        f1 = np.array([f["segment_f1_macro"] for f in folds])
        return {
            "scheme": "leave-one-participant-out over train+validation participants",
            "folds": folds,
            "segment_accuracy_mean": float(acc.mean()), "segment_accuracy_std": float(acc.std()),
            "segment_f1_macro_mean": float(f1.mean()), "segment_f1_macro_std": float(f1.std()),
        }

    def _save_figures(self, version: str, best: str, classes: list[str], validation: dict[str, Any],
                      test_seg: dict[str, Any], test_trial: dict[str, Any], y_test: np.ndarray, test_proba: np.ndarray) -> dict[str, str | None]:
        r = self.reporter
        return {
            "confusion_matrix_segment": r.save_confusion_matrix(
                test_seg["confusion_matrix"], classes, f"{version}_cm_segment.png", f"Test participants, segment level ({best})"),
            "confusion_matrix_trial": r.save_confusion_matrix(
                test_trial["confusion_matrix"], classes, f"{version}_cm_trial.png", f"Test participants, trial level ({best})"),
            "roc_curve": r.save_roc_curve(y_test, test_proba, classes, f"{version}_roc.png", f"ROC, test participants ({best})"),
            "model_comparison": r.save_model_comparison(
                {k: v["segment"] for k, v in validation.items()}, "f1_macro", f"{version}_validation_comparison.png",
                "Validation participants: macro-F1 per candidate"),
        }

    def _save_predictions(self, version: str, table: FeatureTable, y: np.ndarray, proba: np.ndarray, classes: list[str]) -> str:
        path = self.settings.predictions_dir / f"{version}_test_predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["participant", "trial", "segment", "true_label", "predicted_label", *[f"p_{c}" for c in classes]])
            pred = np.asarray(classes)[np.argmax(proba, axis=1)]
            for i in range(len(y)):
                writer.writerow([table.participant[i], int(table.trial[i]), int(table.segment[i]), y[i], pred[i],
                                 *[f"{p:.4f}" for p in proba[i]]])
        return str(path)
