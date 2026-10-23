"""Protocol-aware candidate training (Evaluation Protocol v1.1, docs/EVALUATION_PROTOCOL.md).

What it does:
1. Uses only the configured development participants (default: the 12 trusted participants). The frozen final-test
   participants can never be loaded (dataset guard + settings validation).
2. Fixes the algorithm in advance: Random Forest with the existing configuration. No hyperparameter search and no
   model selection, so there is no selection step that could leak information across folds.
3. Exploratory leave-one-participant-out (LOPO) cross-validation (protocol section 2.1): for each development
   participant, train on all the others and evaluate once on that person. Whole participants only, so no segment
   or trial (video) of the held-out person is ever in training. Feature scaling sits inside the model pipeline and
   is fitted on the fold's training participants only.
4. Metrics per held-out participant at the TRIAL level (segment probabilities averaged within a trial), the
   majority-class reference from the fold's training participants, and the mandatory diagnostics (predicted-class
   share, single-label participants, within-person AUC).
5. Refits the same configuration on all development participants and saves it as an INACTIVE candidate unless
   activate=True is passed explicitly.

Results are labelled "exploratory (12 participants, Protocol v1.1)". They are NOT an independent or held-out test
result: the final evaluation on the frozen group is blocked until the data-access decision (Decision I) is made.
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
from app.ml.dataset import DatasetError, DEAPDataset
from app.ml.evaluation import EvaluationReporter, aggregate_by_trial, compute_metrics
from app.ml.factory import make_feature_extractor, make_label_config, make_preprocessor
from app.ml.feature_table import FeatureTable, build_feature_table
from app.ml.models import ALGORITHM_DESCRIPTIONS, candidate_models
from app.ml.participants import FROZEN_TEST_PARTICIPANTS, PROTOCOL_VERSION
from app.ml.registry import ModelRegistry
from app.ml.stats import mean_ci
from app.ml.training import DISCLAIMER
from app.preprocessing.features import FEATURE_VERSION

logger = get_logger("protocol_training")

ProgressCallback = Callable[[str, float], None]
ALGORITHM = "random_forest"
MIN_DEVELOPMENT_PARTICIPANTS = 3
FEW_TRIALS = 5  # protocol section 3.4
EVALUATION_LABEL = f"exploratory (development participants only, Protocol v{PROTOCOL_VERSION})"
LIMITATIONS = (
    "Exploratory leave-one-participant-out cross-validation on the development participants only. It is not an "
    "independent or held-out test result, and it must not be used to claim that this model outperforms another one. "
    "Single seed; primary comparisons under the protocol need seeds 42-46. The final evaluation on the frozen "
    "participant group is pending the data-access decision (Decision I)."
)


class ProtocolViolation(RuntimeError):
    """Raised if a fold would mix participants between training and evaluation (should never happen)."""


@dataclass
class ProtocolTrainingResult:
    version: str
    algorithm: str
    metadata: dict[str, Any]


def _positive_class(classes: list[str]) -> str:
    return "high_valence" if "high_valence" in classes else classes[0]


class ProtocolTrainer:
    def __init__(self, settings: Settings, dataset: DEAPDataset | None = None, run_cross_validation: bool = True,
                 activate: bool = False) -> None:
        self.settings = settings
        self.development = frozenset(settings.development_participants)
        if self.development & FROZEN_TEST_PARTICIPANTS:  # settings validation already prevents this
            raise ProtocolViolation("Frozen final-test participants cannot be development participants")
        self.dataset = dataset or DEAPDataset(settings.dataset_dir, settings.n_eeg_channels, settings.sampling_rate,
                                              settings.baseline_seconds, allowed_participants=self.development)
        self.preprocessor = make_preprocessor(settings)
        self.extractor = make_feature_extractor(settings)
        self.label_config = make_label_config(settings)
        self.run_cross_validation = run_cross_validation
        self.activate = activate
        self.registry = ModelRegistry(settings.model_dir)
        self.reporter = EvaluationReporter(settings.graphs_dir, settings.metrics_dir)

    # ------------------------------------------------------------------ main entry point
    def run(self, progress: ProgressCallback | None = None) -> ProtocolTrainingResult:
        def report(stage: str, fraction: float) -> None:
            logger.info("Training progress", extra={"stage": stage, "progress": round(fraction, 2)})
            if progress:
                progress(stage, fraction)

        s = self.settings
        s.ensure_output_dirs()
        started = datetime.now(UTC)
        report("Extracting features from the development participants", 0.05)
        full = build_feature_table(self.dataset, self.preprocessor, self.extractor, s.cache_dir)
        table, excluded = self._development_rows(full)
        participants = sorted(set(table.participant.tolist()))
        missing = sorted(self.development - set(participants))
        if len(participants) < MIN_DEVELOPMENT_PARTICIPANTS:
            raise DatasetError(f"Protocol training needs at least {MIN_DEVELOPMENT_PARTICIPANTS} development "
                               f"participants; {len(participants)} are available")
        y = table.labels(self.label_config)
        classes = sorted({str(c) for c in y})
        if len(classes) < 2:
            raise DatasetError("The development data contain a single label class; cannot train a classifier")

        evaluation = None
        if self.run_cross_validation:
            evaluation = self._lopo(table, y, participants, classes, report)

        report("Refitting on all development participants", 0.85)
        final_model = candidate_models(s.random_seed)[ALGORITHM]().fit(table.X, y)
        classes = [str(c) for c in final_model.classes_]
        trained_at = datetime.now(UTC)
        version = f"{self.label_config.strategy.value}-{ALGORITHM}-protocol-{trained_at:%Y%m%d-%H%M%S}"

        report("Saving the candidate model", 0.92)
        lopo_path = self._save_lopo_predictions(version, evaluation) if evaluation else None
        metadata = self._metadata(table, y, participants, classes, missing, excluded, evaluation, started, trained_at, lopo_path)
        self.registry.save(version, final_model, metadata, activate=self.activate)
        saved = self.registry.metadata(version)
        self.reporter.save_metrics(saved | {"feature_schema": {"version": FEATURE_VERSION, "count": len(table.feature_names)}},
                                   f"{version}.json")
        report("Done", 1.0)
        logger.info("Protocol training finished", extra={"version": version, "activated": self.activate})
        return ProtocolTrainingResult(version=version, algorithm=ALGORITHM, metadata=saved)

    # ------------------------------------------------------------------ data
    def _development_rows(self, table: FeatureTable) -> tuple[FeatureTable, list[str]]:
        present = set(table.participant.tolist())
        if present & FROZEN_TEST_PARTICIPANTS:  # the dataset guard makes this unreachable
            raise ProtocolViolation("Frozen final-test participants reached the feature table")
        excluded = sorted(present - self.development)
        keep = np.isin(table.participant, sorted(self.development))
        return table.subset(keep), excluded

    # ------------------------------------------------------------------ evaluation
    def _lopo(self, table: FeatureTable, y: np.ndarray, participants: list[str], classes: list[str],
              report: Callable[[str, float], None]) -> dict[str, Any]:
        factory = candidate_models(self.settings.random_seed)[ALGORITHM]
        positive = _positive_class(classes)
        groups = np.char.add(np.char.add(table.participant.astype(str), ":"), table.trial.astype(str))
        folds: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for i, pid in enumerate(participants):
            report(f"Leave-one-participant-out fold {i + 1}/{len(participants)} (held out: {pid})",
                   0.1 + 0.7 * i / len(participants))
            held = table.participant == pid
            train_participants = sorted(set(table.participant[~held].tolist()))
            # Leakage checks (protocol section 6.1): whole participants only, and no trial group on both sides.
            if pid in train_participants or set(groups[held]) & set(groups[~held]):
                raise ProtocolViolation(f"Fold for {pid} mixes training and evaluation data")
            model = factory().fit(table.X[~held], y[~held])
            model_classes = [str(c) for c in model.classes_]
            proba = model.predict_proba(table.X[held])
            trial_keys, y_trial, p_trial = aggregate_by_trial(groups[held], y[held], proba)
            seg = compute_metrics(y[held], proba, model_classes)
            trial = compute_metrics(y_trial, p_trial, model_classes)

            # Majority-class reference: the most frequent TRIAL label among this fold's training participants.
            _, y_train_trial, _ = aggregate_by_trial(groups[~held], y[~held], np.zeros((int((~held).sum()), 1)))
            values, counts = np.unique(y_train_trial, return_counts=True)
            majority = str(values[np.argmax(counts)])
            ref_proba = np.tile([1.0 if c == majority else 0.0 for c in model_classes], (len(y_trial), 1))
            reference = compute_metrics(y_trial, ref_proba, model_classes)

            pred_trial = np.asarray(model_classes)[np.argmax(p_trial, axis=1)]
            share = float(np.mean(pred_trial == positive))
            support = {c: int((y_trial == c).sum()) for c in model_classes}
            folds.append({
                "held_out": pid,
                "train_participants": train_participants,
                "n_train_participants": len(train_participants),
                "n_trials": int(len(y_trial)),
                "n_segments": int(held.sum()),
                "trial_groups": sorted(int(k.split(":")[1]) for k in trial_keys),
                "trial": {k: trial[k] for k in ("balanced_accuracy", "f1_macro", "accuracy", "roc_auc")},
                "segment": {k: seg[k] for k in ("balanced_accuracy", "f1_macro", "accuracy", "roc_auc")},
                "majority_reference": {"class": majority, "trial_balanced_accuracy": reference["balanced_accuracy"],
                                       "trial_f1_macro": reference["f1_macro"]},
                "predicted_positive_share_trial": share,
                "single_label": share < 0.05 or share > 0.95,
                "trial_support": support,
                "few_trials_flag": min(support.values()) < FEW_TRIALS,
                "within_person_auc": trial["roc_auc"],
            })
            predictions.extend({"participant": pid, "trial": int(k.split(":")[1]), "true_label": str(t),
                                "predicted_label": str(p), **{f"p_{c}": float(v) for c, v in zip(model_classes, row, strict=True)}}
                               for k, t, p, row in zip(trial_keys, y_trial, pred_trial, p_trial, strict=True))

        def agg(path: tuple[str, ...]) -> dict[str, Any]:
            values = []
            for f in folds:
                v: Any = f
                for key in path:
                    v = v[key]
                values.append(v)
            return mean_ci(values)

        return {
            "kind": "lopo_exploratory",
            "protocol_version": PROTOCOL_VERSION,
            "label": EVALUATION_LABEL,
            "unit_of_analysis": "participant (trial-level metrics per held-out participant)",
            "positive_class": positive,
            "folds": folds,
            "aggregate": {
                "trial_balanced_accuracy": agg(("trial", "balanced_accuracy")),
                "trial_f1_macro": agg(("trial", "f1_macro")),
                "trial_accuracy": agg(("trial", "accuracy")),
                "within_person_auc": agg(("within_person_auc",)),
                "segment_balanced_accuracy": agg(("segment", "balanced_accuracy")),
                "majority_trial_balanced_accuracy": agg(("majority_reference", "trial_balanced_accuracy")),
                "majority_trial_f1_macro": agg(("majority_reference", "trial_f1_macro")),
            },
            "diagnostics": {
                "single_label_participants": int(sum(f["single_label"] for f in folds)),
                "few_trials_participants": [f["held_out"] for f in folds if f["few_trials_flag"]],
            },
            "_predictions": predictions,  # written to a CSV, removed from the metadata
        }

    # ------------------------------------------------------------------ outputs
    def _save_lopo_predictions(self, version: str, evaluation: dict[str, Any]) -> str:
        rows = evaluation.pop("_predictions")
        path = self.settings.predictions_dir / f"{version}_lopo_trial_predictions.csv"
        if rows:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return str(path)

    def _metadata(self, table: FeatureTable, y: np.ndarray, participants: list[str], classes: list[str],
                  missing: list[str], excluded: list[str], evaluation: dict[str, Any] | None, started: datetime,
                  trained_at: datetime, lopo_path: str | None) -> dict[str, Any]:
        s = self.settings
        trials_per_participant = {p: sorted({int(t) for t in table.trial[table.participant == p]}) for p in participants}
        return {
            "name": f"NeuroSense {ALGORITHM} ({self.label_config.strategy.value}, protocol candidate)",
            "algorithm": ALGORITHM,
            "algorithm_description": ALGORITHM_DESCRIPTIONS[ALGORITHM],
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
            "split": {
                "train": participants, "validation": [], "test": [], "seed": s.random_seed,
                "segments": {"train": int(table.X.shape[0])},
                "strategy": (f"Protocol v{PROTOCOL_VERSION}: final model fitted on all development participants; "
                             "evaluated by leave-one-participant-out cross-validation (exploratory). No held-out test "
                             "participants exist yet."),
            },
            "protocol": {
                "version": PROTOCOL_VERSION,
                "role": "candidate (inactive unless explicitly activated)",
                "development_participants": participants,
                "missing_development_participants": missing,
                "excluded_non_development_participants": len(excluded),
                "frozen_participants_used": False,
                "trial_groups": trials_per_participant,
                "algorithm_fixed_in_advance": True,
                "search_space": "none (existing Random Forest configuration, no tuning)",
                "seeds": [s.random_seed],
                "leakage_controls": [
                    "whole participants per fold (no segment or trial of a held-out person in training)",
                    "trial groups checked disjoint between training and evaluation in every fold",
                    "feature scaling fitted inside the model pipeline on training participants only",
                    "per-trial preprocessing only (baseline removal, filtering, clipping use the same trial)",
                    "no model selection or threshold tuning (fixed configuration, 0.5 trial threshold)",
                ],
                "limitations": LIMITATIONS,
            },
            "dataset": {
                "dir": str(self.dataset.root),
                "fingerprint": table.fingerprint,
                "participants": participants,
                "segments": int(table.X.shape[0]),
                "rejected_segments": table.rejected_segments,
                "class_balance": {c: int((y == c).sum()) for c in classes},
            },
            "training_config": {"random_seed": s.random_seed, "selection_metric": "none (fixed algorithm)",
                                "candidates": [ALGORITHM], "final_fit": "all development participants"},
            # Compatibility with the historical fixed-split metadata: there is no validation or test split here.
            "validation": {},
            "test": {},
            "test_reference_majority": {},
            "cross_validation": None,
            "evaluation": evaluation,
            "artifacts": {"graphs": {}, "lopo_trial_predictions": lopo_path},
            "disclaimer": DISCLAIMER,
        }
