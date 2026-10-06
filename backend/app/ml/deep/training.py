"""CNN training on real DEAP segments with the Review 2 participant-aware protocol.

1. Segment table (same rows/labels as the feature baseline) and the same participant split
   (split_participants with the configured seed).
2. Selection run: fit on TRAIN participants, evaluate VALIDATION participants every epoch;
   keep the epoch with the best validation macro-F1 (ties -> lower validation loss); stop after
   `patience` epochs without improvement. Best and last checkpoints are saved.
3. Final fit (default, mirrors the baseline's refit): retrain from the same seed on TRAIN + VALIDATION
   for exactly the selected number of epochs, then evaluate TEST participants once.
   The test set is never used for any decision.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from app.core.config import Settings
from app.core.logging import get_logger
from app.ml.dataset import EEG_CHANNEL_NAMES, DEAPDataset
from app.ml.deep.classifier import CNNClassifier
from app.ml.deep.cnn_model import CNNConfig, EEGCNN, build_cnn, layer_summary, parameter_count
from app.ml.deep.data import (
    SEGMENT_CACHE_VERSION,
    ChannelNormalizer,
    EEGSegmentDataset,
    SegmentTable,
    build_segment_table,
    encode_labels,
    make_loader,
)
from app.ml.evaluation import EvaluationReporter, aggregate_by_trial, compute_metrics
from app.ml.factory import make_dataset, make_label_config, make_preprocessor
from app.ml.models import candidate_models
from app.ml.registry import ModelRegistry
from app.ml.splitting import split_participants
from app.ml.training import DISCLAIMER

logger = get_logger("deep.training")

ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class CNNTrainingConfig:
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    class_weighting: bool = True
    selection_metric: str = "f1_macro"
    final_fit: str = "train+validation"  # or "train" (use the selected checkpoint directly)
    seed: int = 42
    num_threads: int = 4

    def __post_init__(self) -> None:
        if self.final_fit not in ("train+validation", "train"):
            raise ValueError("final_fit must be 'train+validation' or 'train'")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("max_epochs, patience and batch_size must be positive")


@dataclass
class CNNTrainingResult:
    version: str
    metadata: dict[str, Any]


def seed_everything(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


def _class_weights(codes: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(codes, minlength=n_classes).astype(np.float64)
    weights = len(codes) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(model: EEGCNN, loader: Any, loss_fn: nn.Module, optimizer: torch.optim.Optimizer | None) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass. With an optimizer: training mode + updates; otherwise eval mode. Returns (mean loss, proba, labels)."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    probas, labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
            probas.append(torch.softmax(logits.detach(), dim=1).numpy())
            labels.append(y.numpy())
    return total / max(n, 1), np.concatenate(probas), np.concatenate(labels)


class CNNTrainer:
    def __init__(self, settings: Settings, model_config: CNNConfig | None = None,
                 training_config: CNNTrainingConfig | None = None, dataset: DEAPDataset | None = None) -> None:
        self.settings = settings
        self.train_cfg = training_config or CNNTrainingConfig(seed=settings.random_seed)
        self.model_cfg = model_config
        self.dataset = dataset or make_dataset(settings)
        self.preprocessor = make_preprocessor(settings)
        self.label_config = make_label_config(settings)
        self.registry = ModelRegistry(settings.model_dir)
        self.reporter = EvaluationReporter(settings.graphs_dir, settings.metrics_dir)

    # ---------------------------------------------------------------- public
    def run(self, progress: ProgressCallback | None = None) -> CNNTrainingResult:
        def report(stage: str, fraction: float) -> None:
            logger.info("CNN training progress", extra={"stage": stage, "progress": round(fraction, 2)})
            if progress:
                progress(stage, fraction)

        s, tc = self.settings, self.train_cfg
        s.ensure_output_dirs()
        seed_everything(tc.seed, tc.num_threads)
        started = datetime.now(UTC)

        report("Loading preprocessed DEAP segments", 0.02)
        table = build_segment_table(self.dataset, self.preprocessor, s.cache_dir)
        X = np.ascontiguousarray(table.X)  # load into memory for fast epochs
        labels = table.labels(self.label_config)
        classes = sorted(set(labels.tolist()))
        codes = encode_labels(labels, classes)
        n_ch, n_samp = table.shape
        model_cfg = self.model_cfg or CNNConfig(n_channels=n_ch, n_samples=n_samp, n_classes=len(classes))
        if (model_cfg.n_channels, model_cfg.n_samples, model_cfg.n_classes) != (n_ch, n_samp, len(classes)):
            raise ValueError("CNN config does not match the data (channels, samples, classes)")

        split = split_participants(sorted(set(table.participant.tolist())), s.test_participants,
                                   s.validation_participants, s.random_seed)
        rows = {p: np.flatnonzero(split.mask(table.participant, p)) for p in ("train", "validation", "test")}
        if not len(rows["validation"]):
            raise ValueError("CNN training needs at least one validation participant for early stopping")
        trained_at = datetime.now(UTC)
        version = f"cnn_{self.label_config.strategy.value}_{trained_at:%Y%m%d-%H%M%S}"
        run_dir = self.registry.version_dir(version)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=False)

        # ---- selection run: train -> validation
        report("Selection run: training on train participants, early stopping on validation", 0.05)
        norm_sel = ChannelNormalizer.fit(X, rows["train"])
        model = build_cnn(model_cfg, tc.seed)
        history, best = self._fit(model, X, codes, rows["train"], rows["validation"], norm_sel, ckpt_dir,
                                  lambda e: report(f"Selection epoch {e}", 0.05 + 0.5 * e / tc.max_epochs))
        best_epoch = best["epoch"]
        model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=True)["state_dict"])
        selected = CNNClassifier(model, classes, norm_sel)
        val_seg, val_trial, _ = self._evaluate(selected, X, labels, table, rows["validation"])

        # ---- final fit
        if tc.final_fit == "train+validation":
            report(f"Final fit: train + validation participants for {best_epoch} epochs", 0.6)
            fit_rows = np.sort(np.concatenate([rows["train"], rows["validation"]]))
            norm_final = ChannelNormalizer.fit(X, fit_rows)
            final_model = build_cnn(model_cfg, tc.seed)
            final_history = self._fit_fixed(final_model, X, codes, fit_rows, norm_final, best_epoch,
                                            lambda e: report(f"Final-fit epoch {e}", 0.6 + 0.25 * e / best_epoch))
            final = CNNClassifier(final_model, classes, norm_final)
        else:
            fit_rows, final_history, final = rows["train"], [], selected

        # ---- single evaluation on held-out test participants
        report("Evaluating on held-out test participants", 0.88)
        test_seg, test_trial, test_proba = self._evaluate(final, X, labels, table, rows["test"])
        reference = candidate_models(s.random_seed)["majority_baseline"]().fit(np.zeros((len(fit_rows), 1)), labels[fit_rows])
        ref_proba = reference.predict_proba(np.zeros((len(rows["test"]), 1)))
        ref_seg = compute_metrics(labels[rows["test"]], ref_proba, classes)

        report("Saving model, history, metrics and figures", 0.94)
        graphs = self._save_figures(version, classes, history, final_history, best_epoch, test_seg, test_trial,
                                    labels[rows["test"]], test_proba)
        predictions_path = self._save_predictions(version, table, rows["test"], labels, test_proba, classes)
        history_path = s.metrics_dir / f"{version}_history.json"
        history_path.write_text(json.dumps({"selection": history, "final_fit": final_history, "best_epoch": best_epoch},
                                           indent=2), encoding="utf-8")

        finished = datetime.now(UTC)
        metadata: dict[str, Any] = {
            "name": f"NeuroSense EEG CNN ({self.label_config.strategy.value})",
            "algorithm": "cnn",
            "model_type": "cnn",
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "algorithm_description": "EEGNet-style CNN: temporal conv, depthwise spatial conv, separable conv, linear head",
            "trained_at": trained_at.isoformat(),
            "training_seconds": round((finished - started).total_seconds(), 1),
            "label": {
                "strategy": self.label_config.strategy.value,
                "description": self.label_config.describe(),
                "valence_threshold": self.label_config.valence_threshold,
                "arousal_threshold": self.label_config.arousal_threshold,
            },
            "class_labels": classes,
            "input": {
                "type": "segments",
                "shape": [1, n_ch, n_samp],
                "channels": list(EEG_CHANNEL_NAMES[:n_ch]),
                "sampling_rate": s.sampling_rate,
                "segment_seconds": s.segment_seconds,
                "normalization": final.normalizer.to_dict(),
            },
            "feature_schema": {"version": SEGMENT_CACHE_VERSION, "count": n_ch * n_samp, "names": []},
            "architecture": {
                "config": model_cfg.to_dict(),
                "parameters": parameter_count(final.model),
                "layers": layer_summary(final.model),
                "reference": "Lawhern et al. 2018, EEGNet (J. Neural Eng. 15:056013)",
            },
            "preprocessing": asdict(self.preprocessor.config),
            "split": {**split.to_dict(), "segments": {p: int(len(r)) for p, r in rows.items()},
                      "strategy": "participant-aware (whole participants per split)"},
            "dataset": {
                "dir": str(self.dataset.root),
                "fingerprint": table.fingerprint,
                "participants": sorted(set(table.participant.tolist())),
                "segments": int(X.shape[0]),
                "rejected_segments": table.rejected_segments,
                "class_balance": {c: int((labels == c).sum()) for c in classes},
            },
            "training_config": {**asdict(tc), "optimizer": "AdamW", "loss": "cross-entropy"
                                + (" (balanced class weights)" if tc.class_weighting else ""),
                                "best_epoch": best_epoch, "epochs_run": len(history),
                                "best_validation": best},
            "history": {"selection": history, "final_fit": final_history},
            "validation": {"cnn": {"segment": val_seg, "trial": val_trial}},
            "test": {"segment": test_seg, "trial": test_trial},
            "test_reference_majority": {"segment": ref_seg, "trial": None},
            "cross_validation": None,
            "artifacts": {"graphs": graphs, "test_predictions": predictions_path, "history": str(history_path),
                          "checkpoints": str(ckpt_dir)},
            "disclaimer": DISCLAIMER,
        }
        self.registry.save(version, final, metadata, activate=False)
        saved = self.registry.metadata(version)
        self.reporter.save_metrics({k: v for k, v in saved.items() if k != "history"}, f"{version}.json")
        report("Done", 1.0)
        logger.info("CNN training finished", extra={"version": version, "test_f1_macro": test_seg["f1_macro"]})
        return CNNTrainingResult(version=version, metadata=saved)

    # ---------------------------------------------------------------- internals
    def _loss(self, codes: np.ndarray, n_classes: int) -> nn.Module:
        weight = _class_weights(codes, n_classes) if self.train_cfg.class_weighting else None
        return nn.CrossEntropyLoss(weight=weight)

    def _optimizer(self, model: EEGCNN) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=self.train_cfg.learning_rate, weight_decay=self.train_cfg.weight_decay)

    def _fit(self, model: EEGCNN, X: np.ndarray, codes: np.ndarray, train_rows: np.ndarray, val_rows: np.ndarray,
             norm: ChannelNormalizer, ckpt_dir: Path, on_epoch: Callable[[int], None]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tc = self.train_cfg
        n_classes = model.config.n_classes
        classes_idx = [str(i) for i in range(n_classes)]
        train_loader = make_loader(EEGSegmentDataset(X, train_rows, codes, norm), tc.batch_size, True, tc.seed)
        val_loader = make_loader(EEGSegmentDataset(X, val_rows, codes, norm), 256, False, tc.seed)
        loss_fn = self._loss(codes[train_rows], n_classes)
        optimizer = self._optimizer(model)
        history: list[dict[str, Any]] = []
        best: dict[str, Any] = {"epoch": 0, "score": -np.inf, "val_loss": np.inf}
        stale = 0
        for epoch in range(1, tc.max_epochs + 1):
            train_loss, train_proba, train_y = _run_epoch(model, train_loader, loss_fn, optimizer)
            val_loss, val_proba, val_y = _run_epoch(model, val_loader, loss_fn, None)
            tm = compute_metrics(train_y.astype(str), train_proba, classes_idx)
            vm = compute_metrics(val_y.astype(str), val_proba, classes_idx)
            row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": tm["accuracy"], "train_f1_macro": tm["f1_macro"],
                   "val_loss": val_loss, "val_accuracy": vm["accuracy"], "val_balanced_accuracy": vm["balanced_accuracy"],
                   "val_f1_macro": vm["f1_macro"], "val_roc_auc": vm["roc_auc"]}
            history.append(row)
            score = vm[tc.selection_metric]
            improved = score > best["score"] + 1e-6 or (abs(score - best["score"]) <= 1e-6 and val_loss < best["val_loss"])
            if improved:
                best = {"epoch": epoch, "score": score, "val_loss": val_loss, "metric": tc.selection_metric}
                torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_dir / "best.pt")
                stale = 0
            else:
                stale += 1
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_dir / "last.pt")
            logger.info("CNN epoch", extra={k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            on_epoch(epoch)
            if stale >= tc.patience:
                logger.info("Early stopping", extra={"epoch": epoch, "best_epoch": best["epoch"]})
                break
        return history, best

    def _fit_fixed(self, model: EEGCNN, X: np.ndarray, codes: np.ndarray, rows: np.ndarray, norm: ChannelNormalizer,
                   epochs: int, on_epoch: Callable[[int], None]) -> list[dict[str, Any]]:
        tc = self.train_cfg
        n_classes = model.config.n_classes
        loader = make_loader(EEGSegmentDataset(X, rows, codes, norm), tc.batch_size, True, tc.seed)
        loss_fn = self._loss(codes[rows], n_classes)
        optimizer = self._optimizer(model)
        history = []
        for epoch in range(1, epochs + 1):
            loss, proba, y = _run_epoch(model, loader, loss_fn, optimizer)
            m = compute_metrics(y.astype(str), proba, [str(i) for i in range(n_classes)])
            history.append({"epoch": epoch, "train_loss": loss, "train_accuracy": m["accuracy"], "train_f1_macro": m["f1_macro"]})
            on_epoch(epoch)
        return history

    @staticmethod
    def _evaluate(clf: CNNClassifier, X: np.ndarray, labels: np.ndarray, table: SegmentTable,
                  rows: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
        classes = [str(c) for c in clf.classes_]
        proba = clf.predict_proba(X[rows])
        y = labels[rows]
        segment = compute_metrics(y, proba, classes)
        groups = np.char.add(np.char.add(table.participant[rows].astype(str), ":"), table.trial[rows].astype(str))
        _, y_trial, proba_trial = aggregate_by_trial(groups, y, proba)
        return segment, compute_metrics(y_trial, proba_trial, classes), proba

    def _save_figures(self, version: str, classes: list[str], history: list[dict[str, Any]], final_history: list[dict[str, Any]],
                      best_epoch: int, test_seg: dict[str, Any], test_trial: dict[str, Any], y_test: np.ndarray,
                      test_proba: np.ndarray) -> dict[str, str | None]:
        graphs_dir = self.settings.graphs_dir
        epochs = [h["epoch"] for h in history]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [h["train_loss"] for h in history], label="train (train participants)")
        ax.plot(epochs, [h["val_loss"] for h in history], label="validation (held-out participants)")
        if final_history:
            ax.plot([h["epoch"] for h in final_history], [h["train_loss"] for h in final_history], "--", label="final fit (train+val)")
        ax.axvline(best_epoch, color="0.5", ls=":", label=f"selected epoch {best_epoch}")
        ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss"); ax.set_title("CNN loss"); ax.legend(fontsize=8)
        fig.tight_layout(); loss_path = graphs_dir / f"{version}_loss.png"; fig.savefig(loss_path, dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [h["train_f1_macro"] for h in history], label="train macro-F1")
        ax.plot(epochs, [h["val_f1_macro"] for h in history], label="validation macro-F1")
        ax.plot(epochs, [h["train_accuracy"] for h in history], alpha=0.5, label="train accuracy")
        ax.plot(epochs, [h["val_accuracy"] for h in history], alpha=0.5, label="validation accuracy")
        ax.axvline(best_epoch, color="0.5", ls=":")
        ax.axhline(0.5, color="0.7", lw=0.8)
        ax.set_ylim(0, 1); ax.set_xlabel("epoch"); ax.set_title("CNN metrics per epoch (selection run)"); ax.legend(fontsize=8)
        fig.tight_layout(); metric_path = graphs_dir / f"{version}_metrics.png"; fig.savefig(metric_path, dpi=130); plt.close(fig)

        r = self.reporter
        return {
            "loss_curve": str(loss_path),
            "metric_curve": str(metric_path),
            "confusion_matrix_segment": r.save_confusion_matrix(test_seg["confusion_matrix"], classes, f"{version}_cm_segment.png",
                                                                "Test participants, segment level (CNN)"),
            "confusion_matrix_trial": r.save_confusion_matrix(test_trial["confusion_matrix"], classes, f"{version}_cm_trial.png",
                                                              "Test participants, trial level (CNN)"),
            "roc_curve": r.save_roc_curve(y_test, test_proba, classes, f"{version}_roc.png", "ROC, test participants (CNN)"),
        }

    def _save_predictions(self, version: str, table: SegmentTable, rows: np.ndarray, labels: np.ndarray,
                          proba: np.ndarray, classes: list[str]) -> str:
        path = self.settings.predictions_dir / f"{version}_test_predictions.csv"
        pred = np.asarray(classes)[np.argmax(proba, axis=1)]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["participant", "trial", "segment", "true_label", "predicted_label", *[f"p_{c}" for c in classes]])
            for i, row in enumerate(rows):
                writer.writerow([table.participant[row], int(table.trial[row]), int(table.segment[row]), labels[row], pred[i],
                                 *[f"{p:.4f}" for p in proba[i]]])
        return str(path)
