"""Robust CNN evaluation: repeated seeds, nested leave-one-participant-out CV, normalization variants.

The single registered CNN run (one split, one seed) cannot say how much of its score is the model and
how much is the seed or the particular 3 test people. This module re-runs the SAME training procedure
(CNNTrainer's selection run + train+validation refit) under two protocols:

1. repeated_seeds -- the fixed participant split (split_participants, settings.random_seed); only the
   training seed changes (weight init, batch order, dropout). The test participants are evaluated once
   per seed; no seed, epoch or setting is chosen from those scores. Seed 42 with "train_stats"
   normalization is the registered CNN's exact procedure and must reproduce its metrics.
2. lopo -- leave-one-participant-out over the non-test participants (the same 9 people the Random
   Forest's cross-validation uses; the 3 test participants are never touched). For each held-out
   person, early stopping uses 2 INNER validation participants drawn from the other 8, then the model
   is refit on all 8 for the selected epoch count and scored once on the held-out person.

Normalization variants (both use no labels):
- train_stats: per-channel mean/std from the fitted participants only (the registered CNN's method).
- per_participant: each participant's segments are z-scored per channel with that participant's own
  statistics before the train_stats step. For a held-out person this uses their unlabelled EEG, i.e.
  it assumes a calibration recording of each new person is available at prediction time.

Runs are evaluation experiments: they are tracked as JSON records, not registered as models.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.core.config import BACKEND_DIR, Settings
from app.core.logging import get_logger
from app.ml.dataset import DEAPDataset
from app.ml.deep.classifier import CNNClassifier
from app.ml.deep.cnn_model import CNNConfig, build_cnn, parameter_count
from app.ml.deep.data import SEGMENT_CACHE_VERSION, ChannelNormalizer, SegmentTable, build_segment_table, encode_labels
from app.ml.deep.training import CNNTrainer, CNNTrainingConfig, seed_everything
from app.ml.evaluation import aggregate_by_trial, compute_metrics
from app.ml.factory import make_dataset, make_label_config, make_preprocessor
from app.ml.models import candidate_models
from app.ml.splitting import ParticipantSplit, split_participants

logger = get_logger("deep.robust_evaluation")

PROTOCOLS = ("repeated_seeds", "lopo")
NORMALIZATIONS = ("train_stats", "per_participant")
NORMALIZATION_DESCRIPTIONS = {
    "train_stats": "per-channel z-score with statistics of the fitted (training) participants only",
    "per_participant": ("per-channel z-score with each participant's own statistics (unlabelled; assumes a "
                        "calibration recording per person), followed by the train_stats step"),
}
RECORD_FORMAT = "neurosense-cnn-eval-run-v1"


def participant_standardize(X: np.ndarray, participant: np.ndarray) -> np.ndarray:
    """Copy of X with every participant's segments z-scored per channel using only that participant's data."""
    out = np.empty(X.shape, dtype=np.float32)
    for pid in np.unique(participant):
        rows = np.flatnonzero(participant == pid)
        norm = ChannelNormalizer.fit(X, rows)
        for start in range(0, len(rows), 512):
            block = rows[start : start + 512]
            out[block] = norm.apply(X[block])
    return out


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    normalization: str
    seed: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    evaluate: tuple[str, ...]
    fold: str | None = None

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS or self.normalization not in NORMALIZATIONS:
            raise ValueError(f"Unknown protocol/normalization: {self.protocol}/{self.normalization}")
        groups = [set(self.train), set(self.validation), set(self.evaluate)]
        if not all(groups) or any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
            raise ValueError("train, validation and evaluated participants must be non-empty and disjoint")

    @property
    def run_id(self) -> str:
        unit = f"fold-{self.fold}" if self.fold else f"seed-{self.seed}"
        return f"{self.protocol}__{self.normalization}__{unit}"


def plan_runs(split: ParticipantSplit, protocols: list[str], normalizations: list[str], seeds: list[int],
              lopo_seed: int, n_inner_validation: int) -> list[RunSpec]:
    """All runs of an experiment. LOPO never includes split.test participants in any role."""
    specs: list[RunSpec] = []
    if "repeated_seeds" in protocols:
        specs += [RunSpec("repeated_seeds", norm, s, tuple(split.train), tuple(split.validation), tuple(split.test))
                  for norm in normalizations for s in seeds]
    for norm in normalizations:
        if "lopo" in protocols:
            pool = sorted(split.train + split.validation)
            order = np.random.default_rng(lopo_seed).permutation(len(pool) - 1)
            for held in pool:
                rest = [p for p in pool if p != held]
                inner_val = sorted(rest[i] for i in order[:n_inner_validation])
                inner_train = sorted(set(rest) - set(inner_val))
                specs.append(RunSpec("lopo", norm, lopo_seed, tuple(inner_train), tuple(inner_val), (held,), fold=held))
    return specs


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class _Data:
    table: SegmentTable
    X: dict[str, np.ndarray]
    labels: np.ndarray
    classes: list[str]
    codes: np.ndarray
    model_cfg: CNNConfig


class RobustCNNEvaluator:
    def __init__(self, settings: Settings, training_config: CNNTrainingConfig | None = None,
                 model_config: CNNConfig | None = None, dataset: DEAPDataset | None = None) -> None:
        self.settings = settings
        self.base_cfg = training_config or CNNTrainingConfig(seed=settings.random_seed)
        self.model_config = model_config
        self.dataset = dataset or make_dataset(settings)
        self.preprocessor = make_preprocessor(settings)
        self.label_config = make_label_config(settings)
        self._data: _Data | None = None

    # ---------------------------------------------------------------- data / plan
    def data(self, normalizations: list[str] | tuple[str, ...] = NORMALIZATIONS) -> _Data:
        if self._data is None:
            table = build_segment_table(self.dataset, self.preprocessor, self.settings.cache_dir)
            labels = table.labels(self.label_config)
            classes = sorted(set(labels.tolist()))
            n_ch, n_samp = table.shape
            cfg = self.model_config or CNNConfig(n_channels=n_ch, n_samples=n_samp, n_classes=len(classes))
            self._data = _Data(table, {"train_stats": np.ascontiguousarray(table.X)}, labels, classes,
                               encode_labels(labels, classes), cfg)
        if "per_participant" in normalizations and "per_participant" not in self._data.X:
            self._data.X["per_participant"] = participant_standardize(self._data.X["train_stats"], self._data.table.participant)
        return self._data

    def split(self) -> ParticipantSplit:
        s = self.settings
        return split_participants(sorted(set(self.data([]).table.participant.tolist())), s.test_participants,
                                  s.validation_participants, s.random_seed)

    def plan(self, protocols: list[str], normalizations: list[str], seeds: list[int]) -> list[RunSpec]:
        return plan_runs(self.split(), protocols, normalizations, seeds, self.settings.random_seed,
                         self.settings.validation_participants)

    def experiment_config(self) -> dict[str, Any]:
        d = self.data([])
        return {
            "label": {"strategy": self.label_config.strategy.value, "description": self.label_config.describe()},
            "class_labels": d.classes,
            "architecture": {"name": "EEGNet-style CNN", "config": d.model_cfg.to_dict(),
                             "parameters": parameter_count(build_cnn(d.model_cfg, 0))},
            "training_config": {**asdict(self.base_cfg), "optimizer": "AdamW", "loss": "cross-entropy"
                                + (" (balanced class weights)" if self.base_cfg.class_weighting else "")},
            "preprocessing": asdict(self.preprocessor.config),
            "input": {"representation": "preprocessed 4 s segments (channels x samples)", "segment_cache": SEGMENT_CACHE_VERSION,
                      "shape": [1, *d.table.shape]},
            "dataset": {"fingerprint": d.table.fingerprint, "participants": sorted(set(d.table.participant.tolist())),
                        "segments": int(d.table.X.shape[0]), "rejected_segments": d.table.rejected_segments,
                        "class_balance": {c: int((d.labels == c).sum()) for c in d.classes}},
            "split": self.split().to_dict(),
            "normalizations": {k: NORMALIZATION_DESCRIPTIONS[k] for k in NORMALIZATIONS},
            "framework": f"pytorch {torch.__version__}",
            "git_commit": git_commit(),
        }

    # ---------------------------------------------------------------- one run
    def run(self, spec: RunSpec, on_epoch: Any = None) -> dict[str, Any]:
        d = self.data([spec.normalization])
        X, table, labels, classes = d.X[spec.normalization], d.table, d.labels, d.classes

        def rows_of(ids: tuple[str, ...]) -> np.ndarray:
            return np.flatnonzero(np.isin(table.participant, ids))

        train_rows, val_rows, eval_rows = rows_of(spec.train), rows_of(spec.validation), rows_of(spec.evaluate)
        cfg = replace(self.base_cfg, seed=spec.seed)
        trainer = CNNTrainer(self.settings, model_config=d.model_cfg, training_config=cfg, dataset=self.dataset)
        # Same call order as CNNTrainer.run, so seed 42 / train_stats reproduces the registered model.
        seed_everything(cfg.seed, cfg.num_threads)
        started = datetime.now(UTC)

        norm_sel = ChannelNormalizer.fit(X, train_rows)
        model = build_cnn(d.model_cfg, cfg.seed)
        with tempfile.TemporaryDirectory() as tmp:
            history, best = trainer._fit(model, X, d.codes, train_rows, val_rows, norm_sel, Path(tmp),
                                         on_epoch or (lambda epoch: None))
            model.load_state_dict(torch.load(Path(tmp) / "best.pt", map_location="cpu", weights_only=True)["state_dict"])
        selected = CNNClassifier(model, classes, norm_sel)
        # Eval-mode (no dropout) scores of the selected checkpoint, for train/validation/test consistency.
        train_seg, train_trial, _ = trainer._evaluate(selected, X, labels, table, train_rows)
        val_seg, val_trial, _ = trainer._evaluate(selected, X, labels, table, val_rows)

        if cfg.final_fit == "train+validation":
            fit_rows = np.sort(np.concatenate([train_rows, val_rows]))
            norm_final = ChannelNormalizer.fit(X, fit_rows)
            final_model = build_cnn(d.model_cfg, cfg.seed)
            final_history = trainer._fit_fixed(final_model, X, d.codes, fit_rows, norm_final, best["epoch"], lambda epoch: None)
            final = CNNClassifier(final_model, classes, norm_final)
        else:
            fit_rows, final_history, final = train_rows, [], selected

        eval_seg, eval_trial, proba = trainer._evaluate(final, X, labels, table, eval_rows)
        reference = candidate_models(cfg.seed)["majority_baseline"]().fit(np.zeros((len(fit_rows), 1)), labels[fit_rows])
        ref_proba = reference.predict_proba(np.zeros((len(eval_rows), 1)))
        ref_seg = compute_metrics(labels[eval_rows], ref_proba, classes)
        ref_groups = np.char.add(np.char.add(table.participant[eval_rows].astype(str), ":"), table.trial[eval_rows].astype(str))
        _, ref_y_trial, ref_p_trial = aggregate_by_trial(ref_groups, labels[eval_rows], ref_proba)

        per_participant = {}
        for pid in spec.evaluate:
            mask = table.participant[eval_rows] == pid
            seg, trial, _ = trainer._evaluate(final, X, labels, table, eval_rows[mask])
            per_participant[pid] = {"segment": seg, "trial": trial,
                                    "predicted_fraction": {c: float(np.mean(np.argmax(proba[mask], 1) == i)) for i, c in enumerate(classes)},
                                    "true_fraction": {c: float(np.mean(labels[eval_rows][mask] == c)) for c in classes}}

        finished = datetime.now(UTC)
        record = {
            "format": RECORD_FORMAT,
            "run_id": spec.run_id,
            "protocol": spec.protocol,
            "normalization": spec.normalization,
            "seed": spec.seed,
            "fold": spec.fold,
            "participants": {"train": list(spec.train), "validation": list(spec.validation), "evaluated": list(spec.evaluate),
                             "evaluated_role": "held-out test participants" if spec.protocol == "repeated_seeds"
                             else "LOPO held-out participant (never used for training or selection in this fold)"},
            "segments": {"train": int(len(train_rows)), "validation": int(len(val_rows)), "evaluated": int(len(eval_rows))},
            "started_at": started.isoformat(),
            "seconds": round((finished - started).total_seconds(), 1),
            "epochs_run": len(history),
            "best_epoch": best["epoch"],
            "best_validation": best,
            "early_stopped": len(history) < cfg.max_epochs,
            "final_fit": cfg.final_fit,
            "history": {"selection": history, "final_fit": final_history},
            "selected_checkpoint": {"train": {"segment": train_seg, "trial": train_trial},
                                    "validation": {"segment": val_seg, "trial": val_trial}},
            "evaluated": {"segment": eval_seg, "trial": eval_trial},
            "per_participant": per_participant,
            "majority_reference": {"segment": ref_seg, "trial": compute_metrics(ref_y_trial, ref_p_trial, classes)},
            "predictions": {"classes": classes,
                            "participant": table.participant[eval_rows].tolist(),
                            "trial": table.trial[eval_rows].astype(int).tolist(),
                            "true_label": labels[eval_rows].tolist(),
                            "proba": np.round(proba, 6).tolist()},
        }
        logger.info("CNN evaluation run finished", extra={"run_id": spec.run_id, "best_epoch": best["epoch"],
                                                          "f1_macro": eval_seg["f1_macro"], "seconds": record["seconds"]})
        return record
