from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from backend.preprocessing.eeg_preprocessor import EEGPreprocessor
from backend.training.model_registry import ModelRegistry
from backend.utils.logging_utils import configure_logging


class EmotionTrainer:
    """Train and compare multiple classifiers for EEG-derived emotion labels."""

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = Path(data_dir or "Dataset")
        self.preprocessor = EEGPreprocessor()
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.logger = configure_logging("outputs/training.log")

    def _load_demo_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        n_samples = 120
        X = rng.normal(size=(n_samples, 1024)).astype(np.float32)
        y = np.array(["happy", "sad", "angry", "relaxed"])[rng.integers(0, 4, size=n_samples)]
        return X, y

    def build_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._load_demo_dataset()
        features: list[np.ndarray] = []
        labels: list[str] = []
        for signal, label in zip(X, y, strict=False):
            feature_dict = self.preprocessor.extract_features(signal)
            feature_vector = np.array([
                feature_dict["mean"],
                feature_dict["variance"],
                feature_dict["std"],
                feature_dict["entropy"],
                feature_dict["band_power"]["delta"],
                feature_dict["band_power"]["theta"],
                feature_dict["band_power"]["alpha"],
                feature_dict["band_power"]["beta"],
                feature_dict["band_power"]["gamma"],
            ], dtype=np.float32)
            features.append(feature_vector)
            labels.append(label)
        return np.stack(features), np.array(labels)

    def train(self) -> dict[str, Any]:
        X, y = self.build_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        model_specs = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
            "xgboost": XGBClassifier(n_estimators=80, eval_metric="mlogloss", random_state=42),
        }

        results: list[dict[str, Any]] = []
        for name, model in model_specs.items():
            self.logger.info("Training %s", name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average="weighted", zero_division=0)
            recall = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, preds, labels=["happy", "sad", "angry", "relaxed"])
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test), average_method="macro")
            except ValueError:
                auc = 0.0
            self.models[name] = model
            self.metrics[name] = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
            }
            results.append({"name": name, **self.metrics[name], "confusion_matrix": cm.tolist()})

        best_name = max(self.metrics, key=lambda key: self.metrics[key]["f1"])
        best_model = self.models[best_name]
        self.models["best_model"] = best_model
        self.models["best_name"] = best_name

        output_dir = Path("outputs/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "best_model.pkl").open("wb") as fh:
            pickle.dump(best_model, fh)
        with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(self.metrics, fh, indent=2)

        registry = ModelRegistry(str(output_dir))
        registry.save_metadata(best_name, self.metrics[best_name])

        return {"best_model": best_name, "results": results, "metrics": self.metrics}
