from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.ml.evaluation import aggregate_by_trial, compute_metrics
from app.ml.models import EncodedLabelClassifier, candidate_models
from app.ml.registry import ModelNotAvailableError, ModelRegistry


def _toy(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.where(rng.random(n) > 0.5, "zebra", "apple")
    X = rng.normal(size=(n, 5))
    X[:, 0] += np.where(y == "zebra", 3.0, -3.0)
    return X, y


@pytest.mark.parametrize("name", ["random_forest", "svm_rbf", "xgboost", "majority_baseline"])
def test_candidates_accept_string_labels(name: str) -> None:
    """Regression: XGBoost failed on string labels; every candidate must accept them."""
    X, y = _toy()
    model = candidate_models(0)[name]().fit(X, y)
    assert list(model.classes_) == ["apple", "zebra"]
    assert set(model.predict(X)) <= {"apple", "zebra"}


def test_probability_columns_follow_model_classes() -> None:
    """Regression: the old API assumed a hard-coded label order for probabilities."""
    X, y = _toy()
    model = candidate_models(0)["random_forest"]().fit(X, y)
    proba = model.predict_proba(X)
    zebra_col = list(model.classes_).index("zebra")
    assert proba[y == "zebra", zebra_col].mean() > 0.8
    assert (model.predict(X) == y).mean() > 0.9


def test_encoded_classifier_detects_order_mismatch() -> None:
    class Weird:
        classes_ = np.array([1, 0])

        def fit(self, X, y):  # type: ignore[no-untyped-def]
            return self

        def get_params(self, deep=True):  # type: ignore[no-untyped-def]
            return {}

    with pytest.raises(RuntimeError):
        EncodedLabelClassifier(Weird()).fit(np.zeros((4, 1)), np.array(["a", "b", "a", "b"]))


def test_binary_metrics_and_auc() -> None:
    y = np.array(["hi", "lo", "hi", "lo"])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    m = compute_metrics(y, proba, ["hi", "lo"])
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]
    assert m["roc_auc"] == pytest.approx(0.75)  # "lo" scores {0.8, 0.3} vs "hi" {0.1, 0.4}: 3 of 4 pairs ranked correctly


def test_multiclass_auc_uses_ovr() -> None:
    """Regression: old code passed an invalid roc_auc_score argument and crashed /train."""
    rng = np.random.default_rng(0)
    classes = ["a", "b", "c", "d"]
    y = np.array(classes * 10)
    proba = rng.dirichlet(np.ones(4), size=40)
    m = compute_metrics(y, proba, classes)
    assert m["roc_auc"] is not None and 0 <= m["roc_auc"] <= 1
    assert len(m["confusion_matrix"]) == 4


def test_auc_undefined_for_single_class() -> None:
    m = compute_metrics(np.array(["hi", "hi"]), np.array([[0.6, 0.4], [0.3, 0.7]]), ["hi", "lo"])
    assert m["roc_auc"] is None


def test_aggregate_by_trial_averages_segments() -> None:
    groups = np.array(["s1:0", "s1:0", "s1:1"])
    y = np.array(["hi", "hi", "lo"])
    proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
    keys, y_trial, agg = aggregate_by_trial(groups, y, proba)
    assert keys.tolist() == ["s1:0", "s1:1"]
    assert y_trial.tolist() == ["hi", "lo"]
    assert agg[0].tolist() == pytest.approx([0.6, 0.4])


def test_registry_without_model(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError):
        ModelRegistry(tmp_path).load()


def test_registry_detects_tampered_model(tmp_path: Path) -> None:
    X, y = _toy()
    model = candidate_models(0)["majority_baseline"]().fit(X, y)
    registry = ModelRegistry(tmp_path)
    registry.save("v1", model, {"trained_at": "2026-01-01T00:00:00+00:00"}, activate=True)
    loaded, meta = registry.load()
    assert meta["version"] == "v1" and list(loaded.classes_) == ["apple", "zebra"]
    (tmp_path / "v1" / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ModelNotAvailableError, match="integrity"):
        registry.load()


# ---------- end-to-end training on synthetic DEAP-format data ----------
def test_training_artifacts_and_metadata(trained_output_dir: Path, fake_deap_dir: Path) -> None:
    registry = ModelRegistry(trained_output_dir / "models")
    model, meta = registry.load()
    split = meta["split"]
    assert not set(split["test"]) & (set(split["train"]) | set(split["validation"]))
    assert len(split["test"]) == 2 and len(split["validation"]) == 1
    assert meta["class_labels"] == [str(c) for c in model.classes_] == ["high_valence", "low_valence"]
    assert meta["feature_schema"]["count"] == 440 == len(meta["feature_schema"]["names"])
    assert meta["test"]["segment"]["n_samples"] == split["segments"]["test"]
    assert meta["algorithm"] != "majority_baseline"
    assert meta["test"]["trial"]["accuracy"] >= 0.75  # synthetic data is separable by alpha power
    assert len(meta["cross_validation"]["folds"]) == 4
    for path in meta["artifacts"]["graphs"].values():
        assert path is None or Path(path).exists()
    assert Path(meta["artifacts"]["test_predictions"]).exists()
    metrics_file = trained_output_dir / "metrics" / f"{meta['version']}.json"
    assert json.loads(metrics_file.read_text())["version"] == meta["version"]


def test_trainer_refuses_missing_dataset(settings: Settings, tmp_path: Path) -> None:
    from app.ml.dataset import DatasetError
    from app.ml.training import BaselineTrainer

    broken = settings.model_copy(update={"dataset_dir": tmp_path / "missing"})
    with pytest.raises(DatasetError):
        BaselineTrainer(broken, run_cross_validation=False).run()
