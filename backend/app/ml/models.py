from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class EncodedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Wraps any classifier so it is trained on integer codes but exposes string class names.

    classes_ comes from the fitted LabelEncoder and predict_proba columns follow classes_, so callers
    never need to assume a class order (XGBoost requires integer labels; sklearn sorts strings).
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> EncodedLabelClassifier:
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        inner = getattr(self.estimator_, "classes_", None)
        if inner is not None and list(inner) != list(range(len(self.classes_))):
            raise RuntimeError("Inner estimator class order does not match the label encoder")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _pipeline(clf: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def candidate_models(seed: int) -> dict[str, Callable[[], EncodedLabelClassifier]]:
    """Fresh, unfitted candidates. The scaler sits inside each pipeline so it only sees training data."""
    return {
        "majority_baseline": lambda: EncodedLabelClassifier(DummyClassifier(strategy="most_frequent")),
        "random_forest": lambda: EncodedLabelClassifier(
            _pipeline(
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                    class_weight="balanced", n_jobs=-1, random_state=seed,
                )
            )
        ),
        "svm_rbf": lambda: EncodedLabelClassifier(
            _pipeline(
                CalibratedClassifierCV(
                    SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed), ensemble=False
                )
            )
        ),
        "xgboost": lambda: EncodedLabelClassifier(
            _pipeline(
                XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                    eval_metric="logloss", random_state=seed, n_jobs=-1,
                )
            )
        ),
    }


ALGORITHM_DESCRIPTIONS: dict[str, str] = {
    "majority_baseline": "Always predicts the most frequent training class (chance-level reference)",
    "random_forest": "StandardScaler + RandomForest (300 trees, min_samples_leaf=5, balanced class weights)",
    "svm_rbf": "StandardScaler + RBF-kernel SVM (C=1, balanced class weights) with sigmoid-calibrated probabilities",
    "xgboost": "StandardScaler + XGBoost (300 trees, depth 4, lr 0.05, subsample 0.8, colsample 0.5)",
}
