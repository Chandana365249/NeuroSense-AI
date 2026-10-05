from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.preprocessing.eeg_preprocessor import EEGPreprocessor
from backend.training.trainer import EmotionTrainer

router = APIRouter(prefix="", tags=["neurosense"])


class PredictionRequest(BaseModel):
    signal: list[float]


class TrainingRequest(BaseModel):
    dataset_path: str | None = None


@router.post("/train")
def train_model(payload: TrainingRequest) -> dict[str, Any]:
    trainer = EmotionTrainer(data_dir=payload.dataset_path)
    result = trainer.train()
    return result


@router.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    model_path = Path("outputs/models/best_model.pkl")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model found.")
    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    preprocessor = EEGPreprocessor()
    features = preprocessor.extract_features(payload.signal)
    feature_vector = [
        features["mean"],
        features["variance"],
        features["std"],
        features["entropy"],
        features["band_power"]["delta"],
        features["band_power"]["theta"],
        features["band_power"]["alpha"],
        features["band_power"]["beta"],
        features["band_power"]["gamma"],
    ]
    prediction = model.predict([feature_vector])[0]
    probabilities = model.predict_proba([feature_vector])[0]
    labels = ["happy", "sad", "angry", "relaxed"]
    confidence = float(max(probabilities))
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {label: float(prob) for label, prob in zip(labels, probabilities, strict=False)},
    }


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    metrics_path = Path("outputs/models/model_metrics.json")
    if not metrics_path.exists():
        return {"status": "not_trained"}
    with metrics_path.open("r", encoding="utf-8") as fh:
        return {"status": "trained", "metrics": json.load(fh)}


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    metrics_path = Path("outputs/graphs/metrics.json")
    if not metrics_path.exists():
        return {"status": "not_available"}
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/docs")
def docs_info() -> dict[str, Any]:
    return {"message": "OpenAPI docs available at /docs"}
