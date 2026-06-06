"""Prediction helpers."""

from __future__ import annotations

from pathlib import Path

from cross_market_regression.features.scaling import StandardScaler

from .model import load_linear_model
from .persistence import load_json


def predict_rows(model_dir: str, feature_rows: list[list[float]]) -> list[float]:
    metadata = load_json(Path(model_dir) / "metadata.json")
    scaler = StandardScaler.load(Path(model_dir) / "scaler.json")
    x_scaled = scaler.transform(feature_rows)
    model = load_linear_model(model_dir, n_features=len(metadata["feature_names"]))
    return [float(value[0]) for value in model.predict(x_scaled, verbose=0)]
