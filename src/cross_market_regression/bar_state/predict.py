"""Prediction helpers for saved bar-state models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from cross_market_regression.features.scalers import StandardScaler1D
from cross_market_regression.modeling.linear_tf_model import build_linear_model
from cross_market_regression.modeling.persistence import load_json


def _load_target_scaler(model_dir: Path, metadata: dict) -> dict[str, float]:
    path = model_dir / "target_scaler.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = metadata.get("target_scaler", {"mean": 0.0, "std": 1.0})
    std = float(data.get("std", 1.0)) or 1.0
    return {"mean": float(data.get("mean", 0.0)), "std": std}


def load_bar_state_model(model_dir: str | Path) -> dict[str, object]:
    """Load a saved bar-state model and its preprocessing artifacts."""

    model_path = Path(model_dir)
    metadata = load_json(model_path / "metadata.json")
    feature_cols = metadata.get("feature_cols") or metadata.get("feature_names")
    if not feature_cols:
        raise ValueError("metadata.json is missing feature_cols")
    scaler = StandardScaler1D.load(model_path / "scaler.json")
    target_scaler = _load_target_scaler(model_path, metadata)
    model = build_linear_model(len(feature_cols), learning_rate=float(metadata.get("learning_rate", 0.01)))
    model.load_weights(str(model_path / "model.weights.h5"))
    return {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "metadata": metadata,
        "model_dir": str(model_path),
    }


def _row_to_frame(current_feature_row: Mapping[str, object] | pd.Series | pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if isinstance(current_feature_row, pd.DataFrame):
        if len(current_feature_row) != 1:
            raise ValueError("current_feature_row DataFrame must contain exactly one row")
        frame = current_feature_row.copy()
    elif isinstance(current_feature_row, pd.Series):
        frame = current_feature_row.to_frame().T
    else:
        frame = pd.DataFrame([dict(current_feature_row)])
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for prediction: {missing}")
    return frame[feature_cols]


def predict_bar_state_return(current_feature_row: Mapping[str, object] | pd.Series | pd.DataFrame, model_dir: str | Path) -> float:
    """Predict an unscaled future log return from a current feature row."""

    artifacts = load_bar_state_model(model_dir)
    metadata = artifacts["metadata"]
    feature_cols = list(metadata.get("feature_cols") or metadata.get("feature_names"))
    frame = _row_to_frame(current_feature_row, feature_cols)
    x_scaled = artifacts["scaler"].transform(frame)
    scaled_prediction = float(artifacts["model"].predict(x_scaled, verbose=0)[0][0])
    target_scaler = artifacts["target_scaler"]
    return float(scaled_prediction * target_scaler["std"] + target_scaler["mean"])


def predicted_fair_value_from_last(current_last: float, predicted_log_return: float) -> float:
    """Convert a predicted log return into an implied fair value."""

    return float(current_last) * float(np.exp(predicted_log_return))
