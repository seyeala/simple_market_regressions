"""Prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cross_market_regression.features.scaling import StandardScaler
from cross_market_regression.features.returns import ratio_return

from .model import load_linear_model
from .persistence import load_json, load_training_artifacts


@dataclass
class PredictionResult:
    predicted_return: float
    predicted_return_pct: float
    implied_target_level: float
    direction: str
    target_name: str
    model_name: str
    raw_features: dict
    standardized_features: dict
    model_dir: str
    metadata: dict


def _resolve_feature(name: str, manual_feature_values, source_signal_price, source_reference_price, fx_signal, fx_reference, optional_prices):
    manual_feature_values = manual_feature_values or {}
    optional_prices = optional_prices or {}
    if name in manual_feature_values:
        return float(manual_feature_values[name])
    lname = name.lower()
    if "fx" in lname:
        if fx_signal is not None and fx_reference is not None:
            return ratio_return(float(fx_signal), float(fx_reference))
    if name in optional_prices:
        prices = optional_prices[name]
        return ratio_return(float(prices["signal"]), float(prices["reference"]))
    if source_signal_price is not None and source_reference_price is not None:
        return ratio_return(float(source_signal_price), float(source_reference_price))
    raise ValueError(f"Missing inputs for feature {name!r}")


def predict_target_from_inputs(
    *,
    model_dir: str,
    target_current_close: float,
    manual_feature_values: dict[str, float] | None = None,
    source_signal_price: float | None = None,
    source_reference_price: float | None = None,
    fx_signal: float | None = None,
    fx_reference: float | None = None,
    optional_prices: dict[str, dict[str, float]] | None = None,
) -> PredictionResult:
    artifacts = load_training_artifacts(model_dir)
    metadata = artifacts["metadata"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    feature_names = metadata["feature_names"]
    raw_features = {
        name: _resolve_feature(name, manual_feature_values, source_signal_price, source_reference_price, fx_signal, fx_reference, optional_prices)
        for name in feature_names
    }
    import pandas as pd

    x = pd.DataFrame([raw_features], columns=feature_names)
    x_scaled = scaler.transform(x)
    standardized_features = {name: float(x_scaled[0, idx]) for idx, name in enumerate(feature_names)}
    predicted_return = float(model.predict(x_scaled, verbose=0)[0][0])
    threshold = float(metadata.get("direction_threshold", 0.002))
    if predicted_return > threshold:
        direction = "bull"
    elif predicted_return < -threshold:
        direction = "bear"
    else:
        direction = "neutral"
    return PredictionResult(
        predicted_return=predicted_return,
        predicted_return_pct=predicted_return * 100.0,
        implied_target_level=float(target_current_close) * (1.0 + predicted_return),
        direction=direction,
        target_name=metadata.get("target_asset", {}).get("name") or metadata.get("target_name", "target"),
        model_name=metadata.get("model_name", Path(model_dir).name),
        raw_features=raw_features,
        standardized_features=standardized_features,
        model_dir=model_dir,
        metadata=metadata,
    )


def predict_rows(model_dir: str, feature_rows: list[list[float]]) -> list[float]:
    metadata = load_json(Path(model_dir) / "metadata.json")
    feature_names = metadata["feature_names"]
    # Historical tests use the original list-based scaler JSON; support both.
    scaler_path = Path(model_dir) / "scaler.json"
    try:
        from cross_market_regression.features.scalers import StandardScaler1D
        import pandas as pd

        scaler_new = StandardScaler1D.load(scaler_path)
        x_scaled = scaler_new.transform(pd.DataFrame(feature_rows, columns=feature_names))
    except Exception:
        scaler_old = StandardScaler.load(scaler_path)
        x_scaled = scaler_old.transform(feature_rows)
    model = load_linear_model(model_dir, n_features=len(feature_names))
    return [float(value[0]) for value in model.predict(x_scaled, verbose=0)]
