"""Coefficient explanation utilities."""

from __future__ import annotations

from pathlib import Path

from cross_market_regression.features.scalers import StandardScaler1D

from .model import load_linear_model
from .persistence import load_json


def coefficients(model_dir: str) -> dict[str, float]:
    metadata = load_json(Path(model_dir) / "metadata.json")
    names = metadata["feature_names"]
    model = load_linear_model(model_dir, n_features=len(names))
    weights, bias = model.layers[-1].get_weights()
    result = {name: float(weights[idx][0]) for idx, name in enumerate(names)}
    result["bias"] = float(bias[0])
    return result


def get_standardized_formula(model_dir: str) -> str:
    coefs = coefficients(model_dir)
    bias = coefs.pop("bias")
    terms = [f"{coef:.10g}*z_{name}" for name, coef in coefs.items()]
    return "y_hat = " + " + ".join([f"{bias:.10g}", *terms])


def get_raw_return_formula(model_dir: str) -> dict:
    metadata = load_json(Path(model_dir) / "metadata.json")
    names = metadata["feature_names"]
    model = load_linear_model(model_dir, n_features=len(names))
    weights, bias = model.layers[-1].get_weights()
    scaler = StandardScaler1D.load(Path(model_dir) / "scaler.json")
    raw = scaler.inverse_transform_coef(weights, float(bias[0]))
    terms = [f"{coef:.10g}*{name}" for name, coef in raw["raw_betas"].items()]
    raw["formula_string"] = "y_hat = " + " + ".join([f"{raw['raw_intercept']:.10g}", *terms])
    return raw


def explain_coefficients(model_dir: str) -> list[dict[str, float | str]]:
    coefs = coefficients(model_dir)
    return [{"feature": key, "coefficient": value} for key, value in coefs.items()]
