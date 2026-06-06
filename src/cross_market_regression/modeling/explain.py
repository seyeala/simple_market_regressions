"""Coefficient explanation utilities."""

from __future__ import annotations

from pathlib import Path

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


def explain_coefficients(model_dir: str) -> list[dict[str, float | str]]:
    coefs = coefficients(model_dir)
    return [{"feature": key, "coefficient": value} for key, value in coefs.items()]
