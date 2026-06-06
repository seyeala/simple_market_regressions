"""Regression metrics."""

from __future__ import annotations


def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    mean = sum(y_true) / len(y_true)
    total = sum((value - mean) ** 2 for value in y_true)
    if total == 0:
        return 0.0
    residual = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return 1.0 - residual / total


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
