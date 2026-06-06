"""Diagnostics helpers."""

from __future__ import annotations


def residuals(y_true: list[float], y_pred: list[float]) -> list[float]:
    return [actual - predicted for actual, predicted in zip(y_true, y_pred)]


def directional_accuracy(y_true: list[float], y_pred: list[float]) -> float:
    hits = sum((a >= 0) == (b >= 0) for a, b in zip(y_true, y_pred))
    return hits / len(y_true)
