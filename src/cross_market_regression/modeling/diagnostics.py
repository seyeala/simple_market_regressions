"""Diagnostics helpers."""

from __future__ import annotations


def residuals(y_true: list[float], y_pred: list[float]) -> list[float]:
    return [actual - predicted for actual, predicted in zip(y_true, y_pred)]


def directional_accuracy(y_true: list[float], y_pred: list[float]) -> float:
    hits = sum((a >= 0) == (b >= 0) for a, b in zip(y_true, y_pred))
    return hits / len(y_true)


def residual_summary(df_predictions) -> dict:
    residual = df_predictions["prediction"].astype(float) - df_predictions["actual"].astype(float)
    return {
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std(ddof=0)),
        "residual_5pct": float(residual.quantile(0.05)),
        "residual_50pct": float(residual.quantile(0.50)),
        "residual_95pct": float(residual.quantile(0.95)),
        "n_observations": int(len(residual)),
    }


def rolling_directional_accuracy(df_predictions, window: int = 60):
    out = df_predictions.copy()
    hits = (out["prediction"].astype(float).ge(0) == out["actual"].astype(float).ge(0)).astype(float)
    out["rolling_directional_accuracy"] = hits.rolling(window=window, min_periods=1).mean()
    return out


def coefficient_stability_report(*args, **kwargs):
    """Placeholder for future rolling/refit coefficient-stability diagnostics."""

    return {"status": "not_implemented"}
