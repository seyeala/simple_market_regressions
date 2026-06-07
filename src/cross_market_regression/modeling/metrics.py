"""Regression and direction metrics."""

from __future__ import annotations

def calculate_metrics(y_true, y_pred, threshold: float = 0.002) -> dict[str, float | int | bool]:
    import numpy as np

    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) == 0:
        return {"n_observations": 0}
    residual = pred - y
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom else 0.0
    directional_accuracy = float(np.mean(np.sign(pred) == np.sign(y)))
    active = np.abs(pred) > threshold
    hit_rate = float(np.mean(np.sign(pred[active]) == np.sign(y[active]))) if np.any(active) else 0.0
    baseline = np.zeros_like(y)
    baseline_residual = baseline - y
    baseline_mae = float(np.mean(np.abs(baseline_residual)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_residual**2)))
    baseline_directional_accuracy = float(np.mean(np.sign(baseline) == np.sign(y)))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "hit_rate_when_abs_prediction_gt_threshold": hit_rate,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_5pct": float(np.quantile(residual, 0.05)),
        "residual_50pct": float(np.quantile(residual, 0.50)),
        "residual_95pct": float(np.quantile(residual, 0.95)),
        "n_observations": int(len(y)),
        "baseline_prediction": 0.0,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_directional_accuracy": baseline_directional_accuracy,
        "beats_zero_return_baseline_mae": bool(mae < baseline_mae),
        "beats_zero_return_baseline_rmse": bool(rmse < baseline_rmse),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float | int | bool]:
    return calculate_metrics(y_true, y_pred)
