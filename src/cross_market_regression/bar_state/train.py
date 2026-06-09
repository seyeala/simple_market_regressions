"""Training workflow for intraday bar-state models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_market_regression.features.scalers import StandardScaler1D
from cross_market_regression.modeling.linear_tf_model import build_linear_model
from cross_market_regression.modeling.metrics import calculate_metrics
from cross_market_regression.modeling.persistence import ensure_model_dir, save_history_csv, save_json

from .dataset import chronological_split_by_date


def _target_scaler(y_train: pd.Series) -> dict[str, float]:
    mean = float(y_train.astype(float).mean())
    std = float(y_train.astype(float).std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return {"mean": mean, "std": std}


def _scale_target(values: pd.Series, scaler: dict[str, float]) -> np.ndarray:
    return ((values.astype(float).to_numpy() - scaler["mean"]) / scaler["std"]).reshape(-1, 1)


def _unscale_target(values: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1) * scaler["std"] + scaler["mean"]


def _write_split_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def train_bar_state_model(
    dataset: pd.DataFrame,
    model_dir: str | Path,
    *,
    feature_cols: list[str] | None = None,
    target_col: str | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float | None = None,
    learning_rate: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    random_seed: int = 42,
    direction_threshold: float = 0.0,
) -> dict[str, object]:
    """Train a linear TensorFlow model on a bar-state dataset and save artifacts."""

    import tensorflow as tf

    feature_cols = list(feature_cols or dataset.attrs.get("feature_cols") or [])
    target_col = target_col or dataset.attrs.get("target_col")
    if not feature_cols:
        raise ValueError("feature_cols must be provided or present in dataset.attrs['feature_cols']")
    if not target_col:
        raise ValueError("target_col must be provided or present in dataset.attrs['target_col']")
    missing = [column for column in [*feature_cols, target_col] if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    rows = dataset.dropna(subset=[*feature_cols, target_col]).copy()
    if rows.empty:
        raise ValueError("No rows remain after dropping NaNs in features/target")

    tf.keras.utils.set_random_seed(random_seed)
    train_df, val_df, test_df = chronological_split_by_date(
        rows,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    if train_df.empty:
        raise ValueError("Training split is empty")
    if val_df.empty:
        val_df = train_df.copy()

    feature_scaler = StandardScaler1D(feature_names=feature_cols)
    x_train = feature_scaler.fit_transform(train_df[feature_cols])
    x_val = feature_scaler.transform(val_df[feature_cols])
    x_test = feature_scaler.transform(test_df[feature_cols]) if not test_df.empty else np.empty((0, len(feature_cols)))

    target_scaler = _target_scaler(train_df[target_col])
    target_scaler["target_col"] = target_col
    y_train = _scale_target(train_df[target_col], target_scaler)
    y_val = _scale_target(val_df[target_col], target_scaler)
    y_test = _scale_target(test_df[target_col], target_scaler) if not test_df.empty else np.empty((0, 1))

    out = ensure_model_dir(model_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out / "model.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=0,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=0,
    )

    model = build_linear_model(len(feature_cols), learning_rate=learning_rate)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[checkpoint, early_stopping],
        verbose=0,
    )
    if (out / "model.weights.h5").exists():
        model.load_weights(str(out / "model.weights.h5"))
    else:
        model.save_weights(str(out / "model.weights.h5"))

    val_pred = _unscale_target(model.predict(x_val, verbose=0), target_scaler)
    metrics = {"validation": calculate_metrics(val_df[target_col], val_pred, threshold=direction_threshold)}
    if not test_df.empty:
        test_pred = _unscale_target(model.predict(x_test, verbose=0), target_scaler)
        metrics["test"] = calculate_metrics(test_df[target_col], test_pred, threshold=direction_threshold)
    else:
        metrics["test"] = {"n_observations": 0}

    weights, bias = model.layers[-1].get_weights()
    target_std = target_scaler["std"]
    target_mean = target_scaler["mean"]
    raw_formula_scaled = feature_scaler.inverse_transform_coef(weights, float(bias[0]))
    raw_formula = {
        "raw_intercept": float(target_mean + target_std * raw_formula_scaled["raw_intercept"]),
        "raw_betas": {name: float(target_std * beta) for name, beta in raw_formula_scaled["raw_betas"].items()},
    }

    metadata = {
        "model_type": "tf_keras_linear_bar_state_regression",
        "feature_cols": feature_cols,
        "feature_names": feature_cols,
        "target_col": target_col,
        "target_name": target_col,
        "target_symbol": dataset.attrs.get("target_symbol"),
        "source_symbols": dataset.attrs.get("source_symbols"),
        "horizon_bars": dataset.attrs.get("horizon_bars"),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "random_seed": random_seed,
        "direction_threshold": direction_threshold,
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "target_scaler": target_scaler,
        "raw_formula": raw_formula,
    }

    feature_scaler.save(out / "scaler.json")
    (out / "target_scaler.json").write_text(json.dumps(target_scaler, indent=2, sort_keys=True), encoding="utf-8")
    save_json(metadata, out / "metadata.json")
    save_json(metrics, out / "metrics.json")
    save_history_csv(out / "training_history.csv", history.history)
    _write_split_csv(train_df, out / "train_split.csv")
    _write_split_csv(val_df, out / "validation_split.csv")
    _write_split_csv(test_df, out / "test_split.csv")

    return {"model_dir": str(out), "metadata": metadata, "metrics": metrics}
