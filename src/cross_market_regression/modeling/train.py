"""Model training orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import subprocess

from cross_market_regression.config import CrossMarketConfig, ModelConfig
from cross_market_regression.data.dataset import build_dataset
from cross_market_regression.data.dataset_builder import load_all_configured_data
from cross_market_regression.data.registry import default_registry
from cross_market_regression.features.feature_builder import build_supervised_dataset
from cross_market_regression.features.scalers import StandardScaler1D
from cross_market_regression.features.scaling import StandardScaler
from cross_market_regression.features.supervised import split_xy

from .linear_tf_model import build_linear_model
from .metrics import calculate_metrics, regression_metrics
from .persistence import ensure_model_dir, save_history_csv, save_json, save_training_artifacts


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _split_by_dates(dataset, config: ModelConfig):
    import pandas as pd

    df = dataset.sort_values("target_next_date").reset_index(drop=True)
    dates = pd.to_datetime(df["target_next_date"])
    train_mask = pd.Series(True, index=df.index)
    if config.train_start:
        train_mask &= dates >= pd.Timestamp(config.train_start)
    if config.train_end:
        train_mask &= dates <= pd.Timestamp(config.train_end)
    val_mask = pd.Series(False, index=df.index)
    if config.validation_start:
        val_mask = dates >= pd.Timestamp(config.validation_start)
        if config.validation_end:
            val_mask &= dates <= pd.Timestamp(config.validation_end)
        if config.test_start:
            val_mask &= dates < pd.Timestamp(config.test_start)
    test_mask = pd.Series(False, index=df.index)
    if config.test_start:
        test_mask = dates >= pd.Timestamp(config.test_start)
        if config.test_end:
            test_mask &= dates <= pd.Timestamp(config.test_end)
    if not val_mask.any() and not test_mask.any():
        split = max(1, int(len(df) * (1.0 - config.validation_fraction)))
        train_mask = pd.Series([idx < split for idx in range(len(df))])
        val_mask = ~train_mask
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def train_cross_market_regression(dataset, feature_names: list[str], target_name: str, config: ModelConfig) -> dict:
    """Train, evaluate, and save a TensorFlow linear cross-market regression."""

    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.random_seed)
    train_df, val_df, test_df = _split_by_dates(dataset, config)
    if train_df.empty:
        raise ValueError("Training split is empty")
    eval_df = val_df if not val_df.empty else train_df
    scaler = StandardScaler1D(feature_names=feature_names)
    x_train = scaler.fit_transform(train_df[feature_names]) if config.standardize else train_df[feature_names].to_numpy(dtype=float)
    x_val = scaler.transform(eval_df[feature_names]) if config.standardize else eval_df[feature_names].to_numpy(dtype=float)
    y_train = train_df[target_name].astype(float).to_numpy()
    y_val = eval_df[target_name].astype(float).to_numpy()
    model = build_linear_model(len(feature_names), config.learning_rate)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config.epochs, batch_size=config.batch_size, shuffle=False, verbose=0)
    val_pred = model.predict(x_val, verbose=0).reshape(-1)
    metrics = calculate_metrics(y_val, val_pred, threshold=config.direction_threshold)
    weights, bias = model.layers[-1].get_weights()
    raw_formula = scaler.inverse_transform_coef(weights, float(bias[0])) if config.standardize else {
        "raw_intercept": float(bias[0]),
        "raw_betas": {name: float(weights[idx][0]) for idx, name in enumerate(feature_names)},
    }
    metadata = {
        "model_name": config.model_name,
        "model_type": "tf_keras_linear_regression",
        "feature_names": feature_names,
        "target_name": target_name,
        "source_assets": {},
        "target_asset": {},
        "fx_assets": {},
        "return_type": "simple",
        "alignment": {},
        "train_start": config.train_start,
        "train_end": config.train_end,
        "validation_start": config.validation_start,
        "validation_end": config.validation_end,
        "test_start": config.test_start,
        "test_end": config.test_end,
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "direction_threshold": config.direction_threshold,
        "learning_rate": config.learning_rate,
        "model_config": asdict(config),
        "package_version": "0.1.0",
        "git_commit": _git_commit(),
        "raw_formula": raw_formula,
    }
    save_training_artifacts(model, scaler, metadata, metrics, history, config.model_dir)
    return {"model_dir": config.model_dir, "metadata": metadata, "metrics": metrics}


def train_from_config(config: CrossMarketConfig, output_dir: str | None = None, registry=None) -> dict:
    """Load configured data, build supervised rows, and train the canonical model."""

    model_config = replace(config.model, model_dir=output_dir) if output_dir else config.model
    configured = load_all_configured_data(config, registry or default_registry())
    dataset = build_supervised_dataset(configured["sources"], configured["target"], configured["fx"], config)
    return train_cross_market_regression(dataset, model_config.feature_names, config.target.effective_name, model_config)


def train_model(config: CrossMarketConfig, output_dir: str) -> dict[str, object]:
    """Backward-compatible train helper for the initial list-of-rows pipeline."""

    rows = build_dataset(config)
    feature_names = [feature.name for feature in config.features]
    label = config.target.effective_name
    x, y = split_xy(rows, feature_names, label)
    split = max(1, int(len(x) * (1.0 - config.model.validation_fraction)))
    x_train, y_train = x[:split], y[:split]
    x_eval, y_eval = x[split:] or x_train, y[split:] or y_train
    scaler = StandardScaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)

    model = build_linear_model(len(feature_names), config.model.learning_rate)
    history = model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_eval_scaled, y_eval),
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        shuffle=False,
        verbose=0,
    )
    predictions = [float(value[0]) for value in model.predict(x_eval_scaled, verbose=0)]
    metrics = regression_metrics(y_eval, predictions)

    out = ensure_model_dir(output_dir)
    model.save_weights(str(out / "model.weights.h5"))
    scaler.save(out / "scaler.json")
    save_json({"config_name": config.name, "feature_names": feature_names, "label": label}, out / "metadata.json")
    save_json(metrics, out / "metrics.json")
    save_history_csv(out / "training_history.csv", history.history)
    return {"rows": len(rows), "metrics": metrics, "output_dir": str(Path(output_dir))}
