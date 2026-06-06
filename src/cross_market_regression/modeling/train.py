"""Model training orchestration."""

from __future__ import annotations

from pathlib import Path

from cross_market_regression.config import CrossMarketConfig
from cross_market_regression.data.dataset import build_dataset
from cross_market_regression.features.scaling import StandardScaler
from cross_market_regression.features.supervised import split_xy

from .metrics import regression_metrics
from .model import build_linear_model
from .persistence import ensure_output_dir, save_history_csv, save_json


def train_model(config: CrossMarketConfig, output_dir: str) -> dict[str, object]:
    rows = build_dataset(config)
    feature_names = [feature.name for feature in config.features]
    label = config.target.label if config.target else "target_next_return"
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
        verbose=0,
    )
    predictions = [float(value[0]) for value in model.predict(x_eval_scaled, verbose=0)]
    metrics = regression_metrics(y_eval, predictions)

    out = ensure_output_dir(output_dir)
    model.save_weights(str(out / "model.weights.h5"))
    scaler.save(out / "scaler.json")
    save_json(out / "metadata.json", {"config_name": config.name, "feature_names": feature_names, "label": label})
    save_json(out / "metrics.json", metrics)
    save_history_csv(out / "training_history.csv", history.history)
    return {"rows": len(rows), "metrics": metrics, "output_dir": str(Path(output_dir))}
