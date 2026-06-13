"""Separate training entry point for 15-action trading models.

This module intentionally does not change the existing scalar regression
training path in :mod:`cross_market_regression.modeling.train`.  It trains a
model whose network inputs are historical/current features only, while future
OHLC values are read separately to construct trading targets and losses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cross_market_regression.features.scalers import StandardScaler1D

from .persistence import ensure_model_dir, save_history_csv, save_json
from .train import _git_commit, _split_by_dates
from .trading_losses import make_15_action_space, n_step_soft_dp_targets_15, one_step_targets_15, loss_from_targets


_FUTURE_OHLC_TOKENS = ("close_next", "open_next", "high_next", "low_next", "future_open", "future_high", "future_low", "future_close")


@dataclass(frozen=True)
class TradingTrainConfig:
    """Configuration for the separate 15-action trading trainer."""

    model_dir: str = "models/trading_model"
    model_name: str = "trading_15_action_model"
    train_start: str | None = None
    train_end: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    validation_fraction: float = 0.2
    standardize: bool = True
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    random_seed: int = 42
    buy_offsets: tuple[float, float] = (0.001, 0.003)
    sell_offsets: tuple[float, float] = (0.001, 0.003)
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    cost_stress_multiplier: float = 1.0
    fill_mode: str = "hard"
    soft_fill_sharpness: float = 100.0
    gamma: float = 1.0
    eta: float = 10.0
    beta_values: tuple[float, ...] = field(default_factory=lambda: (1.0,))
    edge_margin: float = 0.0
    ce_weight: float = 0.0
    n_step_horizon: int = 1


def build_trading_logits_model(input_dim: int, learning_rate: float = 0.001):
    """Build a simple Keras model that outputs logits/Q-values with shape ``[B, 15]``."""

    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    outputs = tf.keras.layers.Dense(15, name="action_logits")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def _validate_no_future_features(feature_names: Sequence[str]) -> None:
    bad = [name for name in feature_names if any(token in name.lower() for token in _FUTURE_OHLC_TOKENS)]
    if bad:
        raise ValueError(f"Future OHLC fields are not allowed in feature tensors: {bad}")


def _require_columns(frame: Any, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in frame]
    if missing:
        raise ValueError(f"Missing required trading training columns: {missing}")


def _series_array(frame: Any, name: str, indices: np.ndarray) -> np.ndarray:
    return frame.iloc[indices][name].astype(float).to_numpy()


def _batch_for_step(frame: Any, indices: np.ndarray, step: int) -> dict[str, np.ndarray]:
    suffix = "" if step == 1 else f"_{step}"
    close_next = f"close_next{suffix}"
    high_next = f"high_next{suffix}"
    low_next = f"low_next{suffix}"
    close_t = "close_t" if step == 1 else f"close_t_{step}"
    if close_t not in frame and step > 1:
        close_t = f"close_next_{step - 1}" if step > 2 else "close_next"
    _require_columns(frame, ["current_pos", close_t, close_next, high_next, low_next])
    return {
        "current_pos": _series_array(frame, "current_pos", indices),
        "close_t": _series_array(frame, close_t, indices),
        "close_next": _series_array(frame, close_next, indices),
        "high_next": _series_array(frame, high_next, indices),
        "low_next": _series_array(frame, low_next, indices),
    }


def _target_config(config: TradingTrainConfig) -> dict[str, float | str | bool]:
    return {
        "fee_rate": config.fee_rate,
        "slippage_rate": config.slippage_rate,
        "x_cost": config.cost_stress_multiplier,
        "fill_mode": config.fill_mode,
        "soft_fill": config.fill_mode == "soft",
        "sharpness": config.soft_fill_sharpness,
        "gamma": config.gamma,
        "eta": config.eta,
    }


def _compute_targets(frame: Any, indices: np.ndarray, actions: Any, config: TradingTrainConfig):
    horizon = int(config.n_step_horizon)
    if horizon < 1:
        raise ValueError("n_step_horizon must be at least 1")
    batches = [_batch_for_step(frame, indices, step) for step in range(1, horizon + 1)]
    if horizon == 1:
        result = one_step_targets_15(batches[0], actions, _target_config(config))
        return result["reward"], result["legal_mask"]
    targets = n_step_soft_dp_targets_15(batches, actions, _target_config(config))
    first = one_step_targets_15(batches[0], actions, _target_config(config))
    return targets, first["legal_mask"]


def _assert_logits_shape(model: Any, sample_x: np.ndarray) -> None:
    output_shape = tuple(model(sample_x[: min(len(sample_x), 2)], training=False).shape)
    if len(output_shape) != 2 or output_shape[-1] != 15:
        raise ValueError(f"Trading model must output logits/Q-values with shape [B, 15], got {output_shape}")


def _split_trading_dataset(dataset: Any, config: TradingTrainConfig):
    if "target_next_date" in dataset:
        return _split_by_dates(dataset, config)  # type: ignore[arg-type]
    split = max(1, int(len(dataset) * (1.0 - config.validation_fraction)))
    return dataset.iloc[:split].copy(), dataset.iloc[split:].copy(), dataset.iloc[0:0].copy()


def train_trading_model(dataset: Any, feature_names: list[str], config: TradingTrainConfig, model: Any | None = None) -> dict[str, Any]:
    """Train and save a 15-action trading model without feature lookahead leakage.

    ``dataset`` must contain feature columns, ``current_pos``, ``close_t``, and
    future OHLC target columns named ``close_next``, ``high_next``, ``low_next``
    for one-step training.  For N-step training, add ``*_next_2`` ...
    ``*_next_N`` columns; ``close_t_k`` is optional because the previous close
    column is used as the next step's current close when absent.
    """

    import tensorflow as tf

    _validate_no_future_features(feature_names)
    tf.keras.utils.set_random_seed(config.random_seed)
    actions = make_15_action_space(config.buy_offsets, config.sell_offsets)
    train_df, val_df, test_df = _split_trading_dataset(dataset, config)
    if train_df.empty:
        raise ValueError("Training split is empty")
    eval_df = val_df if not val_df.empty else train_df

    scaler = StandardScaler1D(feature_names=feature_names)
    if config.standardize:
        x_train = scaler.fit_transform(train_df[feature_names])
        x_eval = scaler.transform(eval_df[feature_names])
    else:
        scaler.fit_identity(feature_names)
        x_train = train_df[feature_names].to_numpy(dtype=float)
        x_eval = eval_df[feature_names].to_numpy(dtype=float)

    model = model or build_trading_logits_model(len(feature_names), config.learning_rate)
    if not hasattr(model, "optimizer") or model.optimizer is None:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate))
    _assert_logits_shape(model, x_train.astype("float32"))

    history = {"loss": [], "val_loss": []}
    for _epoch in range(config.epochs):
        losses: list[float] = []
        for start in range(0, len(train_df), config.batch_size):
            stop = min(start + config.batch_size, len(train_df))
            indices = np.arange(start, stop)
            xb = tf.convert_to_tensor(x_train[indices], dtype=tf.float32)
            targets, legal_mask = _compute_targets(train_df, indices, actions, config)
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss = loss_from_targets(logits, targets, legal_mask)
                if config.ce_weight:
                    beta = float(config.beta_values[0]) if config.beta_values else 1.0
                    labels = tf.nn.softmax((tf.stop_gradient(targets) - config.edge_margin) * beta, axis=-1)
                    loss = loss + config.ce_weight * tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))  # type: ignore[union-attr]
            losses.append(float(loss.numpy()))
        eval_targets, eval_mask = _compute_targets(eval_df, np.arange(len(eval_df)), actions, config)
        val_loss = loss_from_targets(model(tf.convert_to_tensor(x_eval, tf.float32), training=False), eval_targets, eval_mask)
        history["loss"].append(float(np.mean(losses)))
        history["val_loss"].append(float(val_loss.numpy()))

    metrics = {"loss": history["loss"][-1], "val_loss": history["val_loss"][-1]}
    metadata = {
        "model_name": config.model_name,
        "model_type": "tf_keras_trading_15_action_logits",
        "feature_names": feature_names,
        "action_offsets": {"buy_offsets": list(config.buy_offsets), "sell_offsets": list(config.sell_offsets)},
        "fee_rate": config.fee_rate,
        "slippage_rate": config.slippage_rate,
        "cost_stress_multiplier": config.cost_stress_multiplier,
        "fill_mode": config.fill_mode,
        "soft_fill_sharpness": config.soft_fill_sharpness,
        "gamma": config.gamma,
        "eta": config.eta,
        "beta_values": list(config.beta_values),
        "edge_margin": config.edge_margin,
        "ce_weight": config.ce_weight,
        "n_step_horizon": config.n_step_horizon,
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "model_config": asdict(config),
        "package_version": "0.1.0",
        "git_commit": _git_commit(),
    }
    out = ensure_model_dir(config.model_dir)
    model.save_weights(str(out / "model.weights.h5"))
    scaler.save(out / "scaler.json")
    save_json(metadata, out / "metadata.json")
    save_json(metrics, out / "metrics.json")
    save_history_csv(out / "training_history.csv", history)
    return {"model_dir": str(Path(config.model_dir)), "metadata": metadata, "metrics": metrics}
