"""Separate training entry point for 15-action trading models.

This module intentionally does not change the existing scalar regression
training path in :mod:`cross_market_regression.modeling.train`.  It trains a
model whose network inputs are historical/current features only, while future
OHLC values are read separately to construct trading targets and losses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    model_type: str = "dense_logits"
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
    fixed_policy_max_bars: int = 60
    fixed_policy_windows: tuple[int, ...] = (2, 4, 8, 16, 32, 60)
    fixed_policy_utility_dim: int = 12
    fixed_policy_epsilon: float = 1.0e-6
    fixed_policy_alpha: float = 1.0
    fixed_policy_kappa: float = 0.5
    fixed_policy_zeta: float = 0.25
    fixed_policy_tau_pi: float = 1.0
    fixed_policy_ohlcv_feature_names: tuple[str, ...] = ()
    fixed_policy_ohlcv_fields: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    fixed_policy_feature_contract: Mapping[str, int] = field(default_factory=lambda: {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4})
    fixed_policy_current_position_feature_name: str | None = None
    fixed_policy_time_to_close_feature_name: str | None = None


_SUPPORTED_MODEL_TYPES = {"dense_logits", "fixed_multi_window_utility_policy"}


def build_trading_logits_model(input_dim: int, learning_rate: float = 0.001):
    """Build a simple Keras model that outputs logits/Q-values with shape ``[B, 15]``."""

    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    outputs = tf.keras.layers.Dense(15, name="action_logits")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def _flat_feature_fixed_policy_model(
    policy: Any,
    ohlcv_indices: np.ndarray,
    num_ohlcv_fields: int,
    current_position_index: int | None,
    time_to_close_index: int | None,
):
    """Create a Keras adapter from flat feature rows to fixed-policy logits."""

    import tensorflow as tf

    class Adapter(tf.keras.Model):
        def __init__(self) -> None:
            super().__init__()
            self.policy = policy
            self._ohlcv_indices = tf.constant(ohlcv_indices, dtype=tf.int32)
            self._num_ohlcv_fields = int(num_ohlcv_fields)
            self._current_position_index = current_position_index
            self._time_to_close_index = time_to_close_index

        def call(self, inputs: Any, training: bool = False) -> Any:
            del training
            x = tf.convert_to_tensor(inputs, dtype=tf.float32)
            batch = tf.shape(x)[0]
            flat = tf.gather(x, self._ohlcv_indices, axis=1)
            ohlcv = tf.reshape(flat, [batch, policy.config.max_bars, self._num_ohlcv_fields])
            if self._current_position_index is None:
                current_position = tf.zeros([batch], dtype=x.dtype)
            else:
                current_position = x[:, self._current_position_index]
            if self._time_to_close_index is None:
                time_to_close = tf.ones([batch], dtype=x.dtype)
            else:
                time_to_close = x[:, self._time_to_close_index]
            return self.policy.logits(
                {"ohlcv": ohlcv, "current_position": current_position, "time_to_close": time_to_close}
            )

    return Adapter()


def _build_fixed_policy_config(config: TradingTrainConfig):
    from .fixed_multi_window_policy import FixedMultiWindowPolicyConfig, OHLCVFeatureContract

    contract_values = {
        field: int(config.fixed_policy_feature_contract[field])
        for field in ("open", "high", "low", "close", "volume")
    }
    num_fields = len(config.fixed_policy_ohlcv_fields)
    out_of_range = {field: index for field, index in contract_values.items() if index < 0 or index >= num_fields}
    if out_of_range:
        raise ValueError(f"fixed_policy_feature_contract indices must fit fixed_policy_ohlcv_fields: {out_of_range}")
    return FixedMultiWindowPolicyConfig(
        max_bars=config.fixed_policy_max_bars,
        num_actions=3 * (1 + len(config.buy_offsets) + len(config.sell_offsets)),
        buy_offsets=config.buy_offsets,
        sell_offsets=config.sell_offsets,
        windows=config.fixed_policy_windows,
        utility_dim=config.fixed_policy_utility_dim,
        epsilon=config.fixed_policy_epsilon,
        alpha=config.fixed_policy_alpha,
        kappa=config.fixed_policy_kappa,
        zeta=config.fixed_policy_zeta,
        tau_pi=config.fixed_policy_tau_pi,
        feature_contract=OHLCVFeatureContract(**contract_values),
    )


def _feature_index(feature_names: Sequence[str], name: str | None) -> int | None:
    if name is None:
        return None
    try:
        return feature_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Configured fixed-policy feature {name!r} is not present in feature_names") from exc


def _fixed_policy_ohlcv_indices(config: TradingTrainConfig, feature_names: Sequence[str]) -> np.ndarray:
    expected = config.fixed_policy_max_bars * len(config.fixed_policy_ohlcv_fields)
    names = tuple(config.fixed_policy_ohlcv_feature_names)
    if len(names) != expected:
        raise ValueError(
            "fixed_policy_ohlcv_feature_names must contain "
            f"max_bars * len(fixed_policy_ohlcv_fields) names ({expected}), got {len(names)}"
        )
    return np.asarray([_feature_index(feature_names, name) for name in names], dtype=np.int32)


def build_trading_model(config: TradingTrainConfig, feature_names: Sequence[str]):
    """Build the configured 15-action trading model for ``feature_names``.

    ``dense_logits`` preserves the original single dense logits layer.
    ``fixed_multi_window_utility_policy`` adapts configured flat OHLCV feature
    columns into the reusable 12-weight fixed multi-window policy.
    """

    model_type = config.model_type
    if model_type == "dense_logits":
        return build_trading_logits_model(len(feature_names), config.learning_rate)
    if model_type == "fixed_multi_window_utility_policy":
        from .fixed_multi_window_policy import build_fixed_multi_window_policy

        policy = build_fixed_multi_window_policy(_build_fixed_policy_config(config))
        adapter = _flat_feature_fixed_policy_model(
            policy=policy,
            ohlcv_indices=_fixed_policy_ohlcv_indices(config, feature_names),
            num_ohlcv_fields=len(config.fixed_policy_ohlcv_fields),
            current_position_index=_feature_index(feature_names, config.fixed_policy_current_position_feature_name),
            time_to_close_index=_feature_index(feature_names, config.fixed_policy_time_to_close_feature_name),
        )
        import tensorflow as tf

        adapter.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate))
        return adapter
    raise ValueError(f"Unsupported trading model_type {model_type!r}; expected one of {sorted(_SUPPORTED_MODEL_TYPES)}")


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

    model = model or build_trading_model(config, feature_names)
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
        "model_type": config.model_type,
        "model_artifact_type": "tf_keras_trading_15_action_logits",
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
