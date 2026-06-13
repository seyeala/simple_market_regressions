"""Trading target utilities for a 15-action spot/limit action space.

The functions in this module are intentionally TensorFlow-first so they can be
used while building differentiable targets for Keras models.  Batches are plain
mappings (or objects with matching attributes) containing at least
``current_pos``, ``close_t``, ``close_next``, ``low_next`` and ``high_next``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import tensorflow as tf


@dataclass(frozen=True)
class ActionSpace15:
    """Immutable container for the canonical 15 spot/limit actions."""

    spot_delta: Any
    limit_delta: Any
    limit_offset: Any
    labels: tuple[str, ...] = ()

    def as_tensors(self, dtype: tf.dtypes.DType = tf.float32) -> "ActionSpace15":
        return ActionSpace15(
            spot_delta=tf.convert_to_tensor(self.spot_delta, dtype=dtype),
            limit_delta=tf.convert_to_tensor(self.limit_delta, dtype=dtype),
            limit_offset=tf.convert_to_tensor(self.limit_offset, dtype=dtype),
            labels=self.labels,
        )


def make_15_action_space(
    buy_offsets: Sequence[float] = (0.001, 0.003),
    sell_offsets: Sequence[float] = (0.001, 0.003),
) -> ActionSpace15:
    """Create the Cartesian product of 3 spot actions and 5 limit actions."""

    spot_actions = (("NEUTRAL", 0), ("BUY_MARKET", 1), ("SELL_MARKET", -1))
    limit_actions = [("NO_LIMIT", 0, 0.0)]
    limit_actions += [(f"BUY_LIMIT_x{i + 1}", 1, float(offset)) for i, offset in enumerate(buy_offsets)]
    limit_actions += [(f"SELL_LIMIT_y{i + 1}", -1, float(offset)) for i, offset in enumerate(sell_offsets)]

    if len(limit_actions) != 5:
        raise ValueError("buy_offsets and sell_offsets must create exactly four limit actions")

    spot_delta: list[float] = []
    limit_delta: list[float] = []
    limit_offset: list[float] = []
    labels: list[str] = []
    for spot_label, spot in spot_actions:
        for limit_label, limit, offset in limit_actions:
            spot_delta.append(float(spot))
            limit_delta.append(float(limit))
            limit_offset.append(float(offset))
            labels.append(f"{spot_label}+{limit_label}")

    return ActionSpace15(spot_delta, limit_delta, limit_offset, tuple(labels)).as_tensors()


def relu_step(x: Any, sharpness: Any) -> tf.Tensor:
    """Continuous clipped ramp approximation to a step function."""

    return tf.clip_by_value(tf.convert_to_tensor(x) * sharpness + 0.5, 0.0, 1.0)


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping) and name in source:
        return source[name]
    return getattr(source, name, default)


def _cfg(config: Any, name: str, default: Any) -> Any:
    return _get(config, name, default)


def _col(x: Any, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    t = tf.convert_to_tensor(x, dtype=dtype)
    return tf.reshape(t, [-1, 1])


def _action_tensors(actions: ActionSpace15, dtype: tf.dtypes.DType = tf.float32) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    a = actions.as_tensors(dtype)
    return tf.reshape(a.spot_delta, [1, -1]), tf.reshape(a.limit_delta, [1, -1]), tf.reshape(a.limit_offset, [1, -1])


def simulate_one_step_15_actions(batch: Any, actions: ActionSpace15, config: Any | None = None) -> dict[str, tf.Tensor]:
    """Simulate one next bar for every action and return rewards/masks.

    Reward is marked-to-close: existing inventory plus any market/filled limit
    delta earns from its execution price to ``close_next``.  Transaction costs
    are included exactly once here.
    """

    config = config or {}
    dtype = tf.float32
    current_pos = _col(_get(batch, "current_pos"), dtype)
    close_t = _col(_get(batch, "close_t", _get(batch, "close")), dtype)
    close_next = _col(_get(batch, "close_next"), dtype)
    low_next = _col(_get(batch, "low_next"), dtype)
    high_next = _col(_get(batch, "high_next"), dtype)

    spot_delta, limit_delta, limit_offset = _action_tensors(actions, dtype)
    eps = tf.cast(_cfg(config, "epsilon", 1e-8), dtype)
    fee_rate = tf.cast(_cfg(config, "fee_rate", 0.0), dtype)
    slippage_rate = tf.cast(_cfg(config, "slippage_rate", 0.0), dtype)
    x_cost = tf.cast(_cfg(config, "x_cost", 1.0), dtype)
    sharpness = tf.cast(_cfg(config, "sharpness", 100.0), dtype)
    soft_fill = bool(_cfg(config, "soft_fill", _cfg(config, "fill_mode", "hard") == "soft"))

    post_spot_pos = current_pos + spot_delta
    legal_mask = tf.logical_and(post_spot_pos >= -1.0, post_spot_pos <= 1.0)
    final_if_filled = post_spot_pos + limit_delta
    legal_mask = tf.logical_and(legal_mask, tf.logical_and(final_if_filled >= -1.0, final_if_filled <= 1.0))

    buy_limit_price = close_t * (1.0 - limit_offset)
    sell_limit_price = close_t * (1.0 + limit_offset)
    limit_price = tf.where(limit_delta > 0.0, buy_limit_price, tf.where(limit_delta < 0.0, sell_limit_price, close_t))

    buy_fill_raw = (limit_price - low_next) / (close_t + eps)
    sell_fill_raw = (high_next - limit_price) / (close_t + eps)
    if soft_fill:
        buy_fill = relu_step(buy_fill_raw, sharpness)
        sell_fill = relu_step(sell_fill_raw, sharpness)
    else:
        buy_fill = tf.cast(low_next <= limit_price, dtype)
        sell_fill = tf.cast(high_next >= limit_price, dtype)
    fill = tf.where(limit_delta > 0.0, buy_fill, tf.where(limit_delta < 0.0, sell_fill, tf.zeros_like(limit_delta + close_t)))
    fill = tf.where(legal_mask, fill, tf.zeros_like(fill))

    executed_limit_delta = limit_delta * fill
    next_pos = post_spot_pos + executed_limit_delta
    spot_cost = x_cost * (fee_rate + slippage_rate) * tf.abs(spot_delta) * close_t
    limit_cost = x_cost * (fee_rate + slippage_rate) * tf.abs(executed_limit_delta) * limit_price
    reward = (
        current_pos * (close_next - close_t)
        + spot_delta * (close_next - close_t)
        + executed_limit_delta * (close_next - limit_price)
        - spot_cost
        - limit_cost
    )
    reward = tf.where(legal_mask, reward, tf.zeros_like(reward))

    return {"reward": reward, "next_pos": next_pos, "legal_mask": legal_mask, "fill": fill, "limit_price": limit_price}


def one_step_targets_15(batch: Any, actions: ActionSpace15, config: Any | None = None) -> dict[str, tf.Tensor]:
    """Build one-step rewards and legal masks for the 15-action space."""

    return simulate_one_step_15_actions(batch, actions, config)


def soft_best(q_values: Any, legal_mask: Any, eta: Any) -> tf.Tensor:
    """Return a masked soft maximum; as ``eta`` grows this approaches max."""

    q = tf.convert_to_tensor(q_values, dtype=tf.float32)
    mask = tf.convert_to_tensor(legal_mask, dtype=tf.bool)
    eta = tf.cast(eta, q.dtype)
    neg_inf = tf.constant(-1.0e9, dtype=q.dtype)
    masked_q = tf.where(mask, q, neg_inf)
    hard = tf.reduce_max(masked_q, axis=-1)
    weights = tf.nn.softmax(masked_q * eta, axis=-1)
    soft = tf.reduce_sum(tf.where(mask, weights * q, tf.zeros_like(q)), axis=-1)
    return tf.where(eta > 0.0, soft, hard)


def n_step_soft_dp_targets_15(batches: Sequence[Any], actions: ActionSpace15, config: Any | None = None, bootstrap_q: Any | None = None) -> tf.Tensor:
    """Compute backward n-step soft-DP targets for a sequence of batches."""

    config = config or {}
    gamma = tf.cast(_cfg(config, "gamma", 1.0), tf.float32)
    eta = tf.cast(_cfg(config, "eta", 10.0), tf.float32)
    value = None if bootstrap_q is None else tf.convert_to_tensor(bootstrap_q, tf.float32)
    targets = None
    for batch in reversed(tuple(batches)):
        sim = simulate_one_step_15_actions(batch, actions, config)
        reward = sim["reward"]
        if value is None:
            targets = reward
        else:
            targets = reward + gamma * tf.reshape(value, [-1, 1])
        value = soft_best(targets, sim["legal_mask"], eta)
    if targets is None:
        raise ValueError("batches must contain at least one batch")
    return targets


def loss_from_targets(q_values: Any, targets: Any, legal_mask: Any | None = None, reduction: str = "mean") -> tf.Tensor:
    """Squared-error loss against pre-costed targets; no costs are subtracted."""

    q = tf.convert_to_tensor(q_values, dtype=tf.float32)
    y = tf.stop_gradient(tf.convert_to_tensor(targets, dtype=tf.float32))
    err = tf.square(q - y)
    if legal_mask is not None:
        mask = tf.cast(tf.convert_to_tensor(legal_mask, dtype=tf.bool), err.dtype)
        err = err * mask
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
    else:
        denom = tf.cast(tf.size(err), err.dtype)
    if reduction == "none":
        return err
    if reduction == "sum":
        return tf.reduce_sum(err)
    if reduction != "mean":
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    return tf.reduce_sum(err) / denom
