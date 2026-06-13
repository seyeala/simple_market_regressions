"""Fixed multi-window TensorFlow/Keras utility policy.

This module implements a deliberately small policy head for the canonical
15-action trading space.  Market history is summarized by fixed causal
exponential windows; the only trainable parameter is a length-``utility_dim``
vector used to score deterministic action/state interaction features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import tensorflow as tf


@dataclass(frozen=True)
class OHLCVFeatureContract:
    """Column indices for OHLCV tensors passed as ``[batch, bars, features]``."""

    open: int = 0
    high: int = 1
    low: int = 2
    close: int = 3
    volume: int = 4


@dataclass(frozen=True)
class FixedMultiWindowPolicyConfig:
    """Configuration for :class:`FixedMultiWindowUtilityPolicy`."""

    max_bars: int = 60
    num_actions: int = 15
    buy_offsets: tuple[float, ...] = (0.001, 0.003)
    sell_offsets: tuple[float, ...] = (0.001, 0.003)
    windows: tuple[int, ...] = (2, 4, 8, 16, 32, 60)
    utility_dim: int = 12
    epsilon: float = 1.0e-6
    alpha: float = 1.0
    kappa: float = 0.5
    zeta: float = 0.25
    tau_pi: float = 1.0
    feature_contract: OHLCVFeatureContract = field(default_factory=OHLCVFeatureContract)

    def __post_init__(self) -> None:
        if self.max_bars <= 0:
            raise ValueError("max_bars must be positive")
        expected_actions = 3 * (1 + len(self.buy_offsets) + len(self.sell_offsets))
        if self.num_actions != expected_actions:
            raise ValueError(f"FixedMultiWindowUtilityPolicy requires num_actions={expected_actions} for configured offsets")
        required_windows = {4, 8, 16, 32, 60}
        if not required_windows.issubset(set(self.windows)):
            raise ValueError(f"windows must include {sorted(required_windows)}")
        if self.utility_dim != 12:
            raise ValueError("the fixed state/action contract requires utility_dim=12")
        if self.tau_pi <= 0.0:
            raise ValueError("tau_pi must be positive")


class FixedMultiWindowUtilityPolicy(tf.keras.Model):
    """Keras policy with fixed features and one trainable utility vector.

    Inputs may be either a raw OHLCV tensor with shape ``[B, T, F]`` or a mapping
    containing ``ohlcv``/``features`` plus optional ``current_position``,
    ``time_to_close`` and ``legal_mask`` entries.
    """

    def __init__(self, config: FixedMultiWindowPolicyConfig | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config or FixedMultiWindowPolicyConfig()
        self.theta = self.add_weight(
            name="theta",
            shape=(self.config.utility_dim,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        del training
        return self.probabilities(inputs)

    def logits(self, inputs: Any) -> tf.Tensor:
        """Return masked action logits with shape ``[B, 15]``."""

        psi = self.psi(inputs)
        logits = tf.einsum("bak,k->ba", psi, self.theta)
        legal_mask = self.legal_action_mask(inputs, batch_size=tf.shape(logits)[0])
        neg_inf = tf.constant(-1.0e9, dtype=logits.dtype)
        return tf.where(legal_mask, logits / tf.cast(self.config.tau_pi, logits.dtype), neg_inf)

    def probabilities(self, inputs: Any) -> tf.Tensor:
        """Return masked softmax action probabilities with shape ``[B, 15]``."""

        return tf.nn.softmax(self.logits(inputs), axis=-1)

    def psi(self, inputs: Any) -> tf.Tensor:
        """Return fixed action interaction features with shape ``[B, 15, 12]``."""

        ohlcv, current_position, time_to_close, bar_mask = self._unpack_inputs(inputs)
        g = self.state_vector(ohlcv, current_position, time_to_close, bar_mask=bar_mask)
        basis = self._action_basis(g.dtype)
        return g[:, tf.newaxis, :] * basis[tf.newaxis, :, :]

    def count_trainable_parameters(self) -> int:
        """Return the number of trainable scalar parameters."""

        return int(sum(tf.size(var).numpy() for var in self.trainable_variables))

    def state_vector(self, ohlcv: Any, current_position: Any, time_to_close: Any, bar_mask: Any | None = None) -> tf.Tensor:
        """Build ``g_t`` with the documented 12 fixed state features.

        ``bar_mask`` may mark padded/future bars as invalid. Invalid bars are
        excluded from the fixed windows and the state is read from the latest
        valid bar, so masked future padding cannot affect the policy output.
        """

        x = tf.convert_to_tensor(ohlcv, dtype=tf.float32)[:, -self.config.max_bars :, :]
        mask = None
        if bar_mask is not None:
            mask = tf.convert_to_tensor(bar_mask, dtype=tf.bool)[:, -self.config.max_bars :]
            x = tf.where(mask[:, :, tf.newaxis], x, tf.zeros_like(x))
        c = self.config.feature_contract
        open_ = x[:, :, c.open]
        high = x[:, :, c.high]
        low = x[:, :, c.low]
        close = x[:, :, c.close]
        volume = x[:, :, c.volume]
        eps = tf.cast(self.config.epsilon, x.dtype)

        prev_close = tf.concat([close[:, :1], close[:, :-1]], axis=1)
        returns = (close - prev_close) / (tf.abs(prev_close) + eps)
        intrabar = (close - open_) / (high - low + eps)
        energy = tf.math.log1p(tf.maximum(volume, 0.0)) * tf.abs(returns)

        last_close = self._last_value(close, mask)
        mu_4 = self._last_ewma(returns, 4, mask)
        mu_16 = self._last_ewma(returns, 16, mask)
        mu_60 = self._last_ewma(returns, 60, mask)
        i_8 = self._last_ewma(intrabar, 8, mask)
        i_32 = self._last_ewma(intrabar, 32, mask)
        sigma_4 = self._last_ew_std(returns, 4, mask)
        sigma_60 = self._last_ew_std(returns, 60, mask)
        r_16 = last_close / (self._last_ewma(close, 16, mask) + eps) - 1.0
        r_60 = last_close / (self._last_ewma(close, 60, mask) + eps) - 1.0
        e_16 = self._last_ewma(energy, 16, mask)
        e_60 = self._last_ewma(energy, 60, mask)
        pos = tf.reshape(tf.cast(current_position, x.dtype), [-1])
        ttc = tf.reshape(tf.cast(time_to_close, x.dtype), [-1])

        return tf.stack([mu_4, mu_16, mu_60, i_8, i_32, sigma_4 / (sigma_60 + eps), r_16, r_60, e_16, e_60, pos, ttc], axis=-1)

    def legal_action_mask(self, inputs: Any, batch_size: tf.Tensor | None = None) -> tf.Tensor:
        if isinstance(inputs, Mapping) and "legal_mask" in inputs:
            return tf.convert_to_tensor(inputs["legal_mask"], dtype=tf.bool)
        _, current_position, _, _ = self._unpack_inputs(inputs)
        pos = tf.reshape(tf.cast(current_position, tf.float32), [-1, 1])
        if batch_size is None:
            batch_size = tf.shape(pos)[0]
        spot, limit, _ = self._action_components(tf.float32)
        post_spot = pos + spot[tf.newaxis, :]
        final_pos = post_spot + limit[tf.newaxis, :]
        mask = tf.logical_and(post_spot >= -1.0, post_spot <= 1.0)
        mask = tf.logical_and(mask, tf.logical_and(final_pos >= -1.0, final_pos <= 1.0))
        return tf.reshape(mask, [batch_size, self.config.num_actions])

    def _last_ewma(self, values: tf.Tensor, window: int, mask: tf.Tensor | None = None) -> tf.Tensor:
        weighted = self._causal_exponential_window(values, window, mask)
        return self._last_value(weighted, mask)

    def _last_ew_std(self, values: tf.Tensor, window: int, mask: tf.Tensor | None = None) -> tf.Tensor:
        mean = self._causal_exponential_window(values, window, mask)
        var = self._causal_exponential_window(tf.square(values - mean), window, mask)
        return tf.sqrt(tf.maximum(self._last_value(var, mask), 0.0))

    def _last_value(self, values: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        if mask is None:
            return values[:, -1]
        lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        last_indices = tf.maximum(lengths - 1, 0)
        return tf.gather(values, last_indices, batch_dims=1)

    def _causal_exponential_window(self, values: tf.Tensor, window: int, mask: tf.Tensor | None = None) -> tf.Tensor:
        values = tf.convert_to_tensor(values, dtype=tf.float32)
        t = tf.shape(values)[1]
        idx = tf.range(t)
        age = idx[tf.newaxis, :] - idx[:, tf.newaxis]
        causal = age <= 0
        distance = tf.cast(-age, values.dtype)
        within_window = distance < tf.cast(window, values.dtype)
        weights = tf.exp(-distance / tf.cast(window, values.dtype))
        weights = tf.where(tf.logical_and(causal, within_window), weights, tf.zeros_like(weights))
        if mask is None:
            weights = weights / tf.maximum(tf.reduce_sum(weights, axis=-1, keepdims=True), self.config.epsilon)
            return tf.einsum("ts,bs->bt", weights, values)
        source_mask = tf.cast(mask, values.dtype)
        numerator = tf.einsum("ts,bs->bt", weights, values * source_mask)
        denominator = tf.einsum("ts,bs->bt", weights, source_mask)
        return numerator / tf.maximum(denominator, tf.cast(self.config.epsilon, values.dtype))

    def _unpack_inputs(self, inputs: Any) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor | None]:
        if isinstance(inputs, Mapping):
            ohlcv = inputs.get("ohlcv", inputs.get("features"))
            if ohlcv is None:
                raise ValueError("mapping inputs must contain 'ohlcv' or 'features'")
            batch = tf.shape(tf.convert_to_tensor(ohlcv))[0]
            current_position = inputs.get("current_position", inputs.get("position", tf.zeros([batch], tf.float32)))
            time_to_close = inputs.get("time_to_close", tf.ones([batch], tf.float32))
            bar_mask = inputs.get("bar_mask", inputs.get("mask"))
            return tf.convert_to_tensor(ohlcv, tf.float32), current_position, time_to_close, bar_mask
        ohlcv = tf.convert_to_tensor(inputs, tf.float32)
        batch = tf.shape(ohlcv)[0]
        return ohlcv, tf.zeros([batch], tf.float32), tf.ones([batch], tf.float32), None

    def _action_basis(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        spot, limit, offset = self._action_components(dtype)
        alpha = tf.cast(self.config.alpha, dtype)
        kappa = tf.cast(self.config.kappa, dtype)
        zeta = tf.cast(self.config.zeta, dtype)
        ones = tf.ones_like(spot)
        return tf.stack([
            ones + alpha * spot,
            ones + alpha * limit,
            ones + alpha * (spot + limit),
            ones + kappa * tf.abs(spot),
            ones + kappa * tf.abs(limit),
            ones + zeta * offset,
            ones + spot * limit,
            ones - spot * limit,
            ones + tf.sign(limit) * offset,
            ones - tf.sign(limit) * offset,
            ones + spot,
            ones + limit,
        ], axis=-1)

    def _action_components(self, dtype: tf.dtypes.DType) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        limit_template = [0.0] + [1.0] * len(self.config.buy_offsets) + [-1.0] * len(self.config.sell_offsets)
        offset_template = [0.0] + list(self.config.buy_offsets) + list(self.config.sell_offsets)
        spot = tf.constant([0.0] * len(limit_template) + [1.0] * len(limit_template) + [-1.0] * len(limit_template), dtype=dtype)
        limit = tf.constant(limit_template * 3, dtype=dtype)
        offset = tf.constant(offset_template * 3, dtype=dtype)
        return spot, limit, offset


def build_fixed_multi_window_policy(config: FixedMultiWindowPolicyConfig | None = None) -> FixedMultiWindowUtilityPolicy:
    """Convenience factory for the fixed utility policy."""

    return FixedMultiWindowUtilityPolicy(config=config)
