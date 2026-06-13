import importlib.util

import numpy as np
import pytest
import tensorflow as tf

from cross_market_regression.modeling.fixed_multi_window_policy import (
    FixedMultiWindowPolicyConfig,
    FixedMultiWindowUtilityPolicy,
)
from cross_market_regression.modeling.trading_train import TradingTrainConfig, build_trading_model


requires_tensorflow = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")


def _ohlcv(batch=2, bars=60, fields=5):
    base = np.linspace(100.0, 110.0, bars, dtype=np.float32)
    x = np.zeros((batch, bars, fields), dtype=np.float32)
    x[:, :, 0] = base - 0.2
    x[:, :, 1] = base + 0.5
    x[:, :, 2] = base - 0.5
    x[:, :, 3] = base
    x[:, :, 4] = 1000.0
    return x


def _flat_feature_names(bars=60, fields=("open", "high", "low", "close", "volume")):
    return [f"bar_{bar}_{field}" for bar in range(bars) for field in fields]


@requires_tensorflow
def test_policy_logits_have_15_action_shape():
    model = FixedMultiWindowUtilityPolicy()
    inputs = {"ohlcv": _ohlcv(batch=3), "current_position": [0.0, 1.0, -1.0], "time_to_close": [1.0, 0.5, 0.1]}

    logits = model.logits(inputs)

    assert logits.shape == (3, 15)


@requires_tensorflow
def test_policy_probabilities_sum_to_one_across_legal_actions():
    model = FixedMultiWindowUtilityPolicy()
    inputs = {"ohlcv": _ohlcv(batch=2), "current_position": [1.0, -1.0], "time_to_close": [1.0, 0.5]}

    probabilities = model.probabilities(inputs).numpy()
    legal_mask = model.legal_action_mask(inputs).numpy()

    np.testing.assert_allclose(np.sum(np.where(legal_mask, probabilities, 0.0), axis=-1), np.ones(2), rtol=1e-6)


@requires_tensorflow
def test_illegal_actions_receive_zero_probability():
    model = FixedMultiWindowUtilityPolicy()
    inputs = {"ohlcv": _ohlcv(batch=1), "current_position": [1.0], "time_to_close": [1.0]}

    probabilities = model.probabilities(inputs).numpy()[0]
    mask = model.legal_action_mask(inputs).numpy()[0]

    assert not mask[5]  # BUY_MARKET + NO_LIMIT would exceed +1
    assert probabilities[5] == 0.0
    np.testing.assert_allclose(probabilities[~mask], np.zeros(np.sum(~mask)), atol=0.0)


@requires_tensorflow
def test_policy_trainable_parameter_count_is_exactly_12():
    model = FixedMultiWindowUtilityPolicy()

    assert len(model.trainable_variables) == 1
    assert model.theta.shape == (12,)
    assert model.count_trainable_parameters() == 12


@requires_tensorflow
def test_future_padded_bars_do_not_affect_output_when_masked_out():
    model = FixedMultiWindowUtilityPolicy(FixedMultiWindowPolicyConfig(max_bars=65, windows=(2, 4, 8, 16, 32, 60, 65)))
    current_position = [0.0, 0.5]
    time_to_close = [1.0, 0.25]
    valid_bars = _ohlcv(batch=2, bars=60)
    padded_bars = np.concatenate([valid_bars, np.full((2, 5, 5), 999_999.0, dtype=np.float32)], axis=1)
    bar_mask = np.concatenate([np.ones((2, 60), dtype=bool), np.zeros((2, 5), dtype=bool)], axis=1)

    unpadded = model.logits({"ohlcv": valid_bars, "current_position": current_position, "time_to_close": time_to_close})
    padded = model.logits(
        {"ohlcv": padded_bars, "bar_mask": bar_mask, "current_position": current_position, "time_to_close": time_to_close}
    )

    np.testing.assert_allclose(padded.numpy(), unpadded.numpy(), rtol=1e-6, atol=1e-6)


@requires_tensorflow
def test_configurable_dimensions_windows_and_offsets_are_honored():
    config = FixedMultiWindowPolicyConfig(
        max_bars=65,
        windows=(2, 4, 8, 16, 32, 60, 65),
        buy_offsets=(0.002, 0.004),
        sell_offsets=(0.005, 0.007),
    )
    model = FixedMultiWindowUtilityPolicy(config)
    inputs = {"ohlcv": _ohlcv(batch=2, bars=70), "current_position": [0.0, 0.5], "time_to_close": [1.0, 0.25]}

    psi = model.psi(inputs)
    logits = model.logits(inputs)
    probabilities = model.probabilities(inputs)
    _, _, offsets = model._action_components(tf.float32)

    assert model.config.max_bars == 65
    assert 65 in model.config.windows
    assert psi.shape == (2, 15, 12)
    assert logits.shape == (2, 15)
    assert probabilities.shape == (2, 15)
    np.testing.assert_allclose(offsets.numpy()[1:5], [0.002, 0.004, 0.005, 0.007])


@requires_tensorflow
def test_custom_feature_contract_indices_are_supported():
    model = FixedMultiWindowUtilityPolicy(FixedMultiWindowPolicyConfig())
    state = model.state_vector(_ohlcv(batch=2), [0.0, 0.5], [1.0, 0.25])

    assert state.shape == (2, 12)
    np.testing.assert_allclose(state.numpy()[:, 10], [0.0, 0.5])
    np.testing.assert_allclose(state.numpy()[:, 11], [1.0, 0.25])


@requires_tensorflow
def test_build_trading_model_factory_preserves_dense_logits_behavior():
    model = build_trading_model(TradingTrainConfig(model_type="dense_logits"), ["x", "y"])

    output = model(np.zeros((3, 2), dtype=np.float32), training=False)

    assert output.shape == (3, 15)
    assert model.layers[-1].name == "action_logits"
    assert model.layers[-1].units == 15


@requires_tensorflow
def test_build_trading_model_factory_dispatches_to_configured_fixed_policy():
    feature_names = _flat_feature_names()
    feature_names.extend(["position_feature", "time_to_close_feature"])
    config = TradingTrainConfig(
        model_type="fixed_multi_window_utility_policy",
        fixed_policy_ohlcv_feature_names=tuple(feature_names[: 60 * 5]),
        fixed_policy_current_position_feature_name="position_feature",
        fixed_policy_time_to_close_feature_name="time_to_close_feature",
    )
    model = build_trading_model(config, feature_names)
    batch = np.ones((2, len(feature_names)), dtype=np.float32)
    batch[:, -2] = [0.0, 1.0]
    batch[:, -1] = [1.0, 0.5]

    output = model(batch, training=False)

    assert output.shape == (2, 15)
    assert len(model.trainable_variables) == 1
    assert model.trainable_variables[0].shape == (12,)


def test_fixed_policy_requires_configured_ohlcv_feature_mapping():
    config = TradingTrainConfig(model_type="fixed_multi_window_utility_policy")

    with pytest.raises(ValueError, match="fixed_policy_ohlcv_feature_names"):
        build_trading_model(config, ["x"])


def test_build_trading_model_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported trading model_type"):
        build_trading_model(TradingTrainConfig(model_type="unknown"), ["x"])
