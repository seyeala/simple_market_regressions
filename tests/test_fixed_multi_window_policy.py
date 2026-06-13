import numpy as np
import tensorflow as tf

from cross_market_regression.modeling.fixed_multi_window_policy import (
    FixedMultiWindowPolicyConfig,
    FixedMultiWindowUtilityPolicy,
)


def _ohlcv(batch=2, bars=60):
    base = np.linspace(100.0, 110.0, bars, dtype=np.float32)
    x = np.zeros((batch, bars, 5), dtype=np.float32)
    x[:, :, 0] = base - 0.2
    x[:, :, 1] = base + 0.5
    x[:, :, 2] = base - 0.5
    x[:, :, 3] = base
    x[:, :, 4] = 1000.0
    return x


def test_policy_has_one_trainable_theta_vector():
    model = FixedMultiWindowUtilityPolicy()

    assert len(model.trainable_variables) == 1
    assert model.theta.shape == (12,)
    assert model.count_trainable_parameters() == 12


def test_psi_logits_and_probabilities_have_expected_shapes():
    model = FixedMultiWindowUtilityPolicy()
    inputs = {"ohlcv": _ohlcv(batch=3), "current_position": [0.0, 1.0, -1.0], "time_to_close": [1.0, 0.5, 0.1]}

    psi = model.psi(inputs)
    logits = model.logits(inputs)
    probabilities = model.probabilities(inputs)

    assert psi.shape == (3, 15, 12)
    assert logits.shape == (3, 15)
    assert probabilities.shape == (3, 15)
    np.testing.assert_allclose(tf.reduce_sum(probabilities, axis=-1).numpy(), np.ones(3), rtol=1e-6)


def test_legal_action_mask_removes_position_increasing_actions():
    model = FixedMultiWindowUtilityPolicy()
    inputs = {"ohlcv": _ohlcv(batch=1), "current_position": [1.0], "time_to_close": [1.0]}

    probabilities = model.probabilities(inputs).numpy()[0]
    mask = model.legal_action_mask(inputs).numpy()[0]

    assert not mask[5]  # BUY_MARKET + NO_LIMIT would exceed +1
    assert probabilities[5] == 0.0


def test_custom_feature_contract_indices_are_supported():
    config = FixedMultiWindowPolicyConfig()
    model = FixedMultiWindowUtilityPolicy(config)
    state = model.state_vector(_ohlcv(batch=2), [0.0, 0.5], [1.0, 0.25])

    assert state.shape == (2, 12)
    np.testing.assert_allclose(state.numpy()[:, 10], [0.0, 0.5])
    np.testing.assert_allclose(state.numpy()[:, 11], [1.0, 0.25])
import importlib.util

import numpy as np
import pytest

from cross_market_regression.modeling.trading_train import TradingTrainConfig, build_trading_model


requires_tensorflow = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")


@requires_tensorflow
def test_build_trading_model_dispatches_to_dense_logits():
    model = build_trading_model(TradingTrainConfig(model_type="dense_logits"), ["x", "y"])

    output = model(np.zeros((3, 2), dtype=np.float32), training=False)

    assert output.shape == (3, 15)


@requires_tensorflow
def test_build_trading_model_dispatches_to_configured_fixed_policy():
    feature_names = []
    for bar in range(60):
        for field in ("open", "high", "low", "close", "volume"):
            feature_names.append(f"bar_{bar}_{field}")
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
