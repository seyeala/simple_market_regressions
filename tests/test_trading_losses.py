import math

import pytest

np = pytest.importorskip("numpy")
tf = pytest.importorskip("tensorflow")

from cross_market_regression.modeling.trading_losses import (
    UtilityPolicyLossConfig,
    loss_from_targets,
    make_15_action_space,
    n_step_soft_dp_targets_15,
    one_step_targets_15,
    soft_best,
    utility_policy_loss,
    masked_policy_probabilities,
)
from cross_market_regression.modeling.trading_train import _validate_no_future_features


def _batch(current_pos=0.0, close_t=100.0, close_next=101.0, low_next=99.0, high_next=102.0, b=1):
    return {
        "current_pos": np.full(b, current_pos, dtype=np.float32),
        "close_t": np.full(b, close_t, dtype=np.float32),
        "close_next": np.full(b, close_next, dtype=np.float32),
        "low_next": np.full(b, low_next, dtype=np.float32),
        "high_next": np.full(b, high_next, dtype=np.float32),
    }


def _idx(actions, label):
    return actions.labels.index(label)


def test_make_15_action_space_returns_15_actions():
    actions = make_15_action_space()

    assert len(actions.labels) == 15
    assert actions.spot_delta.shape == (15,)
    assert actions.limit_delta.shape == (15,)
    assert actions.limit_offset.shape == (15,)


@pytest.mark.parametrize(
    ("buy_offsets", "sell_offsets"),
    [((0.001,), (0.001, 0.003)), ((0.001, 0.003), (0.001,)), ((0.001, 0.002, 0.003), (0.001, 0.003))],
)
def test_wrong_offset_counts_raise(buy_offsets, sell_offsets):
    with pytest.raises(ValueError, match="exactly four limit actions"):
        make_15_action_space(buy_offsets=buy_offsets, sell_offsets=sell_offsets)


def test_legal_masks_reject_positions_outside_minus_one_plus_one():
    actions = make_15_action_space()
    result = one_step_targets_15(_batch(current_pos=1.0), actions)
    mask = result["legal_mask"].numpy()[0]

    assert not mask[_idx(actions, "BUY_MARKET+NO_LIMIT")]
    assert mask[_idx(actions, "SELL_MARKET+NO_LIMIT")]
    assert not mask[_idx(actions, "NEUTRAL+BUY_LIMIT_x1")]


def test_hard_buy_limits_fill_only_when_low_next_touches_limit_price():
    actions = make_15_action_space(buy_offsets=(0.01, 0.03), sell_offsets=(0.01, 0.03))

    touched = one_step_targets_15(_batch(low_next=99.0), actions, {"fill_mode": "hard"})["fill"].numpy()[0]
    missed = one_step_targets_15(_batch(low_next=99.01), actions, {"fill_mode": "hard"})["fill"].numpy()[0]

    buy_idx = _idx(actions, "NEUTRAL+BUY_LIMIT_x1")
    assert touched[buy_idx] == pytest.approx(1.0)
    assert missed[buy_idx] == pytest.approx(0.0)


def test_hard_sell_limits_fill_only_when_high_next_touches_limit_price():
    actions = make_15_action_space(buy_offsets=(0.01, 0.03), sell_offsets=(0.01, 0.03))

    touched = one_step_targets_15(_batch(high_next=101.0), actions, {"fill_mode": "hard"})["fill"].numpy()[0]
    missed = one_step_targets_15(_batch(high_next=100.99), actions, {"fill_mode": "hard"})["fill"].numpy()[0]

    sell_idx = _idx(actions, "NEUTRAL+SELL_LIMIT_y1")
    assert touched[sell_idx] == pytest.approx(1.0)
    assert missed[sell_idx] == pytest.approx(0.0)


def test_soft_relu_ramp_uses_clip_k_u_plus_half_behavior():
    actions = make_15_action_space(buy_offsets=(0.01, 0.03), sell_offsets=(0.01, 0.03))
    low_next = 99.25  # u = (99 - 99.25) / 100 == -0.0025
    sharpness = 100.0

    fill = one_step_targets_15(
        _batch(low_next=low_next), actions, {"fill_mode": "soft", "sharpness": sharpness}
    )["fill"].numpy()[0]

    expected = np.clip(sharpness * ((99.0 - low_next) / 100.0) + 0.5, 0.0, 1.0)
    assert fill[_idx(actions, "NEUTRAL+BUY_LIMIT_x1")] == pytest.approx(expected)


def test_rewards_subtract_transaction_costs_exactly_once():
    actions = make_15_action_space()
    result = one_step_targets_15(
        _batch(close_t=100.0, close_next=110.0),
        actions,
        {"fee_rate": 0.01, "slippage_rate": 0.02, "x_cost": 1.0},
    )

    buy_market_reward = result["reward"].numpy()[0, _idx(actions, "BUY_MARKET+NO_LIMIT")]
    assert buy_market_reward == pytest.approx(10.0 - 3.0)


def test_one_step_target_outputs_have_shape_b_15():
    actions = make_15_action_space()
    result = one_step_targets_15(_batch(b=4), actions)

    assert result["reward"].shape == (4, 15)
    assert result["legal_mask"].shape == (4, 15)


def test_soft_best_ignores_illegal_actions():
    q = tf.constant([[1.0, 1000.0, 3.0]])
    mask = tf.constant([[True, False, True]])

    value = soft_best(q, mask, eta=100.0).numpy()[0]

    assert value == pytest.approx(3.0)


def test_soft_best_matches_weighted_average_formula():
    q = tf.constant([[1.0, 2.0, 4.0]], dtype=tf.float32)
    mask = tf.constant([[True, False, True]])
    eta = 0.7

    value = soft_best(q, mask, eta=eta).numpy()[0]
    legal = np.array([1.0, 4.0])
    weights = np.exp(eta * legal) / np.exp(eta * legal).sum()
    expected = float((weights * legal).sum())

    assert value == pytest.approx(expected)


def test_n_step_horizon_one_equals_one_step_rewards():
    actions = make_15_action_space()
    batch = _batch(b=2)

    n_step = n_step_soft_dp_targets_15([batch], actions)
    one_step = one_step_targets_15(batch, actions)["reward"]

    np.testing.assert_allclose(n_step.numpy(), one_step.numpy())


def test_n_step_gamma_zero_equals_first_step_rewards():
    actions = make_15_action_space()
    first = _batch(close_next=101.0, b=2)
    second = _batch(close_t=101.0, close_next=200.0, low_next=100.0, high_next=201.0, b=2)

    n_step = n_step_soft_dp_targets_15([first, second], actions, {"gamma": 0.0})
    one_step = one_step_targets_15(first, actions, {"gamma": 0.0})["reward"]

    np.testing.assert_allclose(n_step.numpy(), one_step.numpy())


def test_loss_from_targets_returns_finite_masked_scalar():
    q = tf.constant([[1.0, 2.0, 999.0]])
    targets = tf.constant([[0.0, 4.0, -999.0]])
    mask = tf.constant([[True, True, False]])

    loss = loss_from_targets(q, targets, mask)

    assert loss.shape == ()
    assert math.isfinite(float(loss.numpy()))
    assert float(loss.numpy()) == pytest.approx(((1.0 - 0.0) ** 2 + (2.0 - 4.0) ** 2) / 2.0)


def test_masked_policy_probabilities_zero_illegal_actions():
    logits = tf.constant([[0.0, 100.0, 0.0]], dtype=tf.float32)
    mask = tf.constant([[True, False, True]])

    pi = masked_policy_probabilities(logits, mask).numpy()[0]

    assert pi[1] == pytest.approx(0.0)
    assert pi[0] == pytest.approx(0.5)
    assert pi[2] == pytest.approx(0.5)


def test_utility_policy_loss_components_match_formula():
    logits = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    q_n = tf.constant([[-1.0, 2.0, 4.0]], dtype=tf.float32)
    mask = tf.constant([[True, False, True]])
    cfg = UtilityPolicyLossConfig(
        beta_return=1.0,
        beta_loss=2.0,
        beta_missed=3.0,
        ce_weight=0.5,
        soft_best_temperature=1.0,
        edge_margin=0.25,
    )

    loss, components = utility_policy_loss(logits, q_n, mask, cfg, return_components=True)

    assert components["pi"].numpy()[0, 1] == pytest.approx(0.0)
    r_pi = 0.5 * -1.0 + 0.5 * 4.0
    legal_q = np.array([-1.0, 4.0])
    rho_legal = np.exp(legal_q) / np.exp(legal_q).sum()
    b_n = float((rho_legal * legal_q).sum())
    ce = float(-np.sum(rho_legal * np.log(np.array([0.5, 0.5]))))
    expected = -r_pi + 2.0 * max(-r_pi, 0.0) + 3.0 * max(b_n - r_pi - 0.25, 0.0) + 0.5 * ce

    assert components["R_pi"].numpy()[0] == pytest.approx(r_pi)
    assert components["B_N"].numpy()[0] == pytest.approx(b_n)
    assert float(loss.numpy()) == pytest.approx(expected)


def test_utility_policy_loss_stops_q_gradient_by_default():
    logits = tf.Variable([[0.0, 0.0]], dtype=tf.float32)
    q_n = tf.Variable([[1.0, 2.0]], dtype=tf.float32)
    mask = tf.constant([[True, True]])

    with tf.GradientTape() as tape:
        loss = utility_policy_loss(logits, q_n, mask)
    grad_logits, grad_q = tape.gradient(loss, [logits, q_n])

    assert grad_logits is not None
    assert grad_q is None


@pytest.mark.parametrize("bad_name", ["close_next", "future_open", "my_high_next_feature", "FUTURE_CLOSE"])
def test_training_feature_validation_rejects_future_ohlc_feature_names(bad_name):
    with pytest.raises(ValueError, match="Future OHLC fields"):
        _validate_no_future_features(["current_return", bad_name])
