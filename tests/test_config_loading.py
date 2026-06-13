from pathlib import Path

from cross_market_regression.config import CrossMarketConfig, load_config


def test_load_example_config_without_external_services():
    config = load_config("configs/examples/ewy_kospi.yaml")

    assert isinstance(config, CrossMarketConfig)
    assert config.name == "ewy_kospi_example"
    assert config.auth.provider == "csv"
    assert config.target.asset == "target_index"
    assert [feature.kind for feature in config.features] == ["source_return", "fx_return"]


def test_all_example_configs_load_from_local_yaml_only():
    names = {load_config(str(path)).name for path in Path("configs/examples").glob("*.yaml")}

    assert {
        "ewy_kospi_example",
        "qqq_ndx_example",
        "ewj_nikkei_example",
        "ewt_taiwan_example",
        "soxx_semis_example",
    }.issubset(names)


def _write_minimal_config(path: Path, trading_loss: str = "") -> Path:
    path.write_text(
        f"""
name: trading_loss_test
assets:
  target_index:
    symbol: TEST
    provider: csv
target:
  asset: target_index
{trading_loss}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_trading_loss_defaults_load(tmp_path):
    config = load_config(str(_write_minimal_config(tmp_path / "config.yaml")))

    assert config.trading_loss.buy_offsets == (0.001, 0.003)
    assert config.trading_loss.sell_offsets == (0.001, 0.003)
    assert config.trading_loss.fill_mode == "hard"
    assert config.trading_loss.soft_fill_sharpness == 100.0
    assert config.trading_loss.gamma == 1.0
    assert config.trading_loss.eta == 10.0
    assert config.trading_loss.beta_1 == 1.0
    assert config.trading_loss.beta_2 == 1.0
    assert config.trading_loss.edge_margin == 0.0
    assert config.trading_loss.policy_temperature == 1.0
    assert config.trading_loss.ce_weight == 0.0
    assert config.trading_loss.n_step_horizon == 1
    assert config.trading_loss.epsilon == 1e-8


def test_trading_loss_yaml_overrides(tmp_path):
    config = load_config(
        str(
            _write_minimal_config(
                tmp_path / "config.yaml",
                """
trading_loss:
  buy_offsets: [0.002, 0.004]
  sell_offsets: [0.003, 0.005]
  fee_rate: 0.0001
  slippage_rate: 0.0002
  cost_stress_multiplier: 2.0
  fill_mode: soft
  soft_fill_sharpness: 50.0
  gamma: 0.9
  eta: 5.0
  beta_1: 0.5
  beta_2: 2.0
  edge_margin: 0.001
  policy_temperature: 0.75
  ce_weight: 0.25
  n_step_horizon: 3
  epsilon: 0.000001
""",
            )
        )
    )

    assert config.trading_loss.buy_offsets == (0.002, 0.004)
    assert config.trading_loss.sell_offsets == (0.003, 0.005)
    assert config.trading_loss.fee_rate == 0.0001
    assert config.trading_loss.slippage_rate == 0.0002
    assert config.trading_loss.cost_stress_multiplier == 2.0
    assert config.trading_loss.fill_mode == "soft"
    assert config.trading_loss.soft_fill_sharpness == 50.0
    assert config.trading_loss.gamma == 0.9
    assert config.trading_loss.eta == 5.0
    assert config.trading_loss.beta_1 == 0.5
    assert config.trading_loss.beta_2 == 2.0
    assert config.trading_loss.edge_margin == 0.001
    assert config.trading_loss.policy_temperature == 0.75
    assert config.trading_loss.ce_weight == 0.25
    assert config.trading_loss.n_step_horizon == 3
    assert config.trading_loss.epsilon == 0.000001


def test_trading_loss_metadata_export(tmp_path):
    config = load_config(
        str(
            _write_minimal_config(
                tmp_path / "config.yaml",
                """
trading_loss:
  fill_mode: soft
  policy_temperature: 2.0
""",
            )
        )
    )

    metadata = config.to_metadata_dict()

    assert metadata["trading_loss"]["fill_mode"] == "soft"
    assert metadata["trading_loss"]["policy_temperature"] == 2.0
    assert metadata["trading_loss"]["buy_offsets"] == (0.001, 0.003)


def test_trading_loss_invalid_values(tmp_path):
    invalid_cases = [
        ("fill_mode: midpoint", "fill_mode"),
        ("buy_offsets: [0.001]", "buy_offsets"),
        ("sell_offsets: [0.001, 0.002, 0.003]", "sell_offsets"),
        ("soft_fill_sharpness: 0", "soft_fill_sharpness"),
        ("policy_temperature: 0", "policy_temperature"),
        ("epsilon: 0", "epsilon"),
        ("fee_rate: -0.1", "fee_rate"),
        ("buy_offsets: [-0.001, 0.002]", "buy_offsets"),
        ("eta: -1", "eta"),
        ("ce_weight: -0.5", "ce_weight"),
        ("n_step_horizon: 0", "n_step_horizon"),
    ]

    for index, (trading_loss_line, expected_message) in enumerate(invalid_cases):
        path = _write_minimal_config(
            tmp_path / f"invalid_{index}.yaml",
            f"""
trading_loss:
  {trading_loss_line}
""",
        )
        try:
            load_config(str(path))
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected {trading_loss_line!r} to be invalid")
