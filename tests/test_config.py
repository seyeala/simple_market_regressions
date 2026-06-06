from cross_market_regression.config import CrossMarketConfig, load_config


def test_load_config_example():
    config = load_config("configs/examples/ewy_kospi.yaml")
    assert isinstance(config, CrossMarketConfig)
    assert config.name == "ewy_kospi_example"
    assert config.target.asset == "target_index"
    assert [feature.kind for feature in config.features] == ["source_return", "fx_return"]
