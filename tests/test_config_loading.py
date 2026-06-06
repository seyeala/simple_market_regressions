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
