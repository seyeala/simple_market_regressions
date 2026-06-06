from pathlib import Path

from cross_market_regression.config import load_config


def test_reusable_source_code_has_no_market_specific_constants():
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/cross_market_regression").rglob("*.py"))
    forbidden = ["EWY", "KOSPI", "Korea", "USDKRW"]

    assert not any(token in source for token in forbidden)


def test_multiple_market_configs_reuse_the_same_schema():
    config_paths = sorted(Path("configs/examples").glob("*.yaml"))
    configs = [load_config(str(path)) for path in config_paths]

    assert len(configs) >= 5
    assert all(config.target.label == "target_next_return" for config in configs)
    assert all(config.features for config in configs)
    assert all(feature.kind in {"source_return", "fx_return"} for config in configs for feature in config.features)
