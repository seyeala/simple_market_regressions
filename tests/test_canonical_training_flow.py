import importlib.util

import pytest

from cross_market_regression.config import (
    AlignmentConfig,
    AssetCollection,
    AssetConfig,
    CrossMarketConfig,
    FeatureConfig,
    FeaturesConfig,
    ModelConfig,
    TargetConfig,
)
from cross_market_regression.data.dataset_builder import load_all_configured_data
from cross_market_regression.features.feature_builder import build_supervised_dataset


requires_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")


def _pair_schema_config() -> CrossMarketConfig:
    return CrossMarketConfig(
        name="pair_schema",
        assets=AssetCollection(
            {
                "source_signal": AssetConfig(name="source_signal", symbol="SRC", provider="memory"),
                "source_reference": AssetConfig(name="source_reference", symbol="REF", provider="memory"),
                "fx_signal": AssetConfig(name="fx_signal", symbol="FXS", provider="memory"),
                "fx_reference": AssetConfig(name="fx_reference", symbol="FXR", provider="memory"),
                "target": AssetConfig(name="target", symbol="TGT", provider="memory"),
            }
        ),
        features=FeaturesConfig(
            feature_specs=[
                FeatureConfig(
                    name="source_relative_return",
                    kind="source_return",
                    signal_asset="source_signal",
                    reference_asset="source_reference",
                ),
                FeatureConfig(
                    name="fx_relative_return",
                    kind="fx_return",
                    signal_asset="fx_signal",
                    reference_asset="fx_reference",
                ),
            ]
        ),
        target=TargetConfig(asset="target", label="target_next_return"),
        alignment=AlignmentConfig(mapping_mode="same_market_next_session"),
        model=ModelConfig(feature_names=["source_relative_return", "fx_relative_return"]),
    )


@requires_pandas
def test_load_all_configured_data_loads_pair_schema_assets():
    class MemoryProvider:
        def load_prices(self, asset_config):
            return [{"date": "2024-01-01", "close": 1.0}]

    class MemoryRegistry:
        def get(self, name):
            assert name == "memory"
            return MemoryProvider()

    loaded = load_all_configured_data(_pair_schema_config(), MemoryRegistry())

    assert sorted(loaded["sources"]) == ["source_reference", "source_signal"]
    assert sorted(loaded["fx"]) == ["fx_reference", "fx_signal"]
    assert loaded["target"]["symbol"].tolist() == ["TGT"]


@requires_pandas
def test_build_supervised_dataset_supports_pair_schema_features():
    import pandas as pd

    cfg = _pair_schema_config()
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    source_data = {
        "source_signal": pd.DataFrame({"date": dates, "close": [110.0, 121.0, 132.0]}),
        "source_reference": pd.DataFrame({"date": dates, "close": [100.0, 110.0, 120.0]}),
    }
    fx_data = {
        "fx_signal": pd.DataFrame({"date": dates, "close": [1.10, 1.20, 1.30]}),
        "fx_reference": pd.DataFrame({"date": dates, "close": [1.00, 1.00, 1.00]}),
    }
    target = pd.DataFrame({"date": dates, "close": [200.0, 220.0, 242.0]})

    dataset = build_supervised_dataset(source_data, target, fx_data, cfg)

    assert dataset["source_date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert dataset["target_next_date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert dataset["source_relative_return"].tolist() == pytest.approx([0.10, 0.10])
    assert dataset["fx_relative_return"].tolist() == pytest.approx([0.10, 0.20])
    assert dataset["target_next_return"].tolist() == pytest.approx([0.10, 0.10])


def test_train_cli_uses_canonical_config_training_path(monkeypatch, capsys):
    from cross_market_regression.cli import train_model as cli

    calls = {}

    def fake_load_config(path):
        calls["config_path"] = path
        return "config-object"

    def fake_train_from_config(config, output_dir):
        calls["config"] = config
        calls["output_dir"] = output_dir
        return {"model_dir": output_dir}

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "train_from_config", fake_train_from_config)

    cli.main(["--config", "config.yaml", "--output-dir", "artifacts/model"])

    assert calls == {"config_path": "config.yaml", "config": "config-object", "output_dir": "artifacts/model"}
    assert "artifacts/model" in capsys.readouterr().out
