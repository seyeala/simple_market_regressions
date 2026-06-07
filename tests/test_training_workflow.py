import importlib.util
import json

import pytest

from cross_market_regression.config import load_config
from cross_market_regression.modeling.persistence import load_json
from cross_market_regression.modeling.predict import predict_target_from_inputs
from cross_market_regression.modeling.train import train_model


requires_pandas_and_tensorflow = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("tensorflow") is None,
    reason="pandas and tensorflow are required for end-to-end training",
)
requires_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")


def _write_price_csv(path, closes):
    lines = ["date,close"]
    for idx, close in enumerate(closes, start=1):
        lines.append(f"2024-01-0{idx},{close}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@requires_pandas_and_tensorflow
def test_train_model_artifacts_are_compatible_with_live_prediction(tmp_path):
    _write_price_csv(tmp_path / "source.csv", [101, 102, 103, 104, 105])
    _write_price_csv(tmp_path / "reference.csv", [100, 100, 100, 100, 100])
    _write_price_csv(tmp_path / "target.csv", [200, 202, 204, 206, 208])
    model_dir = tmp_path / "model"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "fixture_pair_asset_model",
                "auth": {"provider": "csv"},
                "assets": [
                    {"name": "source", "symbol": "SRC", "provider": "csv", "csv_path": str(tmp_path / "source.csv")},
                    {"name": "reference", "symbol": "REF", "provider": "csv", "csv_path": str(tmp_path / "reference.csv")},
                    {"name": "target", "symbol": "TGT", "provider": "csv", "csv_path": str(tmp_path / "target.csv")},
                ],
                "features": [
                    {
                        "name": "source_relative_return",
                        "kind": "source_return",
                        "signal_asset": "source",
                        "reference_asset": "reference",
                    }
                ],
                "target": {"asset": "target", "label": "target_next_return"},
                "model": {
                    "epochs": 1,
                    "batch_size": 2,
                    "validation_fraction": 0.4,
                    "learning_rate": 0.01,
                    "model_dir": str(model_dir),
                },
            }
        ),
        encoding="utf-8",
    )

    result = train_model(load_config(str(config_path)), str(model_dir))
    metadata = load_json(model_dir / "metadata.json")
    scaler = load_json(model_dir / "scaler.json")
    prediction = predict_target_from_inputs(
        model_dir=str(model_dir),
        target_current_close=208.0,
        manual_feature_values={"source_relative_return": 0.06},
    )

    assert result["rows"] == 4
    assert metadata["feature_names"] == ["source_relative_return"]
    assert metadata["target_name"] == "target_next_return"
    assert scaler["feature_names"] == ["source_relative_return"]
    assert prediction.model_dir == str(model_dir)


@requires_pandas
def test_date_splits_are_mutually_exclusive():
    import pandas as pd

    from cross_market_regression.config import ModelConfig
    from cross_market_regression.modeling.train import _split_by_dates

    dataset = pd.DataFrame(
        {
            "target_next_date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [0.1, 0.2, 0.3, 0.4],
        }
    )

    train_df, val_df, test_df = _split_by_dates(
        dataset,
        ModelConfig(validation_start="2024-01-03", validation_end="2024-01-03", test_start="2024-01-04"),
    )

    assert train_df["target_next_date"].tolist() == ["2024-01-02"]
    assert val_df["target_next_date"].tolist() == ["2024-01-03"]
    assert test_df["target_next_date"].tolist() == ["2024-01-04", "2024-01-05"]
