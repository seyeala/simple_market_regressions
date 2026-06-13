from __future__ import annotations

import argparse

import pandas as pd
import yaml

from cross_market_regression.modeling.trading_train import TradingTrainConfig, train_trading_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a 15-action trading model")
    parser.add_argument("--data", required=True, help="CSV containing historical/current features and future OHLC target columns")
    parser.add_argument("--features", nargs="+", required=True, help="Feature columns to feed into the model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", help="Optional YAML mapping of TradingTrainConfig fields")
    args = parser.parse_args(argv)

    cfg_values = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            cfg_values = yaml.safe_load(handle) or {}
    cfg_values["model_dir"] = args.output_dir
    result = train_trading_model(pd.read_csv(args.data), args.features, TradingTrainConfig(**cfg_values))
    print(result)


if __name__ == "__main__":
    main()
