from __future__ import annotations

import argparse

from cross_market_regression.config import load_config
from cross_market_regression.modeling.train import train_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a cross-market regression model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = train_model(load_config(args.config), args.output_dir)
    print(result)


if __name__ == "__main__":
    main()
