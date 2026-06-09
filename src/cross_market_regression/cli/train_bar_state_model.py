"""CLI for training cross-asset intraday bar-state models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cross_market_regression.bar_state.dataset import build_cross_asset_bar_state_dataset
from cross_market_regression.bar_state.train import train_bar_state_model


def _parse_symbol_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SYMBOL=PATH")
    symbol, path = value.split("=", 1)
    symbol = symbol.strip()
    path = path.strip()
    if not symbol:
        raise argparse.ArgumentTypeError("SYMBOL must not be empty")
    if not path:
        raise argparse.ArgumentTypeError("PATH must not be empty")
    return symbol, Path(path)


def _paths_by_symbol(entries: list[tuple[str, Path]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for symbol, path in entries:
        paths[symbol] = path
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a cross-asset intraday bar-state model")
    parser.add_argument("--target-symbol", required=True)
    parser.add_argument("--source-symbols", nargs="+", required=True)
    parser.add_argument("--csv", action="append", type=_parse_symbol_path, required=True, metavar="SYMBOL=PATH")
    parser.add_argument("--horizon-bars", type=int, required=True)
    parser.add_argument("--include-target-state", action="store_true")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    args = parser.parse_args(argv)

    paths_by_symbol = _paths_by_symbol(args.csv)
    dataset = build_cross_asset_bar_state_dataset(
        paths_by_symbol,
        target_symbol=args.target_symbol,
        source_symbols=args.source_symbols,
        horizon_bars=args.horizon_bars,
        include_target_state=args.include_target_state,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(model_dir / "paired_dataset.csv", index=False)

    result = train_bar_state_model(
        dataset,
        model_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
