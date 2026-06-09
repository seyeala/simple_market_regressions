"""CLI for live prediction with saved bar-state models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cross_market_regression.bar_state.features import build_bar_state_features
from cross_market_regression.bar_state.io import load_intraday_bars
from cross_market_regression.bar_state.predict import (
    load_bar_state_model,
    predict_bar_state_return,
    predicted_fair_value_from_last,
)

_KEY_COLUMNS = ["timestamp", "date", "time"]


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


def _load_features_json(value: str) -> dict[str, Any]:
    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        text = value
    else:
        text = Path(value).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError("--features-json list input must contain exactly one feature row")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("--features-json must be a JSON object or a one-element JSON array")
    return data


def _latest_aligned_feature_row(csv_entries: list[tuple[str, Path]], feature_cols: list[str]) -> pd.Series:
    if not csv_entries:
        raise ValueError("At least one --csv SYMBOL=PATH entry is required when --features-json is not supplied")

    merged: pd.DataFrame | None = None
    for symbol, path in csv_entries:
        symbol_key = symbol.lower()
        bars = load_intraday_bars(path, symbol=symbol)
        features = build_bar_state_features(bars, symbol=symbol_key)
        merged = features if merged is None else merged.merge(features, on=_KEY_COLUMNS, how="inner")

    if merged is None or merged.empty:
        raise ValueError("No shared timestamps were found across supplied CSV files")
    missing = [column for column in feature_cols if column not in merged.columns]
    if missing:
        raise ValueError(f"Supplied CSV files do not provide required model features: {missing}")

    aligned = merged.sort_values("timestamp").dropna(subset=feature_cols)
    if aligned.empty:
        raise ValueError("No aligned rows contain all required model features")
    return aligned.iloc[-1]


def _prediction_payload(
    feature_row: dict[str, Any] | pd.Series,
    model_dir: str,
    current_target_last: float | None,
) -> dict[str, Any]:
    predicted_log_return = predict_bar_state_return(feature_row, model_dir)
    output: dict[str, Any] = {"predicted_log_return": predicted_log_return}
    if isinstance(feature_row, pd.Series):
        timestamp = feature_row.get("timestamp")
        if timestamp is not None:
            output["timestamp"] = str(timestamp)
    if current_target_last is not None:
        output["predicted_target_fair_value"] = predicted_fair_value_from_last(
            current_target_last,
            predicted_log_return,
        )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict from a persisted bar-state model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--features-json", help="JSON object, one-element JSON array, or path containing one feature row")
    parser.add_argument("--csv", action="append", type=_parse_symbol_path, metavar="SYMBOL=PATH")
    parser.add_argument("--current-target-last", type=float)
    args = parser.parse_args(argv)

    if args.features_json is not None and args.csv:
        raise SystemExit("Use either --features-json or --csv inputs, not both")
    if args.features_json is None and not args.csv:
        raise SystemExit("Either --features-json or at least one --csv SYMBOL=PATH is required")

    if args.features_json is not None:
        feature_row: dict[str, Any] | pd.Series = _load_features_json(args.features_json)
    else:
        artifacts = load_bar_state_model(args.model_dir)
        metadata = artifacts["metadata"]
        feature_cols = list(metadata.get("feature_cols") or metadata.get("feature_names") or [])
        feature_row = _latest_aligned_feature_row(args.csv or [], feature_cols)

    print(
        json.dumps(
            _prediction_payload(feature_row, args.model_dir, args.current_target_last),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
