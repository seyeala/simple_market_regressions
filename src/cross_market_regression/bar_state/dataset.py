"""Dataset assembly for cross-asset intraday bar-state models."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .features import build_bar_state_features
from .io import load_intraday_bars

_KEY_COLUMNS = ["timestamp", "date", "time"]
_FEATURE_SUFFIXES = [
    "last",
    "log_last",
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "realized_vol_6",
    "realized_vol_12",
    "normalized_range",
    "intrabar_return",
    "log_volume",
    "volume_z_20",
    "vwap_deviation",
    "time_sin",
    "time_cos",
]


def _symbol_key(symbol: str) -> str:
    return str(symbol).lower()


def _load_or_copy(value: pd.DataFrame | str | Path, symbol: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return load_intraday_bars(value, symbol=symbol)


def chronological_split_by_date(
    dataset: pd.DataFrame,
    *,
    date_col: str = "date",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float | None = None,
    train_end_date: str | None = None,
    validation_end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically by whole dates.

    Fractions are applied to unique sorted dates when explicit end dates are not
    supplied.  Splitting by dates prevents bars from the same session appearing
    in multiple splits.
    """

    if dataset.empty:
        empty = dataset.copy()
        return empty, empty.copy(), empty.copy()
    if date_col not in dataset.columns:
        raise ValueError(f"Dataset is missing date column {date_col!r}")

    df = dataset.sort_values(
        [date_col, "timestamp"] if "timestamp" in dataset.columns else [date_col]
    ).reset_index(drop=True)
    dates = pd.Series(sorted(pd.unique(df[date_col])))
    if train_end_date is not None or validation_end_date is not None:
        train_mask = pd.Series(True, index=df.index)
        if train_end_date is not None:
            train_mask = df[date_col] <= train_end_date
        val_mask = pd.Series(False, index=df.index)
        if train_end_date is not None:
            val_mask = df[date_col] > train_end_date
        if validation_end_date is not None:
            val_mask &= df[date_col] <= validation_end_date
        test_mask = ~(train_mask | val_mask)
        return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

    n_dates = len(dates)
    if n_dates == 1:
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()
    test_fraction = (
        1.0 - train_fraction - validation_fraction
        if test_fraction is None
        else test_fraction
    )
    if min(train_fraction, validation_fraction, test_fraction) < 0:
        raise ValueError("Split fractions must be non-negative")

    train_count = max(1, int(n_dates * train_fraction))
    validation_count = int(n_dates * validation_fraction)
    if validation_fraction > 0 and validation_count == 0 and n_dates - train_count > 0:
        validation_count = 1
    if train_count + validation_count > n_dates:
        validation_count = max(0, n_dates - train_count)

    train_dates = set(dates.iloc[:train_count])
    validation_dates = set(dates.iloc[train_count : train_count + validation_count])
    test_dates = set(dates.iloc[train_count + validation_count :])
    return (
        df[df[date_col].isin(train_dates)].copy(),
        df[df[date_col].isin(validation_dates)].copy(),
        df[df[date_col].isin(test_dates)].copy(),
    )


def build_cross_asset_bar_state_dataset(
    bars_by_symbol: Mapping[str, pd.DataFrame | str | Path],
    *,
    target_symbol: str,
    source_symbols: list[str] | tuple[str, ...] | None = None,
    horizon_bars: int = 1,
    include_target_state: bool = False,
) -> pd.DataFrame:
    """Build an inner-joined cross-asset bar-state supervised dataset."""

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1")
    target_key = _symbol_key(target_symbol)
    normalized_symbols = {_symbol_key(symbol): symbol for symbol in bars_by_symbol}
    if target_key not in normalized_symbols:
        raise ValueError(
            f"target_symbol {target_symbol!r} is not present in bars_by_symbol"
        )
    if source_symbols is None:
        source_keys = [key for key in normalized_symbols if key != target_key]
    else:
        source_keys = [_symbol_key(symbol) for symbol in source_symbols]
    missing_sources = [
        symbol for symbol in source_keys if symbol not in normalized_symbols
    ]
    if missing_sources:
        raise ValueError(
            f"source_symbols not present in bars_by_symbol: {missing_sources}"
        )

    feature_symbols = [*source_keys]
    if include_target_state:
        feature_symbols.append(target_key)

    frames: list[pd.DataFrame] = []
    for symbol_key in dict.fromkeys([*feature_symbols, target_key]):
        original_symbol = normalized_symbols[symbol_key]
        bars = _load_or_copy(bars_by_symbol[original_symbol], original_symbol)
        frames.append(build_bar_state_features(bars, symbol=symbol_key))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=_KEY_COLUMNS, how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    target_last_col = f"{target_key}_last"
    if target_last_col not in merged.columns:
        target_features = build_bar_state_features(
            _load_or_copy(
                bars_by_symbol[normalized_symbols[target_key]],
                normalized_symbols[target_key],
            ),
            symbol=target_key,
        )
        merged = merged.merge(
            target_features[[*_KEY_COLUMNS, target_last_col]],
            on=_KEY_COLUMNS,
            how="inner",
        )
    target_col = f"{target_key}_future_log_return_{horizon_bars}"
    merged[target_col] = np.log(
        merged[target_last_col].shift(-horizon_bars) / merged[target_last_col]
    )

    selected_feature_symbols = [
        *source_keys,
        *([target_key] if include_target_state else []),
    ]
    feature_cols = [
        f"{symbol}_{suffix}"
        for symbol in selected_feature_symbols
        for suffix in _FEATURE_SUFFIXES
    ]
    selected = [*_KEY_COLUMNS, *feature_cols, target_col]
    dataset = (
        merged[selected]
        .dropna(subset=[*feature_cols, target_col])
        .reset_index(drop=True)
    )

    train_df, validation_df, test_df = chronological_split_by_date(dataset)
    split_by_date: dict[str, str] = {}
    split_by_date.update({date: "train" for date in train_df["date"].unique()})
    split_by_date.update(
        {date: "validation" for date in validation_df["date"].unique()}
    )
    split_by_date.update({date: "test" for date in test_df["date"].unique()})
    dataset["split"] = dataset["date"].map(split_by_date)

    dataset.attrs["feature_cols"] = feature_cols
    dataset.attrs["target_col"] = target_col
    dataset.attrs["target_symbol"] = target_key
    dataset.attrs["source_symbols"] = source_keys
    dataset.attrs["horizon_bars"] = int(horizon_bars)
    return dataset
