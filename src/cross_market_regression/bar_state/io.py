"""Intraday bar loading and normalization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_REQUIRED_NUMERIC_COLUMNS = ["open", "high", "low", "last", "volume"]
_SCHWAB_COLUMNS = {
    "Date & Time": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Last": "last",
    "Volume": "volume",
}


def clean_numeric(value: object) -> float:
    """Convert Schwab-style numeric strings to floats.

    Dollar signs, commas, and whitespace are ignored.  Missing or unparseable
    values are returned as ``numpy.nan`` so callers can use normal pandas null
    handling.
    """

    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    for token in ("$", ",", " "):
        text = text.replace(token, "")
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    return -number if negative else number


def parse_intraday_timestamp(series: pd.Series) -> pd.Series:
    """Parse intraday timestamp strings into pandas timestamps."""

    parsed = pd.to_datetime(series, errors="coerce")
    return pd.Series(parsed, index=series.index, name=series.name)


def load_intraday_bars(path: str | Path, symbol: str) -> pd.DataFrame:
    """Load and normalize a Schwab-style intraday OHLCV CSV file."""

    df = pd.read_csv(path)
    missing = [name for name in _SCHWAB_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required Schwab intraday columns: {missing}")

    normalized = df.rename(columns=_SCHWAB_COLUMNS)[list(_SCHWAB_COLUMNS.values())].copy()
    normalized["timestamp"] = parse_intraday_timestamp(normalized["timestamp"])
    for column in _REQUIRED_NUMERIC_COLUMNS:
        normalized[column] = normalized[column].map(clean_numeric)

    normalized = normalized.dropna(subset=["timestamp", *_REQUIRED_NUMERIC_COLUMNS])
    normalized = normalized.sort_values("timestamp", kind="mergesort")
    normalized = normalized.drop_duplicates(subset=["timestamp"], keep="last")
    normalized["date"] = normalized["timestamp"].dt.strftime("%Y-%m-%d")
    normalized["time"] = normalized["timestamp"].dt.strftime("%H:%M:%S")
    normalized["symbol"] = symbol

    columns = ["timestamp", "date", "time", "symbol", "open", "high", "low", "last", "volume"]
    return normalized[columns].reset_index(drop=True)
