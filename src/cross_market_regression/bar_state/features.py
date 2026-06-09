"""Feature engineering for per-bar market state."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_symbol(bars: pd.DataFrame, symbol: str | None) -> str:
    if symbol is not None:
        return symbol.lower()
    if "symbol" not in bars.columns:
        raise ValueError("symbol must be provided when bars has no 'symbol' column")
    symbols = bars["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        raise ValueError("bars must contain exactly one symbol or symbol must be provided")
    return symbols[0].lower()


def _seconds_since_midnight(timestamp: pd.Series) -> pd.Series:
    return timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second


def build_bar_state_features(bars: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Build current/past-only bar-state features for one symbol.

    VWAP is computed from cumulative same-date price-volume totals, so it resets
    at each new trading date and uses no future bars.
    """

    required = {"timestamp", "date", "time", "open", "high", "low", "last", "volume"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"Missing required bar columns: {missing}")

    prefix = _resolve_symbol(bars, symbol)
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
    df = df.reset_index(drop=True)

    last = df["last"].astype(float)
    volume = df["volume"].astype(float)
    log_last = np.log(last)
    log_return_1 = log_last.diff(1)

    by_date = df.groupby("date", sort=False)
    cumulative_dollars = (last * volume).groupby(df["date"], sort=False).cumsum()
    cumulative_volume = volume.groupby(df["date"], sort=False).cumsum()
    vwap = cumulative_dollars / cumulative_volume.replace(0.0, np.nan)

    seconds = _seconds_since_midnight(df["timestamp"])
    angle = 2.0 * np.pi * seconds / 86_400.0
    rolling_volume_mean = volume.rolling(20, min_periods=20).mean()
    rolling_volume_std = volume.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)

    out = df[["timestamp", "date", "time"]].copy()
    out[f"{prefix}_last"] = last
    out[f"{prefix}_log_last"] = log_last
    out[f"{prefix}_log_return_1"] = log_return_1
    out[f"{prefix}_log_return_3"] = log_last.diff(3)
    out[f"{prefix}_log_return_6"] = log_last.diff(6)
    out[f"{prefix}_realized_vol_6"] = log_return_1.rolling(6, min_periods=6).std(ddof=0)
    out[f"{prefix}_realized_vol_12"] = log_return_1.rolling(12, min_periods=12).std(ddof=0)
    out[f"{prefix}_normalized_range"] = (df["high"].astype(float) - df["low"].astype(float)) / last.replace(0.0, np.nan)
    out[f"{prefix}_intrabar_return"] = np.log(last / df["open"].astype(float).replace(0.0, np.nan))
    out[f"{prefix}_log_volume"] = np.log1p(volume.clip(lower=0.0))
    out[f"{prefix}_volume_z_20"] = (volume - rolling_volume_mean) / rolling_volume_std
    out[f"{prefix}_vwap_deviation"] = np.log(last / vwap)
    out[f"{prefix}_time_sin"] = np.sin(angle)
    out[f"{prefix}_time_cos"] = np.cos(angle)
    return out
