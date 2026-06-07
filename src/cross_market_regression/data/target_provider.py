"""Target series provider wrapper."""

from __future__ import annotations


class TargetSeriesProvider:
    def __init__(self, provider_registry):
        self.provider_registry = provider_registry

    def load_target_series(self, asset_config, start: str | None = None, end: str | None = None):
        import pandas as pd

        provider = self.provider_registry.get(asset_config.provider)
        if hasattr(provider, "get_daily_ohlcv") and asset_config.csv_path:
            df = provider.get_daily_ohlcv(asset_config.csv_path, symbol=asset_config.symbol)
        elif hasattr(provider, "get_daily_ohlcv"):
            df = provider.get_daily_ohlcv(asset_config.symbol, start or "", end or "")
        else:
            frame = provider.load_prices(asset_config)
            df = pd.DataFrame(list(frame))
            df["symbol"] = asset_config.symbol
            df["source"] = asset_config.provider
        df = df.copy()
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        return pd.DataFrame(
            {
                "target_name": asset_config.name,
                "target_symbol": asset_config.symbol,
                "date": df["date"],
                "close": df["close"].astype(float),
                "source": df.get("source", asset_config.provider),
            }
        ).sort_values("date").reset_index(drop=True)


class TargetProvider:
    name = "target_csv"

    def __init__(self, delegate):
        self.delegate = delegate

    def load_prices(self, asset_config):
        return self.delegate.load_prices(asset_config)
