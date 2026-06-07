"""FX provider wrapper."""

from __future__ import annotations


class FxProvider:
    def __init__(self, provider_registry):
        self.provider_registry = provider_registry

    def load_fx_series(self, asset_config, start: str | None = None, end: str | None = None):
        import pandas as pd

        provider = self.provider_registry.get(asset_config.provider)
        if hasattr(provider, "get_daily_ohlcv") and asset_config.csv_path:
            df = provider.get_daily_ohlcv(asset_config.csv_path, symbol=asset_config.symbol)
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
            {"symbol": asset_config.symbol, "date": df["date"], "close": df["close"].astype(float), "source": df.get("source", asset_config.provider)}
        ).sort_values("date").reset_index(drop=True)

    def get_latest_fx(self, asset_config) -> float:
        raise NotImplementedError("Latest FX requires a live provider implementation or manual input")


class FXProvider:
    name = "fx_csv"

    def __init__(self, delegate):
        self.delegate = delegate

    def load_prices(self, asset_config):
        return self.delegate.load_prices(asset_config)
