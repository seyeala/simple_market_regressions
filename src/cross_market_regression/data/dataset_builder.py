"""Load all configured data from an explicit provider registry."""

from __future__ import annotations


def _load_daily(asset, registry):
    import pandas as pd

    provider = registry.get(asset.provider)
    if hasattr(provider, "get_daily_ohlcv") and asset.csv_path:
        return provider.get_daily_ohlcv(asset.csv_path, symbol=asset.symbol)
    if hasattr(provider, "get_daily_ohlcv"):
        return provider.get_daily_ohlcv(asset.symbol, "", "")
    frame = provider.load_prices(asset)
    df = pd.DataFrame(list(frame))
    df["symbol"] = asset.symbol
    df["source"] = asset.provider
    return df


def load_all_configured_data(config, registry):
    """Load source/market, target, and FX frames referenced by config."""

    target_asset = config.assets[config.target.asset]
    target = _load_daily(target_asset, registry)
    sources: dict[str, pd.DataFrame] = {}
    fx: dict[str, pd.DataFrame] = {}
    for feature in config.features:
        if not feature.enabled:
            continue
        if feature.asset:
            asset = config.assets[feature.asset]
            if feature.kind == "fx_return":
                fx[feature.asset] = _load_daily(asset, registry)
            else:
                sources[feature.asset] = _load_daily(asset, registry)
    return {"sources": sources, "target": target, "fx": fx}
