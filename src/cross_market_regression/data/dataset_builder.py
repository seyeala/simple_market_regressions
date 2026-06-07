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


def _add_feature_asset(collection, config, asset_key, registry):
    if asset_key and asset_key not in collection:
        collection[asset_key] = _load_daily(config.assets[asset_key], registry)


def load_all_configured_data(config, registry):
    """Load source/market, target, and FX frames referenced by config."""

    target_asset = config.assets[config.target.asset]
    target = _load_daily(target_asset, registry)
    sources = {}
    fx = {}
    for feature in config.features:
        if not feature.enabled:
            continue
        collection = fx if feature.kind == "fx_return" else sources
        _add_feature_asset(collection, config, feature.asset, registry)
        _add_feature_asset(collection, config, feature.signal_asset, registry)
        _add_feature_asset(collection, config, feature.reference_asset, registry)
    return {"sources": sources, "target": target, "fx": fx}
