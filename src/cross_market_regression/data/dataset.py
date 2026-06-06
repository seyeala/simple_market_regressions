"""Dataset loading/building helpers."""

from __future__ import annotations

from cross_market_regression.config import CrossMarketConfig
from cross_market_regression.features.supervised import build_supervised_rows

from .alignment import inner_join_by_date
from .registry import ProviderRegistry, default_registry


def load_asset_frames(config: CrossMarketConfig, registry: ProviderRegistry | None = None):
    registry = registry or default_registry()
    frames = {}
    for asset in config.assets:
        provider = registry.get(asset.provider)
        frames[asset.name] = provider.load_prices(asset)
    return frames


def build_dataset(config: CrossMarketConfig, registry: ProviderRegistry | None = None) -> list[dict[str, object]]:
    frames = load_asset_frames(config, registry)
    aligned = inner_join_by_date(frames)
    return build_supervised_rows(aligned, config.features, config.target)
