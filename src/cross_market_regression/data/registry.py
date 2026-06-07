"""Explicit provider registry."""

from __future__ import annotations

from typing import Protocol

from .frame import PriceFrame


class PriceProvider(Protocol):
    name: str

    def load_prices(self, asset_config) -> PriceFrame: ...


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, PriceProvider] = {}

    def register(self, name: str, provider: PriceProvider) -> None:
        if not name:
            raise ValueError("Provider name is required")
        self._providers[name] = provider

    def get(self, name: str) -> PriceProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._providers)) or "<none>"
            raise KeyError(f"Unknown provider '{name}'. Registered providers: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._providers)


def default_registry() -> ProviderRegistry:
    from .csv_provider import CSVProvider
    from .fx_provider import FXProvider
    from .schwab_provider import SchwabPriceProvider
    from .target_provider import TargetProvider

    registry = ProviderRegistry()
    registry.register("csv", CSVProvider())
    registry.register("fx_csv", FXProvider(CSVProvider()))
    registry.register("target_csv", TargetProvider(CSVProvider()))
    registry.register("schwab", SchwabPriceProvider())
    return registry
