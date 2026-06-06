import pytest

from cross_market_regression.data.registry import ProviderRegistry, default_registry


class DummyProvider:
    name = "dummy"

    def load_prices(self, asset_config):
        return [{"date": "2024-01-01", "close": 1.0}]


def test_provider_registry_explicit_lookup_uses_registered_instances():
    registry = ProviderRegistry()
    provider = DummyProvider()

    registry.register("dummy", provider)

    assert registry.get("dummy") is provider
    assert registry.names() == ["dummy"]


def test_provider_registry_missing_provider_reports_known_names():
    registry = ProviderRegistry()
    registry.register("dummy", DummyProvider())

    with pytest.raises(KeyError, match="dummy"):
        registry.get("missing")


def test_default_registry_is_constructed_without_credentials_or_network():
    names = default_registry().names()

    assert {"csv", "fx_csv", "target_csv", "schwab"}.issubset(names)
