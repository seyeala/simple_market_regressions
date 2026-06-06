import pytest

from cross_market_regression.data.registry import ProviderRegistry, default_registry


class DummyProvider:
    name = "dummy"

    def load_prices(self, asset_config):
        return None


def test_provider_registry_explicit_lookup():
    registry = ProviderRegistry()
    registry.register("dummy", DummyProvider())
    assert registry.get("dummy").name == "dummy"
    with pytest.raises(KeyError):
        registry.get("missing")


def test_default_registry_names():
    names = default_registry().names()
    assert {"csv", "fx_csv", "target_csv", "schwab"}.issubset(names)
