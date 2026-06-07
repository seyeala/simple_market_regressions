"""Schwab price provider interface.

The class defines the reusable framework contract.  Live API wiring is kept out
of tests and notebooks so credentials are never required for local imports.
"""

from __future__ import annotations

from pathlib import Path

class SchwabPriceProvider:
    name = "schwab"

    def __init__(self, client=None, cache_dir: str = "artifacts/raw/schwab"):
        self.client = client
        self.cache_dir = Path(cache_dir)

    def get_daily_ohlcv(self, symbol: str, start: str, end: str, use_cache: bool = True):
        raise NotImplementedError("Schwab daily history requires a project-specific authenticated client")

    def get_intraday_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency_minutes: int = 1,
        extended_hours: bool = True,
        use_cache: bool = True,
    ):
        raise NotImplementedError("Schwab intraday history requires a project-specific authenticated client")

    def get_quote_snapshot(self, symbols: list[str]):
        raise NotImplementedError("Schwab quotes require a project-specific authenticated client")

    def load_prices(self, asset_config):
        return self.get_daily_ohlcv(asset_config.symbol, "", "")
