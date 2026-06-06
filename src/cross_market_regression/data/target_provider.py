"""Target provider wrapper."""

from __future__ import annotations


class TargetProvider:
    name = "target_csv"

    def __init__(self, delegate):
        self.delegate = delegate

    def load_prices(self, asset_config):
        return self.delegate.load_prices(asset_config)
