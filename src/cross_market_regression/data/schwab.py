"""Backward-compatible Schwab module."""

from __future__ import annotations

from .schwab_auth import SchwabClientConfig as SchwabAuth, create_schwab_client
from .schwab_provider import SchwabPriceProvider

__all__ = ["SchwabAuth", "SchwabPriceProvider", "create_schwab_client"]
