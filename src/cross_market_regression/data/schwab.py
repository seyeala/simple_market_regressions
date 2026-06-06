"""Schwab authentication and provider stubs.

The reusable framework keeps Schwab support explicit but does not perform live
brokerage calls without an application-specific implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SchwabAuth:
    client_id: str | None
    client_secret: str | None
    redirect_uri: str | None
    token_path: str | None

    @classmethod
    def from_config(cls, auth_config) -> "SchwabAuth":
        return cls(
            client_id=os.getenv(auth_config.client_id_env or ""),
            client_secret=os.getenv(auth_config.client_secret_env or ""),
            redirect_uri=os.getenv(auth_config.redirect_uri_env or ""),
            token_path=auth_config.token_path,
        )


class SchwabPriceProvider:
    name = "schwab"

    def load_prices(self, asset_config):
        raise NotImplementedError(
            "Schwab provider is a stub. Supply an implementation or use the CSV provider."
        )
