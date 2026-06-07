"""Schwab client creation helpers.

This module deliberately does not print or persist secrets.  It returns a small
credential bundle when no project-specific Schwab SDK client is injected.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class SchwabClientConfig:
    api_key: str | None
    app_secret: str | None
    callback_url: str | None
    token_path: str | None


def create_schwab_client(auth_config):
    """Create a Schwab client/credential bundle from environment variables only."""

    token_path = getattr(auth_config, "token_path", None)
    if token_path:
        Path(token_path).parent.mkdir(parents=True, exist_ok=True)
    return SchwabClientConfig(
        api_key=os.getenv(getattr(auth_config, "api_key_env", None) or getattr(auth_config, "client_id_env", None) or ""),
        app_secret=os.getenv(getattr(auth_config, "app_secret_env", None) or getattr(auth_config, "client_secret_env", None) or ""),
        callback_url=os.getenv(getattr(auth_config, "callback_url_env", None) or getattr(auth_config, "redirect_uri_env", None) or ""),
        token_path=token_path,
    )
