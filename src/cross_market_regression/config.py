"""Typed configuration objects and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuthConfig:
    provider: str = "csv"
    client_id_env: str | None = None
    client_secret_env: str | None = None
    redirect_uri_env: str | None = None
    token_path: str | None = None


@dataclass(frozen=True)
class AssetConfig:
    name: str
    symbol: str
    provider: str
    price_column: str = "close"
    calendar: str | None = None
    timezone: str | None = None
    csv_path: str | None = None


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    signal_asset: str
    reference_asset: str
    kind: str = "source_return"


@dataclass(frozen=True)
class TargetConfig:
    asset: str
    label: str = "target_next_return"


@dataclass(frozen=True)
class ModelConfig:
    train_start: str | None = None
    train_end: str | None = None
    validation_fraction: float = 0.2
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    random_seed: int = 7


@dataclass(frozen=True)
class CrossMarketConfig:
    name: str
    auth: AuthConfig = field(default_factory=AuthConfig)
    assets: list[AssetConfig] = field(default_factory=list)
    features: list[FeatureConfig] = field(default_factory=list)
    target: TargetConfig | None = None
    model: ModelConfig = field(default_factory=ModelConfig)

    def asset_by_name(self) -> dict[str, AssetConfig]:
        return {asset.name: asset for asset in self.assets}


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def load_config(path: str) -> CrossMarketConfig:
    """Load a cross-market regression config from YAML or JSON."""

    raw = _load_mapping(Path(path))
    assets = [AssetConfig(**item) for item in raw.get("assets", [])]
    features = [FeatureConfig(**item) for item in raw.get("features", [])]
    target_raw = raw.get("target")
    target = TargetConfig(**target_raw) if target_raw else None
    return CrossMarketConfig(
        name=raw["name"],
        auth=AuthConfig(**raw.get("auth", {})),
        assets=assets,
        features=features,
        target=target,
        model=ModelConfig(**raw.get("model", {})),
    )
