"""Typed configuration objects and loading helpers.

The loader accepts both the original lightweight JSON/YAML examples in this
repository and the newer reusable cross-market YAML schema.  Runtime code should
use the typed objects rather than hard-coded market names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Iterator, Literal


@dataclass(frozen=True)
class AuthConfig:
    provider: str
    api_key_env: str | None = None
    app_secret_env: str | None = None
    callback_url_env: str | None = None
    token_path: str | None = None
    # Backward-compatible aliases used by the initial repo skeleton.
    client_id_env: str | None = None
    client_secret_env: str | None = None
    redirect_uri_env: str | None = None


@dataclass(frozen=True)
class AssetConfig:
    name: str
    symbol: str
    provider: str
    currency: str = ""
    asset_type: str = ""
    timezone: str | None = None
    exchange_calendar: str | None = None
    csv_path: str | None = None
    price_column: str = "close"
    calendar: str | None = None


@dataclass(frozen=True)
class PriceModeConfig:
    reference_price_mode: Literal["open", "close", "manual", "quote_field"] = "open"
    signal_price_mode: Literal["close", "after_hours", "latest_quote", "manual", "quote_field"] = "close"
    quote_field: str = "last"


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    kind: Literal["source_return", "fx_return", "market_return", "custom_return"] = "source_return"
    asset: str | None = None
    reference_price_mode: str = "open"
    signal_price_mode: str = "close"
    enabled: bool = True
    # Backward-compatible pair-of-assets feature schema.
    signal_asset: str | None = None
    reference_asset: str | None = None


@dataclass(frozen=True)
class FeaturesConfig:
    return_type: Literal["simple", "log"] = "simple"
    feature_specs: list[FeatureConfig] = field(default_factory=list)

    def __iter__(self) -> Iterator[FeatureConfig]:
        return iter(self.feature_specs)

    def __len__(self) -> int:
        return len(self.feature_specs)

    def __getitem__(self, index: int) -> FeatureConfig:
        return self.feature_specs[index]


@dataclass(frozen=True)
class TargetConfig:
    name: str = "y_target_next_return"
    asset: str = ""
    current_close_column: str = "close"
    horizon: str = "next_target_close"
    target_return_type: Literal["simple", "log"] = "simple"
    label: str | None = None

    @property
    def effective_name(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class AlignmentConfig:
    source_session_timezone: str | None = None
    target_session_timezone: str | None = None
    mapping_mode: str = "same_market_next_session"
    no_lookahead: bool = True


@dataclass(frozen=True)
class TradingLossConfig:
    """Configuration for trading-target loss construction."""

    buy_offsets: tuple[float, float] = (0.001, 0.003)
    sell_offsets: tuple[float, float] = (0.001, 0.003)
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    x_cost: float = 1.0
    fill_mode: str = "soft"
    fill_sharpness: float = 1000.0
    gamma: float = 1.0
    eta: float = 1.0
    beta_1: float = 1.0
    beta_2: float = 1.0
    edge_margin: float = 0.0
    policy_temperature: float = 1.0
    ce_weight: float = 0.0
    n_steps: int = 1
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        buy_offsets = tuple(float(offset) for offset in self.buy_offsets)
        sell_offsets = tuple(float(offset) for offset in self.sell_offsets)
        object.__setattr__(self, "buy_offsets", buy_offsets)
        object.__setattr__(self, "sell_offsets", sell_offsets)

        if self.fill_mode not in {"soft", "hard"}:
            raise ValueError("fill_mode must be one of: 'soft', 'hard'")
        if self.fill_sharpness <= 0.0:
            raise ValueError("fill_sharpness must be positive")
        if self.policy_temperature <= 0.0:
            raise ValueError("policy_temperature must be positive")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if len(buy_offsets) != 2:
            raise ValueError("buy_offsets must contain exactly two offsets")
        if len(sell_offsets) != 2:
            raise ValueError("sell_offsets must contain exactly two offsets")
        if any(offset < 0.0 for offset in buy_offsets):
            raise ValueError("buy_offsets must be non-negative")
        if any(offset < 0.0 for offset in sell_offsets):
            raise ValueError("sell_offsets must be non-negative")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "default_linear"
    feature_names: list[str] = field(default_factory=list)
    target_name: str = "y_target_next_return"
    train_start: str | None = None
    train_end: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    epochs: int = 500
    learning_rate: float = 0.01
    batch_size: int = 32
    random_seed: int = 42
    standardize: bool = True
    model_dir: str = "artifacts/models/default_linear"
    direction_threshold: float = 0.002
    # Backward-compatible static split knob.
    validation_fraction: float = 0.2


class AuthCollection(dict[str, AuthConfig]):
    """Mapping of provider auth configs with legacy `.provider` access."""

    @property
    def provider(self) -> str:
        if not self:
            return ""
        return next(iter(self.values())).provider


class AssetCollection(dict[str, AssetConfig]):
    """Mapping that iterates over values for legacy list-style loops."""

    def __iter__(self) -> Iterator[AssetConfig]:  # type: ignore[override]
        return iter(self.values())


@dataclass(frozen=True)
class CrossMarketConfig:
    name: str
    auth: AuthCollection = field(default_factory=AuthCollection)
    assets: AssetCollection = field(default_factory=AssetCollection)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading_loss: TradingLossConfig = field(default_factory=TradingLossConfig)

    def asset_by_name(self) -> dict[str, AssetConfig]:
        return dict(self.assets)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "assets": {key: asdict(value) for key, value in self.assets.items()},
            "features": {
                "return_type": self.features.return_type,
                "feature_specs": [asdict(feature) for feature in self.features.feature_specs],
            },
            "target": asdict(self.target),
            "alignment": asdict(self.alignment),
            "model": asdict(self.model),
            "trading_loss": asdict(self.trading_loss),
        }


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


def _load_auth(raw_auth: Any) -> AuthCollection:
    if not raw_auth:
        return AuthCollection({"default": AuthConfig(provider="csv")})
    if isinstance(raw_auth, dict) and "provider" in raw_auth:
        provider = str(raw_auth.get("provider", "csv"))
        return AuthCollection({provider: AuthConfig(**raw_auth)})
    if isinstance(raw_auth, dict):
        return AuthCollection({key: AuthConfig(**value) for key, value in raw_auth.items()})
    raise ValueError("auth must be a mapping")


def _load_assets(raw_assets: Any) -> AssetCollection:
    if isinstance(raw_assets, list):
        items = [AssetConfig(**item) for item in raw_assets]
        return AssetCollection({asset.name: asset for asset in items})
    if isinstance(raw_assets, dict):
        out = AssetCollection()
        for key, value in raw_assets.items():
            item = dict(value)
            item.setdefault("name", item.get("symbol", key))
            out[key] = AssetConfig(**item)
        return out
    return AssetCollection()


def _load_features(raw_features: Any) -> FeaturesConfig:
    if isinstance(raw_features, list):
        return FeaturesConfig(feature_specs=[FeatureConfig(**item) for item in raw_features])
    if isinstance(raw_features, dict):
        return FeaturesConfig(
            return_type=raw_features.get("return_type", "simple"),
            feature_specs=[FeatureConfig(**item) for item in raw_features.get("feature_specs", [])],
        )
    return FeaturesConfig()


def load_config(path: str) -> CrossMarketConfig:
    """Load YAML/JSON config, validate required links, and return typed config."""

    raw = _load_mapping(Path(path))
    assets = _load_assets(raw.get("assets", []))
    features = _load_features(raw.get("features", []))
    target = TargetConfig(**raw.get("target", {}))
    model_raw = dict(raw.get("model", {}))
    if not model_raw.get("feature_names"):
        model_raw["feature_names"] = [feature.name for feature in features if feature.enabled]
    if not model_raw.get("target_name"):
        model_raw["target_name"] = target.effective_name
    model = ModelConfig(**model_raw)
    cfg = CrossMarketConfig(
        name=raw.get("name") or model.model_name,
        auth=_load_auth(raw.get("auth")),
        assets=assets,
        features=features,
        target=target,
        alignment=AlignmentConfig(**raw.get("alignment", {})),
        model=model,
        trading_loss=TradingLossConfig(**raw.get("trading_loss", {})),
    )
    _validate_config(cfg)
    return cfg


def _validate_config(config: CrossMarketConfig) -> None:
    if not config.target.asset:
        raise ValueError("target.asset is required")
    if config.target.asset not in config.assets:
        raise ValueError(f"Unknown target asset: {config.target.asset}")
    for feature in config.features:
        if not feature.enabled:
            continue
        if feature.asset and feature.asset not in config.assets:
            raise ValueError(f"Feature {feature.name} references unknown asset {feature.asset}")
        if feature.signal_asset and feature.signal_asset not in config.assets:
            raise ValueError(f"Feature {feature.name} references unknown signal_asset {feature.signal_asset}")
        if feature.reference_asset and feature.reference_asset not in config.assets:
            raise ValueError(f"Feature {feature.name} references unknown reference_asset {feature.reference_asset}")
