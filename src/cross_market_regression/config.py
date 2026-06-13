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
class TradingLossConfig:
    buy_offsets: tuple[float, float] = (0.001, 0.003)
    sell_offsets: tuple[float, float] = (0.001, 0.003)
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    cost_stress_multiplier: float = 1.0
    fill_mode: Literal["hard", "soft"] = "hard"
    soft_fill_sharpness: float = 100.0
    gamma: float = 1.0
    eta: float = 10.0
    beta_1: float = 1.0
    beta_2: float = 1.0
    edge_margin: float = 0.0
    policy_temperature: float = 1.0
    ce_weight: float = 0.0
    n_step_horizon: int = 1
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(self, "buy_offsets", tuple(self.buy_offsets))
        object.__setattr__(self, "sell_offsets", tuple(self.sell_offsets))


@dataclass(frozen=True)
class AlignmentConfig:
    source_session_timezone: str | None = None
    target_session_timezone: str | None = None
    mapping_mode: str = "same_market_next_session"
    no_lookahead: bool = True


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
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _load_simple_yaml_mapping(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    return data


def _load_simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the simple YAML mappings used by tests when PyYAML is unavailable."""

    def parse_scalar(value: str) -> Any:
        value = value.strip()
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        if value in {"null", "None", "~"}:
            return None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(item) for item in inner.split(",")]
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip("\"'")

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


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
    _validate_trading_loss_config(config.trading_loss)


def _validate_trading_loss_config(config: TradingLossConfig) -> None:
    if config.fill_mode not in {"hard", "soft"}:
        raise ValueError("trading_loss.fill_mode must be one of: hard, soft")
    if len(config.buy_offsets) != 2:
        raise ValueError("trading_loss.buy_offsets must contain exactly two offsets")
    if len(config.sell_offsets) != 2:
        raise ValueError("trading_loss.sell_offsets must contain exactly two offsets")

    positive_values = {
        "soft_fill_sharpness": config.soft_fill_sharpness,
        "policy_temperature": config.policy_temperature,
        "epsilon": config.epsilon,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"trading_loss.{name} must be positive")

    non_negative_values = {
        "fee_rate": config.fee_rate,
        "slippage_rate": config.slippage_rate,
        "cost_stress_multiplier": config.cost_stress_multiplier,
        "gamma": config.gamma,
        "eta": config.eta,
        "beta_1": config.beta_1,
        "beta_2": config.beta_2,
        "edge_margin": config.edge_margin,
        "ce_weight": config.ce_weight,
        **{f"buy_offsets[{index}]": value for index, value in enumerate(config.buy_offsets)},
        **{f"sell_offsets[{index}]": value for index, value in enumerate(config.sell_offsets)},
    }
    for name, value in non_negative_values.items():
        if value < 0:
            raise ValueError(f"trading_loss.{name} must be non-negative")

    if config.n_step_horizon < 1:
        raise ValueError("trading_loss.n_step_horizon must be at least 1")
