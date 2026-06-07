"""Config-driven feature and supervised dataset construction."""

from __future__ import annotations

from cross_market_regression.data.alignment import build_session_map, validate_no_lookahead


def _normalize_daily(df):
    import pandas as pd

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values("date").reset_index(drop=True)


def _price_for_mode(df, mode: str):
    if mode in {"open", "close", "high", "low", "last", "mark", "bid", "ask"}:
        if mode not in df.columns:
            raise ValueError(f"Price mode {mode!r} requires column {mode!r}")
        return df[mode].astype(float)
    if mode == "previous_close":
        if "close" not in df.columns:
            raise ValueError("previous_close requires close column")
        return df["close"].astype(float).shift(1)
    if mode in {"after_hours", "latest_quote", "quote_field"}:
        column = "close" if "close" in df.columns else "last"
        if column not in df.columns:
            raise ValueError(f"Price mode {mode!r} requires close or last column")
        return df[column].astype(float)
    if mode == "manual":
        raise ValueError("manual mode is only supported in live prediction inputs")
    raise ValueError(f"Unsupported price mode: {mode}")


def _ratio(signal, reference, return_type: str):
    import numpy as np

    ratio = signal.astype(float) / reference.astype(float)
    if return_type == "log":
        return np.log(ratio)
    if return_type == "simple":
        return ratio - 1.0
    raise ValueError(f"Unsupported return_type: {return_type}")


def build_source_return_feature(source_df, session_map, feature_spec, return_type: str = "simple"):
    df = _normalize_daily(source_df)
    merged = session_map[["source_date"]].merge(df, left_on="source_date", right_on="date", how="left")
    return _ratio(_price_for_mode(merged, feature_spec.signal_price_mode), _price_for_mode(merged, feature_spec.reference_price_mode), return_type)


def build_fx_return_feature(fx_df, session_map, feature_spec, return_type: str = "simple"):
    df = _normalize_daily(fx_df)
    # FX is aligned by target_current_date by default because it is an optional
    # predictor available before target_next_close.
    merged = session_map[["target_current_date"]].merge(df, left_on="target_current_date", right_on="date", how="left")
    return _ratio(_price_for_mode(merged, feature_spec.signal_price_mode), _price_for_mode(merged, feature_spec.reference_price_mode), return_type)


def build_supervised_dataset(source_data, target_data, fx_data, config):
    import numpy as np

    """Build supervised rows from configured source, target, and FX frames."""

    target = _normalize_daily(target_data)
    source_asset_key = None
    for feature in config.features:
        if feature.enabled and feature.kind in {"source_return", "market_return", "custom_return"}:
            source_asset_key = feature.asset
            break
    if source_asset_key is None:
        source_asset_key = next(iter(source_data))
    source_dates = _normalize_daily(source_data[source_asset_key])["date"]
    session_map = build_session_map(source_dates, target["date"], config.alignment.mapping_mode)

    target_lookup = target.set_index("date")["close"].astype(float)
    dataset = session_map.copy()
    dataset["target_current_close"] = dataset["target_current_date"].map(target_lookup)
    dataset["target_next_close"] = dataset["target_next_date"].map(target_lookup)
    ratio = dataset["target_next_close"] / dataset["target_current_close"]
    target_name = config.target.effective_name
    dataset[target_name] = np.log(ratio) if config.target.target_return_type == "log" else ratio - 1.0

    fx_data = fx_data or {}
    return_type = config.features.return_type
    for feature in config.features:
        if not feature.enabled:
            continue
        if feature.kind == "fx_return":
            if feature.asset not in fx_data:
                raise ValueError(f"Missing FX data for feature {feature.name}: {feature.asset}")
            dataset[feature.name] = build_fx_return_feature(fx_data[feature.asset], session_map, feature, return_type)
        else:
            if feature.asset not in source_data:
                raise ValueError(f"Missing source data for feature {feature.name}: {feature.asset}")
            dataset[feature.name] = build_source_return_feature(source_data[feature.asset], session_map, feature, return_type)
    dataset = dataset.dropna(subset=[target_name, *config.model.feature_names]).reset_index(drop=True)
    validate_no_lookahead(dataset)
    return dataset
