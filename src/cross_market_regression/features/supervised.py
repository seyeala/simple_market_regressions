"""Supervised row construction."""

from __future__ import annotations

from cross_market_regression.config import FeatureConfig, TargetConfig

from .returns import fx_return, source_return, target_next_return


def compute_feature(row: dict[str, object], feature: FeatureConfig) -> float:
    signal = float(row[feature.signal_asset])
    reference = float(row[feature.reference_asset])
    if feature.kind == "fx_return":
        return fx_return(signal, reference)
    if feature.kind == "source_return":
        return source_return(signal, reference)
    raise ValueError(f"Unsupported feature kind: {feature.kind}")


def build_supervised_rows(
    aligned_rows: list[dict[str, object]],
    features: list[FeatureConfig],
    target: TargetConfig | None,
) -> list[dict[str, object]]:
    if target is None:
        raise ValueError("Target config is required")
    rows = sorted(aligned_rows, key=lambda row: str(row["date"]))
    output = []
    for idx in range(len(rows) - 1):
        current = rows[idx]
        nxt = rows[idx + 1]
        item: dict[str, object] = {"date": current["date"], "target_date": nxt["date"]}
        for feature in features:
            item[feature.name] = compute_feature(current, feature)
        item[target.effective_name] = target_next_return(float(nxt[target.asset]), float(current[target.asset]))
        output.append(item)
    return output


def split_xy(rows: list[dict[str, object]], feature_names: list[str], label: str):
    x = [[float(row[name]) for name in feature_names] for row in rows]
    y = [float(row[label]) for row in rows]
    return x, y
