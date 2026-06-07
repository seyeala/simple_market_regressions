"""Calendar-free alignment utilities."""

from __future__ import annotations

from collections.abc import Mapping

from .frame import PriceFrame


def inner_join_by_date(frames: Mapping[str, PriceFrame]) -> list[dict[str, object]]:
    if not frames:
        return []
    common = set.intersection(*(set(frame.dates()) for frame in frames.values()))
    rows = []
    for date in sorted(common):
        row: dict[str, object] = {"date": date}
        for name, frame in frames.items():
            row[name] = float(frame.by_date()[date]["close"])
        rows.append(row)
    return rows


def map_signal_to_target_calendar(signal_rows: list[dict[str, object]], target_dates: list[str]) -> list[dict[str, object]]:
    """Map already-observed signal rows onto target dates without hard-coded calendars.

    For each target date, use the most recent signal row with `signal.date <= target.date`.
    """

    signals = sorted(signal_rows, key=lambda row: str(row["date"]))
    out = []
    idx = -1
    for target_date in sorted(target_dates):
        while idx + 1 < len(signals) and str(signals[idx + 1]["date"]) <= target_date:
            idx += 1
        if idx >= 0:
            mapped = dict(signals[idx])
            mapped["target_date"] = target_date
            mapped["signal_date"] = mapped.pop("date")
            out.append(mapped)
    return out

# Pandas-based generic session mapping for the reusable framework.
def _as_date_series(values):
    import pandas as pd

    return pd.Series(pd.to_datetime(values).dt.strftime("%Y-%m-%d") if hasattr(values, "dt") else pd.to_datetime(list(values)).strftime("%Y-%m-%d"))


def build_session_map(source_dates, target_dates, mapping_mode: str):
    """Return source -> target-current -> target-next session rows.

    The implementation is intentionally calendar-name agnostic: configured and
    observed source/target date series provide the tradable sessions, so holidays
    and weekends are skipped by selecting the next available target date.
    """

    import pandas as pd

    source = sorted(pd.to_datetime(source_dates).strftime("%Y-%m-%d"))
    target = sorted(pd.to_datetime(target_dates).strftime("%Y-%m-%d"))
    rows: list[dict[str, str]] = []
    if len(target) < 2:
        return pd.DataFrame(columns=["source_date", "target_current_date", "target_next_date"])

    for source_date in source:
        if mapping_mode == "same_market_next_session":
            currents = [date for date in target if date <= source_date]
            if not currents:
                continue
            current = currents[-1]
        elif mapping_mode in {"source_close_to_next_target_session", "custom_next_available"}:
            currents = [date for date in target if date <= source_date]
            current = currents[-1] if currents else target[0]
        else:
            raise ValueError(f"Unsupported mapping_mode: {mapping_mode}")
        nexts = [date for date in target if date > current]
        if not nexts:
            continue
        rows.append({"source_date": source_date, "target_current_date": current, "target_next_date": nexts[0]})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def validate_no_lookahead(dataset) -> None:
    """Validate generic no-lookahead invariants for a supervised dataset."""

    import pandas as pd

    required = {"source_date", "target_current_date", "target_next_date"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset missing no-lookahead columns: {sorted(missing)}")
    if (pd.to_datetime(dataset["target_next_date"]) <= pd.to_datetime(dataset["target_current_date"])).any():
        raise ValueError("target_next_date must be after target_current_date")
    raw_feature_columns = [column for column in dataset.columns if column.startswith("raw_feature") or column.startswith("feature_raw")]
    if "target_next_close" in raw_feature_columns:
        raise ValueError("target_next_close may not appear in raw feature columns")
    if dataset.duplicated(["source_date", "target_current_date", "target_next_date"]).any():
        raise ValueError("Duplicate source/target mapping rows detected")
    if "feature_timestamp" in dataset.columns:
        if (pd.to_datetime(dataset["feature_timestamp"]) > pd.to_datetime(dataset["target_next_date"])).any():
            raise ValueError("Feature timestamp is after target_next_date")
