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
