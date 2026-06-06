"""Small dependency-light table helpers used by tests and providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class PricePoint:
    date: str
    close: float


class PriceFrame:
    def __init__(self, rows: Iterable[dict[str, object]]):
        self.rows = sorted([dict(row) for row in rows], key=lambda row: str(row["date"]))

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def by_date(self) -> dict[str, dict[str, object]]:
        return {str(row["date"]): row for row in self.rows}

    def dates(self) -> list[str]:
        return [str(row["date"]) for row in self.rows]
