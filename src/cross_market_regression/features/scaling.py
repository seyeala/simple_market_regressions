"""Train-only standardization with JSON persistence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass
class StandardScaler:
    means: list[float]
    scales: list[float]

    @classmethod
    def fit(cls, rows: list[list[float]]) -> "StandardScaler":
        if not rows:
            raise ValueError("Cannot fit scaler on empty data")
        n_features = len(rows[0])
        means = [sum(row[i] for row in rows) / len(rows) for i in range(n_features)]
        scales = []
        for i, mean in enumerate(means):
            variance = sum((row[i] - mean) ** 2 for row in rows) / len(rows)
            scale = math.sqrt(variance)
            scales.append(scale if scale > 0 else 1.0)
        return cls(means=means, scales=scales)

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [[(value - self.means[i]) / self.scales[i] for i, value in enumerate(row)] for row in rows]

    def to_json(self) -> dict[str, list[float]]:
        return {"means": self.means, "scales": self.scales}

    @classmethod
    def from_json(cls, data: dict[str, list[float]]) -> "StandardScaler":
        return cls(means=list(data["means"]), scales=list(data["scales"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
