"""Artifact persistence helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACTS = ["model.weights.h5", "scaler.json", "metadata.json", "metrics.json", "training_history.csv"]


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_history_csv(path: str | Path, history: dict[str, list[float]]) -> None:
    keys = list(history)
    rows = max((len(values) for values in history.values()), default=0)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *keys])
        writer.writeheader()
        for idx in range(rows):
            row = {"epoch": idx + 1}
            row.update({key: values[idx] if idx < len(values) else "" for key, values in history.items()})
            writer.writerow(row)
