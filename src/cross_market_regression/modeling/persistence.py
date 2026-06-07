"""Artifact persistence helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .linear_tf_model import build_linear_model

REQUIRED_ARTIFACTS = ["model.weights.h5", "scaler.json", "metadata.json", "metrics.json", "training_history.csv"]


def ensure_model_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


ensure_output_dir = ensure_model_dir


def save_json(obj_or_path, path_or_obj) -> None:
    """Save JSON with both historical `(path, obj)` and new `(obj, path)` order."""

    if isinstance(obj_or_path, (str, Path)):
        path = Path(obj_or_path)
        obj = path_or_obj
    else:
        obj = obj_or_path
        path = Path(path_or_obj)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


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


def save_training_artifacts(model, scaler, metadata: dict, metrics: dict, history, model_dir: str) -> None:
    out = ensure_model_dir(model_dir)
    model.save_weights(str(out / "model.weights.h5"))
    scaler.save(out / "scaler.json")
    save_json(metadata, out / "metadata.json")
    save_json(metrics, out / "metrics.json")
    history_dict = getattr(history, "history", history) or {}
    save_history_csv(out / "training_history.csv", history_dict)


def load_training_artifacts(model_dir: str):
    metadata = load_json(Path(model_dir) / "metadata.json")
    from cross_market_regression.features.scalers import StandardScaler1D

    scaler = StandardScaler1D.load(Path(model_dir) / "scaler.json")
    model = build_linear_model(len(metadata["feature_names"]), metadata.get("learning_rate", 0.01))
    model.load_weights(str(Path(model_dir) / "model.weights.h5"))
    return {"metadata": metadata, "scaler": scaler, "model": model}
