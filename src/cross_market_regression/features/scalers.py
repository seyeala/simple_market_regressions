"""Pandas/numpy standard scaler with coefficient inverse transform."""

from __future__ import annotations

import json
from pathlib import Path
class StandardScaler1D:
    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names or []
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def fit(self, X) -> "StandardScaler1D":
        if X.empty:
            raise ValueError("Cannot fit scaler on empty data")
        self.feature_names = self.feature_names or list(X.columns)
        frame = X[self.feature_names].astype(float)
        self.means = {name: float(frame[name].mean()) for name in self.feature_names}
        self.stds = {name: float(frame[name].std(ddof=0)) or 1.0 for name in self.feature_names}
        self.stds = {name: (std if std > 0 else 1.0) for name, std in self.stds.items()}
        return self

    def transform(self, X):
        import numpy as np

        frame = X[self.feature_names].astype(float)
        return np.column_stack([(frame[name] - self.means[name]) / self.stds[name] for name in self.feature_names])

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform_coef(self, weights, bias: float) -> dict:
        import numpy as np

        flat = np.asarray(weights, dtype=float).reshape(-1)
        raw_betas = {name: float(flat[idx] / self.stds[name]) for idx, name in enumerate(self.feature_names)}
        raw_intercept = float(bias - sum(flat[idx] * self.means[name] / self.stds[name] for idx, name in enumerate(self.feature_names)))
        return {"raw_intercept": raw_intercept, "raw_betas": raw_betas}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"feature_names": self.feature_names, "means": self.means, "stds": self.stds}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler1D":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = cls(feature_names=list(data["feature_names"]))
        scaler.means = {key: float(value) for key, value in data["means"].items()}
        scaler.stds = {key: float(value) for key, value in data["stds"].items()}
        return scaler
