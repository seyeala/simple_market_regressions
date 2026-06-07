"""TensorFlow/Keras linear model factory compatibility module."""

from __future__ import annotations

from pathlib import Path

from .linear_tf_model import build_linear_model


def load_linear_model(model_dir: str, n_features: int, learning_rate: float = 0.01):
    model = build_linear_model(n_features=n_features, learning_rate=learning_rate)
    model.load_weights(str(Path(model_dir) / "model.weights.h5"))
    return model
