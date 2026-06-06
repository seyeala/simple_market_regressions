"""TensorFlow/Keras linear model factory."""

from __future__ import annotations


def build_linear_model(n_features: int, learning_rate: float = 0.01):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(1, activation=None, use_bias=True),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def load_linear_model(model_dir: str, n_features: int, learning_rate: float = 0.01):
    from pathlib import Path

    model = build_linear_model(n_features=n_features, learning_rate=learning_rate)
    model.load_weights(str(Path(model_dir) / "model.weights.h5"))
    return model
