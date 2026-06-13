"""TensorFlow/Keras linear model factory."""

from __future__ import annotations

from types import MethodType


def build_linear_model(n_features: int, learning_rate: float = 0.01):
    import numpy as np
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(1, activation=None, use_bias=True),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    _install_keras_numpy_compatibility(model, tf, np)
    return model


def _install_keras_numpy_compatibility(model, tf, np) -> None:
    """Normalize legacy list/numpy inputs for newer Keras data adapters."""

    dense = model.layers[-1]
    original_set_weights = dense.set_weights

    def set_weights_with_arrays(self, weights):
        return original_set_weights([np.asarray(weight, dtype=float) for weight in weights])

    dense.set_weights = MethodType(set_weights_with_arrays, dense)

    original_predict = model.predict

    def predict_with_tensor_inputs(self, x, *args, **kwargs):
        return original_predict(tf.convert_to_tensor(np.asarray(x, dtype=float)), *args, **kwargs)

    model.predict = MethodType(predict_with_tensor_inputs, model)

    original_fit = model.fit

    def fit_with_tensor_inputs(self, x, y=None, *args, **kwargs):
        x_tensor = tf.convert_to_tensor(np.asarray(x, dtype=float))
        y_tensor = None if y is None else tf.convert_to_tensor(np.asarray(y, dtype=float))
        validation_data = kwargs.get("validation_data")
        if validation_data is not None:
            val_x, val_y, *rest = validation_data
            kwargs["validation_data"] = (
                tf.convert_to_tensor(np.asarray(val_x, dtype=float)),
                tf.convert_to_tensor(np.asarray(val_y, dtype=float)),
                *rest,
            )
        return original_fit(x_tensor, y_tensor, *args, **kwargs)

    model.fit = MethodType(fit_with_tensor_inputs, model)
