import importlib.util
import inspect

import pytest

from cross_market_regression.modeling import model as model_module


def test_linear_model_factory_uses_required_keras_architecture():
    source = inspect.getsource(model_module.build_linear_model)
    assert "Input(shape=(n_features,))" in source
    assert "Dense(1, activation=None, use_bias=True)" in source


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")
def test_model_save_load_roundtrip(tmp_path):
    model = model_module.build_linear_model(2)
    model.save_weights(str(tmp_path / "model.weights.h5"))
    reloaded = model_module.load_linear_model(str(tmp_path), 2)
    assert len(reloaded.layers[-1].get_weights()) == 2
