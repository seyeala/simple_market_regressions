import importlib.util

import pytest

from cross_market_regression.modeling.model import build_linear_model, load_linear_model


pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")


def test_model_save_load_roundtrip_with_synthetic_weights(tmp_path):
    model = build_linear_model(2)
    model.layers[-1].set_weights([
        [[1.5], [-0.5]],
        [0.25],
    ])
    model.save_weights(str(tmp_path / "model.weights.h5"))

    reloaded = load_linear_model(str(tmp_path), 2)
    weights, bias = reloaded.layers[-1].get_weights()

    assert weights.tolist() == [[1.5], [-0.5]]
    assert bias.tolist() == [0.25]
