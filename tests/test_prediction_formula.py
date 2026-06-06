import importlib.util

import pytest

from cross_market_regression.features.scaling import StandardScaler
from cross_market_regression.modeling.model import build_linear_model
from cross_market_regression.modeling.persistence import save_json
from cross_market_regression.modeling.predict import predict_rows


pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not installed")


def test_prediction_uses_scaled_features_linear_weights_and_bias(tmp_path):
    save_json(tmp_path / "metadata.json", {"feature_names": ["a", "b"], "label": "y"})
    StandardScaler(means=[10.0, 100.0], scales=[2.0, 10.0]).save(tmp_path / "scaler.json")
    model = build_linear_model(2)
    model.layers[-1].set_weights([
        [[2.0], [-1.0]],
        [0.5],
    ])
    model.save_weights(str(tmp_path / "model.weights.h5"))

    predictions = predict_rows(str(tmp_path), [[12.0, 90.0]])

    # scaled row is [1.0, -1.0], so 2.0 * 1.0 + -1.0 * -1.0 + 0.5 = 3.5
    assert predictions == pytest.approx([3.5])
