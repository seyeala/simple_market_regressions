import pytest

from cross_market_regression.config import FeatureConfig, TargetConfig
from cross_market_regression.features.returns import fx_return, source_return, target_next_return
from cross_market_regression.features.supervised import build_supervised_rows


def test_feature_math_ratio_formula():
    assert source_return(110.0, 100.0) == pytest.approx(0.1)
    assert fx_return(99.0, 100.0) == pytest.approx(-0.01)
    assert target_next_return(105.0, 100.0) == pytest.approx(0.05)


def test_supervised_target_uses_next_close_no_lookahead_features_current_row():
    rows = [
        {"date": "2024-01-01", "source": 110.0, "reference": 100.0, "target": 200.0},
        {"date": "2024-01-02", "source": 999.0, "reference": 1.0, "target": 220.0},
    ]
    out = build_supervised_rows(
        rows,
        [FeatureConfig(name="rel", signal_asset="source", reference_asset="reference")],
        TargetConfig(asset="target", label="y"),
    )
    assert len(out) == 1
    assert out[0]["rel"] == pytest.approx(0.1)
    assert out[0]["y"] == pytest.approx(0.1)
    assert out[0]["target_date"] == "2024-01-02"
