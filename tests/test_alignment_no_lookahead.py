import pytest

from cross_market_regression.config import FeatureConfig, TargetConfig
from cross_market_regression.data.alignment import inner_join_by_date
from cross_market_regression.data.frame import PriceFrame
from cross_market_regression.features.supervised import build_supervised_rows


def test_inner_join_by_date_only_keeps_overlapping_dates():
    frames = {
        "source": PriceFrame([{"date": "2024-01-01", "close": 10}, {"date": "2024-01-03", "close": 30}]),
        "target": PriceFrame([{"date": "2024-01-02", "close": 20}, {"date": "2024-01-03", "close": 40}]),
    }

    assert inner_join_by_date(frames) == [{"date": "2024-01-03", "source": 30.0, "target": 40.0}]


def test_supervised_features_use_current_row_and_label_uses_next_target_only():
    rows = [
        {"date": "2024-01-01", "source": 110.0, "reference": 100.0, "target": 200.0},
        {"date": "2024-01-02", "source": 999.0, "reference": 1.0, "target": 220.0},
    ]

    out = build_supervised_rows(
        rows,
        [FeatureConfig(name="relative_source", signal_asset="source", reference_asset="reference")],
        TargetConfig(asset="target", label="next_target_return"),
    )

    assert len(out) == 1
    assert out[0]["relative_source"] == pytest.approx(0.10)
    assert out[0]["next_target_return"] == pytest.approx(0.10)
    assert out[0]["date"] == "2024-01-01"
    assert out[0]["target_date"] == "2024-01-02"
