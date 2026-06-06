from cross_market_regression.data.alignment import inner_join_by_date, map_signal_to_target_calendar
from cross_market_regression.data.frame import PriceFrame


def test_inner_join_by_date_no_calendar_assumptions():
    frames = {
        "a": PriceFrame([{"date": "2024-01-01", "close": 1}, {"date": "2024-01-03", "close": 3}]),
        "b": PriceFrame([{"date": "2024-01-02", "close": 2}, {"date": "2024-01-03", "close": 4}]),
    }
    assert inner_join_by_date(frames) == [{"date": "2024-01-03", "a": 3.0, "b": 4.0}]


def test_calendar_mapping_uses_latest_observed_signal_only():
    signals = [{"date": "2024-01-01", "value": 1.0}, {"date": "2024-01-03", "value": 3.0}]
    mapped = map_signal_to_target_calendar(signals, ["2024-01-02", "2024-01-04"])
    assert mapped == [
        {"signal_date": "2024-01-01", "target_date": "2024-01-02", "value": 1.0},
        {"signal_date": "2024-01-03", "target_date": "2024-01-04", "value": 3.0},
    ]
