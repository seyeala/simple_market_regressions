from cross_market_regression.data.alignment import map_signal_to_target_calendar


def test_calendar_mapping_uses_latest_observed_signal_not_future_signal():
    signals = [
        {"date": "2024-01-01", "value": 1.0},
        {"date": "2024-01-03", "value": 3.0},
        {"date": "2024-01-08", "value": 8.0},
    ]

    mapped = map_signal_to_target_calendar(signals, ["2024-01-02", "2024-01-04", "2024-01-05"])

    assert mapped == [
        {"signal_date": "2024-01-01", "target_date": "2024-01-02", "value": 1.0},
        {"signal_date": "2024-01-03", "target_date": "2024-01-04", "value": 3.0},
        {"signal_date": "2024-01-03", "target_date": "2024-01-05", "value": 3.0},
    ]


def test_calendar_mapping_skips_targets_before_first_observed_signal():
    mapped = map_signal_to_target_calendar([{"date": "2024-01-03", "value": 3.0}], ["2024-01-02"])

    assert mapped == []
