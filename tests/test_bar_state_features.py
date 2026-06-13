import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cross_market_regression.bar_state.features import build_bar_state_features

FEATURE_SUFFIXES = [
    "last",
    "log_last",
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "realized_vol_6",
    "realized_vol_12",
    "normalized_range",
    "intrabar_return",
    "log_volume",
    "volume_z_20",
    "vwap_deviation",
    "time_sin",
    "time_cos",
]


def _fake_bars(symbol="KORU", dates=("2026-06-01", "2026-06-02"), bars_per_day=15):
    rows = []
    counter = 0
    for date in dates:
        session_start = pd.Timestamp(f"{date} 09:30:00")
        for bar in range(bars_per_day):
            timestamp = session_start + pd.Timedelta(minutes=bar)
            last = 100.0 + counter * 0.25
            rows.append(
                {
                    "timestamp": timestamp,
                    "date": date,
                    "time": timestamp.strftime("%H:%M:%S"),
                    "symbol": symbol,
                    "open": last - 0.05,
                    "high": last + 0.20,
                    "low": last - 0.20,
                    "last": last,
                    "volume": 1_000 + counter * 10,
                }
            )
            counter += 1
    return pd.DataFrame(rows)


def test_build_bar_state_features_includes_required_columns_and_resets_vwap_by_date():
    bars = _fake_bars()
    features = build_bar_state_features(bars)

    required_columns = [
        "timestamp",
        "date",
        "time",
        *(f"koru_{suffix}" for suffix in FEATURE_SUFFIXES),
    ]
    for column in required_columns:
        assert column in features.columns

    first_row_each_date = features.groupby("date", sort=False).head(1)
    assert first_row_each_date["koru_vwap_deviation"].to_list() == pytest.approx(
        [0.0, 0.0]
    )


def test_rolling_features_use_current_and_past_rows_only():
    bars = _fake_bars()
    features = build_bar_state_features(bars)
    row_number = 21

    log_last = np.log(bars["last"].astype(float))
    log_return_1 = log_last.diff(1)
    volume_window = bars.loc[row_number - 19 : row_number, "volume"].astype(float)
    return_window = log_return_1.loc[row_number - 5 : row_number]

    assert features.loc[row_number, "koru_log_return_6"] == pytest.approx(
        log_last.loc[row_number] - log_last.loc[row_number - 6]
    )
    assert features.loc[row_number, "koru_realized_vol_6"] == pytest.approx(
        return_window.std(ddof=0)
    )
    assert features.loc[row_number, "koru_volume_z_20"] == pytest.approx(
        (bars.loc[row_number, "volume"] - volume_window.mean())
        / volume_window.std(ddof=0)
    )


def test_time_features_are_finite():
    features = build_bar_state_features(_fake_bars())

    assert np.isfinite(features["koru_time_sin"]).all()
    assert np.isfinite(features["koru_time_cos"]).all()
