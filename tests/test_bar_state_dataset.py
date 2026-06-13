import pytest

pd = pytest.importorskip("pandas")

from cross_market_regression.bar_state.dataset import (
    build_cross_asset_bar_state_dataset,
)

SYMBOLS = ["KORU", "QQQ", "SOXL", "TECL", "USO"]


def _write_fake_intraday_csv(path, symbol, dates, bars_per_day=25):
    symbol_offset = SYMBOLS.index(symbol) * 10.0
    rows = ["Date & Time,Open,High,Low,Last,Volume"]
    counter = 0
    for date in dates:
        session_start = pd.Timestamp(f"{date} 09:30:00")
        for bar in range(bars_per_day):
            timestamp = session_start + pd.Timedelta(minutes=bar)
            last = 100.0 + symbol_offset + counter * 0.10
            rows.append(
                f'"{timestamp.strftime("%I:%M:%S %p ET, %m/%d/%Y")}",'
                f"${last - 0.05:.2f},${last + 0.20:.2f},${last - 0.20:.2f},${last:.2f},"
                f'"{1_000 + counter:,}"'
            )
            counter += 1
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_cross_asset_bar_state_dataset_from_aligned_csvs(tmp_path):
    dates = pd.bdate_range("2026-06-01", periods=5).strftime("%Y-%m-%d").to_list()
    bars_by_symbol = {}
    for symbol in SYMBOLS:
        path = tmp_path / f"{symbol.lower()}.csv"
        _write_fake_intraday_csv(path, symbol, dates)
        bars_by_symbol[symbol] = path

    dataset = build_cross_asset_bar_state_dataset(
        bars_by_symbol, target_symbol="KORU", horizon_bars=1
    )

    assert "koru_future_log_return_1" in dataset.columns
    assert any(column.startswith("qqq_") for column in dataset.attrs["feature_cols"])
    assert any(column.startswith("soxl_") for column in dataset.attrs["feature_cols"])
    assert any(column.startswith("tecl_") for column in dataset.attrs["feature_cols"])
    assert any(column.startswith("uso_") for column in dataset.attrs["feature_cols"])
    assert all(
        not column.startswith("koru_") for column in dataset.attrs["feature_cols"]
    )
    assert set(dataset["split"]) == {"train", "validation", "test"}
    assert dataset["timestamp"].max() < pd.Timestamp(f"{dates[-1]} 09:54:00")
