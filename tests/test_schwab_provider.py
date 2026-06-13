import importlib.util

import pytest

from cross_market_regression.config import AssetConfig
from cross_market_regression.data.schwab_provider import SchwabPriceProvider


requires_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")


class FakeSchwabClient:
    def __init__(self):
        self.daily_calls = 0
        self.intraday_calls = 0
        self.quote_calls = 0

    def get_daily_ohlcv(self, symbol, start, end):
        self.daily_calls += 1
        assert symbol == "EWY"
        assert (start, end) in {("2024-01-01", "2024-01-03"), ("", "")}
        return {
            "candles": [
                {"datetime": 1704153600000, "open": 64.0, "high": 65.0, "low": 63.5, "close": 64.5, "volume": 123},
                {"datetime": 1704067200000, "open": 63.0, "high": 64.0, "low": 62.5, "close": 63.5, "volume": 456},
            ]
        }

    def get_intraday_ohlcv(self, symbol, start, end, frequency_minutes, extended_hours):
        self.intraday_calls += 1
        assert (symbol, frequency_minutes, extended_hours) == ("EWY", 5, False)
        return [
            {"datetime": 1704215700000, "open": 65.0, "high": 65.5, "low": 64.8, "close": 65.25, "volume": 10}
        ]

    def get_quote_snapshot(self, symbols):
        self.quote_calls += 1
        assert symbols == ["EWY", "SPY"]
        return {
            "EWY": {"quote": {"lastPrice": 64.75, "bidPrice": 64.7, "askPrice": 64.8, "totalVolume": 1000}},
            "SPY": {"quote": {"lastPrice": 501.0, "closePrice": 500.5}},
        }


@requires_pandas
def test_schwab_daily_history_normalizes_and_caches_like_csv_schema(tmp_path):
    client = FakeSchwabClient()
    provider = SchwabPriceProvider(client=client, cache_dir=str(tmp_path))

    df = provider.get_daily_ohlcv("EWY", "2024-01-01", "2024-01-03")
    cached_df = provider.get_daily_ohlcv("EWY", "2024-01-01", "2024-01-03")

    assert df.columns.tolist() == ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
    assert df["date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert df["symbol"].tolist() == ["EWY", "EWY"]
    assert df["source"].tolist() == ["schwab", "schwab"]
    assert cached_df.equals(df)
    assert client.daily_calls == 1
    assert (tmp_path / "daily_EWY_2024-01-01_2024-01-03.json").exists()


@requires_pandas
def test_schwab_intraday_history_normalizes_timestamp_schema(tmp_path):
    provider = SchwabPriceProvider(client=FakeSchwabClient(), cache_dir=str(tmp_path))

    df = provider.get_intraday_ohlcv("EWY", "2024-01-02T14:30:00Z", "2024-01-02T21:00:00Z", frequency_minutes=5, extended_hours=False)

    assert df.columns.tolist() == ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
    assert df.loc[0, "date"] == "2024-01-02T17:15:00+00:00"
    assert df.loc[0, "close"] == 65.25


@requires_pandas
def test_schwab_quote_snapshot_normalizes_fake_client_response(tmp_path):
    provider = SchwabPriceProvider(client=FakeSchwabClient(), cache_dir=str(tmp_path))

    df = provider.get_quote_snapshot(["EWY", "SPY"])

    assert df["symbol"].tolist() == ["EWY", "SPY"]
    assert df["source"].tolist() == ["schwab", "schwab"]
    assert df.loc[df["symbol"] == "EWY", "last"].iloc[0] == 64.75
    assert df.loc[df["symbol"] == "SPY", "close"].iloc[0] == 500.5


def test_schwab_without_client_reports_explicit_actionable_error():
    provider = SchwabPriceProvider()

    with pytest.raises(NotImplementedError, match="not implemented unless.*authenticated client is supplied"):
        provider.get_daily_ohlcv("EWY", "2024-01-01", "2024-01-03")


@requires_pandas
def test_schwab_load_prices_returns_price_frame_rows(tmp_path):
    asset = AssetConfig(name="ewy", symbol="EWY", provider="schwab")
    rows = list(SchwabPriceProvider(client=FakeSchwabClient(), cache_dir=str(tmp_path)).load_prices(asset))

    assert rows == [{"date": "2024-01-01", "close": 63.5}, {"date": "2024-01-02", "close": 64.5}]
