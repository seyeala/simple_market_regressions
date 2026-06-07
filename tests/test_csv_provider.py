import importlib.util

import pytest

from cross_market_regression.config import AssetConfig
from cross_market_regression.data.csv_provider import CsvPriceProvider
from cross_market_regression.data.fx_provider import FXProvider


requires_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")


@requires_pandas
def test_csv_provider_normalizes_daily_ohlcv_without_import_error(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "Date,Close,Open\n"
        "2024-01-02,101.5,100.0\n"
        "2024-01-01,99.5,98.0\n",
        encoding="utf-8",
    )

    df = CsvPriceProvider().get_daily_ohlcv(str(csv_path), symbol="TEST")

    assert df["date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert df["symbol"].tolist() == ["TEST", "TEST"]
    assert df["close"].tolist() == [99.5, 101.5]
    assert df["source"].tolist() == ["csv", "csv"]


@requires_pandas
def test_csv_provider_load_prices_uses_configured_price_column(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("date,close,adjusted_close\n2024-01-01,100.0,101.0\n", encoding="utf-8")
    asset = AssetConfig(
        name="asset",
        symbol="AST",
        provider="csv",
        csv_path=str(csv_path),
        price_column="adjusted_close",
    )

    rows = list(CsvPriceProvider().load_prices(asset))

    assert rows == [{"date": "2024-01-01", "close": 101.0}]


@requires_pandas
def test_csv_provider_missing_configured_price_column_reports_file(tmp_path):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("date,close\n2024-01-01,100.0\n", encoding="utf-8")
    asset = AssetConfig(name="asset", symbol="AST", provider="csv", csv_path=str(csv_path), price_column="missing")

    with pytest.raises(ValueError, match="missing"):
        CsvPriceProvider().load_prices(asset)


def test_fx_csv_provider_can_return_latest_configured_close():
    class DummyDelegate:
        def load_prices(self, asset_config):
            return [{"date": "2024-01-01", "close": 1.1}, {"date": "2024-01-02", "close": 1.2}]

    asset = AssetConfig(name="fx", symbol="FX", provider="fx_csv")

    assert FXProvider(DummyDelegate()).get_latest_fx(asset) == 1.2
