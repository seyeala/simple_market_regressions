import pytest

pd = pytest.importorskip("pandas")

from cross_market_regression.bar_state.io import (
    clean_numeric,
    load_intraday_bars,
    parse_intraday_timestamp,
)


def test_clean_numeric_handles_currency_and_commas():
    assert clean_numeric("$1,234.56") == pytest.approx(1234.56)
    assert clean_numeric("2,741,096.93") == pytest.approx(2741096.93)


def test_parse_intraday_timestamp_handles_schwab_et_format():
    parsed = parse_intraday_timestamp(pd.Series(["09:30:00 AM ET, 06/02/2026"]))

    assert parsed.iloc[0] == pd.Timestamp("2026-06-02 09:30:00")


def test_load_intraday_bars_returns_normalized_columns(tmp_path):
    csv_path = tmp_path / "koru.csv"
    csv_path.write_text(
        "Date & Time,Open,High,Low,Last,Volume\n"
        '"09:30:00 AM ET, 06/02/2026",$1.00,$1.20,$0.90,$1.10,"1,000"\n',
        encoding="utf-8",
    )

    bars = load_intraday_bars(csv_path, symbol="KORU")

    assert list(bars.columns) == [
        "timestamp",
        "date",
        "time",
        "symbol",
        "open",
        "high",
        "low",
        "last",
        "volume",
    ]
