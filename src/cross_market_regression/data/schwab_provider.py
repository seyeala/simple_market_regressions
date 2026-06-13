"""Schwab price provider backed by an injected authenticated client.

The provider intentionally does not create or refresh Schwab credentials itself.
Callers supply a small authenticated client abstraction whose methods return either
plain dictionaries/lists or SDK response objects with a ``json()`` method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .csv_provider import STANDARD_COLUMNS
from .frame import PriceFrame


class SchwabPriceProvider:
    """Load Schwab OHLCV and quote data through an injected client.

    Supported client abstraction methods, in priority order:

    * ``get_daily_ohlcv(symbol, start, end)`` for daily candles.
    * ``get_intraday_ohlcv(symbol, start, end, frequency_minutes, extended_hours)`` for minute candles.
    * ``get_quote_snapshot(symbols)`` for quote snapshots.

    For compatibility with Schwab SDK-style clients, the provider also falls back
    to ``get_price_history(...)`` and ``get_quotes(...)`` when the project-specific
    abstraction methods are not present.
    """

    name = "schwab"

    def __init__(self, client: Any | None = None, cache_dir: str = "artifacts/raw/schwab"):
        self.client = client
        self.cache_dir = Path(cache_dir)

    def get_daily_ohlcv(self, symbol: str, start: str, end: str, use_cache: bool = True):
        """Return daily bars normalized to the same columns as CSV providers."""

        payload = self._history_payload(
            cache_key=f"daily_{symbol}_{start}_{end}.json",
            use_cache=use_cache,
            fetch=lambda: self._fetch_daily(symbol=symbol, start=start, end=end),
        )
        return self._normalize_ohlcv(payload, symbol=symbol, intraday=False)

    def get_intraday_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency_minutes: int = 1,
        extended_hours: bool = True,
        use_cache: bool = True,
    ):
        """Return intraday bars normalized to the same columns as CSV providers."""

        payload = self._history_payload(
            cache_key=f"intraday_{symbol}_{start}_{end}_{frequency_minutes}_{extended_hours}.json",
            use_cache=use_cache,
            fetch=lambda: self._fetch_intraday(
                symbol=symbol,
                start=start,
                end=end,
                frequency_minutes=frequency_minutes,
                extended_hours=extended_hours,
            ),
        )
        return self._normalize_ohlcv(payload, symbol=symbol, intraday=True)

    def get_quote_snapshot(self, symbols: list[str]):
        """Return current quote snapshots in a normalized pandas DataFrame."""

        import pandas as pd

        self._require_client("quote snapshots")
        if hasattr(self.client, "get_quote_snapshot"):
            payload = self.client.get_quote_snapshot(symbols)
        elif hasattr(self.client, "get_quotes"):
            payload = self.client.get_quotes(symbols)
        else:
            raise NotImplementedError(
                "Schwab quotes require an authenticated client with get_quote_snapshot(symbols) "
                "or get_quotes(symbols)"
            )
        payload = self._json_payload(payload)
        rows = self._quote_rows(payload, symbols)
        df = pd.DataFrame(rows)
        columns = ["symbol", "bid", "ask", "last", "close", "total_volume", "quote_time", "source"]
        for column in columns:
            if column not in df.columns:
                df[column] = pd.NA
        return df[columns].sort_values("symbol").reset_index(drop=True)

    def load_prices(self, asset_config) -> PriceFrame:
        start = getattr(asset_config, "start", "") or ""
        end = getattr(asset_config, "end", "") or ""
        df = self.get_daily_ohlcv(asset_config.symbol, start, end)
        return PriceFrame(df[["date", "close"]].to_dict("records"))

    def _require_client(self, operation: str) -> None:
        if self.client is None:
            raise NotImplementedError(
                f"Schwab {operation} is not implemented unless a project-specific authenticated client is supplied"
            )

    def _history_payload(self, cache_key: str, use_cache: bool, fetch):
        cache_path = self.cache_dir / cache_key
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = self._json_payload(fetch())
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _fetch_daily(self, symbol: str, start: str, end: str):
        self._require_client("daily history")
        if hasattr(self.client, "get_daily_ohlcv"):
            return self.client.get_daily_ohlcv(symbol, start, end)
        if hasattr(self.client, "get_price_history_every_day"):
            return self.client.get_price_history_every_day(symbol, start_datetime=start, end_datetime=end)
        if hasattr(self.client, "get_price_history"):
            return self.client.get_price_history(
                symbol,
                period_type="year",
                frequency_type="daily",
                start_datetime=start,
                end_datetime=end,
            )
        raise NotImplementedError(
            "Schwab daily history requires an authenticated client with get_daily_ohlcv(symbol, start, end) "
            "or a Schwab SDK-style price history method"
        )

    def _fetch_intraday(self, symbol: str, start: str, end: str, frequency_minutes: int, extended_hours: bool):
        self._require_client("intraday history")
        if hasattr(self.client, "get_intraday_ohlcv"):
            return self.client.get_intraday_ohlcv(symbol, start, end, frequency_minutes, extended_hours)
        if hasattr(self.client, "get_price_history"):
            return self.client.get_price_history(
                symbol,
                frequency_type="minute",
                frequency=frequency_minutes,
                start_datetime=start,
                end_datetime=end,
                need_extended_hours_data=extended_hours,
            )
        raise NotImplementedError(
            "Schwab intraday history requires an authenticated client with "
            "get_intraday_ohlcv(symbol, start, end, frequency_minutes, extended_hours) "
            "or get_price_history(...)"
        )

    @staticmethod
    def _json_payload(payload):
        if hasattr(payload, "json"):
            return payload.json()
        return payload

    @staticmethod
    def _candles(payload) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            if "candles" in payload:
                return list(payload["candles"] or [])
            if "data" in payload:
                return list(payload["data"] or [])
        if isinstance(payload, list):
            return payload
        raise ValueError("Schwab history response must contain a candles/data list or be a list of bars")

    def _normalize_ohlcv(self, payload, symbol: str, intraday: bool):
        import pandas as pd

        rows = []
        for candle in self._candles(payload):
            value = candle.get("date", candle.get("datetime", candle.get("timestamp")))
            if value is None:
                raise ValueError("Schwab candle is missing date/datetime")
            timestamp = (
                pd.to_datetime(value, unit="ms", utc=True)
                if isinstance(value, (int, float))
                else pd.to_datetime(value, utc=intraday)
            )
            rows.append(
                {
                    "symbol": candle.get("symbol", symbol),
                    "date": timestamp.isoformat() if intraday else timestamp.strftime("%Y-%m-%d"),
                    "open": candle.get("open"),
                    "high": candle.get("high"),
                    "low": candle.get("low"),
                    "close": candle.get("close"),
                    "volume": candle.get("volume"),
                    "source": "schwab",
                }
            )
        df = pd.DataFrame(rows)
        for column in STANDARD_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        return df[STANDARD_COLUMNS].sort_values("date").reset_index(drop=True)

    @staticmethod
    def _quote_rows(payload, requested_symbols: list[str]) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            records = payload.items()
        elif isinstance(payload, list):
            records = [(item.get("symbol"), item) for item in payload]
        else:
            raise ValueError("Schwab quote response must be a mapping or list")

        rows = []
        requested = {symbol.upper(): symbol for symbol in requested_symbols}
        for key, quote in records:
            quote_body = quote.get("quote", quote) if isinstance(quote, dict) else {}
            symbol = quote.get("symbol") or quote_body.get("symbol") or key
            canonical_symbol = requested.get(str(symbol).upper(), symbol)
            rows.append(
                {
                    "symbol": canonical_symbol,
                    "bid": quote_body.get("bidPrice", quote_body.get("bid")),
                    "ask": quote_body.get("askPrice", quote_body.get("ask")),
                    "last": quote_body.get("lastPrice", quote_body.get("last")),
                    "close": quote_body.get("closePrice", quote_body.get("close")),
                    "total_volume": quote_body.get("totalVolume", quote_body.get("volume")),
                    "quote_time": quote_body.get("quoteTime", quote_body.get("quote_time")),
                    "source": "schwab",
                }
            )
        return rows
