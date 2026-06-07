"""Generic CSV price provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .frame import PriceFrame

if TYPE_CHECKING:
    import pandas as pd

STANDARD_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]


class CsvPriceProvider:
    name = "csv"

    def get_daily_ohlcv(self, path: str, symbol: str | None = None, date_col: str = "date") -> "pd.DataFrame":
        """Load daily bars from CSV and normalize common OHLCV columns."""

        import pandas as pd

        df = pd.read_csv(path)
        lower_map = {column: column.lower() for column in df.columns}
        df = df.rename(columns=lower_map)
        date_col = date_col.lower()
        if date_col not in df.columns:
            raise ValueError(f"CSV {path} is missing date column {date_col!r}")
        if "close" not in df.columns:
            raise ValueError(f"CSV {path} must contain a close column")
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        if symbol is not None:
            df["symbol"] = df["symbol"].fillna(symbol)
        for column in ["open", "high", "low", "volume"]:
            if column not in df.columns:
                df[column] = pd.NA
        df["source"] = "csv"
        return df[STANDARD_COLUMNS].sort_values("date").reset_index(drop=True)

    def load_prices(self, asset_config) -> PriceFrame:
        if not asset_config.csv_path:
            raise ValueError(f"Asset {asset_config.name} requires csv_path for CSV provider")

        import pandas as pd

        df = self.get_daily_ohlcv(asset_config.csv_path, symbol=asset_config.symbol)
        close_column = getattr(asset_config, "price_column", "close") or "close"
        if close_column != "close":
            raw = pd.read_csv(asset_config.csv_path)
            raw.columns = [column.lower() for column in raw.columns]
            if close_column not in raw.columns:
                raise ValueError(f"CSV {asset_config.csv_path} is missing price_column {close_column!r}")
            df["close"] = raw[close_column].astype(float)
        return PriceFrame(df[["date", "close"]].to_dict("records"))


# Backward-compatible name used by existing tests.
CSVProvider = CsvPriceProvider
