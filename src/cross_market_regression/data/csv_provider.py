"""CSV price provider."""

from __future__ import annotations

import csv
from pathlib import Path

from .frame import PriceFrame


class CSVProvider:
    name = "csv"

    def load_prices(self, asset_config) -> PriceFrame:
        if not asset_config.csv_path:
            raise ValueError(f"Asset {asset_config.name} requires csv_path for CSV provider")
        rows = []
        with Path(asset_config.csv_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"date": row["date"], "close": float(row[asset_config.price_column])})
        return PriceFrame(rows)
