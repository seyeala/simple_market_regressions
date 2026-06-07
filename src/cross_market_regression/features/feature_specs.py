"""Feature build result objects."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class BuiltFeature:
    name: str
    values: pd.Series
    metadata: dict
