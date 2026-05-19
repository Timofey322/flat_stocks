from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd


OutputSize = Literal["compact", "full"]


class DataProvider(ABC):
    """Common interface for market and fundamental data sources."""

    @abstractmethod
    def fetch_daily_ohlcv(self, symbol: str, output_size: OutputSize = "compact") -> pd.DataFrame:
        """Return daily OHLCV data indexed by date."""

    @abstractmethod
    def fetch_quarterly_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Return normalized quarterly financial statement data indexed by period end."""
