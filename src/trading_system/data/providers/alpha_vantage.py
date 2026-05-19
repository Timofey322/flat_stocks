from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import requests

from trading_system.config import settings
from trading_system.data.providers.base import DataProvider, OutputSize


class AlphaVantageError(RuntimeError):
    """Raised when Alpha Vantage returns an unusable response."""


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage provider for US equities.

    The provider keeps API-specific naming at the boundary and returns normalized
    dataframes that the rest of the system can consume without knowing the vendor.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or settings.alpha_vantage_api_key
        self.base_url = base_url or settings.alpha_vantage_base_url
        self.timeout = timeout
        if not self.api_key:
            raise AlphaVantageError("ALPHAVANTAGE_API_KEY is required for Alpha Vantage calls.")

    def fetch_daily_ohlcv(self, symbol: str, output_size: OutputSize = "compact") -> pd.DataFrame:
        payload = self._request(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": output_size,
            }
        )
        key = "Time Series (Daily)"
        if key not in payload:
            raise AlphaVantageError(f"Daily time series is missing for {symbol}: {payload}")

        frame = pd.DataFrame.from_dict(payload[key], orient="index")
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        frame = frame.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. adjusted close": "adjusted_close",
                "6. volume": "volume",
                "7. dividend amount": "dividend",
                "8. split coefficient": "split_coefficient",
            }
        )
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "dividend",
            "split_coefficient",
        ]
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        return frame.sort_index()

    def fetch_quarterly_fundamentals(self, symbol: str) -> pd.DataFrame:
        income = self._quarterly_reports("INCOME_STATEMENT", symbol)
        balance = self._quarterly_reports("BALANCE_SHEET", symbol)
        cash_flow = self._quarterly_reports("CASH_FLOW", symbol)
        overview = self._request({"function": "OVERVIEW", "symbol": symbol})

        merged = income.join(balance, how="outer", rsuffix="_balance").join(
            cash_flow, how="outer", rsuffix="_cash_flow"
        )
        merged["symbol"] = symbol
        merged["shares_outstanding"] = self._number_or_none(overview.get("SharesOutstanding"))
        return merged.sort_index()

    def _quarterly_reports(self, function: str, symbol: str) -> pd.DataFrame:
        payload = self._request({"function": function, "symbol": symbol})
        reports = payload.get("quarterlyReports")
        if not isinstance(reports, list):
            raise AlphaVantageError(f"{function} quarterly reports are missing for {symbol}: {payload}")

        frame = pd.DataFrame(reports)
        if frame.empty:
            return pd.DataFrame()

        frame = frame.rename(columns={"fiscalDateEnding": "period_end"})
        frame["period_end"] = pd.to_datetime(frame["period_end"])
        frame = frame.set_index("period_end")
        for column in frame.columns:
            if column != "reportedCurrency":
                frame[column] = frame[column].map(self._number_or_none)
        return frame

    def _request(self, params: Mapping[str, Any]) -> dict[str, Any]:
        query = dict(params)
        query["apikey"] = self.api_key
        response = requests.get(self.base_url, params=query, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        if "Error Message" in payload:
            raise AlphaVantageError(str(payload["Error Message"]))
        if "Note" in payload:
            raise AlphaVantageError(str(payload["Note"]))
        if "Information" in payload:
            raise AlphaVantageError(str(payload["Information"]))
        return payload

    @staticmethod
    def _number_or_none(value: Any) -> float | None:
        if value in (None, "", "None", "null"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
