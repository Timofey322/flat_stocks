from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class FinancialSnapshot:
    """Normalized latest financial state used by valuation models."""

    symbol: str
    period_end: date
    revenue: float
    operating_income: float
    income_tax_expense: float
    capital_expenditures: float
    total_debt: float
    cash: float
    shares_outstanding: float
    free_cash_flow: float | None = None

    @property
    def operating_margin(self) -> float:
        if self.revenue == 0:
            return 0.0
        return self.operating_income / self.revenue

    @property
    def tax_rate(self) -> float:
        if self.operating_income <= 0:
            return 0.0
        return max(0.0, min(self.income_tax_expense / self.operating_income, 1.0))


@dataclass(frozen=True)
class FinancialAssumptions:
    """Explicit assumptions for a compact DCF valuation."""

    revenue_growth_rate: float = 0.06
    operating_margin: float | None = None
    tax_rate: float | None = None
    reinvestment_rate: float = 0.25
    discount_rate: float = 0.10
    terminal_growth_rate: float = 0.025
    forecast_years: int = 5
    margin_of_safety: float = 0.25

    def validate(self) -> None:
        if self.forecast_years < 1:
            raise ValueError("forecast_years must be at least 1.")
        if self.discount_rate <= self.terminal_growth_rate:
            raise ValueError("discount_rate must be greater than terminal_growth_rate.")
        if not 0 <= self.margin_of_safety < 1:
            raise ValueError("margin_of_safety must be in the [0, 1) range.")


def latest_snapshot_from_frame(
    symbol: str,
    fundamentals: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> FinancialSnapshot:
    """Build a normalized snapshot from provider fundamentals.

    Alpha Vantage returns quarterly reports ordered newest-first, but cached data may
    be sorted differently, so we sort by the index before selecting the latest period.
    """

    frame = fundamentals.sort_index()
    if as_of is not None:
        frame = frame.loc[frame.index <= as_of]
    if frame.empty:
        raise ValueError("fundamentals dataframe is empty for the requested as_of date.")

    latest = frame.iloc[-1]
    period_end = pd.Timestamp(frame.index[-1]).date()

    return FinancialSnapshot(
        symbol=symbol,
        period_end=period_end,
        revenue=_value(latest, "total_revenue", "totalRevenue"),
        operating_income=_value(latest, "operating_income", "operatingIncome"),
        income_tax_expense=abs(_value(latest, "income_tax_expense", "incomeTaxExpense")),
        capital_expenditures=abs(
            _value(latest, "capital_expenditures", "capitalExpenditures")
        ),
        total_debt=_value(
            latest,
            "total_debt",
            "shortLongTermDebtTotal",
            "totalDebt",
            "longTermDebt",
        ),
        cash=_value(
            latest,
            "cash_and_equivalents",
            "cashAndCashEquivalentsAtCarryingValue",
            "cashAndShortTermInvestments",
        )
        + _optional_value(latest, "marketable_securities"),
        shares_outstanding=_value(
            latest,
            "shares_outstanding",
            "commonStockSharesOutstanding",
        ),
        free_cash_flow=(
            _optional_value(latest, "free_cash_flow")
            if "free_cash_flow" in latest and pd.notna(latest["free_cash_flow"])
            else _optional_value(latest, "operating_cashflow", "operatingCashflow")
            - abs(_optional_value(latest, "capital_expenditures", "capitalExpenditures"))
        ),
    )


def _value(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return 0.0


def _optional_value(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return 0.0
