from __future__ import annotations

from datetime import date

import pandas as pd

from trading_system.fundamental import FinancialAssumptions, FinancialSnapshot, select_valuation_signal
from trading_system.fundamental.profiler import AssetProfiler
from trading_system.types import AssetProfileResult


def test_range_band_valuation() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    close = 20 + 3 * (pd.Series(range(120), index=dates) % 20) / 20
    market = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=dates,
    )
    snapshot = FinancialSnapshot(
        symbol="TEST",
        period_end=date(2024, 12, 31),
        revenue=1e9,
        operating_income=1e8,
        income_tax_expense=1e7,
        capital_expenditures=1e7,
        total_debt=1e8,
        cash=5e8,
        shares_outstanding=1e8,
    )
    profile = AssetProfileResult(
        symbol="TEST",
        profile="range_bound",
        growth_rate=0.05,
        operating_margin=0.1,
        volatility=0.3,
        trend_strength=0.05,
        debt_to_revenue=0.1,
    )
    signal = select_valuation_signal(snapshot, profile, FinancialAssumptions(), market_data=market)

    assert signal.model == "range_band"
    assert signal.buy_below_price < signal.fair_value_per_share
    assert 0 <= signal.fundamental_score <= 1


def test_assumptions_reject_invalid_terminal_growth() -> None:
    assumptions = FinancialAssumptions(discount_rate=0.03, terminal_growth_rate=0.03)
    try:
        assumptions.validate()
    except ValueError as exc:
        assert "discount_rate" in str(exc)
    else:
        raise AssertionError("Expected invalid assumptions to raise ValueError.")
