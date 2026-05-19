from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_system.backtest import WalkForwardBacktester
from trading_system.fundamental import (
    AssetProfiler,
    FinancialAssumptions,
    FinancialSnapshot,
    select_valuation_signal,
)
from trading_system.config import BEST_STRATEGY_PARAMETERS
from trading_system.strategy.range_synergy import RangeSynergyStrategy
from trading_system.technical.indicators import rolling_range
from trading_system.types import StrategyParameters


def _li_like_snapshot() -> FinancialSnapshot:
    return FinancialSnapshot(
        symbol="LI",
        period_end=date(2025, 12, 31),
        revenue=4_100_000_000,
        operating_income=-63_000_000,
        income_tax_expense=0,
        capital_expenditures=146_000_000,
        total_debt=500_000_000,
        cash=14_500_000_000,
        shares_outstanding=1_671_855_018,
    )


def _sideways_market(length: int = 280) -> pd.DataFrame:
    dates = pd.date_range("2023-01-03", periods=length, freq="B")
    close = 24 + 4 * np.sin(np.linspace(0, 14 * np.pi, length))
    close += np.random.default_rng(42).normal(0, 0.15, length)
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 2_000_000,
        },
        index=dates,
    )


def _flat_fundamentals() -> pd.DataFrame:
    return pd.DataFrame(
        {"totalRevenue": [3.6e9, 3.8e9, 4.0e9, 4.1e9, 4.0e9]},
        index=pd.date_range("2024-03-31", periods=5, freq="QE"),
    )


def test_profiler_detects_range_bound_channel() -> None:
    market = _sideways_market()
    profile = AssetProfiler().profile(_li_like_snapshot(), market, _flat_fundamentals())

    assert profile.profile == "range_bound"
    assert abs(profile.trend_strength) < 0.35


def test_range_band_valuation_uses_price_channel() -> None:
    market = _sideways_market()
    profile = AssetProfiler().profile(_li_like_snapshot(), market, _flat_fundamentals())
    signal = select_valuation_signal(
        _li_like_snapshot(), profile, FinancialAssumptions(), market_data=market
    )

    assert signal.model == "range_band"
    assert signal.buy_below_price < signal.fair_value_per_share
    assert signal.allows_entry(current_price=signal.buy_below_price)


def test_range_bound_strategy_generates_entries() -> None:
    market = _sideways_market()
    profile = AssetProfiler().profile(_li_like_snapshot(), market, _flat_fundamentals())
    valuation = select_valuation_signal(
        _li_like_snapshot(), profile, FinancialAssumptions(), market_data=market
    )
    parameters = BEST_STRATEGY_PARAMETERS
    signals = RangeSynergyStrategy(profile, valuation, parameters).build_signals(market)

    assert {"range_position", "nw_position"}.issubset(signals.columns)
    assert signals["entry_signal"].sum() > 0


def test_rolling_range_position_between_zero_and_one() -> None:
    close = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 12, 11])
    frame = rolling_range(close, window=5).dropna()

    assert frame["range_position"].between(0, 1).all()


def test_walk_forward_runs_for_range_bound_profile() -> None:
    market = _sideways_market(200)
    profile = AssetProfiler().profile(_li_like_snapshot(), market, _flat_fundamentals())
    valuation = select_valuation_signal(
        _li_like_snapshot(), profile, FinancialAssumptions(), market_data=market
    )

    result = WalkForwardBacktester(train_size=100, test_size=40).run(market, "LI")

    assert profile.profile == "range_bound"
    assert len(result.windows) >= 1


def test_best_parameters_use_nw_envelope() -> None:
    assert BEST_STRATEGY_PARAMETERS.use_nw_envelope is True
    assert BEST_STRATEGY_PARAMETERS.synergy_min_votes_entry == 3
