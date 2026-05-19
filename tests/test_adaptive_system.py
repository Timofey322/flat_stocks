from __future__ import annotations

from datetime import date

import pandas as pd

from trading_system.backtest import WalkForwardBacktester
from trading_system.config import BEST_STRATEGY_PARAMETERS
from trading_system.fundamental import FinancialSnapshot, select_valuation_signal
from trading_system.fundamental.profiler import AssetProfiler
from trading_system.pipeline import run_backtest
from trading_system.strategy.range_synergy import RangeSynergyStrategy


def _snapshot() -> FinancialSnapshot:
    return FinancialSnapshot(
        symbol="TEST",
        period_end=date(2025, 1, 31),
        revenue=4_000_000_000,
        operating_income=200_000_000,
        income_tax_expense=0,
        capital_expenditures=100_000_000,
        total_debt=500_000_000,
        cash=10_000_000_000,
        shares_outstanding=1_000_000_000,
    )


def _sideways_market(length: int = 280) -> pd.DataFrame:
    import numpy as np

    dates = pd.date_range("2023-01-03", periods=length, freq="B")
    close = 24 + 4 * np.sin(np.linspace(0, 14 * np.pi, length))
    close += np.random.default_rng(42).normal(0, 0.15, length)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 2_000_000,
        },
        index=dates,
    )


def test_profiler_detects_range_bound() -> None:
    market = _sideways_market()
    profile = AssetProfiler().profile(_snapshot(), market, None)
    assert profile.profile == "range_bound"


def test_range_synergy_generates_signals() -> None:
    market = _sideways_market()
    profile = AssetProfiler().profile(_snapshot(), market, None)
    valuation = select_valuation_signal(_snapshot(), profile, market_data=market)
    signals = RangeSynergyStrategy(profile, valuation, BEST_STRATEGY_PARAMETERS).build_signals(
        market
    )
    assert {"nw_position", "range_position"}.issubset(signals.columns)


def test_pipeline_backtest_runs() -> None:
    market = _sideways_market()
    profile = AssetProfiler().profile(_snapshot(), market, None)
    valuation = select_valuation_signal(_snapshot(), profile, market_data=market)
    _, result, alpha = run_backtest(market, _snapshot(), profile, valuation)
    assert result.metrics.trade_count >= 0
    assert isinstance(alpha.alpha_annualized, float)


def test_walk_forward_with_fixed_best_params() -> None:
    market = _sideways_market(180)
    result = WalkForwardBacktester(train_size=80, test_size=30).run(market, "TEST", None)
    assert "total_return" in result.aggregate_metrics
