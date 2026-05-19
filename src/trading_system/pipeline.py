from __future__ import annotations

from datetime import date

import pandas as pd

from trading_system.backtest import BacktestConfig, BacktestEngine, calculate_alpha_metrics
from trading_system.config import BEST_STRATEGY_PARAMETERS
from trading_system.fundamental import (
    AssetProfiler,
    latest_snapshot_from_frame,
    select_valuation_signal,
)
from trading_system.fundamental.model import FinancialSnapshot
from trading_system.strategy.range_synergy import RangeSynergyStrategy
from trading_system.types import AssetProfileResult, StrategyParameters, ValuationSignal


def price_only_snapshot(symbol: str, market: pd.DataFrame) -> FinancialSnapshot:
    """Minimal snapshot when quarterly fundamentals are unavailable."""

    close = float(market["close"].iloc[-1])
    return FinancialSnapshot(
        symbol=symbol,
        period_end=pd.Timestamp(market.index[-1]).date(),
        revenue=1.0,
        operating_income=0.0,
        income_tax_expense=0.0,
        capital_expenditures=0.0,
        total_debt=0.0,
        cash=close * 1e8,
        shares_outstanding=1.0,
    )


def build_context(
    symbol: str,
    market: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> tuple[FinancialSnapshot, AssetProfileResult, ValuationSignal]:
    as_of = as_of or pd.Timestamp(market.index[-1])
    market_slice = market.loc[market.index <= as_of]

    if fundamentals is not None and not fundamentals.empty:
        fund = fundamentals.loc[fundamentals.index <= as_of]
        snapshot = latest_snapshot_from_frame(symbol, fund, as_of=as_of)
        profile = AssetProfiler().profile(snapshot, market_slice, fund)
    else:
        snapshot = price_only_snapshot(symbol, market_slice)
        profile = AssetProfiler().profile(snapshot, market_slice, None)

    valuation = select_valuation_signal(
        snapshot, profile, market_data=market_slice, as_of=as_of
    )
    return snapshot, profile, valuation


def run_backtest(
    market: pd.DataFrame,
    snapshot: FinancialSnapshot,
    profile: AssetProfileResult,
    valuation: ValuationSignal,
    parameters: StrategyParameters | None = None,
):
    params = parameters or BEST_STRATEGY_PARAMETERS
    signals = RangeSynergyStrategy(profile, valuation, params).build_signals(market)
    result = BacktestEngine(
        BacktestConfig(
            position_size=params.position_size,
            atr_stop_multiple=params.atr_stop_multiple,
            max_holding_days=params.max_holding_days,
        )
    ).run(signals)
    alpha = calculate_alpha_metrics(result.equity_curve["equity"], market["close"])
    return signals, result, alpha
