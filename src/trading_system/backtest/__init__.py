"""Backtesting engine and performance metrics."""

from trading_system.backtest.alpha import AlphaMetrics, buy_and_hold_equity, calculate_alpha_metrics
from trading_system.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, Trade
from trading_system.backtest.metrics import PerformanceMetrics, calculate_metrics
from trading_system.backtest.walk_forward import (
    WalkForwardBacktester,
    WalkForwardResult,
    WalkForwardWindow,
)

__all__ = [
    "AlphaMetrics",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "Trade",
    "WalkForwardBacktester",
    "WalkForwardResult",
    "WalkForwardWindow",
    "buy_and_hold_equity",
    "calculate_alpha_metrics",
    "calculate_metrics",
]
