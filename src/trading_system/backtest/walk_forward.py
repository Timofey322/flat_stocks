from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_system.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from trading_system.config import BEST_STRATEGY_PARAMETERS
from trading_system.pipeline import build_context
from trading_system.strategy.range_synergy import RangeSynergyStrategy
from trading_system.types import StrategyParameters, ValuationSignal


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_return: float
    test_max_drawdown: float
    test_trade_count: int


@dataclass(frozen=True)
class WalkForwardResult:
    windows: tuple[WalkForwardWindow, ...]
    trades: pd.DataFrame
    aggregate_metrics: dict[str, float]

    @property
    def profitable_windows(self) -> int:
        return sum(1 for window in self.windows if window.test_return > 0)


class WalkForwardBacktester:
    """Out-of-sample evaluation with fixed best parameters (no re-tuning on test)."""

    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int | None = None,
        parameters: StrategyParameters | None = None,
    ) -> None:
        if train_size < 50 or test_size < 10:
            raise ValueError("train_size and test_size are too small.")
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.parameters = parameters or BEST_STRATEGY_PARAMETERS

    def run(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        fundamentals: pd.DataFrame | None = None,
    ) -> WalkForwardResult:
        frame = market_data.sort_index()
        min_fund_date = (
            pd.Timestamp(fundamentals.index.min())
            if fundamentals is not None and not fundamentals.empty
            else None
        )

        windows: list[WalkForwardWindow] = []
        trades: list[pd.DataFrame] = []
        start = 0

        while start + self.train_size + self.test_size <= len(frame):
            train = frame.iloc[start : start + self.train_size]
            test = frame.iloc[start + self.train_size : start + self.train_size + self.test_size]
            train_end = pd.Timestamp(train.index[-1])

            if min_fund_date is not None and train_end < min_fund_date:
                start += self.step_size
                continue

            _, profile, valuation = build_context(
                symbol, frame.iloc[: start + self.train_size], fundamentals, as_of=train_end
            )
            if profile.profile != "range_bound":
                start += self.step_size
                continue

            test_result = self._run_test_window(train, test, profile, valuation)

            if not test_result.trades.empty:
                trade_frame = test_result.trades.copy()
                trade_frame["window"] = len(windows)
                trades.append(trade_frame)

            windows.append(
                WalkForwardWindow(
                    train_start=pd.Timestamp(train.index[0]),
                    train_end=train_end,
                    test_start=pd.Timestamp(test.index[0]),
                    test_end=pd.Timestamp(test.index[-1]),
                    test_return=test_result.metrics.total_return,
                    test_max_drawdown=test_result.metrics.max_drawdown,
                    test_trade_count=test_result.metrics.trade_count,
                )
            )
            start += self.step_size

        combined = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        return WalkForwardResult(
            windows=tuple(windows),
            trades=combined,
            aggregate_metrics=self._aggregate_metrics(windows),
        )

    def _run_test_window(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        profile,
        valuation: ValuationSignal,
    ) -> BacktestResult:
        warmup = max(
            self.parameters.slow_sma_window,
            self.parameters.nw_lookback,
            self.parameters.range_window,
        )
        context = pd.concat([train.tail(warmup), test])
        signals = RangeSynergyStrategy(profile, valuation, self.parameters).build_signals(context)
        return BacktestEngine(
            BacktestConfig(
                position_size=self.parameters.position_size,
                atr_stop_multiple=self.parameters.atr_stop_multiple,
                max_holding_days=self.parameters.max_holding_days,
            )
        ).run(signals.loc[test.index])

    def _aggregate_metrics(self, windows: list[WalkForwardWindow]) -> dict[str, float]:
        if not windows:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "trade_count": 0.0,
                "profitable_window_rate": 0.0,
            }
        compounded = 1.0
        trade_count = 0
        max_drawdown = 0.0
        profitable = 0
        for window in windows:
            compounded *= 1 + window.test_return
            trade_count += window.test_trade_count
            max_drawdown = min(max_drawdown, window.test_max_drawdown)
            profitable += int(window.test_return > 0)
        return {
            "total_return": compounded - 1,
            "max_drawdown": max_drawdown,
            "trade_count": float(trade_count),
            "profitable_window_rate": profitable / len(windows),
            "window_count": float(len(windows)),
        }
