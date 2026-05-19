from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from trading_system.backtest.alpha import calculate_alpha_metrics
from trading_system.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from trading_system.strategy.range_synergy import RangeSynergyStrategy
from trading_system.types import AssetProfileResult, StrategyParameters, ValuationSignal


@dataclass(frozen=True)
class SynergySearchResult:
    parameters: StrategyParameters
    backtest: BacktestResult
    train_alpha: float
    candidates_evaluated: int


class FeatureSynergySearch:
    """Grid-search synergy thresholds (NW + range + RSI votes) on train data."""

    def __init__(
        self,
        min_trades: int = 1,
        max_drawdown: float = -0.40,
        initial_cash: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        optimize_for: str = "alpha",
    ) -> None:
        self.min_trades = min_trades
        self.max_drawdown = max_drawdown
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        if optimize_for not in {"alpha", "return"}:
            raise ValueError("optimize_for must be 'alpha' or 'return'.")
        self.optimize_for = optimize_for

    def search(
        self,
        market_data: pd.DataFrame,
        profile: AssetProfileResult,
        valuation: ValuationSignal,
        base_parameters: StrategyParameters | None = None,
    ) -> SynergySearchResult:
        base = base_parameters or StrategyParameters()
        candidates = self._synergy_grid(base)
        best_score = float("-inf")
        best_result: BacktestResult | None = None
        best_params: StrategyParameters | None = None
        best_alpha = 0.0

        benchmark = market_data["close"]

        for params in candidates:
            result = self._run(market_data, profile, valuation, params)
            if result.metrics.trade_count < self.min_trades:
                continue
            if result.metrics.max_drawdown < self.max_drawdown:
                continue

            alpha = calculate_alpha_metrics(
                result.equity_curve["equity"], benchmark
            ).alpha_annualized
            score = alpha if self.optimize_for == "alpha" else result.metrics.total_return

            if score > best_score:
                best_score = score
                best_result = result
                best_params = params
                best_alpha = alpha

        if best_result is None or best_params is None:
            fallback = self._run(market_data, profile, valuation, base)
            return SynergySearchResult(
                parameters=base,
                backtest=fallback,
                train_alpha=0.0,
                candidates_evaluated=len(candidates),
            )

        return SynergySearchResult(
            parameters=best_params,
            backtest=best_result,
            train_alpha=best_alpha,
            candidates_evaluated=len(candidates),
        )

    def _synergy_grid(self, base: StrategyParameters) -> list[StrategyParameters]:
        grid: list[StrategyParameters] = []
        for bw, mult, lb, nw_entry, nw_exit, votes_in, votes_out in product(
            [6.0, 8.0, 10.0],
            [2.0, 2.5, 3.0],
            [24, 32, 48],
            [0.25, 0.30, 0.35],
            [0.68, 0.72, 0.78],
            [2, 3, 4],
            [1, 2],
        ):
            if nw_exit <= nw_entry + 0.30:
                continue
            grid.append(
                StrategyParameters(
                    fast_sma_window=base.fast_sma_window,
                    slow_sma_window=base.slow_sma_window,
                    rsi_window=base.rsi_window,
                    rsi_entry_threshold=base.rsi_entry_threshold,
                    rsi_exit_threshold=base.rsi_exit_threshold,
                    atr_window=base.atr_window,
                    atr_stop_multiple=base.atr_stop_multiple,
                    position_size=base.position_size,
                    use_macd=base.use_macd,
                    max_holding_days=base.max_holding_days,
                    range_window=base.range_window,
                    range_entry_percentile=base.range_entry_percentile,
                    range_exit_percentile=base.range_exit_percentile,
                    nw_bandwidth=bw,
                    nw_multiplier=mult,
                    nw_lookback=lb,
                    nw_entry_position_max=nw_entry,
                    nw_exit_position_min=nw_exit,
                    synergy_min_votes_entry=votes_in,
                    synergy_min_votes_exit=votes_out,
                    use_nw_envelope=True,
                )
            )
        return grid

    def _run(
        self,
        market_data: pd.DataFrame,
        profile: AssetProfileResult,
        valuation: ValuationSignal,
        parameters: StrategyParameters,
    ) -> BacktestResult:
        signals = RangeSynergyStrategy(profile, valuation, parameters).build_signals(market_data)
        return BacktestEngine(
            BacktestConfig(
                initial_cash=self.initial_cash,
                position_size=parameters.position_size,
                commission_rate=self.commission_rate,
                slippage_rate=self.slippage_rate,
                atr_stop_multiple=parameters.atr_stop_multiple,
                max_holding_days=parameters.max_holding_days,
            )
        ).run(signals)
