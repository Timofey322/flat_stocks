from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlphaMetrics:
    """Strategy performance vs a buy-and-hold benchmark on the same calendar."""

    strategy_total_return: float
    benchmark_total_return: float
    excess_return: float
    alpha_annualized: float
    beta: float
    information_ratio: float
    tracking_error: float
    sharpe_strategy: float
    sharpe_benchmark: float


def align_returns(
    strategy_equity: pd.Series,
    benchmark_close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    strat = strategy_equity.astype(float).pct_change().dropna()
    bench = benchmark_close.astype(float).pct_change().dropna()
    aligned = pd.concat([strat.rename("strategy"), bench.rename("benchmark")], axis=1).dropna()
    return aligned["strategy"], aligned["benchmark"]


def calculate_alpha_metrics(
    strategy_equity: pd.Series,
    benchmark_close: pd.Series,
    risk_free_rate_annual: float = 0.04,
) -> AlphaMetrics:
    strat_returns, bench_returns = align_returns(strategy_equity, benchmark_close)
    if strat_returns.empty:
        return AlphaMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rf_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    strat_excess = strat_returns - rf_daily
    bench_excess = bench_returns - rf_daily

    bench_var = float(bench_excess.var(ddof=0))
    if bench_var > 0:
        beta = float(bench_excess.cov(strat_excess) / bench_var)
    else:
        beta = 0.0

    alpha_daily = float(strat_excess.mean() - beta * bench_excess.mean())
    alpha_annualized = (1 + alpha_daily) ** 252 - 1

    active = strat_returns - bench_returns
    tracking_error = float(active.std(ddof=0) * np.sqrt(252)) if len(active) > 1 else 0.0
    information_ratio = (
        float(active.mean() / active.std(ddof=0) * np.sqrt(252))
        if len(active) > 1 and active.std(ddof=0) > 0
        else 0.0
    )

    strategy_total = float(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1)
    benchmark_total = float(benchmark_close.iloc[-1] / benchmark_close.iloc[0] - 1)

    def _sharpe(series: pd.Series) -> float:
        excess = series - rf_daily
        if excess.std(ddof=0) == 0:
            return 0.0
        return float(np.sqrt(252) * excess.mean() / excess.std(ddof=0))

    return AlphaMetrics(
        strategy_total_return=strategy_total,
        benchmark_total_return=benchmark_total,
        excess_return=strategy_total - benchmark_total,
        alpha_annualized=alpha_annualized,
        beta=beta,
        information_ratio=information_ratio,
        tracking_error=tracking_error,
        sharpe_strategy=_sharpe(strat_returns),
        sharpe_benchmark=_sharpe(bench_returns),
    )


def buy_and_hold_equity(
    close: pd.Series,
    initial_cash: float = 100_000.0,
) -> pd.Series:
    close = close.astype(float)
    shares = initial_cash / close.iloc[0]
    return shares * close
