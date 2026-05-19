from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    exposure: float
    trade_count: int


def calculate_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> PerformanceMetrics:
    if equity_curve.empty:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(trades))

    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    elapsed_days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = elapsed_days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    sharpe_ratio = 0.0
    if not returns.empty and returns.std(ddof=0) > 0:
        sharpe_ratio = float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))

    if trades.empty:
        win_rate = 0.0
    else:
        win_rate = float((trades["pnl"] > 0).mean())

    exposure = float((equity_curve["position_value"] > 0).mean())

    return PerformanceMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        exposure=exposure,
        trade_count=len(trades),
    )
