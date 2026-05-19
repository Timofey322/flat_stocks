from __future__ import annotations

from trading_system.backtest import PerformanceMetrics, WalkForwardResult
from trading_system.types import AssetProfileResult, ReportStatus, StrategyReport, ValuationSignal


def build_strategy_report(
    profile: AssetProfileResult,
    valuation: ValuationSignal,
    walk_forward: WalkForwardResult,
) -> StrategyReport:
    metrics = walk_forward.aggregate_metrics
    total_return = metrics.get("total_return", 0.0)
    trade_count = metrics.get("trade_count", 0.0)
    profitable_window_rate = metrics.get("profitable_window_rate", 0.0)

    status: ReportStatus
    reasons: list[str] = []
    if trade_count == 0:
        status = "no_trade"
        reasons.append("strategy produced no out-of-sample trades")
    elif total_return > 0 and profitable_window_rate >= 0.50:
        status = "accepted"
        reasons.append("positive out-of-sample return with at least half profitable windows")
    elif total_return > 0:
        status = "weak"
        reasons.append("positive out-of-sample return but unstable window distribution")
    else:
        status = "rejected"
        reasons.append("negative out-of-sample return")

    from trading_system.config import BEST_STRATEGY_PARAMETERS

    parameters = BEST_STRATEGY_PARAMETERS

    return StrategyReport(
        symbol=valuation.symbol,
        profile=profile.profile,
        valuation_model=valuation.model,
        parameters=parameters,
        status=status,
        train_metrics={},
        test_metrics=metrics,
        profitable_windows=walk_forward.profitable_windows,
        total_windows=len(walk_forward.windows),
        reasons=tuple(reasons + list(profile.reasons) + list(valuation.reasons)),
    )


def metrics_to_dict(metrics: PerformanceMetrics) -> dict[str, float]:
    return {
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "sharpe_ratio": metrics.sharpe_ratio,
        "win_rate": metrics.win_rate,
        "exposure": metrics.exposure,
        "trade_count": float(metrics.trade_count),
    }


def valuation_default_parameters():
    from trading_system.types import StrategyParameters

    return StrategyParameters()
