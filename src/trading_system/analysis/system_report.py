from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_system.analysis.leakage_audit import LeakageAuditReport, run_leakage_audit
from trading_system.backtest.alpha import AlphaMetrics, calculate_alpha_metrics
from trading_system.backtest.engine import BacktestResult
from trading_system.backtest.walk_forward import WalkForwardResult
from trading_system.fundamental.model import FinancialSnapshot
from trading_system.strategy.feature_synergy import (
    nw_entry_vote,
    nw_exit_vote,
    range_entry_vote,
    range_exit_vote,
    rsi_entry_vote,
    rsi_exit_vote,
    vote_count,
)
from trading_system.strategy.range_synergy import RangeSynergyStrategy
from trading_system.types import AssetProfileResult, StrategyParameters, ValuationSignal


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    entry_hit_rate: float
    exit_hit_rate: float


@dataclass(frozen=True)
class SystemAnalysisReport:
    profile: AssetProfileResult
    valuation: ValuationSignal
    parameters: StrategyParameters
    in_sample: BacktestResult
    out_of_sample: WalkForwardResult | None
    alpha_in_sample: AlphaMetrics
    alpha_oos: AlphaMetrics | None
    leakage: LeakageAuditReport
    feature_contributions: tuple[FeatureContribution, ...] = ()
    summary_lines: tuple[str, ...] = field(default_factory=tuple)


def feature_contributions(
    signals: pd.DataFrame,
    parameters: StrategyParameters,
    fundamental_gate: pd.Series,
) -> tuple[FeatureContribution, ...]:
    entry_mask = signals["entry_signal"]
    exit_mask = signals["exit_signal"]
    n = max(len(signals), 1)

    def _rate(vote: pd.Series, mask: pd.Series) -> float:
        subset = vote[mask]
        return float(subset.mean()) if len(subset) else 0.0

    items = [
        ("fundamental", fundamental_gate, pd.Series(False, index=signals.index)),
        ("range_channel", range_entry_vote(signals, parameters), range_exit_vote(signals, parameters)),
        ("rsi", rsi_entry_vote(signals, parameters), rsi_exit_vote(signals, parameters)),
    ]
    if parameters.use_nw_envelope:
        items.append(
            ("nadaraya_watson", nw_entry_vote(signals, parameters), nw_exit_vote(signals, parameters))
        )

    rows: list[FeatureContribution] = []
    for name, entry_vote, exit_vote in items:
        rows.append(
            FeatureContribution(
                feature=name,
                entry_hit_rate=_rate(entry_vote, entry_mask),
                exit_hit_rate=_rate(exit_vote, exit_mask),
            )
        )

    votes_entry = [fundamental_gate, range_entry_vote(signals, parameters), rsi_entry_vote(signals, parameters)]
    if parameters.use_nw_envelope:
        votes_entry.append(nw_entry_vote(signals, parameters))
    synergy_hits = vote_count(votes_entry) >= parameters.synergy_min_votes_entry
    rows.append(
        FeatureContribution(
            feature="synergy_entry_composite",
            entry_hit_rate=float(synergy_hits[entry_mask].mean()) if entry_mask.any() else 0.0,
            exit_hit_rate=0.0,
        )
    )
    return tuple(rows)


def analyze_system(
    market: pd.DataFrame,
    snapshot: FinancialSnapshot,
    profile: AssetProfileResult,
    valuation: ValuationSignal,
    parameters: StrategyParameters,
    in_sample: BacktestResult,
    out_of_sample: WalkForwardResult | None = None,
) -> SystemAnalysisReport:
    rules = RangeSynergyStrategy(profile, valuation, parameters)
    signals = rules.build_signals(market)
    fund_gate = rules._fundamental_gate(signals)

    alpha_is = calculate_alpha_metrics(in_sample.equity_curve["equity"], market["close"])
    alpha_oos = None
    if out_of_sample is not None and not out_of_sample.trades.empty:
        oos_equity = _reconstruct_oos_equity(out_of_sample, market)
        if oos_equity is not None:
            alpha_oos = calculate_alpha_metrics(oos_equity, market["close"].loc[oos_equity.index])

    leakage = run_leakage_audit(market, snapshot, profile, valuation, parameters)
    contributions = feature_contributions(signals, parameters, fund_gate)

    lines: list[str] = []
    lines.append(f"Profile: {profile.profile} | Valuation: {valuation.model}")
    lines.append(
        f"In-sample: return={alpha_is.strategy_total_return:.2%} "
        f"benchmark={alpha_is.benchmark_total_return:.2%} "
        f"excess={alpha_is.excess_return:.2%} "
        f"alpha_ann={alpha_is.alpha_annualized:.2%} beta={alpha_is.beta:.2f} IR={alpha_is.information_ratio:.2f}"
    )
    if alpha_oos:
        lines.append(
            f"OOS walk-forward: excess={alpha_oos.excess_return:.2%} "
            f"alpha_ann={alpha_oos.alpha_annualized:.2%} IR={alpha_oos.information_ratio:.2f}"
        )
    lines.append(f"Leakage audit: {'PASS' if leakage.passed else 'FAIL'}")
    for check in leakage.checks:
        status = "OK" if check.passed else "FAIL"
        lines.append(f"  [{status}] {check.name}: {check.detail}")

    return SystemAnalysisReport(
        profile=profile,
        valuation=valuation,
        parameters=parameters,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        alpha_in_sample=alpha_is,
        alpha_oos=alpha_oos,
        leakage=leakage,
        feature_contributions=contributions,
        summary_lines=tuple(lines),
    )


def _reconstruct_oos_equity(
    wf: WalkForwardResult,
    market: pd.DataFrame,
) -> pd.Series | None:
    if not wf.windows:
        return None
    start = wf.windows[0].test_start
    end = wf.windows[-1].test_end
    close = market["close"].loc[start:end]
    if close.empty:
        return None
    compounded = 1.0
    equity_values = []
    for window in wf.windows:
        compounded *= 1 + window.test_return
        equity_values.append(compounded)
    index = [window.test_end for window in wf.windows]
    return pd.Series(equity_values, index=index).reindex(close.index, method="ffill").dropna() * 100_000
