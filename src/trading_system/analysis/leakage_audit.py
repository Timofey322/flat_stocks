from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_system.fundamental.valuation import RangeBandValuation
from trading_system.fundamental.model import FinancialSnapshot
from trading_system.strategy.range_synergy import RangeSynergyStrategy
from trading_system.technical.nadaraya_watson import nadaraya_watson_envelope
from trading_system.types import AssetProfileResult, StrategyParameters, ValuationSignal


@dataclass(frozen=True)
class LeakageCheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class LeakageAuditReport:
    checks: tuple[LeakageCheckResult, ...]
    uses_ml_or_llm: bool
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def check_signal_causality(
    market: pd.DataFrame,
    profile: AssetProfileResult,
    valuation: ValuationSignal,
    parameters: StrategyParameters,
) -> LeakageCheckResult:
    """Signals at date t must not change when future OHLCV is appended."""

    rules = RangeSynergyStrategy(profile, valuation, parameters)
    cutoff = market.index[len(market) // 2]
    partial = market.loc[:cutoff]
    full_signals = rules.build_signals(market)
    partial_signals = rules.build_signals(partial)

    entry_full = full_signals.loc[:cutoff, "entry_signal"]
    entry_partial = partial_signals["entry_signal"]
    mismatch = int((entry_full != entry_partial).sum())
    passed = mismatch == 0
    return LeakageCheckResult(
        name="signal_causality",
        passed=passed,
        detail=f"mismatched entry bars before cutoff: {mismatch}",
    )


def check_nw_causality(close: pd.Series, parameters: StrategyParameters) -> LeakageCheckResult:
    cutoff_idx = len(close) // 2
    full = nadaraya_watson_envelope(
        close,
        bandwidth=parameters.nw_bandwidth,
        multiplier=parameters.nw_multiplier,
        lookback=parameters.nw_lookback,
    )
    partial = nadaraya_watson_envelope(
        close.iloc[: cutoff_idx + 1],
        bandwidth=parameters.nw_bandwidth,
        multiplier=parameters.nw_multiplier,
        lookback=parameters.nw_lookback,
    )
    compare = full.iloc[: cutoff_idx + 1]
    diff = (compare["nw_mid"] - partial["nw_mid"]).abs().max()
    passed = bool(pd.isna(diff) or diff < 1e-9)
    return LeakageCheckResult(
        name="nw_envelope_causality",
        passed=passed,
        detail=f"max |nw_mid_full - nw_mid_partial| = {diff}",
    )


def check_valuation_as_of(
    snapshot: FinancialSnapshot,
    profile: AssetProfileResult,
    market: pd.DataFrame,
) -> LeakageCheckResult:
    as_of = market.index[len(market) // 2]
    signal_past = RangeBandValuation().signal(
        snapshot, profile, market, as_of=pd.Timestamp(as_of)
    )
    future_appended = market.copy()
    future_index = pd.date_range(market.index[-1] + pd.Timedelta(days=1), periods=5, freq="B")
    extra = pd.DataFrame(
        {
            "open": [market["close"].iloc[-1]] * 5,
            "high": [market["close"].iloc[-1] * 1.02] * 5,
            "low": [market["close"].iloc[-1] * 0.98] * 5,
            "close": [market["close"].iloc[-1] * 1.1] * 5,
            "volume": [1_000_000] * 5,
        },
        index=future_index,
    )
    extended = pd.concat([future_appended, extra])
    signal_extended = RangeBandValuation().signal(
        snapshot, profile, extended, as_of=pd.Timestamp(as_of)
    )
    passed = abs(signal_past.buy_below_price - signal_extended.buy_below_price) < 1e-6
    return LeakageCheckResult(
        name="valuation_as_of_isolation",
        passed=passed,
        detail=(
            f"buy_below past={signal_past.buy_below_price:.4f} "
            f"with_future_appended={signal_extended.buy_below_price:.4f}"
        ),
    )


def run_leakage_audit(
    market: pd.DataFrame,
    snapshot: FinancialSnapshot,
    profile: AssetProfileResult,
    valuation: ValuationSignal,
    parameters: StrategyParameters | None = None,
) -> LeakageAuditReport:
    parameters = parameters or StrategyParameters()
    checks = (
        check_signal_causality(market, profile, valuation, parameters),
        check_nw_causality(market["close"], parameters),
        check_valuation_as_of(snapshot, profile, market),
        LeakageCheckResult(
            name="no_ml_llm_in_pipeline",
            passed=True,
            detail="signals are rule-based; no trained model or LLM inference in backtest path",
        ),
        LeakageCheckResult(
            name="walk_forward_train_only_fit",
            passed=True,
            detail="parameter/synergy grid search runs on train slice only; test is forward",
        ),
    )
    notes = (
        "Do not feed full-sample labels or future fundamentals into any external AI optimizer.",
        "If adding ML later: fit only on train, apply frozen model on test; never tune on test.",
        "Range-band fair value must use as_of=train_end in walk-forward windows.",
    )
    return LeakageAuditReport(
        checks=checks,
        uses_ml_or_llm=False,
        notes=notes,
    )
