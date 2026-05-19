from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.analysis.leakage_audit import run_leakage_audit
from trading_system.backtest.alpha import calculate_alpha_metrics
from trading_system.fundamental import FinancialSnapshot, select_valuation_signal
from trading_system.optimization.feature_synergy import FeatureSynergySearch
from trading_system.technical.nadaraya_watson import nadaraya_watson_envelope
from trading_system.types import AssetProfileResult, StrategyParameters, ValuationSignal
from datetime import date


def _market(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 20 + 3 * np.sin(np.linspace(0, 10, n))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=dates,
    )


def _profile() -> AssetProfileResult:
    return AssetProfileResult(
        symbol="T",
        profile="range_bound",
        growth_rate=0.05,
        operating_margin=0.1,
        volatility=0.3,
        trend_strength=0.05,
        debt_to_revenue=0.2,
    )


def _valuation() -> ValuationSignal:
    return ValuationSignal(
        symbol="T",
        model="range_band",
        fair_value_per_share=28.0,
        buy_below_price=18.0,
        fundamental_score=0.6,
        profile="range_bound",
    )


def _market_for_valuation(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 20 + 3 * (pd.Series(range(n), index=dates) % 20) / 20
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=dates,
    )


def _snapshot() -> FinancialSnapshot:
    return FinancialSnapshot(
        symbol="T",
        period_end=date(2024, 12, 31),
        revenue=1e9,
        operating_income=1e8,
        income_tax_expense=1e7,
        capital_expenditures=1e7,
        total_debt=1e8,
        cash=5e8,
        shares_outstanding=1e8,
    )


def test_nw_envelope_is_causal() -> None:
    close = _market(80)["close"]
    mid = len(close) // 2
    full = nadaraya_watson_envelope(close, bandwidth=8, multiplier=2.5, lookback=32)
    part = nadaraya_watson_envelope(close.iloc[: mid + 1], bandwidth=8, multiplier=2.5, lookback=32)
    diff = (full.iloc[: mid + 1]["nw_mid"] - part["nw_mid"]).abs().max()
    assert bool(pd.isna(diff) or diff < 1e-9)


def test_synergy_search_returns_parameters() -> None:
    market = _market(140)
    valuation = select_valuation_signal(_snapshot(), _profile(), market_data=market)
    result = FeatureSynergySearch(min_trades=0, optimize_for="alpha").search(
        market, _profile(), valuation
    )
    assert result.parameters.use_nw_envelope is True
    assert result.candidates_evaluated > 0


def test_alpha_metrics_vs_benchmark() -> None:
    market = _market(100)
    equity = (market["close"] / market["close"].iloc[0]) * 100_000
    equity = equity * (1 + pd.Series(np.linspace(0, 0.05, len(equity)), index=equity.index))
    alpha = calculate_alpha_metrics(equity, market["close"])
    assert alpha.benchmark_total_return != 0
    assert isinstance(alpha.alpha_annualized, float)


def test_leakage_audit_passes_on_synthetic() -> None:
    market = _market(160)
    params = StrategyParameters(use_nw_envelope=True, synergy_min_votes_entry=2)
    valuation = select_valuation_signal(_snapshot(), _profile(), market_data=market)
    report = run_leakage_audit(market, _snapshot(), _profile(), valuation, params)
    assert report.passed
