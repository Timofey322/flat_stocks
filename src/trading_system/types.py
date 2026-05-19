from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AssetProfile = Literal["range_bound", "not_range"]
ReportStatus = Literal["accepted", "weak", "rejected", "no_trade"]
ValuationModelName = Literal["range_band"]


@dataclass(frozen=True)
class AssetProfileResult:
    symbol: str
    profile: AssetProfile
    growth_rate: float
    operating_margin: float
    volatility: float
    trend_strength: float
    debt_to_revenue: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValuationSignal:
    symbol: str
    model: ValuationModelName
    fair_value_per_share: float
    buy_below_price: float
    fundamental_score: float
    profile: AssetProfile
    reasons: tuple[str, ...] = ()

    def allows_entry(self, current_price: float, threshold: float = 0.55) -> bool:
        return current_price <= self.buy_below_price or self.fundamental_score >= threshold


@dataclass(frozen=True)
class StrategyParameters:
    fast_sma_window: int = 20
    slow_sma_window: int = 50
    rsi_window: int = 14
    rsi_entry_threshold: float = 30.0
    rsi_exit_threshold: float = 70.0
    atr_window: int = 14
    atr_stop_multiple: float = 2.0
    position_size: float = 0.30
    use_macd: bool = False
    max_holding_days: int = 63
    range_window: int = 90
    range_entry_percentile: float = 0.30
    range_exit_percentile: float = 0.80
    nw_bandwidth: float = 6.0
    nw_multiplier: float = 2.5
    nw_lookback: int = 32
    nw_entry_position_max: float = 0.25
    nw_exit_position_min: float = 0.68
    synergy_min_votes_entry: int = 3
    synergy_min_votes_exit: int = 2
    use_nw_envelope: bool = True


@dataclass(frozen=True)
class StrategyReport:
    symbol: str
    profile: AssetProfile
    valuation_model: ValuationModelName
    parameters: StrategyParameters
    status: ReportStatus
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    profitable_windows: int = 0
    total_windows: int = 0
    reasons: tuple[str, ...] = ()
