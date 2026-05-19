from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_system.fundamental.model import FinancialAssumptions, FinancialSnapshot
from trading_system.types import AssetProfileResult, ValuationSignal


@dataclass(frozen=True)
class IntrinsicValueResult:
    symbol: str
    fair_value_per_share: float
    buy_below_price: float


class RangeBandValuation:
    """Support/resistance from historical price channel + balance-sheet quality score."""

    def signal(
        self,
        snapshot: FinancialSnapshot,
        profile: AssetProfileResult,
        market_data: pd.DataFrame,
        assumptions: FinancialAssumptions | None = None,
        as_of: pd.Timestamp | None = None,
    ) -> ValuationSignal:
        assumptions = assumptions or FinancialAssumptions()
        frame = market_data.sort_index()
        if as_of is not None:
            frame = frame.loc[frame.index <= as_of]
        close = frame["close"].dropna().astype(float)
        lookback = close.tail(min(len(close), 252))
        range_low = float(lookback.quantile(0.10))
        range_high = float(lookback.quantile(0.90))
        channel_mid = (range_low + range_high) / 2

        balance_sheet_score = _clamp(1 - profile.debt_to_revenue) * 0.5
        cash_score = _clamp(snapshot.cash / max(snapshot.revenue, 1.0) / 3.0) * 0.5
        fundamental_score = _clamp(balance_sheet_score + cash_score)

        buy_below = range_low * (1 + assumptions.margin_of_safety * 0.15)
        fair_value = max(range_high, channel_mid)

        return ValuationSignal(
            symbol=snapshot.symbol,
            model="range_band",
            fair_value_per_share=fair_value,
            buy_below_price=buy_below,
            fundamental_score=fundamental_score,
            profile=profile.profile,
            reasons=(
                "range-band: buy near channel support, sell near resistance",
                f"channel_low={range_low:.2f}",
                f"channel_high={range_high:.2f}",
                f"fundamental_score={fundamental_score:.2f}",
            ),
        )


def select_valuation_signal(
    snapshot: FinancialSnapshot,
    profile: AssetProfileResult,
    assumptions: FinancialAssumptions | None = None,
    market_data: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> ValuationSignal:
    if market_data is None or market_data.empty:
        raise ValueError("market_data is required for range-band valuation.")
    return RangeBandValuation().signal(
        snapshot, profile, market_data, assumptions, as_of=as_of
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(float(value), upper))
