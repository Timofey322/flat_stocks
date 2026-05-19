from __future__ import annotations

import pandas as pd

from trading_system.fundamental.model import FinancialSnapshot
from trading_system.types import AssetProfile, AssetProfileResult


class AssetProfiler:
    """Detect whether an asset trades in a sideways channel (range-bound)."""

    def profile(
        self,
        snapshot: FinancialSnapshot,
        market_data: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> AssetProfileResult:
        growth_rate = self._revenue_growth_rate(fundamentals)
        operating_margin = snapshot.operating_margin
        volatility = self._annualized_volatility(market_data)
        trend_strength = self._trend_strength(market_data)
        debt_to_revenue = snapshot.total_debt / snapshot.revenue if snapshot.revenue > 0 else 0.0

        reasons: list[str] = []
        if self._is_range_bound(market_data, trend_strength, volatility):
            profile: AssetProfile = "range_bound"
            reasons.append("sideways price channel with weak directional trend")
        else:
            profile = "not_range"
            reasons.append("trend or volatility outside range-trading regime")

        return AssetProfileResult(
            symbol=snapshot.symbol,
            profile=profile,
            growth_rate=growth_rate,
            operating_margin=operating_margin,
            volatility=volatility,
            trend_strength=trend_strength,
            debt_to_revenue=debt_to_revenue,
            reasons=tuple(reasons),
        )

    def _revenue_growth_rate(self, fundamentals: pd.DataFrame | None) -> float:
        if fundamentals is None or fundamentals.empty:
            return 0.0
        revenue_column = "total_revenue" if "total_revenue" in fundamentals.columns else "totalRevenue"
        if revenue_column not in fundamentals.columns:
            return 0.0
        revenues = (
            fundamentals.sort_index()[revenue_column]
            .dropna()
            .astype(float)
            .loc[lambda series: series > 0]
        )
        if len(revenues) >= 5:
            return revenues.iloc[-1] / revenues.iloc[-5] - 1
        if len(revenues) >= 2:
            return revenues.iloc[-1] / revenues.iloc[0] - 1
        return 0.0

    def _annualized_volatility(self, market_data: pd.DataFrame) -> float:
        if market_data.empty or "close" not in market_data.columns:
            return 0.0
        returns = market_data["close"].astype(float).pct_change().dropna()
        if returns.empty:
            return 0.0
        return float(returns.std(ddof=0) * (252**0.5))

    def _trend_strength(self, market_data: pd.DataFrame) -> float:
        if market_data.empty or "close" not in market_data.columns:
            return 0.0
        close = market_data["close"].dropna().astype(float)
        if len(close) < 2 or close.iloc[0] == 0:
            return 0.0
        lookback = close.tail(min(len(close), 252))
        return float(lookback.iloc[-1] / lookback.iloc[0] - 1)

    def _is_range_bound(
        self,
        market_data: pd.DataFrame,
        trend_strength: float,
        volatility: float,
    ) -> bool:
        if market_data.empty or "close" not in market_data.columns:
            return False
        close = market_data["close"].dropna().astype(float)
        if len(close) < 60:
            return False
        lookback = close.tail(min(len(close), 252))
        mean_price = float(lookback.mean())
        if mean_price <= 0:
            return False
        range_coefficient = float((lookback.max() - lookback.min()) / mean_price)
        return (
            abs(trend_strength) < 0.35
            and 0.15 <= range_coefficient <= 1.0
            and 0.15 <= volatility <= 0.75
        )
