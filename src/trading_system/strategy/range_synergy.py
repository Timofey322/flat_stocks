from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_system.strategy.feature_synergy import synergy_entry_mask, synergy_exit_mask
from trading_system.technical.indicators import atr, rolling_range, rsi
from trading_system.technical.nadaraya_watson import nadaraya_watson_envelope
from trading_system.types import AssetProfileResult, StrategyParameters, ValuationSignal


@dataclass(frozen=True)
class RangeSynergyStrategy:
    """Single production strategy: range channel + NW envelope + feature synergy."""

    profile: AssetProfileResult
    valuation: ValuationSignal
    parameters: StrategyParameters

    def build_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(market_data.columns)
        if missing:
            raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")

        params = self.parameters
        signals = market_data.sort_index().copy()
        signals["rsi"] = rsi(signals["close"], params.rsi_window)
        signals["atr"] = atr(signals, params.atr_window)
        signals["atr_14"] = signals["atr"]
        signals = signals.join(rolling_range(signals["close"], params.range_window))
        signals = signals.join(
            nadaraya_watson_envelope(
                signals["close"],
                bandwidth=params.nw_bandwidth,
                multiplier=params.nw_multiplier,
                lookback=params.nw_lookback,
            )
        )

        signals["fair_value_per_share"] = self.valuation.fair_value_per_share
        signals["buy_below_price"] = self.valuation.buy_below_price
        signals["fundamental_score"] = self.valuation.fundamental_score

        fund_gate = self._fundamental_gate(signals)
        fair_value_exit = signals["close"] >= self.valuation.fair_value_per_share
        bounce = signals["close"] >= signals["range_low"]

        signals["entry_signal"] = (
            synergy_entry_mask(signals, params, fund_gate) & bounce
        ).fillna(False)
        signals["exit_signal"] = synergy_exit_mask(signals, params, fair_value_exit).fillna(
            False
        )
        return signals

    def _fundamental_gate(self, signals: pd.DataFrame) -> pd.Series:
        near_support = signals["close"] <= self.valuation.buy_below_price * 1.05
        quality_ok = signals["fundamental_score"] >= 0.40
        return near_support | quality_ok
