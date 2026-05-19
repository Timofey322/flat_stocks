from __future__ import annotations

import pandas as pd

from trading_system.technical.indicators import add_default_indicators, rolling_range, rsi, sma


def test_sma_uses_requested_window() -> None:
    values = pd.Series([1, 2, 3, 4, 5], dtype=float)

    result = sma(values, 3)

    assert pd.isna(result.iloc[1])
    assert result.iloc[-1] == 4


def test_rsi_stays_in_expected_range() -> None:
    close = pd.Series([10, 11, 10, 12, 13, 12, 14, 15, 15, 16, 17, 16, 18, 19, 20])

    result = rsi(close, window=5).dropna()

    assert not result.empty
    assert result.between(0, 100).all()


def test_rolling_range_adds_channel_columns() -> None:
    close = pd.Series([20, 21, 22, 21, 20, 19, 20, 21, 22, 23, 22, 21], dtype=float)

    result = rolling_range(close, window=5)

    assert {"range_low", "range_high", "range_position"}.issubset(result.columns)


def test_default_indicators_add_expected_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=220, freq="B")
    frame = pd.DataFrame(
        {
            "open": range(220),
            "high": range(1, 221),
            "low": range(220),
            "close": range(1, 221),
            "volume": 1_000,
        },
        index=dates,
    )

    result = add_default_indicators(frame)

    assert {"sma_50", "sma_200", "rsi_14", "atr_14", "macd", "macd_signal"}.issubset(
        result.columns
    )
