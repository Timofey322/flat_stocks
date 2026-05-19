from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + relative_strength))


def macd(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    macd_line = ema(close, fast_span) - ema(close, slow_span)
    signal_line = ema(macd_line, signal_span)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        },
        index=close.index,
    )


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def rolling_range(close: pd.Series, window: int) -> pd.DataFrame:
    """Rolling price channel and normalized position inside the channel (0=low, 1=high)."""

    range_low = close.rolling(window=window, min_periods=window).min()
    range_high = close.rolling(window=window, min_periods=window).max()
    width = range_high - range_low
    position = (close - range_low) / width.replace(0, pd.NA)
    return pd.DataFrame(
        {
            "range_low": range_low,
            "range_high": range_high,
            "range_position": position,
        },
        index=close.index,
    )


def add_default_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the default indicator set used by the MVP strategy."""

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")

    enriched = frame.copy()
    enriched["sma_50"] = sma(enriched["close"], 50)
    enriched["sma_200"] = sma(enriched["close"], 200)
    enriched["rsi_14"] = rsi(enriched["close"], 14)
    enriched["atr_14"] = atr(enriched, 14)
    return enriched.join(macd(enriched["close"]))
