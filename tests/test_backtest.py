from __future__ import annotations

import pandas as pd
import pytest

from trading_system.backtest import BacktestConfig, BacktestEngine


def test_backtest_opens_and_closes_position() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    signals = pd.DataFrame(
        {
            "close": [100.0, 110.0, 120.0, 130.0],
            "entry_signal": [True, False, False, False],
            "exit_signal": [False, False, False, True],
        },
        index=dates,
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_cash=100_000,
            position_size=1.0,
            commission_rate=0.0,
            slippage_rate=0.0,
        )
    ).run(signals)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["pnl"] == 30_000
    assert result.equity_curve.iloc[-1]["equity"] == 130_000
    assert result.metrics.total_return == pytest.approx(0.3)


def test_backtest_requires_signal_columns() -> None:
    signals = pd.DataFrame({"close": [100.0]})

    try:
        BacktestEngine().run(signals)
    except ValueError as exc:
        assert "entry_signal" in str(exc)
    else:
        raise AssertionError("Expected missing signal columns to raise ValueError.")
