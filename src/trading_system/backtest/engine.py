from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_system.backtest.metrics import PerformanceMetrics, calculate_metrics


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    position_size: float = 0.25
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    atr_stop_multiple: float = 3.0
    max_holding_days: int = 252

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if not 0 < self.position_size <= 1:
            raise ValueError("position_size must be in the (0, 1] range.")
        if self.commission_rate < 0 or self.slippage_rate < 0:
            raise ValueError("commission_rate and slippage_rate must be non-negative.")


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: PerformanceMetrics


class BacktestEngine:
    """Long-only single-asset backtest engine for the MVP strategy."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.config.validate()

    def run(self, signals: pd.DataFrame) -> BacktestResult:
        required = {"close", "entry_signal", "exit_signal"}
        missing = required.difference(signals.columns)
        if missing:
            raise ValueError(f"Signal frame is missing columns: {sorted(missing)}")

        frame = signals.sort_index()
        cash = self.config.initial_cash
        quantity = 0.0
        entry_price = 0.0
        entry_date: pd.Timestamp | None = None
        stop_price: float | None = None

        equity_rows: list[dict[str, float]] = []
        trades: list[Trade] = []

        for date, row in frame.iterrows():
            timestamp = pd.Timestamp(date)
            close = float(row["close"])
            position_value = quantity * close

            if quantity > 0:
                exit_reason = self._exit_reason(row, close, stop_price, entry_date, timestamp)
                if exit_reason:
                    exit_price = close * (1 - self.config.slippage_rate)
                    gross_value = quantity * exit_price
                    commission = gross_value * self.config.commission_rate
                    cash += gross_value - commission
                    pnl = (exit_price - entry_price) * quantity - commission
                    return_pct = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
                    trades.append(
                        Trade(
                            entry_date=entry_date or timestamp,
                            exit_date=timestamp,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=quantity,
                            pnl=pnl,
                            return_pct=return_pct,
                            exit_reason=exit_reason,
                        )
                    )
                    quantity = 0.0
                    entry_price = 0.0
                    entry_date = None
                    stop_price = None
                    position_value = 0.0

            if quantity == 0 and bool(row["entry_signal"]):
                entry_price = close * (1 + self.config.slippage_rate)
                allocation = cash * self.config.position_size
                commission = allocation * self.config.commission_rate
                notional = max(allocation - commission, 0.0)
                quantity = notional / entry_price if entry_price > 0 else 0.0
                cash -= notional + commission
                entry_date = timestamp
                atr_value = float(row["atr_14"]) if "atr_14" in row and pd.notna(row["atr_14"]) else 0.0
                stop_price = (
                    entry_price - self.config.atr_stop_multiple * atr_value if atr_value > 0 else None
                )
                position_value = quantity * close

            equity_rows.append(
                {
                    "cash": cash,
                    "position_value": position_value,
                    "equity": cash + position_value,
                }
            )

        if quantity > 0 and entry_date is not None and equity_rows:
            final_date = pd.Timestamp(frame.index[-1])
            final_close = float(frame.iloc[-1]["close"])
            exit_price = final_close * (1 - self.config.slippage_rate)
            gross_value = quantity * exit_price
            commission = gross_value * self.config.commission_rate
            cash += gross_value - commission
            pnl = (exit_price - entry_price) * quantity - commission
            return_pct = (exit_price / entry_price - 1) if entry_price > 0 else 0.0
            trades.append(
                Trade(
                    entry_date=entry_date,
                    exit_date=final_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=pnl,
                    return_pct=return_pct,
                    exit_reason="end_of_period",
                )
            )
            equity_rows[-1] = {
                "cash": cash,
                "position_value": 0.0,
                "equity": cash,
            }

        equity_curve = pd.DataFrame(equity_rows, index=frame.index)
        trades_frame = pd.DataFrame([trade.__dict__ for trade in trades])
        metrics = calculate_metrics(equity_curve, trades_frame)
        return BacktestResult(equity_curve=equity_curve, trades=trades_frame, metrics=metrics)

    def _exit_reason(
        self,
        row: pd.Series,
        close: float,
        stop_price: float | None,
        entry_date: pd.Timestamp | None,
        current_date: pd.Timestamp,
    ) -> str | None:
        if stop_price is not None and close <= stop_price:
            return "atr_stop"
        if bool(row["exit_signal"]):
            return "exit_signal"
        if entry_date is not None and (current_date - entry_date).days >= self.config.max_holding_days:
            return "max_holding_days"
        return None
