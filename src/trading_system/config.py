from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from trading_system.types import StrategyParameters


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    database_path: Path
    default_commission_rate: float = 0.001
    default_slippage_rate: float = 0.0005

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.getenv("FLAT_STOCKS_ROOT", Path.cwd()))
        data_dir = root / "data"
        return cls(
            project_root=root,
            data_dir=data_dir,
            database_path=data_dir / "fundamentals.db",
            default_commission_rate=float(os.getenv("DEFAULT_COMMISSION_RATE", "0.001")),
            default_slippage_rate=float(os.getenv("DEFAULT_SLIPPAGE_RATE", "0.0005")),
        )


# Best variant: NW envelope + feature synergy (highest train return on LI, May 2026).
BEST_STRATEGY_PARAMETERS = StrategyParameters(
    fast_sma_window=20,
    slow_sma_window=50,
    rsi_window=14,
    rsi_entry_threshold=30,
    rsi_exit_threshold=70,
    atr_window=14,
    atr_stop_multiple=2.0,
    position_size=0.30,
    use_macd=False,
    max_holding_days=63,
    range_window=90,
    range_entry_percentile=0.30,
    range_exit_percentile=0.80,
    nw_bandwidth=6.0,
    nw_multiplier=2.5,
    nw_lookback=32,
    nw_entry_position_max=0.25,
    nw_exit_position_min=0.68,
    synergy_min_votes_entry=3,
    synergy_min_votes_exit=2,
    use_nw_envelope=True,
)

settings = Settings.from_env()
