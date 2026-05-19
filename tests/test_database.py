from __future__ import annotations

from pathlib import Path

from trading_system.data.database import FundamentalDatabase
from trading_system.data.seed_loader import load_quarterly_seed


def test_load_li_seed_into_sqlite(tmp_path: Path) -> None:
    database = FundamentalDatabase(path=tmp_path / "fundamentals.db")
    database.initialize()
    counts = load_quarterly_seed(
        database,
        symbol="LI",
        company_name="Li Auto Inc.",
        seed_filename="li_quarterly.json",
    )

    assert counts["income_statement"] == 8
    frame = database.load_quarterly_fundamentals("LI")
    assert len(frame) == 8
    assert frame["total_revenue"].iloc[-1] > 0
