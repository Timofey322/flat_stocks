from __future__ import annotations

from trading_system.data.database import FundamentalDatabase
from trading_system.data.seed_loader import load_quarterly_seed

LI_CIK = "0001791706"


def main() -> None:
    database = FundamentalDatabase()
    database.initialize()

    print("Loading Li Auto (LI) quarterly seed data...")
    counts = load_quarterly_seed(
        database,
        symbol="LI",
        company_name="Li Auto Inc.",
        seed_filename="li_quarterly.json",
        cik=LI_CIK,
        source="li_auto_press_releases_seed",
    )
    print(counts)

    summary = database.summary("LI")
    print("\nDatabase summary for LI:")
    for table, count in summary.items():
        print(f"  {table}: {count}")

    fundamentals = database.load_quarterly_fundamentals("LI")
    print(f"\nMerged fundamentals rows: {len(fundamentals)}")
    if not fundamentals.empty:
        print(fundamentals.to_string())


if __name__ == "__main__":
    main()
