from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_system.data.database import FundamentalDatabase

SEEDS_DIR = Path(__file__).resolve().parent / "seeds"


def load_quarterly_seed(
    database: FundamentalDatabase,
    *,
    symbol: str,
    company_name: str,
    seed_filename: str,
    cik: str | None = None,
    source: str = "earnings_press_releases_seed",
) -> dict[str, int]:
    seed_path = SEEDS_DIR / seed_filename
    payload: list[dict[str, Any]] = json.loads(seed_path.read_text(encoding="utf-8"))

    database.upsert_company(
        symbol,
        company_name,
        cik=cik,
        source=source,
        updated_at=datetime.now(UTC).isoformat(),
    )

    income_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    cashflow_rows: list[dict[str, Any]] = []
    filing_rows: list[dict[str, Any]] = []

    for row in payload:
        period_end = row["period_end"]
        income_rows.append(
            {
                "symbol": symbol,
                "period_end": period_end,
                "fiscal_year": row["fiscal_year"],
                "fiscal_quarter": row["fiscal_quarter"],
                "reported_currency": "USD",
                "total_revenue": row.get("total_revenue"),
                "operating_income": row.get("operating_income"),
                "net_income": row.get("net_income"),
                "operating_expenses": row.get("operating_expenses"),
                "income_tax_expense": row.get("income_tax_expense"),
                "source": source,
            }
        )
        if any(
            row.get(field)
            for field in ("cash_and_equivalents", "total_debt", "shares_outstanding")
        ):
            balance_rows.append(
                {
                    "symbol": symbol,
                    "period_end": period_end,
                    "fiscal_year": row["fiscal_year"],
                    "fiscal_quarter": row["fiscal_quarter"],
                    "reported_currency": "USD",
                    "cash_and_equivalents": row.get("cash_and_equivalents"),
                    "marketable_securities": row.get("marketable_securities"),
                    "total_debt": row.get("total_debt"),
                    "shares_outstanding": row.get("shares_outstanding"),
                    "source": source,
                }
            )
        if row.get("operating_cashflow") is not None:
            capex = abs(float(row.get("capital_expenditures") or 0))
            operating = float(row["operating_cashflow"])
            cashflow_rows.append(
                {
                    "symbol": symbol,
                    "period_end": period_end,
                    "fiscal_year": row["fiscal_year"],
                    "fiscal_quarter": row["fiscal_quarter"],
                    "reported_currency": "USD",
                    "operating_cashflow": operating,
                    "capital_expenditures": capex,
                    "free_cash_flow": operating - capex,
                    "source": source,
                }
            )
        filing_rows.append(
            {
                "symbol": symbol,
                "fiscal_year": row["fiscal_year"],
                "fiscal_quarter": row["fiscal_quarter"],
                "period_end": period_end,
                "form_type": "EarningsRelease",
                "filed_date": None,
                "source_url": f"seed://{seed_filename}",
                "title": f"{company_name} {row['fiscal_quarter']} {row['fiscal_year']} seed",
            }
        )

    return {
        "filings": database.upsert_rows("filings", filing_rows),
        "income_statement": database.upsert_rows("income_statement", income_rows),
        "balance_sheet": database.upsert_rows("balance_sheet", balance_rows),
        "cash_flow": database.upsert_rows("cash_flow", cashflow_rows),
    }
