from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from trading_system.config import settings


@dataclass(frozen=True)
class FundamentalDatabase:
    """SQLite store for company fundamentals and filing metadata."""

    path: Path = settings.data_dir / "fundamentals.db"

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS companies (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    cik TEXT,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS filings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    fiscal_year INTEGER,
                    fiscal_quarter TEXT,
                    period_end TEXT NOT NULL,
                    form_type TEXT,
                    filed_date TEXT,
                    source_url TEXT,
                    title TEXT,
                    UNIQUE(symbol, period_end, form_type)
                );

                CREATE TABLE IF NOT EXISTS income_statement (
                    symbol TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    fiscal_year INTEGER,
                    fiscal_quarter TEXT,
                    reported_currency TEXT DEFAULT 'USD',
                    total_revenue REAL,
                    cost_of_revenue REAL,
                    gross_profit REAL,
                    operating_expenses REAL,
                    operating_income REAL,
                    income_tax_expense REAL,
                    net_income REAL,
                    eps_diluted REAL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (symbol, period_end)
                );

                CREATE TABLE IF NOT EXISTS balance_sheet (
                    symbol TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    fiscal_year INTEGER,
                    fiscal_quarter TEXT,
                    reported_currency TEXT DEFAULT 'USD',
                    cash_and_equivalents REAL,
                    marketable_securities REAL,
                    total_assets REAL,
                    total_debt REAL,
                    total_liabilities REAL,
                    shareholders_equity REAL,
                    shares_outstanding REAL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (symbol, period_end)
                );

                CREATE TABLE IF NOT EXISTS cash_flow (
                    symbol TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    fiscal_year INTEGER,
                    fiscal_quarter TEXT,
                    reported_currency TEXT DEFAULT 'USD',
                    operating_cashflow REAL,
                    capital_expenditures REAL,
                    investing_cashflow REAL,
                    financing_cashflow REAL,
                    free_cash_flow REAL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (symbol, period_end)
                );

                CREATE INDEX IF NOT EXISTS idx_income_symbol_period
                    ON income_statement(symbol, period_end);
                CREATE INDEX IF NOT EXISTS idx_balance_symbol_period
                    ON balance_sheet(symbol, period_end);
                CREATE INDEX IF NOT EXISTS idx_cashflow_symbol_period
                    ON cash_flow(symbol, period_end);
                """
            )

    def upsert_company(
        self,
        symbol: str,
        name: str,
        *,
        cik: str | None = None,
        source: str,
        updated_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO companies(symbol, name, cik, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    name=excluded.name,
                    cik=excluded.cik,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (symbol, name, cik, source, updated_at),
            )

    def upsert_rows(self, table: str, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0

        columns = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(columns)
        values = [tuple(row[column] for column in columns) for row in rows]

        if table == "filings":
            update_sql = ", ".join(
                f"{column}=excluded.{column}"
                for column in columns
                if column not in {"symbol", "period_end", "form_type"}
            )
            sql = f"""
                INSERT INTO filings ({column_sql})
                VALUES ({placeholders})
                ON CONFLICT(symbol, period_end, form_type) DO UPDATE SET
                {update_sql}
            """
        else:
            update_sql = ", ".join(
                f"{column}=excluded.{column}"
                for column in columns
                if column not in {"symbol", "period_end"}
            )
            sql = f"""
                INSERT INTO {table} ({column_sql})
                VALUES ({placeholders})
                ON CONFLICT(symbol, period_end) DO UPDATE SET
                {update_sql}
            """

        with self.connect() as connection:
            connection.executemany(sql, values)
        return len(rows)

    def load_quarterly_fundamentals(self, symbol: str) -> pd.DataFrame:
        query = """
            SELECT
                i.period_end,
                i.fiscal_year,
                i.fiscal_quarter,
                i.total_revenue,
                i.operating_income,
                i.income_tax_expense,
                i.net_income,
                b.cash_and_equivalents,
                b.marketable_securities,
                b.total_debt,
                b.shares_outstanding,
                c.capital_expenditures,
                c.operating_cashflow,
                c.free_cash_flow
            FROM income_statement i
            LEFT JOIN balance_sheet b
                ON i.symbol = b.symbol AND i.period_end = b.period_end
            LEFT JOIN cash_flow c
                ON i.symbol = c.symbol AND i.period_end = c.period_end
            WHERE i.symbol = ?
            ORDER BY i.period_end
        """
        with self.connect() as connection:
            frame = pd.read_sql_query(query, connection, params=(symbol,))
        if frame.empty:
            return frame
        frame["period_end"] = pd.to_datetime(frame["period_end"])
        return frame.set_index("period_end")

    def summary(self, symbol: str) -> dict[str, int]:
        tables = ("income_statement", "balance_sheet", "cash_flow", "filings")
        counts: dict[str, int] = {}
        with self.connect() as connection:
            for table in tables:
                row = connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE symbol = ?",
                    (symbol,),
                ).fetchone()
                counts[table] = int(row["count"]) if row else 0
        return counts
