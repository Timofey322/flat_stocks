from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

from trading_system.analysis import run_leakage_audit
from trading_system.backtest import WalkForwardBacktester, calculate_alpha_metrics
from trading_system.config import BEST_STRATEGY_PARAMETERS
from trading_system.data import FundamentalDatabase
from trading_system.pipeline import build_context, run_backtest
from trading_system.types import AssetProfileResult

RESEARCH_UNIVERSE = [
    {"symbol": "LI", "name": "Li Auto", "has_seed": True},
    {"symbol": "NIO", "name": "NIO", "has_seed": False},
    {"symbol": "XPEV", "name": "XPeng", "has_seed": False},
    {"symbol": "BABA", "name": "Alibaba", "has_seed": False},
]


def fetch_yahoo_daily(symbol: str, start: dt.datetime) -> pd.DataFrame:
    period1 = int(start.timestamp())
    period2 = int(dt.datetime.now().timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    payload = response.json()["chart"]["result"][0]
    quotes = payload["indicators"]["quote"][0]
    adjusted = payload["indicators"].get("adjclose", [{}])[0].get("adjclose")
    index = pd.to_datetime(payload["timestamp"], unit="s")
    return pd.DataFrame(
        {
            "open": quotes["open"],
            "high": quotes["high"],
            "low": quotes["low"],
            "close": adjusted if adjusted else quotes["close"],
            "volume": quotes["volume"],
        },
        index=index,
    ).dropna()


def load_fundamentals(database: FundamentalDatabase, symbol: str) -> pd.DataFrame:
    if not database.path.exists():
        return pd.DataFrame()
    return database.load_quarterly_fundamentals(symbol)


def analyze_symbol(
    symbol: str,
    name: str,
    market: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> dict:
    fund = fundamentals if fundamentals is not None and not fundamentals.empty else None
    snapshot, profile, valuation = build_context(symbol, market, fund)
    range_fit = profile.profile == "range_bound"
    if not range_fit:
        profile = AssetProfileResult(
            symbol=profile.symbol,
            profile="range_bound",
            growth_rate=profile.growth_rate,
            operating_margin=profile.operating_margin,
            volatility=profile.volatility,
            trend_strength=profile.trend_strength,
            debt_to_revenue=profile.debt_to_revenue,
            reasons=("research run: channel strategy applied despite weak range classifier",),
        )

    _, result, alpha = run_backtest(market, snapshot, profile, valuation, BEST_STRATEGY_PARAMETERS)
    wf = WalkForwardBacktester().run(market, symbol, fundamentals if not fundamentals.empty else None)
    leakage = run_leakage_audit(market, snapshot, profile, valuation, BEST_STRATEGY_PARAMETERS)

    return {
        "symbol": symbol,
        "name": name,
        "tradeable": True,
        "range_classifier_fit": range_fit,
        "profile": profile.profile,
        "channel_low": valuation.buy_below_price,
        "channel_high": valuation.fair_value_per_share,
        "fundamental_score": valuation.fundamental_score,
        "total_return": result.metrics.total_return,
        "benchmark_return": alpha.benchmark_total_return,
        "excess_return": alpha.excess_return,
        "alpha_annualized": alpha.alpha_annualized,
        "beta": alpha.beta,
        "information_ratio": alpha.information_ratio,
        "max_drawdown": result.metrics.max_drawdown,
        "sharpe": result.metrics.sharpe_ratio,
        "trades": result.metrics.trade_count,
        "oos_return": wf.aggregate_metrics.get("total_return", 0.0),
        "oos_windows": len(wf.windows),
        "leakage_passed": leakage.passed,
    }


def main() -> None:
    database = FundamentalDatabase()
    start = dt.datetime(2023, 1, 1)
    results: list[dict] = []

    print("Range Synergy Strategy — multi-company research")
    print(f"Parameters: fixed best variant (NW + synergy, LI-optimized train)\n")

    for item in RESEARCH_UNIVERSE:
        symbol = item["symbol"]
        print(f"Processing {symbol} ({item['name']})...")
        market = fetch_yahoo_daily(symbol, start)
        fundamentals = load_fundamentals(database, symbol) if item["has_seed"] else pd.DataFrame()
        if not fundamentals.empty:
            market = market.loc[market.index >= pd.Timestamp(fundamentals.index.min())]
        row = analyze_symbol(symbol, item["name"], market, fundamentals)
        results.append(row)

    print("\n" + "=" * 88)
    print(
        f"{'Symbol':<8}{'Name':<12}{'Fit':<6}{'Return':>10}{'B&H':>10}"
        f"{'Excess':>10}{'Alpha':>10}{'MDD':>8}{'Trades':>7}"
    )
    print("-" * 88)
    for row in results:
        fit = "yes" if row.get("range_classifier_fit") else "weak"
        print(
            f"{row['symbol']:<8}{row['name']:<12}{fit:<6}"
            f"{row['total_return']:>9.1%}{row['benchmark_return']:>9.1%}"
            f"{row['excess_return']:>9.1%}{row['alpha_annualized']:>9.1%}"
            f"{row['max_drawdown']:>7.1%}{row['trades']:>7.0f}"
        )

    out = Path("data/reports/multi_company_research.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
