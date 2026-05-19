# flat_stocks

Исследовательский бэктестер для **акций в ценовом канале** (range-bound). Один лучший вариант стратегии: **Range Synergy** (канал + RSI + Nadaraya–Watson envelope + фундаментальный фильтр).

Репозиторий: [github.com/Timofey322/flat_stocks](https://github.com/Timofey322/flat_stocks)

## Быстрый старт

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python examples/load_li_reports.py
python examples/run_research.py
pytest
```

## Документация

- **[RESEARCH.md](RESEARCH.md)** — как считаются фундаментал и техника, как они объединяются, исследование на LI / NIO / XPEV / BABA, альфа-метрики, аудит утечек.

## Архитектура

| Слой | Модули |
|------|--------|
| Данные | `data/database.py`, `li_quarterly.json` |
| Фундаментал | `fundamental/profiler.py`, `valuation.py` (range_band) |
| Техника | `technical/indicators.py`, `nadaraya_watson.py` |
| Стратегия | `strategy/range_synergy.py`, `feature_synergy.py` |
| Бэктест | `backtest/engine.py`, `alpha.py`, `walk_forward.py` |
| Параметры | `config.BEST_STRATEGY_PARAMETERS` |

## Границы MVP

Только research backtest. Не брокер, не live trading. Не финансовая рекомендация.
