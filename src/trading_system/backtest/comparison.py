"""Several strategies, the same data, the same costs and the same risk engine, side by side.

What differs between the rows is only which bars the strategy turns into signals. Sizing, the
position limit, the daily loss guard, the entry plan, fees and slippage are the same for all,
so a difference in the result is the strategy's (or noise: a few dozen trades in one market
regime do not separate a good strategy from a lucky one, and the table says so)."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Sequence

import pandas as pd

from trading_system.backtest.analysis import t_statistic as _t_stat
from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig
from trading_system.config import Settings
from trading_system.strategy.registry import get_strategy


@dataclass(frozen=True)
class ComparisonRow:
    name: str
    version: str
    signals: int
    trades: int
    total_return_pct: float
    profit_factor: float
    win_rate: float
    max_drawdown_pct: float
    avg_trade_pct: float
    # The average trade divided by its standard error: how many "noises" away from zero it is.
    # Below 2 (in absolute value) the average cannot be told from luck.
    t_stat: float
    costs: Decimal  # fees plus slippage, both already inside the return


def compare_strategies(
    settings: Settings,
    bt_config: BacktestConfig,
    df_15m: Dict[str, pd.DataFrame],
    df_1h: Dict[str, pd.DataFrame],
    names: Sequence[str],
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for name in names:
        strategy = get_strategy(name)  # raises ValueError for an unknown name, before any run
        result = BacktestEngine(settings, bt_config, df_15m, df_1h, strategy=strategy).run()
        trade_returns = [t.pnl_pct for t in result.trades if t.pnl_pct is not None]
        rows.append(
            ComparisonRow(
                name=name,
                version=strategy.version,
                signals=result.signals,
                trades=result.trade_count,
                total_return_pct=result.total_return_pct,
                profit_factor=result.profit_factor,
                win_rate=result.win_rate,
                max_drawdown_pct=result.max_drawdown_pct,
                avg_trade_pct=sum(trade_returns) / len(trade_returns) if trade_returns else 0.0,
                t_stat=_t_stat(trade_returns),
                costs=result.total_fees + result.total_slippage,
            )
        )
    return rows


def format_comparison(rows: Sequence[ComparisonRow]) -> str:
    width = max([len("strategy"), *(len(r.name) for r in rows)]) + 2
    header = (
        f"{'strategy':<{width}}{'signals':>8}{'trades':>7}{'return':>9}{'PF':>6}{'win':>7}"
        f"{'max DD':>8}{'avg trade':>11}{'t':>7}{'costs $':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.name:<{width}}{r.signals:>8}{r.trades:>7}{r.total_return_pct * 100:>+8.2f}%"
            f"{r.profit_factor:>6.2f}{r.win_rate * 100:>6.1f}%{r.max_drawdown_pct * 100:>7.2f}%"
            f"{r.avg_trade_pct * 100:>+10.2f}%{r.t_stat:>+7.1f}{float(r.costs):>9.0f}"
        )
    return "\n".join(lines)
