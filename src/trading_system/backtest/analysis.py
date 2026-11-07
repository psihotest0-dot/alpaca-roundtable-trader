"""Reading a backtest result (Phase 3.4 and 3.5): what a trade is worth on average, how steady
that is over time and across symbols, how it holds up under higher costs, and whether the
Oracle kept us out of bad trades.

Everything here reads a finished `BacktestResult`, except the cost sweep, which runs more
backtests. No strategy is fitted to anything: a strategy has fixed parameters, so the
"out of sample" question is whether its result is steady across the periods of one run, and
`by_period` shows exactly that. Fitting parameters on one stretch and testing on the next
needs a search over parameters, which belongs to the work on the strategies themselves."""

import math
import statistics
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

import pandas as pd

from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.models import BacktestConfig, BacktestResult, BacktestTrade
from trading_system.config import Settings
from trading_system.risk.daily_equity import trading_day
from trading_system.risk.entry_plan import AtrExit
from trading_system.strategy.base import BaseStrategy

TRADING_DAYS_PER_YEAR = 252


def welch_t(first: Sequence[float], second: Sequence[float]) -> float:
    """How many standard errors apart the averages of two groups are (Welch's t, which does not
    assume equal spread). 0 with fewer than two values in a group. If neither group has any
    spread, the averages either agree (0) or differ for certain (infinity)."""
    if len(first) < 2 or len(second) < 2:
        return 0.0
    error = math.sqrt(
        statistics.variance(first) / len(first) + statistics.variance(second) / len(second)
    )
    difference = statistics.mean(first) - statistics.mean(second)
    if error == 0:
        return 0.0 if difference == 0 else math.copysign(math.inf, difference)
    return difference / error


def t_statistic(returns: Sequence[float]) -> float:
    """The average over its standard error. Below 2 in absolute value the average cannot be
    told from luck. 0 when there are fewer than two values or no spread."""
    if len(returns) < 2:
        return 0.0
    spread = statistics.stdev(returns)
    if spread == 0:
        return 0.0
    return statistics.mean(returns) / (spread / math.sqrt(len(returns)))


# --- what a trade is worth ------------------------------------------------------------------


@dataclass(frozen=True)
class Expectancy:
    trades: int
    avg_pnl: Decimal  # money per trade
    avg_pct: float  # per trade, on its cost basis
    avg_win_pct: float
    avg_loss_pct: float  # negative
    payoff: float  # average win over average loss; 0 if there is no loss to divide by
    t_stat: float


def expectancy(trades: Sequence[BacktestTrade]) -> Expectancy:
    pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    wins = [p for p in pcts if p > 0]
    losses = [p for p in pcts if p <= 0]
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    pnl = sum((t.pnl for t in trades if t.pnl is not None), Decimal("0"))
    return Expectancy(
        trades=len(trades),
        avg_pnl=pnl / len(trades) if trades else Decimal("0"),
        avg_pct=statistics.mean(pcts) if pcts else 0.0,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        payoff=avg_win / abs(avg_loss) if avg_loss < 0 else 0.0,
        t_stat=t_statistic(pcts),
    )


def sharpe_ratio(result: BacktestResult) -> float:
    """Mean over standard deviation of the daily returns, scaled to a year of 252 days. The
    risk-free rate is taken as zero. 0 with fewer than two days."""
    returns = [r for _, r in result.daily_returns()]
    if len(returns) < 2:
        return 0.0
    spread = statistics.stdev(returns)
    if spread == 0:
        return 0.0
    return statistics.mean(returns) / spread * math.sqrt(TRADING_DAYS_PER_YEAR)


# --- how steady it is -----------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolRow:
    symbol: str
    trades: int
    pnl: Decimal
    win_rate: float
    avg_pct: float


def by_symbol(result: BacktestResult) -> List[SymbolRow]:
    """One row per symbol that traded, best money result first."""
    groups: Dict[str, List[BacktestTrade]] = {}
    for trade in result.trades:
        groups.setdefault(trade.symbol, []).append(trade)
    rows = []
    for symbol, trades in groups.items():
        pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
        rows.append(
            SymbolRow(
                symbol=symbol,
                trades=len(trades),
                pnl=sum((t.pnl for t in trades if t.pnl is not None), Decimal("0")),
                win_rate=sum(1 for t in trades if t.pnl and t.pnl > 0) / len(trades),
                avg_pct=statistics.mean(pcts) if pcts else 0.0,
            )
        )
    return sorted(rows, key=lambda r: r.pnl, reverse=True)


@dataclass(frozen=True)
class PeriodRow:
    first_day: str
    last_day: str
    days: int
    trades: int  # closed in the period
    return_pct: float  # the account's change over the period, open positions included
    win_rate: float


def by_period(result: BacktestResult, periods: int = 4) -> List[PeriodRow]:
    """The replay cut into `periods` consecutive stretches of equal length (in trading days),
    each with its own result. Parameters are the same throughout, so a result that holds in
    every stretch is steadier evidence than the same total made in one."""
    days = sorted(result.daily_equity)
    if not days or periods < 1:
        return []
    periods = min(periods, len(days))
    size, extra = divmod(len(days), periods)
    rows: List[PeriodRow] = []
    previous_equity = float(result.initial_capital)
    start = 0
    for index in range(periods):
        stop = start + size + (1 if index < extra else 0)
        chunk = days[start:stop]
        end_equity = result.daily_equity[chunk[-1]]
        closed = [
            t
            for t in result.trades
            if t.exit_time and trading_day(t.exit_time).isoformat() in chunk
        ]
        rows.append(
            PeriodRow(
                first_day=chunk[0],
                last_day=chunk[-1],
                days=len(chunk),
                trades=len(closed),
                return_pct=end_equity / previous_equity - 1 if previous_equity else 0.0,
                win_rate=sum(1 for t in closed if t.pnl and t.pnl > 0) / len(closed)
                if closed
                else 0.0,
            )
        )
        previous_equity = end_equity
        start = stop
    return rows


# --- how it holds up under higher costs -----------------------------------------------------


@dataclass(frozen=True)
class CostRow:
    slippage_pct: float
    trades: int
    total_return_pct: float
    profit_factor: float


def cost_sweep(
    settings: Settings,
    bt_config: BacktestConfig,
    df_15m: Dict[str, pd.DataFrame],
    df_1h: Dict[str, pd.DataFrame],
    strategy: BaseStrategy,
    slippage_levels: Sequence[float] = (0.0, 0.0005, 0.001, 0.002),
    atr_exit: Optional[AtrExit] = None,
) -> List[CostRow]:
    """The same replay at several levels of slippage per side. Costs are the most certain
    thing a backtest knows about the future, so a result that disappears at twice the
    assumed cost is not a result."""
    rows = []
    for level in slippage_levels:
        config = bt_config.model_copy(update={"slippage_pct": level})
        run = BacktestEngine(
            settings, config, df_15m, df_1h, strategy=strategy, atr_exit=atr_exit
        ).run()
        rows.append(CostRow(level, run.trade_count, run.total_return_pct, run.profit_factor))
    return rows


# --- did the Oracle keep us out of bad trades -----------------------------------------------


@dataclass(frozen=True)
class GroupStats:
    trades: int
    avg_pct: float
    win_rate: float
    t_stat: float


def _group(pcts: Sequence[float]) -> GroupStats:
    return GroupStats(
        trades=len(pcts),
        avg_pct=statistics.mean(pcts) if pcts else 0.0,
        win_rate=sum(1 for p in pcts if p > 0) / len(pcts) if pcts else 0.0,
        t_stat=t_statistic(pcts),
    )


@dataclass(frozen=True)
class OracleValue:
    approved: GroupStats  # the trades that were made
    not_approved: GroupStats  # the signals the Oracle did not approve, played out as shadow trades
    still_open: int  # shadow trades neither level had reached when the data ended
    edge_pct: float  # approved minus not approved, per trade: positive means the Oracle helped
    # The edge over its standard error. Below 2 in absolute value the two groups cannot be told
    # apart from luck, whatever the sign of the edge.
    edge_t_stat: float


def oracle_value(result: BacktestResult) -> Optional[OracleValue]:
    """None if there were no shadow trades (no Oracle replayed, or it approved everything)."""
    if not result.shadow_trades:
        return None
    approved_pcts = [t.pnl_pct for t in result.trades if t.pnl_pct is not None]
    shadow_pcts = [t.pnl_pct for t in result.shadow_trades]
    approved, shadow = _group(approved_pcts), _group(shadow_pcts)
    return OracleValue(
        approved=approved,
        not_approved=shadow,
        still_open=sum(1 for t in result.shadow_trades if t.exit_reason == "OPEN"),
        edge_pct=approved.avg_pct - shadow.avg_pct,
        edge_t_stat=welch_t(approved_pcts, shadow_pcts),
    )


# --- text -----------------------------------------------------------------------------------


def format_analysis(result: BacktestResult, periods: int = 4, symbols: int = 5) -> str:
    lines: List[str] = []
    e = expectancy(result.trades)
    lines.append("=== Per trade ===")
    lines.append(
        f"Expectancy:      {e.avg_pct * 100:+.2f}% per trade (${e.avg_pnl:+,.2f}), "
        f"t = {e.t_stat:+.1f}"
    )
    lines.append(
        f"Average win/loss: {e.avg_win_pct * 100:+.2f}% / {e.avg_loss_pct * 100:+.2f}% "
        f"(payoff {e.payoff:.2f})"
    )
    lines.append(f"Sharpe (daily):  {sharpe_ratio(result):+.2f}, annualised, risk-free rate 0")

    period_rows = by_period(result, periods)
    if period_rows:
        lines.append(
            f"\n=== Stability: {len(period_rows)} consecutive periods, same parameters ==="
        )
        lines.append(f"{'period':<25}{'days':>5}{'trades':>7}{'return':>9}{'win':>7}")
        for period in period_rows:
            lines.append(
                f"{period.first_day + ' .. ' + period.last_day:<25}{period.days:>5}{period.trades:>7}"
                f"{period.return_pct * 100:>+8.2f}%{period.win_rate * 100:>6.0f}%"
            )
        positive = sum(1 for period in period_rows if period.return_pct > 0)
        lines.append(f"{positive} of {len(period_rows)} periods ended up.")

    symbol_rows = by_symbol(result)
    if symbol_rows:
        shown = symbol_rows[:symbols] + [
            row for row in symbol_rows[-symbols:] if row not in symbol_rows[:symbols]
        ]
        lines.append(
            f"\n=== By symbol ({len(symbol_rows)} traded; best {symbols} and worst {symbols}) ==="
        )
        lines.append(f"{'symbol':<8}{'trades':>7}{'pnl $':>10}{'win':>7}{'avg trade':>11}")
        for row in shown:
            lines.append(
                f"{row.symbol:<8}{row.trades:>7}{float(row.pnl):>+10.0f}{row.win_rate * 100:>6.0f}%"
                f"{row.avg_pct * 100:>+10.2f}%"
            )
    return "\n".join(lines)


def format_cost_sweep(rows: Sequence[CostRow]) -> str:
    lines = ["=== The same replay at higher costs (slippage per side) ==="]
    lines.append(f"{'slippage':>9}{'trades':>8}{'return':>9}{'PF':>6}")
    for r in rows:
        lines.append(
            f"{r.slippage_pct * 100:>8.2f}%{r.trades:>8}{r.total_return_pct * 100:>+8.2f}%"
            f"{r.profit_factor:>6.2f}"
        )
    return "\n".join(lines)


def format_oracle_value(value: OracleValue) -> str:
    a, n = value.approved, value.not_approved
    lines = ["=== Did the Oracle keep us out of bad trades? ==="]
    lines.append(
        f"Approved (traded):      {a.trades:>4} trades, {a.avg_pct * 100:+.2f}% each, "
        f"{a.win_rate * 100:.0f}% won, t = {a.t_stat:+.1f}"
    )
    lines.append(
        f"Not approved (shadow):  {n.trades:>4} trades, {n.avg_pct * 100:+.2f}% each, "
        f"{n.win_rate * 100:.0f}% won, t = {n.t_stat:+.1f}"
        + (f" ({value.still_open} still open at the end)" if value.still_open else "")
    )
    if value.edge_t_stat >= 2:
        verdict = "the Oracle helped"
    elif value.edge_t_stat <= -2:
        verdict = "the Oracle hurt"
    else:
        verdict = "no difference that can be told from luck"
    lines.append(
        f"Edge of the Oracle: {value.edge_pct * 100:+.2f}% per trade, t = {value.edge_t_stat:+.1f}: "
        f"{verdict}."
    )
    lines.append(
        "Shadow trades ignore the position limit and the cash: they say whether the signal was "
        "good, not whether there was room for it."
    )
    return "\n".join(lines)
