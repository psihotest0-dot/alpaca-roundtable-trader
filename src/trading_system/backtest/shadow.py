"""What a signal that was not traded would have done (Phase 3.5).

The Oracle can only be judged by what it kept us out of. A vetoed signal never becomes an
order, so its outcome is simulated here: enter at the planned price, hold with the planned stop
and target, and see which the bars reach first. It is a per-trade percentage on the cost basis,
the same measure as a real trade's `pnl_pct`, and it does not depend on the account (no
position limit, no cash): the question is whether the signal was good, not whether there was
room for it."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from trading_system.backtest.exits import resolve_exit
from trading_system.backtest.models import BacktestConfig, ShadowTrade
from trading_system.risk.entry_plan import (
    AtrExit,
    InvalidEntryPlan,
    atr_exit_percentages,
    build_entry_plan,
)
from trading_system.storage.repositories import MarketBarRepository
from trading_system.strategy.atr import atr_and_close

BAR = timedelta(minutes=15)


def simulate_shadow_trade(
    *,
    symbol: str,
    signal_id: str,
    verdict: str,
    bars: pd.DataFrame,
    entry_time: datetime,
    entry_price: Decimal,
    stop_loss: Decimal,
    take_profit: Decimal,
    slippage_pct: float,
    fee_pct: float,
) -> ShadowTrade:
    """`bars`: the symbol's 15-minute bars, sorted, indexed by their start time (UTC). The
    position is held from the bar that starts at `entry_time`; the stop and the target are
    watched on that bar too, as for a real position. A position that neither level reaches
    before the data ends is closed at the last close and marked "OPEN"."""
    first = int(bars.index.searchsorted(entry_time, side="left"))
    opens, highs, lows, closes = (
        bars[name].to_numpy(dtype=float) for name in ("open", "high", "low", "close")
    )

    reason = "OPEN"
    exit_time = bars.index[-1]
    raw_exit = Decimal(str(closes[-1]))
    for i in range(first, len(bars)):
        hit = resolve_exit(
            Decimal(str(opens[i])),
            Decimal(str(highs[i])),
            Decimal(str(lows[i])),
            stop_loss,
            take_profit,
        )
        if hit is not None:
            reason, raw_exit = hit
            exit_time = bars.index[i]
            break

    fee = Decimal(str(fee_pct))
    slippage = Decimal(str(slippage_pct)) if reason != "OPEN" else Decimal("0")
    proceeds = raw_exit * (Decimal("1") - slippage) * (Decimal("1") - fee)
    cost = entry_price * (Decimal("1") + fee)
    return ShadowTrade(
        symbol=symbol,
        signal_id=signal_id,
        verdict=verdict,
        entry_time=entry_time,
        entry_price=entry_price,
        exit_time=exit_time.to_pydatetime(),
        exit_price=raw_exit * (Decimal("1") - slippage),
        exit_reason=reason,
        pnl_pct=float(proceeds / cost - Decimal("1")),
    )


@dataclass(frozen=True)
class SignalToPlay:
    """A signal to be played out. `label` is carried to the trade's `verdict` field: what the Oracle
    said for `judge-oracle`, the strategy's name for the candidates' judge."""

    signal_id: str
    symbol: str
    closed_bar: pd.Timestamp  # the start of the bar that closed and gave the signal (UTC)
    label: str


@dataclass
class PlayedOut:
    trades: list[ShadowTrade] = field(
        default_factory=list
    )  # OPEN ones are marked at the last close
    no_bars: int = 0  # signals with no bar to enter on (or no valid plan)
    skipped_busy: int = 0  # signals dropped because the symbol's earlier trade was still open
    no_atr: int = 0  # with an ATR exit: signals whose ATR is not defined


def play_out(
    signals: Sequence[SignalToPlay],
    bars: MarketBarRepository,
    settings: Any,
    *,
    one_at_a_time: bool = False,
    exit_rule: AtrExit | None = None,
) -> PlayedOut:
    """Plays each signal out as the backtest would trade it: entry at the open of the bar after the
    signal's bar, the entry plan of the live settings (limit buffer, stop, target), the backtest's
    slippage and fee, and the first of stop and target the stored bars reach.

    One definition for every judge (the Oracle's and the candidates'). A trade that neither level has
    reached by the end of the data comes back with exit_reason "OPEN", valued at the last close: the
    caller decides whether to drop it (the Oracle judge does) or to count it (the candidates' judge).

    `one_at_a_time`: a signal on a symbol whose earlier trade (of the SAME call) is still open on the
    entry bar is skipped, the way the live loop holds one position per symbol. Give one strategy's
    signals per call.

    `exit_rule`: a CANDIDATE (research plan, section 13.1). Stop and target are multiples of the ATR
    of the live window of bars up to the signal bar (`settings.lookback_days` days) instead of the
    live percentages; a signal whose ATR is not defined is counted in `no_atr` and not played."""
    costs = BacktestConfig()
    by_symbol: dict[str, list[SignalToPlay]] = {}
    for signal in signals:
        by_symbol.setdefault(signal.symbol, []).append(signal)
    out = PlayedOut()
    # An ATR needs the window the live loop shows its indicators, not just the hours before.
    warmup = timedelta(days=settings.lookback_days) if exit_rule else timedelta(hours=2)
    for symbol, group in by_symbol.items():
        group = sorted(group, key=lambda s: s.closed_bar)
        first = group[0].closed_bar - warmup
        frame = bars.load([symbol], "15m", start=first.to_pydatetime()).get(symbol)
        last: ShadowTrade | None = None
        for signal in group:
            entry_time = signal.closed_bar + BAR
            if frame is None or entry_time not in frame.index:
                out.no_bars += 1
                continue
            if one_at_a_time and last is not None:
                if last.exit_reason == "OPEN" or entry_time.to_pydatetime() <= last.exit_time:
                    out.skipped_busy += 1
                    continue
            price = Decimal(str(frame.loc[entry_time, "open"]))
            stop_pct = Decimal(str(settings.stop_loss_pct))
            take_pct = Decimal(str(settings.take_profit_pct))
            try:
                if exit_rule is not None:
                    window = frame[
                        (frame.index > signal.closed_bar - timedelta(days=settings.lookback_days))
                        & (frame.index <= signal.closed_bar)
                    ]
                    pair = atr_and_close(window)
                    if pair is None:
                        out.no_atr += 1
                        continue
                    stop_pct, take_pct = atr_exit_percentages(*pair, exit_rule)
                plan = build_entry_plan(
                    price,
                    limit_buffer_pct=Decimal(str(settings.entry_limit_buffer_pct)),
                    stop_loss_pct=stop_pct,
                    take_profit_pct=take_pct,
                )
            except InvalidEntryPlan:
                out.no_bars += 1
                continue
            fill = min(plan.limit_price, price * (Decimal("1") + Decimal(str(costs.slippage_pct))))
            last = simulate_shadow_trade(
                symbol=symbol,
                signal_id=signal.signal_id,
                verdict=signal.label,
                bars=frame,
                entry_time=entry_time.to_pydatetime(),
                entry_price=fill,
                stop_loss=plan.stop_price,
                take_profit=plan.take_profit_price,
                slippage_pct=costs.slippage_pct,
                fee_pct=costs.fee_pct_per_trade,
            )
            out.trades.append(last)
    return out
