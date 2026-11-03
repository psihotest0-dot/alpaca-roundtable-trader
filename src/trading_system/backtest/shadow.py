"""What a signal that was not traded would have done (Phase 3.5).

The Oracle can only be judged by what it kept us out of. A vetoed signal never becomes an
order, so its outcome is simulated here: enter at the planned price, hold with the planned stop
and target, and see which the bars reach first. It is a per-trade percentage on the cost basis,
the same measure as a real trade's `pnl_pct`, and it does not depend on the account (no
position limit, no cash): the question is whether the signal was good, not whether there was
room for it."""

from datetime import datetime
from decimal import Decimal

import pandas as pd

from trading_system.backtest.exits import resolve_exit
from trading_system.backtest.models import ShadowTrade


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
