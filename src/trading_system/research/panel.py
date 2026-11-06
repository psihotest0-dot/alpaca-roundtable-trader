"""The daily bars of the research universe as aligned tables (dates down, symbols across).

**The hold-out is never loaded.** `load_panel` stops at HOLDOUT_START by default and has no flag
for going past it; a forward return that would need a day beyond that date is simply missing, so
no measurement on the panel can peek. The hold-out is opened by a separate, deliberate step
(docs/strategy-research-plan.md, step C3), never by a parameter of this one.

**Known artifacts are cut out.** A day on which the adjusted prices show a move that shareholders
did not live through (see data/universe.py: RTX on 2020-04-03) is blanked: the daily return and the
gap of that day are missing, so every window that contains the day is missing too.

All prices are adjusted for splits and dividends (total return). Times are New York dates.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from trading_system.data.universe import MARKET_SYMBOL, research_universe
from trading_system.storage.repositories import MarketBarRepository

TIMEFRAME = "1d"

# The first day of the hold-out: the last months of the data, fixed on 2026-11-05 when the data
# ended on 2026-11-04. It does not move as new days arrive, so the hold-out only grows.
HOLDOUT_START = date(2026, 5, 5)

# (symbol, day): moves in the adjusted prices that are corporate actions, not market moves.
ARTIFACTS: dict[str, tuple[date, ...]] = {"RTX": (date(2020, 4, 3),)}

MARKET_AVERAGE_DAYS = 200


@dataclass(frozen=True)
class Panel:
    open: pd.DataFrame
    close: pd.DataFrame
    log_return: pd.DataFrame  # close to close; missing where unknown or an artifact
    overnight_gap: pd.DataFrame  # log(open / previous close); missing likewise
    market_log_return: pd.Series
    market_gap: pd.Series
    market_up: pd.Series  # SPY closes above its 200-day average (False while unknown)
    market_known: pd.Series  # the 200-day average exists

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def symbols(self) -> list[str]:
        return [str(s) for s in self.close.columns]


def build_panel(
    frames: Mapping[str, pd.DataFrame],
    artifacts: Optional[Mapping[str, Sequence[date]]] = None,
) -> Panel:
    """A panel from `open` and `close` columns per symbol, indexed by date. SPY (the market) sets
    the calendar and is kept apart from the stocks."""
    artifacts = ARTIFACTS if artifacts is None else artifacts
    if MARKET_SYMBOL not in frames:
        raise ValueError(f"no daily bars for {MARKET_SYMBOL}; run `fetch-daily` first")
    market = frames[MARKET_SYMBOL]
    dates = pd.DatetimeIndex(market.index)
    stocks = [s for s in frames if s != MARKET_SYMBOL]

    opens = pd.DataFrame({s: frames[s]["open"] for s in stocks}).reindex(dates)
    closes = pd.DataFrame({s: frames[s]["close"] for s in stocks}).reindex(dates)
    log_return = np.log(closes / closes.shift(1))
    gap = np.log(opens / closes.shift(1))
    for symbol, days in artifacts.items():
        if symbol not in log_return.columns:
            continue
        for day in days:
            stamp = pd.Timestamp(day)
            if stamp in log_return.index:
                log_return.loc[stamp, symbol] = np.nan
                gap.loc[stamp, symbol] = np.nan

    market_close = market["close"].reindex(dates)
    average = market_close.rolling(MARKET_AVERAGE_DAYS, min_periods=MARKET_AVERAGE_DAYS).mean()
    return Panel(
        open=opens,
        close=closes,
        log_return=log_return,
        overnight_gap=gap,
        market_log_return=np.log(market_close / market_close.shift(1)),
        market_gap=np.log(market["open"].reindex(dates) / market_close.shift(1)),
        market_up=(market_close > average).fillna(False),
        market_known=average.notna(),
    )


def frames_from_stored(stored: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """`open` and `close` per symbol, indexed by the New York date of each stored daily bar."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol, bars in stored.items():
        day = (
            pd.DatetimeIndex(bars.index)
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize()
        )
        frames[symbol] = pd.DataFrame(
            {"open": bars["open"].to_numpy(), "close": bars["close"].to_numpy()}, index=day
        )
    return frames


def load_panel(
    repo: MarketBarRepository,
    symbols: Optional[Sequence[str]] = None,
    *,
    end: date = HOLDOUT_START,
) -> Panel:
    """The development part of the stored daily bars: every day before `end`, which is the start
    of the hold-out and cannot be later."""
    if end > HOLDOUT_START:
        raise ValueError(f"the panel stops at the hold-out ({HOLDOUT_START}); asked for {end}")
    wanted = list(symbols) if symbols else research_universe()
    if MARKET_SYMBOL not in wanted:
        wanted.append(MARKET_SYMBOL)
    stored = repo.load(
        wanted, TIMEFRAME, end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    )
    return build_panel(frames_from_stored(stored))


def big_moves(panel: Panel, threshold: float = 0.30) -> list[tuple[str, date, float]]:
    """Daily moves beyond `threshold` that are still in the panel, largest first. The report lists
    them: each one is either a real move or an artifact that must be cut out."""
    change = np.expm1(panel.log_return)
    found = change.stack()
    found = found[found.abs() > threshold].sort_values(key=lambda s: s.abs(), ascending=False)
    return [(str(sym), ts.date(), float(move)) for (ts, sym), move in found.items()]
