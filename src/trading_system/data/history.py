"""Keeps the candle store up to date from Alpaca (Phase 3.2).

**Price policy.** Bars are downloaded adjusted for splits and dividends (`Adjustment.ALL`), the
same basis the live loop reads, so a backtest sees the same series the system trades on. The
catch is that Alpaca re-adjusts the whole past whenever a split or a dividend goes ex, so a
stored copy turns stale: old rows on the old basis and new rows on the new one would look like
a price gap that never happened. Every update therefore downloads the last OVERLAP days again
and compares the oldest bars the two copies share. If their prices disagree, the symbol's
history is downloaded again from the start. The overlap also takes in the late corrections
Alpaca makes to recent bars. Only closed bars are stored.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from trading_system.data.alpaca_data import FIFTEEN_MINUTES, ONE_HOUR, AlpacaData
from trading_system.storage.repositories import MarketBarRepository

logger = logging.getLogger(__name__)

# The price basis of every stored bar. Must match AlpacaData.fetch_bars_bulk.
ADJUSTMENT = "all"

# How far back an update looks again, to catch corrections and re-adjustments.
OVERLAP = timedelta(days=7)
# The oldest bars both copies share that are compared, and how far apart two prices of the
# same bar may be (a dividend is far more than a basis point; a re-download never differs).
ANCHOR_BARS = 20
MAX_PRICE_DRIFT = 1e-4
# Stored history that starts later than the requested start by more than this (a long weekend
# leaves a few days without bars) is taken to be too short and is downloaded from the start.
BACKFILL_SLACK = timedelta(days=5)


@dataclass(frozen=True)
class TimeframeSpec:
    frame: TimeFrame
    length: timedelta
    # None: the configured feed (DATA_FEED). Set where a series must come from one feed only.
    feed: Optional[str] = None


# What `fetch-history` keeps in trading.db: the bars the live loop trades on (the configured feed).
TIMEFRAMES: dict[str, TimeframeSpec] = {
    "15m": TimeframeSpec(FIFTEEN_MINUTES, timedelta(minutes=15)),
    "1h": TimeframeSpec(ONE_HOUR, timedelta(hours=1)),
}

# What `fetch-daily` keeps in research.db. Daily bars come from SIP, the full-market feed, always:
# the free IEX feed starts in 2020 and reports 2 to 3 % of the real volume, so a daily signal
# measured on it would measure the wrong thing. A bar counts as closed from the next midnight New
# York time (Alpaca stamps a daily bar with the start of its day), a few hours after the close.
RESEARCH_TIMEFRAMES: dict[str, TimeframeSpec] = {
    "1d": TimeframeSpec(TimeFrame.Day, timedelta(days=1), feed="sip"),
}

_SPECS: dict[str, TimeframeSpec] = {**TIMEFRAMES, **RESEARCH_TIMEFRAMES}


@dataclass
class HistoryReport:
    rows: dict[tuple[str, str], int] = field(default_factory=dict)  # (symbol, timeframe) -> rows
    downloaded_again: list[tuple[str, str]] = field(default_factory=list)
    no_data: list[tuple[str, str]] = field(default_factory=list)

    @property
    def rows_written(self) -> int:
        return sum(self.rows.values())


def prices_drifted(stored: Optional[pd.DataFrame], fresh: pd.DataFrame) -> bool:
    """True if the oldest bars the two copies share disagree about price, or if nothing can
    be compared (then the stored copy cannot be trusted either)."""
    if stored is None or stored.empty:
        return True
    shared = stored.index.intersection(fresh.index).sort_values()
    if shared.empty:
        return True
    anchors = shared[:ANCHOR_BARS]
    ratio = fresh.loc[anchors, "close"] / stored.loc[anchors, "close"]
    return bool(((ratio - 1).abs() > MAX_PRICE_DRIFT).any())


def update_history(
    data: AlpacaData,
    repo: MarketBarRepository,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    *,
    start: datetime,
    now: datetime,
) -> HistoryReport:
    """Brings the store to hold the closed bars of `symbols` from `start` to `now`, in the
    given timeframes (TIMEFRAMES: "15m", "1h"; RESEARCH_TIMEFRAMES: "1d"). Downloads only what
    is missing, as one request for all symbols that need the same thing."""
    report = HistoryReport()
    for name in timeframes:
        _update_timeframe(data, repo, list(symbols), name, start, now, report)
    return report


def _update_timeframe(
    data: AlpacaData,
    repo: MarketBarRepository,
    symbols: list[str],
    name: str,
    start: datetime,
    now: datetime,
    report: HistoryReport,
) -> None:
    spec = _SPECS[name]
    from_start: list[str] = []  # download the whole history and replace what is stored
    extend: dict[str, datetime] = {}  # download from here on and add to what is stored
    had_history: set[str] = set()

    for symbol in symbols:
        span = repo.span(symbol, name)
        if span is None:
            from_start.append(symbol)
            continue
        had_history.add(symbol)
        if span[0] - start > BACKFILL_SLACK or repo.has_other_adjustment(symbol, name, ADJUSTMENT):
            from_start.append(symbol)
        else:
            extend[symbol] = span[1] - OVERLAP

    if extend:
        fresh: dict[str, pd.DataFrame] = {}
        for chunk in _chunks(list(extend)):
            fresh.update(_download(data, chunk, spec, min(extend[s] for s in chunk), now))
        for symbol in extend:
            new = fresh.get(symbol)
            if new is None or new.empty:
                continue  # nothing new to add; what is stored stays
            stored = repo.load([symbol], name, start=new.index[0]).get(symbol)
            if prices_drifted(stored, new):
                logger.info("%s %s: stored prices are out of date, downloading again", symbol, name)
                from_start.append(symbol)
            else:
                report.rows[(symbol, name)] = repo.replace(
                    symbol, name, new, adjustment=ADJUSTMENT, fetched_at=now
                )

    # In chunks, each one written as soon as it has arrived: one request for ~150 symbols of 15-minute
    # bars paged for a quarter of an hour at a few KB/s and all of it was lost if it failed late.
    for chunk in _chunks(from_start):
        full = _download(data, chunk, spec, start, now)
        for symbol in chunk:
            bars = full.get(symbol)
            if bars is None or bars.empty:
                report.no_data.append((symbol, name))
                continue
            report.rows[(symbol, name)] = repo.replace(
                symbol, name, bars, adjustment=ADJUSTMENT, fetched_at=now, everything=True
            )
            if symbol in had_history:
                report.downloaded_again.append((symbol, name))


# Symbols per request. The watchlist (22) is one request, the daily universe (145) a few.
DOWNLOAD_CHUNK = 25


def _chunks(symbols: Sequence[str]) -> list[list[str]]:
    return [list(symbols[i : i + DOWNLOAD_CHUNK]) for i in range(0, len(symbols), DOWNLOAD_CHUNK)]


def _download(
    data: AlpacaData,
    symbols: list[str],
    spec: TimeframeSpec,
    start: datetime,
    now: datetime,
) -> dict[str, pd.DataFrame]:
    """The closed bars only: the bar still forming at `now` is not a candle yet."""
    frames = data.fetch_bars_bulk(symbols, spec.frame, start, now, feed=spec.feed)
    return {symbol: bars[bars.index + spec.length <= now] for symbol, bars in frames.items()}


def stale_symbols(
    repo: MarketBarRepository,
    symbols: Sequence[str],
    timeframe: str,
    *,
    tolerance: timedelta,
) -> list[tuple[str, Optional[datetime]]]:
    """Symbols whose newest stored bar is more than `tolerance` older than the newest bar of
    any of them, with that bar's time (None: nothing is stored at all).

    A symbol that was renamed, delisted or halted stops getting bars while the rest go on, and
    nothing in the download says so (the old name just ends); a run of daily research on it
    would see a gap that is not real. This is how the update reports it."""
    last: dict[str, datetime] = {}
    for symbol in symbols:
        span = repo.span(symbol, timeframe)
        if span is not None:
            last[symbol] = span[1]
    newest = max(last.values(), default=None)
    flagged: list[tuple[str, Optional[datetime]]] = []
    for symbol in symbols:
        if symbol not in last:
            flagged.append((symbol, None))
        elif newest is not None and newest - last[symbol] > tolerance:
            flagged.append((symbol, last[symbol]))
    return flagged
