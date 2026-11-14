"""Daily crypto bars for the crypto research (docs/strategy-research-plan.md, section 15).

**Free, and no key.** Alpaca's historical crypto bars need no authentication at all (a key only
raises the rate limit), so this feed is built without one: the crypto research never touches the
stock account. History starts on 2021-01-01. A day without a single trade has no bar at all, and a
bar with zero volume is a placeholder: the panel treats both as missing (research/crypto.py).

**The universe is a rule, not a choice.** `eligible_pairs` is the rule fixed on 2026-11-14, before
any return was looked at, applied to the history counts of the 36 tradable /USD pairs of that day.
`CRYPTO_UNIVERSE` is its result and is written down so that it does not move when Alpaca lists a new
coin; a new coin is a new decision, not a drift.

**Only closed bars.** A daily bar is stamped with the start of its UTC day and is closed from the
next UTC midnight; today's bar is never stored. Nothing is adjusted (there are no splits), so a
download simply replaces the stored series: about 2,200 rows a coin, a second or two.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence, cast

import pandas as pd
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from trading_system.data.bars import normalize_bars
from trading_system.storage.repositories import MarketBarRepository

logger = logging.getLogger(__name__)

HISTORY_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
TIMEFRAME = "1d"
ADJUSTMENT = "none"

# The result of `eligible_pairs` on the counts of 2026-11-14 (plan section 15.2).
CRYPTO_UNIVERSE: tuple[str, ...] = (
    "BAT/USD",
    "BCH/USD",
    "BTC/USD",
    "DOGE/USD",
    "ETH/USD",
    "LINK/USD",
    "LTC/USD",
)
# What the crypto "market" is: the regime and the "vs market" column are read from it.
MARKET_COIN = "BTC/USD"

# The rule: a daily bar on at least this share of the days since HISTORY_START, and at most this
# share of those bars with zero volume.
MIN_COVERAGE = 0.95
MAX_ZERO_VOLUME = 0.05
# Left out by the rule whatever their history: pegged to something, so they do not trend.
PEGGED = frozenset({"USDC/USD", "USDT/USD", "USDG/USD", "PAXG/USD"})


def eligible_pairs(counts: Mapping[str, tuple[int, int]], days: int) -> list[str]:
    """The pairs the rule keeps. `counts` maps a pair to (bars since HISTORY_START, bars with zero
    volume); `days` is how many days that span has. Sorted by name."""
    kept = []
    for pair, (bars, zero_volume) in counts.items():
        if pair in PEGGED or bars <= 0:
            continue
        if bars < MIN_COVERAGE * days or zero_volume > MAX_ZERO_VOLUME * bars:
            continue
        kept.append(pair)
    return sorted(kept)


@dataclass
class CryptoUpdate:
    rows_written: int = 0
    no_data: list[str] = field(default_factory=list)


class CryptoDailyFeed:
    """Alpaca's crypto daily bars, with no key (or with one, if a client is given)."""

    def __init__(self, client: Optional[Any] = None):
        self.client = client if client is not None else CryptoHistoricalDataClient()

    def fetch(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> dict[str, pd.DataFrame]:
        """Daily bars per pair from `start` to `end` (UTC). A pair with no bars is left out."""
        response = self.client.get_crypto_bars(
            CryptoBarsRequest(
                symbol_or_symbols=list(symbols), timeframe=TimeFrame.Day, start=start, end=end
            )
        )
        frame = cast(Any, response).df
        if frame is None or frame.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        present = set(frame.index.get_level_values("symbol"))
        for symbol in symbols:
            if symbol in present:
                out[symbol] = normalize_bars(frame.xs(symbol, level="symbol"))
        return out


def closed_only(bars: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Only the daily bars that had ended by `now` (a bar's day is over at the next UTC midnight)."""
    if bars.empty:
        return bars
    return bars[bars.index + timedelta(days=1) <= pd.Timestamp(now)]


def update_crypto_daily(
    feed: CryptoDailyFeed,
    repo: MarketBarRepository,
    symbols: Sequence[str],
    *,
    start: datetime = HISTORY_START,
    now: datetime,
) -> CryptoUpdate:
    """Download the closed daily bars of `symbols` from `start` and store them, replacing what was
    stored for each pair: the history is cheap to fetch and never re-adjusted."""
    report = CryptoUpdate()
    fetched = feed.fetch(symbols, start, now)
    for symbol in symbols:
        bars = closed_only(fetched.get(symbol, pd.DataFrame()), now)
        if bars.empty:
            report.no_data.append(symbol)
            continue
        report.rows_written += repo.replace(
            symbol,
            TIMEFRAME,
            bars,
            adjustment=ADJUSTMENT,
            fetched_at=now,
            everything=True,
        )
    return report
