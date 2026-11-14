"""The crypto research: one signal on a few coins, measured the way the stock signals are
(docs/strategy-research-plan.md, section 15, registered on 2026-11-14 before any result).

Nothing here trades, and nothing here touches the stocks: the bars live in their own file
(`crypto.db`), the variants have their own count (research/trials.py, market "crypto", "counter C"),
and the measurement is the same code as for stocks (research/measure.py) with the crypto rules.

**What is different from the stocks, all of it written in the plan before a result:**
- days are UTC days and every day of the calendar has a bar for BTC (the coin that sets it);
- a bar with zero volume is a placeholder: it is treated as missing, and so is every window that
  touches it (the same way the stock panel blanks a corporate-action day);
- the "market" the regime and "vs market" are read from is BTC, which is also one of the coins;
- a day is an observation only if at least 2 coins are on, at least 1 is off (all on compares
  nothing) and at least 6 of the 7 coins have a window (the last two numbers: plan section 15.7);
- the cost is a flat 0.50 % a round trip (Alpaca's taker fee of 0.25 % a side at the lowest tier);
- at least 300 non-overlapping events, not 1000: seven coins cannot give 1000 (section 15.4).

**The hold-out is not loaded** (from 2026-05-05, as for the stocks): `load_crypto_panel` stops there
and has no flag to go past it.
"""

from datetime import date, datetime, timezone
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from trading_system.data.crypto import CRYPTO_UNIVERSE, MARKET_COIN, TIMEFRAME
from trading_system.research.measure import MeasureRules
from trading_system.research.panel import HOLDOUT_START, MARKET_AVERAGE_DAYS, Panel
from trading_system.research.signals import Signal
from trading_system.storage.repositories import MarketBarRepository

TREND_DAYS = 30  # `crypto_trend_30`: the close is above the close 30 calendar days earlier
CONTROL_SEED = 20261114
ROUND_TRIP_COST = 0.005  # 0.25 % taker fee on each side, the lowest tier (docs of 2026-11-14)

CRYPTO_RULES = MeasureRules(
    min_signal=2,
    min_universe=6,
    min_events=300,
    need_one_off=True,
    asset="coins",
    market="BTC",
)


def build_crypto_panel(frames: Mapping[str, pd.DataFrame]) -> Panel:
    """A panel from `open`, `close` and `volume` per pair, indexed by UTC date. BTC sets the
    calendar, which must have every day; the other coins are laid on it."""
    if MARKET_COIN not in frames:
        raise ValueError(f"no daily bars for {MARKET_COIN}; run `fetch-crypto-daily` first")
    dates = pd.DatetimeIndex(frames[MARKET_COIN].index)
    if len(dates) > 1 and not (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError(f"the {MARKET_COIN} calendar has a missing day: run `fetch-crypto-daily`")

    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        real = frame["volume"] > 0  # a zero-volume bar is a placeholder, not a price
        opens[symbol] = frame["open"].where(real).reindex(dates)
        closes[symbol] = frame["close"].where(real).reindex(dates)
    open_ = pd.DataFrame(opens)
    close = pd.DataFrame(closes)

    log_return = np.log(close / close.shift(1))
    gap = np.log(open_ / close.shift(1))
    market_close = close[MARKET_COIN]
    average = market_close.rolling(MARKET_AVERAGE_DAYS, min_periods=MARKET_AVERAGE_DAYS).mean()
    return Panel(
        open=open_,
        close=close,
        log_return=log_return,
        overnight_gap=gap,
        market_log_return=log_return[MARKET_COIN],
        market_gap=gap[MARKET_COIN],
        market_up=(market_close > average).fillna(False),
        market_known=average.notna(),
    )


def frames_from_stored(stored: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """`open`, `close` and `volume` per pair, indexed by the UTC date of each stored daily bar."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol, bars in stored.items():
        day = pd.DatetimeIndex(bars.index).tz_convert("UTC").tz_localize(None).normalize()
        frames[symbol] = pd.DataFrame(
            {
                "open": bars["open"].to_numpy(),
                "close": bars["close"].to_numpy(),
                "volume": bars["volume"].to_numpy(),
            },
            index=day,
        )
    return frames


def load_crypto_panel(
    repo: MarketBarRepository,
    symbols: Optional[Sequence[str]] = None,
    *,
    end: date = HOLDOUT_START,
) -> Panel:
    """The development part of the stored crypto bars: every day before `end`, which is the start
    of the hold-out and cannot be later."""
    if end > HOLDOUT_START:
        raise ValueError(f"the panel stops at the hold-out ({HOLDOUT_START}); asked for {end}")
    wanted = list(symbols) if symbols else list(CRYPTO_UNIVERSE)
    stored = repo.load(
        wanted, TIMEFRAME, end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    )
    return build_crypto_panel(frames_from_stored(stored))


def zero_volume_share(frames: Mapping[str, pd.DataFrame]) -> float:
    """The share of all stored bars that are placeholders (for the report)."""
    total = sum(len(f) for f in frames.values())
    return sum(int((f["volume"] <= 0).sum()) for f in frames.values()) / total if total else 0.0


# ------------------------------------------------------------------------------------ signals


def crypto_trend_30(panel: Panel) -> pd.DataFrame:
    """A coin is "on" on a day if its close is above its close 30 calendar days earlier (a positive
    30-day return). A missing bar on either date means "off": there is no signal without both."""
    return panel.close > panel.close.shift(TREND_DAYS)


def random_half(panel: Panel) -> pd.DataFrame:
    """A random half of the coins each day. The control: whatever the measurement says about it is
    what chance says, and the measurement must say about zero."""
    draw = np.random.default_rng(CONTROL_SEED).random(panel.close.shape)
    return pd.DataFrame(
        (draw < 0.5) & panel.close.notna().to_numpy(),
        index=panel.close.index,
        columns=panel.close.columns,
    )


CRYPTO_SIGNALS: dict[str, Signal] = {
    signal.name: signal
    for signal in (
        Signal(
            "crypto_trend_30",
            "the coins whose close is above the close 30 days earlier",
            (10, 20),
            crypto_trend_30,
        ),
        Signal(
            "random_half",
            "a random half of the coins each day (control)",
            (10, 20),
            random_half,
            control=True,
        ),
    )
}


def development_note() -> str:
    return (
        f"Development part only: the hold-out from {HOLDOUT_START} is not loaded. Seven coins that "
        "still trade on Alpaca today (survivorship), strongly correlated with each other; about "
        "one and a half market cycles; volume is Alpaca's, not the whole market's."
    )
