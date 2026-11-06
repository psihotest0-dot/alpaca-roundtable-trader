"""What prices do in the hours after an intraday signal, against what they do anyway.

The same method as the daily signals (research/measure.py), on 15-minute bars: for every day, the
stocks the signal fires on are followed from the open of the next bar to the close of the session, and
the difference to the average of ALL stocks over the same window is the day's one observation (stocks
of a day move together, so they are not counted as independent). The windows of two days never
overlap, so the Newey-West lag is 0. A flat round-trip cost is taken off. Results come by month.

The bars are the observation store's (IEX, like the live loop): the opening range is the range of
IEX's own prints, a small share of the volume. That is the data a live strategy would have, and it
is written under every result.

**The hold-out is never loaded.** `load_bars` stops at INTRADAY_HOLDOUT_START and has no flag to go
past it; the last two months stay closed until a candidate has passed (docs/strategy-research-plan.md,
sections 6 and 11).

Every signal here is one more variant in the count (research/trials.py); `random_entry` is the
control that must show nothing and is not counted.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from trading_system.data.universe import MARKET_SYMBOL, observation_universe
from trading_system.research.measure import (
    CRITERIA_MIN_INDEPENDENT_EVENTS,
    CRITERIA_MIN_T,
    MIN_SIGNAL_STOCKS,
    MIN_UNIVERSE,
    ROUND_TRIP_COST,
    Slice,
)
from trading_system.research.signals import CONTROL_SEED, DECILE
from trading_system.research.stats import hac_mean
from trading_system.storage.repositories import AuditRepository, MarketBarRepository

MARKET_TZ = "America/New_York"
TIMEFRAME = "15m"

# The last two months of the 180 days fetched on 2026-11-06 (question 4: for the hours the hold-out is
# the last two months, the history being short). It does not move as days arrive.
INTRADAY_HOLDOUT_START = date(2026, 9, 8)

SLOTS = 26  # 15-minute bars of the regular session, starting 09:30 ... 15:45
OPENING_SLOTS = 2  # the opening range: the bars starting 09:30 and 09:45
FIRST_SIGNAL_SLOT = 2  # 10:00
LAST_SIGNAL_SLOT = 17  # 13:45, which closes at 14:00
EXIT_SLOT = SLOTS - 1  # the bar starting 15:45; its close is the end of the session
CONTROL_ENTRY_SLOT = 2  # the control enters at the open of the 10:00 bar

SIGNALS = ("orb_30m_long", "random_entry")
CONTROLS = ("random_entry",)
DESCRIPTIONS = {
    "orb_30m_long": "first 15-minute close above the high of the first 30 minutes (10:00 to 13:45), "
    "long, in at the next open, out at the close of the session",
    "random_entry": "a random tenth of the stocks each day, in at the 10:00 open, out at the close "
    "of the session (control)",
}


@dataclass(frozen=True)
class Session:
    """The regular sessions as arrays: days down, the 26 bar slots, then the stocks. Missing = NaN."""

    days: pd.DatetimeIndex
    symbols: list[str]
    open: np.ndarray  # (days, SLOTS, stocks)
    high: np.ndarray
    close: np.ndarray


def load_bars(
    repo: MarketBarRepository,
    symbols: Optional[Sequence[str]] = None,
    *,
    end: date = INTRADAY_HOLDOUT_START,
) -> Session:
    """The development part of the stored 15-minute bars, as `Session` arrays. Refuses to go past
    the start of the hold-out."""
    if end > INTRADAY_HOLDOUT_START:
        raise ValueError(
            f"the intraday bars stop at the hold-out ({INTRADAY_HOLDOUT_START}); asked for {end}"
        )
    wanted = [s for s in (symbols or observation_universe([])) if s != MARKET_SYMBOL]
    stored = repo.load(
        wanted, TIMEFRAME, end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    )
    return build_session(stored)


def build_session(stored: dict[str, pd.DataFrame]) -> Session:
    """`Session` arrays from bars per symbol (UTC index; open, high, close). Only the regular
    session counts: bars starting 09:30 to 15:45 New York time, on the quarter hour."""
    symbols = sorted(stored)
    frames: dict[str, pd.DataFrame] = {}
    all_days: set[pd.Timestamp] = set()
    for symbol in symbols:
        bars = stored[symbol]
        local = pd.DatetimeIndex(bars.index).tz_convert(MARKET_TZ)
        minutes = local.hour * 60 + local.minute - (9 * 60 + 30)
        keep = (minutes >= 0) & (minutes < SLOTS * 15) & (minutes % 15 == 0)
        frame = pd.DataFrame(
            {
                "day": local[keep].tz_localize(None).normalize(),
                "slot": (minutes[keep] // 15).astype(int),
                "open": bars["open"].to_numpy()[keep],
                "high": bars["high"].to_numpy()[keep],
                "close": bars["close"].to_numpy()[keep],
            }
        )
        frames[symbol] = frame
        all_days.update(frame["day"].unique())
    days = pd.DatetimeIndex(sorted(all_days))
    shape = (len(days), SLOTS, len(symbols))
    opens, highs, closes = (np.full(shape, np.nan) for _ in range(3))
    for j, symbol in enumerate(symbols):
        frame = frames[symbol]
        if frame.empty:
            continue
        rows = days.get_indexer(pd.DatetimeIndex(frame["day"]))
        slots = frame["slot"].to_numpy()
        opens[rows, slots, j] = frame["open"].to_numpy()
        highs[rows, slots, j] = frame["high"].to_numpy()
        closes[rows, slots, j] = frame["close"].to_numpy()
    return Session(days, symbols, opens, highs, closes)


def forward_to_close(session: Session) -> np.ndarray:
    """Per day, entry slot and stock: the return from that bar's open to the close of the session."""
    return np.asarray(session.close[:, EXIT_SLOT : EXIT_SLOT + 1, :] / session.open - 1.0)


def orb_30m_long(session: Session) -> tuple[np.ndarray, np.ndarray]:
    """(fired, entry_slot): per day and stock, whether the signal fired, and the slot of the bar it
    enters at. The first close above the opening range's high, from 10:00 to 13:45."""
    high = np.fmax(session.high[:, 0, :], session.high[:, 1, :])
    complete = np.isfinite(session.high[:, 0, :]) & np.isfinite(session.high[:, 1, :])
    days, _, stocks = session.close.shape
    fired = np.zeros((days, stocks), dtype=bool)
    entry = np.full((days, stocks), -1, dtype=int)
    open_ = np.ones((days, stocks), dtype=bool)  # no signal yet
    for slot in range(FIRST_SIGNAL_SLOT, LAST_SIGNAL_SLOT + 1):
        above = complete & open_ & (session.close[:, slot, :] > high)
        # entered at the next bar's open: that bar and the exit bar must exist
        usable = above & np.isfinite(session.open[:, slot + 1, :])
        usable &= np.isfinite(session.close[:, EXIT_SLOT, :])
        fired |= usable
        entry[usable] = slot + 1
        open_ &= ~above  # only the first close above counts, usable or not
    return fired, entry


def random_entry(session: Session) -> tuple[np.ndarray, np.ndarray]:
    """The control: a random tenth of the stocks each day, drawn from the date alone."""
    days, _, stocks = session.close.shape
    fired = np.zeros((days, stocks), dtype=bool)
    for d, day in enumerate(session.days):
        draw = np.random.default_rng([CONTROL_SEED, int(day.strftime("%Y%m%d"))]).random(stocks)
        fired[d] = draw < DECILE
    has_prices = np.isfinite(session.open[:, CONTROL_ENTRY_SLOT, :]) & np.isfinite(
        session.close[:, EXIT_SLOT, :]
    )
    fired &= has_prices
    entry = np.where(fired, CONTROL_ENTRY_SLOT, -1)
    return fired, entry


BUILDERS: dict[str, Callable[[Session], tuple[np.ndarray, np.ndarray]]] = {
    "orb_30m_long": orb_30m_long,
    "random_entry": random_entry,
}


@dataclass(frozen=True)
class IntradayCriteria:
    enough_events: bool
    t_high_enough: bool
    not_one_month: bool  # without the month that contributed most, the mean is still above zero
    most_months_positive: bool

    @property
    def all_met(self) -> bool:
        return (
            self.enough_events
            and self.t_high_enough
            and self.not_one_month
            and self.most_months_positive
        )


@dataclass(frozen=True)
class IntradayMeasurement:
    signal: str
    cost: float
    first: Optional[date]
    last: Optional[date]
    events: int
    days_with_signal: int
    universe_mean: float
    signal_mean: float
    vs_universe: Slice
    net: Slice
    by_month: tuple[Slice, ...]
    criteria: IntradayCriteria

    @property
    def usable(self) -> bool:
        return self.vs_universe.dates > 0


def measure_intraday(
    session: Session, signal: str, *, cost: float = ROUND_TRIP_COST
) -> IntradayMeasurement:
    if signal not in BUILDERS:
        raise ValueError(f"unknown intraday signal {signal!r}; known: {', '.join(BUILDERS)}")
    fired, entry = BUILDERS[signal](session)
    forward = forward_to_close(session)  # (days, slots, stocks)
    days = len(session.days)

    # what all stocks did from the same bar to the close, per day and entry slot
    covered = np.isfinite(forward).sum(axis=2)
    with np.errstate(all="ignore"):
        universe = np.where(covered >= MIN_UNIVERSE, np.nanmean(forward, axis=2), np.nan)

    values = np.full(days, np.nan)
    signal_returns = np.full(days, np.nan)
    universe_returns = np.full(days, np.nan)
    counts = np.zeros(days, dtype=int)
    for d in range(days):
        stocks = np.flatnonzero(fired[d])
        if len(stocks) < MIN_SIGNAL_STOCKS:
            continue
        slots = entry[d, stocks]
        own = forward[d, slots, stocks]
        base = universe[d, slots]
        ok = np.isfinite(own) & np.isfinite(base)
        if int(ok.sum()) < MIN_SIGNAL_STOCKS:
            continue
        values[d] = float(np.mean(own[ok] - base[ok]))
        signal_returns[d] = float(np.mean(own[ok]))
        universe_returns[d] = float(np.mean(base[ok]))
        counts[d] = int(ok.sum())

    usable = np.isfinite(values)
    net_values = values - cost
    months = np.array([f"{t.year}-{t.month:02d}" for t in session.days])

    def piece(label: str, vals: np.ndarray, mask: np.ndarray) -> Slice:
        stat = hac_mean(vals, mask, 0)
        chosen = mask & np.isfinite(vals)
        positive = float((vals[chosen] > 0).mean()) if chosen.any() else float("nan")
        return Slice(
            label, int(chosen.sum()), int(counts[chosen].sum()), stat.mean, stat.t, positive
        )

    labels = sorted(set(months[usable]))
    by_month = tuple(piece(m, net_values, usable & (months == m)) for m in labels)
    net = piece("net of cost", net_values, usable)
    contributions = {m: float(np.nansum(net_values[usable & (months == m)])) for m in labels}
    rest_positive = False
    if len(contributions) > 1:
        best = max(contributions, key=lambda m: contributions[m])
        rest = usable & (months != best)
        rest_positive = bool(np.nanmean(net_values[rest]) > 0)
    positive_months = sum(1 for s in by_month if s.mean > 0)
    events = int(counts[usable].sum())
    dates_used = session.days[usable]
    return IntradayMeasurement(
        signal=signal,
        cost=cost,
        first=dates_used[0].date() if len(dates_used) else None,
        last=dates_used[-1].date() if len(dates_used) else None,
        events=events,
        days_with_signal=int(usable.sum()),
        universe_mean=float(np.nanmean(universe_returns)) if usable.any() else float("nan"),
        signal_mean=float(np.nanmean(signal_returns)) if usable.any() else float("nan"),
        vs_universe=piece("vs all stocks", values, usable),
        net=net,
        by_month=by_month,
        criteria=IntradayCriteria(
            enough_events=events >= CRITERIA_MIN_INDEPENDENT_EVENTS,
            t_high_enough=bool(net.t >= CRITERIA_MIN_T),
            not_one_month=rest_positive,
            most_months_positive=bool(by_month) and positive_months * 2 >= len(by_month),
        ),
    )


def record_intraday_trial(
    audit: AuditRepository, m: IntradayMeasurement, *, universe_size: int
) -> None:
    """One more variant tried (same journal as the daily ones, so the count is one count)."""
    audit.record(
        "research_trial",
        {
            "variant": f"{m.signal}|no filter|session close",
            "signal": m.signal,
            "regime_filter": None,
            "horizon": "session close",
            "cost": m.cost,
            "universe_size": universe_size,
            "first": m.first,
            "last": m.last,
            "independent_events": m.events,
            "net_mean": m.net.mean,
            "net_t": m.net.t,
            "meets_criteria": m.criteria.all_met,
        },
    )


def _pct(value: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.{digits}f}%"


def _t(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.1f}"


def format_intraday(m: IntradayMeasurement) -> str:
    lines = [f"{m.signal}: {DESCRIPTIONS.get(m.signal, '')}"]
    if not m.usable:
        lines.append("  no day has enough signals and enough stocks to compare with")
        return "\n".join(lines)
    v = m.vs_universe
    lines += [
        f"  {m.first} to {m.last}: {m.days_with_signal} days with a signal, {m.events:,} stock-events",
        f"  stocks that were on: {_pct(m.signal_mean)} on average from entry to close; all stocks "
        f"over the same windows: {_pct(m.universe_mean)}",
        f"  vs all stocks   {_pct(v.mean)} per day  t {_t(v.t)}  positive on {v.positive * 100:.0f}% of days",
        f"  net of {m.cost * 100:.2f}% cost  {_pct(m.net.mean)} per day  t {_t(m.net.t)}",
        "  by month (net): "
        + "  ".join(f"{s.label} {_pct(s.mean, 1)} (t {_t(s.t)}, {s.dates}d)" for s in m.by_month),
    ]
    c = m.criteria

    def mark(ok: bool) -> str:
        return "yes" if ok else "NO"

    lines.append(
        f"  criteria on the development part: {CRITERIA_MIN_INDEPENDENT_EVENTS}+ events "
        f"{mark(c.enough_events)}; net t >= {CRITERIA_MIN_T:g} {mark(c.t_high_enough)}; "
        f"without the best month still positive {mark(c.not_one_month)}; half of the months "
        f"positive {mark(c.most_months_positive)}  ->  "
        + ("meets them" if c.all_met else "does not meet them")
    )
    return "\n".join(lines)


def development_note() -> str:
    return (
        f"Development part only: the hold-out from {INTRADAY_HOLDOUT_START} is not loaded. 15-minute "
        "bars from the IEX feed (a small share of the volume): the opening range is IEX's own prints."
    )
