"""The signal, forward in time: pick today, judge later (docs/strategy-research-plan.md, section 10).

Everything the development part of the data could say about the weekly signal is said (section 9):
a weak, same-direction sign that does not reach the bar, flattered by survivorship. What no past
data can give is a test without look-back bias, and that is only possible in one way: write down
what the signal picks BEFORE the prices that will judge it exist, and judge it when they do.

* Every evening after the close, `record_picks` stores the stocks the signal selects (one journal
  entry per signal and day, in `audit_events` of trading.db, which is backed up).
* `evaluate` looks at those entries once their prices exist and measures them with the SAME code as
  the development part (`research.measure.measure`): in at the next open, out at the close after the
  horizon, minus the average of all stocks, Newey-West t, a flat round-trip cost.

**What keeps it honest.** A pick counts only if it was recorded before the open at which it would
have been entered (a late record is kept and reported, never counted); a day that was missed is
missed, nothing is filled in afterwards; nothing before FORWARD_START is evaluated here (the
development part and its hold-out are the business of `measure-signals`, which cannot see this
period); the variants and the criteria below are fixed in advance and change only as a written
change. The control (a random tenth, drawn each day from that day's date alone) must show nothing.

**What it cannot do.** The effect found on the development part is small (about +0.5 % over 20 days);
a few months of days cannot tell it from nothing (the plan works the numbers out: a real effect of
that size gives an expected t of about 0.4 in six months). A pass is a reason to go on looking, a
miss is not a refutation. It is a learning experiment, and it places no order.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_system.data.universe import MARKET_SYMBOL, research_universe
from trading_system.research.measure import Measurement, measure
from trading_system.research.panel import Panel, build_panel, frames_from_stored
from trading_system.research.signals import (
    CONTROL_SEED,
    DECILE,
    high_52w,
    momentum_12_1,
)
from trading_system.storage.repositories import AuditRepository, MarketBarRepository

MARKET_TZ = ZoneInfo("America/New_York")
EVENT_TYPE = "forward_picks"
TIMEFRAME = "1d"

# The first day whose close a pick is made at. Nothing earlier is judged here.
FORWARD_START = date(2026, 11, 6)

# Fixed in advance (2026-11-06): the weekly signal and its placebo, each held 10, 15 and 20 days, each
# with and without "SPY above its 200-day average" (read from the day's own record). These are the
# six variants of section 9 that showed something on the development part, so no new variant is added
# to the count; the placebo does not count.
SIGNALS = ("momentum_12_1", "random_tenth", "high_52w")
REGIME_FILTERS: tuple[Optional[str], ...] = (None, "spy200")
HORIZONS = (10, 15, 20)

# Variant 20 (plan section 14, registered 2026-11-09 before any code): the 52-week-high signal, ONE
# variant: 20 days, no filter. It has its own start (the first close it could be recorded at after it
# was built); the clock of its minimum counts from there, and a day it missed stays missed.
SIGNAL_START: dict[str, date] = {
    "momentum_12_1": FORWARD_START,
    "random_tenth": FORWARD_START,
    "high_52w": date(2026, 11, 9),
}
SIGNAL_VARIANTS: dict[str, tuple[tuple[Optional[str], ...], tuple[int, ...]]] = {
    "momentum_12_1": (REGIME_FILTERS, HORIZONS),
    "random_tenth": (REGIME_FILTERS, HORIZONS),
    "high_52w": ((None,), (20,)),
}

# Before this much time there is no verdict, whatever the numbers say.
MIN_ELAPSED_DAYS = (
    60  # trading days since FORWARD_START (the paper period the plan fixed: 12 weeks)
)
MIN_MATURED_DAYS = 40  # days whose pick has its full holding period behind it
# The bar for "same direction": the net difference above zero and t at least this (question 4,
# criterion 3). It is a low bar on purpose; see "What it cannot do".
PASS_NET_T = 1.0
CONTROL_MAX_T = 2.0  # the placebo must stay under this, or the instrument is suspect

# A day's bar is taken as final this long after the close (the free feed lags 16 minutes).
SETTLE = timedelta(minutes=30)
# Fewer stocks with a value than this on the day and the signal says nothing.
MIN_RANKED = 50


def close_time(day: date) -> datetime:
    return datetime.combine(day, time(16, 0), tzinfo=MARKET_TZ).astimezone(timezone.utc)


def open_time(day: date) -> datetime:
    return datetime.combine(day, time(9, 30), tzinfo=MARKET_TZ).astimezone(timezone.utc)


def next_weekday(day: date) -> date:
    following = day + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    return following


def last_complete_day(dates: Sequence[pd.Timestamp], now: datetime) -> Optional[date]:
    """The newest day of `dates` whose bar can be taken as final at `now`."""
    for stamp in reversed(list(dates)):
        if close_time(stamp.date()) + SETTLE <= now:
            return date(stamp.year, stamp.month, stamp.day)
    return None


def load_all_panel(repo: MarketBarRepository, symbols: Optional[Sequence[str]] = None) -> Panel:
    """Every stored daily bar, up to the newest. Only this module uses it, and only for days from
    FORWARD_START: the development loader (`panel.load_panel`) still stops at the hold-out."""
    wanted = list(symbols) if symbols else research_universe()
    if MARKET_SYMBOL not in wanted:
        wanted.append(MARKET_SYMBOL)
    return build_panel(frames_from_stored(repo.load(wanted, TIMEFRAME)))


def random_tenth_on(day: date, symbols: Sequence[str]) -> list[str]:
    """The placebo's pick for one day: a tenth of the symbols, drawn from the date alone, so the
    same day always draws the same stocks and no later price can matter."""
    ordered = sorted(symbols)
    draw = np.random.default_rng([CONTROL_SEED, int(day.strftime("%Y%m%d"))]).random(len(ordered))
    return [symbol for symbol, value in zip(ordered, draw) if value < DECILE]


def picks_on(panel: Panel, signal: str, day: date) -> list[str]:
    """The stocks `signal` selects at the close of `day`, from the prices up to that close."""
    stamp = pd.Timestamp(day)
    if stamp not in panel.close.index:
        raise ValueError(f"no bar for {day}")
    has_price = panel.close.loc[stamp].notna()
    if int(has_price.sum()) < MIN_RANKED:
        return []
    if signal == "momentum_12_1":
        row = momentum_12_1(panel).loc[stamp]
        return sorted(str(s) for s in row.index[row.to_numpy(dtype=bool)])
    if signal == "high_52w":
        row = high_52w(panel).loc[stamp]
        return sorted(str(s) for s in row.index[row.to_numpy(dtype=bool)])
    if signal == "random_tenth":
        return sorted(random_tenth_on(day, [str(s) for s in has_price.index[has_price]]))
    raise ValueError(f"unknown forward signal {signal!r}")


@dataclass
class RecordResult:
    day: Optional[date] = None
    recorded: list[str] = field(default_factory=list)  # signals written now
    already: list[str] = field(default_factory=list)  # signals that had been written before
    skipped: Optional[str] = None  # why nothing was written, when that is the case

    def line(self) -> str:
        if self.skipped:
            return f"Nothing recorded: {self.skipped}"
        parts = []
        if self.recorded:
            parts.append("recorded " + ", ".join(self.recorded))
        if self.already:
            parts.append("already on record: " + ", ".join(self.already))
        return f"Picks for the close of {self.day}: " + "; ".join(parts)


def _entries(audit: AuditRepository, now: datetime) -> list[tuple[datetime, dict[str, Any]]]:
    start = datetime.combine(FORWARD_START, time.min, tzinfo=timezone.utc)
    return audit.events(EVENT_TYPE, start, now + timedelta(days=2))


def record_picks(audit: AuditRepository, repo: MarketBarRepository, now: datetime) -> RecordResult:
    """Writes the picks of the newest complete day, once. Safe to call every hour."""
    panel = load_all_panel(repo)
    if len(panel.symbols) < MIN_RANKED:
        return RecordResult(
            skipped=f"only {len(panel.symbols)} stocks have daily bars (run `fetch-daily`)"
        )
    day = last_complete_day(panel.dates, now)
    if day is None:
        return RecordResult(skipped="no day is complete yet")
    result = RecordResult(day=day)
    if day < FORWARD_START:
        result.skipped = f"the newest complete day ({day}) is before the start ({FORWARD_START})"
        return result
    if now >= open_time(next_weekday(day)):
        result.skipped = (
            f"too late for {day}: its picks would be recorded after the next open, so they could "
            "not count"
        )
        return result

    done = {
        (str(details.get("signal")), str(details.get("day"))) for _, details in _entries(audit, now)
    }
    stamp = pd.Timestamp(day)
    market_up = bool(panel.market_up.loc[stamp])
    market_known = bool(panel.market_known.loc[stamp])
    for signal in SIGNALS:
        if day < SIGNAL_START[signal]:
            continue
        if (signal, day.isoformat()) in done:
            result.already.append(signal)
            continue
        symbols = picks_on(panel, signal, day)
        audit.record(
            EVENT_TYPE,
            {
                "signal": signal,
                "day": day.isoformat(),
                "symbols": symbols,
                "stocks": int(panel.close.loc[stamp].notna().sum()),
                "market_up": market_up,
                "market_known": market_known,
            },
            when=now,
        )
        result.recorded.append(signal)
    return result


# --------------------------------------------------------------------------- judging them later


@dataclass(frozen=True)
class Variant:
    signal: str
    regime_filter: Optional[str]
    horizon: int
    measurement: Measurement
    verdict: str  # "too early", "pass", "no pass", "control ok", "control suspect"

    @property
    def label(self) -> str:
        return f"{self.signal}, {self.regime_filter or 'no filter'}, {self.horizon} days"


@dataclass(frozen=True)
class ForwardReport:
    start: date
    through: Optional[date]  # newest day in the data
    elapsed_days: int  # trading days since the start
    recorded_days: dict[str, int]  # signal -> days with a valid record
    late: int  # records that came after the entry open (kept, not counted)
    missed_days: int  # trading days since the start with no record at all
    variants: tuple[Variant, ...]
    # Descriptive only: on the days both were validly recorded, the days and the mean share of
    # `high_52w`'s picks that `momentum_12_1` also picked (how much new information the second brings).
    overlap: Optional[tuple[int, float]] = None


def _valid_picks(
    entries: Sequence[tuple[datetime, dict[str, Any]]], panel: Panel
) -> tuple[dict[str, dict[date, dict[str, Any]]], int]:
    """Per signal and day, the first record that was made in time; and how many were late."""
    dates = [t.date() for t in panel.dates]
    chosen: dict[str, dict[date, dict[str, Any]]] = {s: {} for s in SIGNALS}
    late = 0
    for when, details in entries:
        signal, day_text = str(details.get("signal")), str(details.get("day"))
        if signal not in chosen:
            continue
        day = date.fromisoformat(day_text)
        if day < SIGNAL_START[signal] or day in chosen[signal]:
            continue
        following = next((d for d in dates if d > day), None)
        deadline = open_time(following if following is not None else next_weekday(day))
        if when > deadline:
            late += 1
            continue
        chosen[signal][day] = {**details, "when": when}
    return chosen, late


def _on_frame(panel: Panel, picks: Mapping[date, dict[str, Any]], regime: Optional[str]) -> Any:
    on = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    for day, details in picks.items():
        if regime == "spy200" and not (details["market_known"] and details["market_up"]):
            continue
        stamp = pd.Timestamp(day)
        if stamp in on.index:
            for symbol in details["symbols"]:
                if symbol in on.columns:
                    on.loc[stamp, symbol] = True
    return on


def evaluate(audit: AuditRepository, repo: MarketBarRepository, now: datetime) -> ForwardReport:
    panel = load_all_panel(repo)
    entries = _entries(audit, now)
    picks, late = _valid_picks(entries, panel)
    since = [t for t in panel.dates if t.date() >= FORWARD_START]
    complete = [t for t in since if close_time(t.date()) + SETTLE <= now]
    elapsed = len(complete)
    # Each signal's clock starts at its own start (the older two share FORWARD_START).
    elapsed_of = {s: len([t for t in complete if t.date() >= SIGNAL_START[s]]) for s in SIGNALS}
    any_record = {day for per_signal in picks.values() for day in per_signal}
    missed = len([t for t in complete if t.date() not in any_record])

    variants: list[Variant] = []
    for signal in SIGNALS:
        regimes, horizons = SIGNAL_VARIANTS[signal]
        for regime in regimes:
            on = _on_frame(panel, picks[signal], regime)
            for horizon in horizons:
                m = measure(panel, on, signal=signal, horizon=horizon)
                variants.append(
                    Variant(signal, regime, horizon, m, _verdict(signal, m, elapsed_of[signal]))
                )
    shared = [
        len(set(new["symbols"]) & set(picks["momentum_12_1"][day]["symbols"])) / len(new["symbols"])
        for day, new in picks["high_52w"].items()
        if day in picks["momentum_12_1"] and new["symbols"]
    ]
    return ForwardReport(
        start=FORWARD_START,
        through=panel.dates[-1].date() if len(panel.dates) else None,
        elapsed_days=elapsed,
        recorded_days={s: len(picks[s]) for s in SIGNALS},
        late=late,
        missed_days=missed,
        variants=tuple(variants),
        overlap=(len(shared), float(np.mean(shared))) if shared else None,
    )


def _verdict(signal: str, m: Measurement, elapsed: int) -> str:
    matured = m.vs_universe.dates
    if elapsed < MIN_ELAPSED_DAYS or matured < MIN_MATURED_DAYS:
        return "too early"
    if signal == "random_tenth":
        gross_t = m.vs_universe.t
        return (
            "control ok"
            if np.isfinite(gross_t) and abs(gross_t) < CONTROL_MAX_T
            else ("control suspect")
        )
    ok = np.isfinite(m.net.t) and m.net.mean > 0 and m.net.t >= PASS_NET_T
    return "pass" if ok else "no pass"


def _pct(value: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.{digits}f}%"


def _t(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.1f}"


def format_report(report: ForwardReport) -> str:
    lines = [
        f"Forward test since {report.start} (docs/strategy-research-plan.md, section 10)",
        f"  newest day in the data: {report.through}; {report.elapsed_days} trading day(s) since "
        f"the start (a verdict needs {MIN_ELAPSED_DAYS} and {MIN_MATURED_DAYS} matured days)",
        "  days with a valid record: "
        + ", ".join(f"{s} {n}" for s, n in report.recorded_days.items())
        + f"; missed days: {report.missed_days}; late records (kept, not counted): {report.late}",
    ]
    for v in report.variants:
        m = v.measurement
        if not m.usable:
            lines.append(f"  {v.label}: no matured day yet  [{v.verdict}]")
            continue
        lines.append(
            f"  {v.label}: {m.vs_universe.dates} matured day(s), vs all stocks "
            f"{_pct(m.vs_universe.mean)} (t {_t(m.vs_universe.t)}), net {_pct(m.net.mean)} "
            f"(t {_t(m.net.t)}), vs SPY {_pct(m.vs_market.mean)}  [{v.verdict}]"
        )
    if report.overlap is not None:
        days, share = report.overlap
        lines.append(
            f"  high_52w picks that momentum_12_1 also picked: {share * 100:.0f}% on average over "
            f"{days} day(s) (descriptive; the more they share, the less new the second one is)"
        )
    lines.append(
        "  A pass means the same direction (net above zero, net t >= "
        f"{PASS_NET_T:g}) after the time above, with the control quiet; it is a reason to look "
        "further, not proof. Nothing here places an order."
    )
    return "\n".join(lines)
