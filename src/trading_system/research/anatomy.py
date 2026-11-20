"""The anatomy of a closed trade (docs/strategy-research-plan.md, section 16.2 and 16.3, "A2").

Pure arithmetic over closed round trips: the R of the plan, the R in units of the stock's own ATR, the
split of the price move into the nights and the rest (the days), and the closed list of five splits of
section 16.3. Nothing here reads a network or a database; `research/paper_report.py` gathers the facts.

What the numbers are NOT: they are not a test of profit. `paper3`'s fills are the best-quote fills of
Alpaca's paper account (no impact, no queue), so they say how this path works, not what it costs.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
MIN_CLOSED_TRADES = 300  # numbers about the result appear only from here (section 16.2)
SECOND_LOOK = 600
MIN_SIDE = 50  # trades a side of a split needs
SPLIT_T = 2.6  # Bonferroni for five splits at 5 %
STOP, TARGET, OTHER = "stop", "target", "other"


@dataclass(frozen=True)
class Trade:
    """One closed round trip of one symbol."""

    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # STOP, TARGET or OTHER
    quantity: float
    limit_price: Optional[float] = None  # the entry's limit, from the order intent
    stop_price: Optional[float] = None  # the planned stop, from the order intent
    signal_close: Optional[float] = None  # the signal bar's close (from the context)
    context: Mapping[str, Any] = field(default_factory=dict)  # the signal_context event, if any

    @property
    def log_return(self) -> float:
        return math.log(self.exit_price / self.entry_price)

    @property
    def entry_day(self) -> date:
        return self.entry_time.astimezone(MARKET_TZ).date()

    @property
    def group(self) -> str:
        return str(self.context.get("group", "other"))

    @property
    def hold_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600


# --------------------------------------------------------------------------- round trips


def pair_round_trips(fills: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Round trips from the account's fills: per symbol, in time order, a trip opens when the position
    goes from nothing to something and closes when it is nothing again. The entry price is the
    volume-weighted price of the buys, the exit price that of the sells. A trip still open is left out.

    Each fill is a dict with `symbol`, `side` (buy or sell), `price`, `qty`, `order_id` and
    `transaction_time` (as Alpaca reports them)."""
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for fill in fills:
        if fill.get("type") not in (None, "fill", "partial_fill"):
            continue
        by_symbol.setdefault(str(fill["symbol"]), []).append(fill)
    trips: list[dict[str, Any]] = []
    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda r: str(r["transaction_time"]))
        position = 0.0
        buys: list[Mapping[str, Any]] = []
        sells: list[Mapping[str, Any]] = []
        for row in rows:
            qty = float(row["qty"])
            if str(row["side"]).lower() == "buy":
                position += qty
                buys.append(row)
            else:
                position -= qty
                sells.append(row)
            if buys and position <= 1e-9:
                trips.append({"symbol": symbol, "buys": buys, "sells": sells})
                position, buys, sells = 0.0, [], []
    return trips


def _vwap(rows: Sequence[Mapping[str, Any]]) -> float:
    quantity = sum(float(r["qty"]) for r in rows)
    return sum(float(r["price"]) * float(r["qty"]) for r in rows) / quantity


def _instant(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    moment: datetime = stamp.to_pydatetime()
    return moment


def trades_from_trips(
    trips: Iterable[Mapping[str, Any]],
    exit_reason: Callable[[str], str],
    entry_facts: Callable[[str], Mapping[str, Any]],
) -> list[Trade]:
    """`exit_reason(order_id)` names how an exit order ended the trip (STOP, TARGET or OTHER);
    `entry_facts(order_id)` gives what is known about the entry order: `limit_price`, `stop_price` and
    `context` (all optional)."""
    trades: list[Trade] = []
    for trip in trips:
        buys, sells = trip["buys"], trip["sells"]
        facts = dict(entry_facts(str(buys[0]["order_id"])))
        context = facts.get("context") or {}
        trades.append(
            Trade(
                symbol=trip["symbol"],
                entry_time=_instant(buys[0]["transaction_time"]),
                entry_price=_vwap(buys),
                exit_time=_instant(sells[-1]["transaction_time"]),
                exit_price=_vwap(sells),
                exit_reason=exit_reason(str(sells[-1]["order_id"])),
                quantity=sum(float(r["qty"]) for r in buys),
                limit_price=facts.get("limit_price"),
                stop_price=facts.get("stop_price"),
                signal_close=context.get("close"),
                context=context,
            )
        )
    return trades


# --------------------------------------------------------------------------- the numbers


def r_plan(trade: Trade) -> Optional[float]:
    """(exit - entry) / (limit - stop): the move in units of the risk the plan took."""
    if trade.limit_price is None or trade.stop_price is None:
        return None
    risk = trade.limit_price - trade.stop_price
    if risk <= 0:
        return None
    return (trade.exit_price - trade.entry_price) / risk


def r_atr(trade: Trade) -> Optional[float]:
    """The trade's return divided by the ATR percent of its signal bar, if the context has one."""
    atr = trade.context.get("atr_pct")
    if not atr:
        return None
    return (trade.exit_price / trade.entry_price - 1) * 100 / float(atr)


def leg_split(trade: Trade, daily: pd.DataFrame) -> Optional[tuple[float, float]]:
    """(nights, days) as log returns, summing to the trade's. The nights are the sum over every night
    between the entry day and the exit day of ln(next open / close of the day); the days are the rest.
    `daily` has `open` and `close`, indexed by the New York date. None if a day is missing."""
    first, last = trade.entry_day, trade.exit_time.astimezone(MARKET_TZ).date()
    if last <= first:
        return 0.0, trade.log_return
    days = [d for d in daily.index if first <= d.date() <= last]
    days = sorted(days)
    if not days or days[0].date() != first or days[-1].date() != last:
        return None
    nights = 0.0
    for before, after in zip(days, days[1:]):
        nights += math.log(float(daily.loc[after, "open"]) / float(daily.loc[before, "close"]))
    return nights, trade.log_return - nights


# --------------------------------------------------------------------------- the splits

SPLITS: dict[str, Callable[[Trade], Optional[bool]]] = {
    "group: risky vs normal": lambda t: (
        None if t.group not in ("risky", "normal") else t.group == "risky"
    ),
    "market: SPY above its 200-day average": lambda t: t.context.get("spy_above_200d"),
    "stress: SPY volatility percentile 67 or more": lambda t: (
        None
        if t.context.get("spy_vol20_pctile") is None
        else float(t.context["spy_vol20_pctile"]) >= 67
    ),
    "hour: first 30 minutes": lambda t: (
        None
        if t.context.get("minutes_since_open") is None
        else int(t.context["minutes_since_open"]) < 30
    ),
    "earnings within 2 sessions": lambda t: (
        t.context.get("earnings_in_2_sessions") if t.context.get("earnings_known") else None
    ),
}


@dataclass(frozen=True)
class SplitResult:
    name: str
    n_true: int
    n_false: int
    mean_true: float
    mean_false: float
    t: float
    same_sign_in_halves: bool
    shows_something: bool


def _daily_means(trades: Iterable[Trade]) -> dict[date, float]:
    by_day: dict[date, list[float]] = {}
    for trade in trades:
        value = r_plan(trade)
        if value is not None:
            by_day.setdefault(trade.entry_day, []).append(value)
    return {day: mean(values) for day, values in by_day.items()}


def _welch(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = sum((x - mean(a)) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mean(b)) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (mean(a) - mean(b)) / se if se > 0 else float("nan")


def _side_mean(trades: Iterable[Trade]) -> Optional[float]:
    values = [v for v in (r_plan(t) for t in trades) if v is not None]
    return mean(values) if values else None


def run_split(name: str, trades: Sequence[Trade]) -> SplitResult:
    """One split of section 16.3: the mean R of the two sides, the t of the difference of their daily
    means (the unit is the day), the sign in each half of the sample by time, and the verdict."""
    side = SPLITS[name]
    true_side = [t for t in trades if side(t) is True]
    false_side = [t for t in trades if side(t) is False]
    daily_true = list(_daily_means(true_side).values())
    daily_false = list(_daily_means(false_side).values())
    t_value = _welch(daily_true, daily_false)

    ordered = sorted(trades, key=lambda t: t.entry_time)
    middle = len(ordered) // 2
    signs = []
    for half in (ordered[:middle], ordered[middle:]):
        a = _side_mean(t for t in half if side(t) is True)
        b = _side_mean(t for t in half if side(t) is False)
        signs.append(None if a is None or b is None else a - b > 0)
    same = signs[0] is not None and signs[0] == signs[1]
    mean_true = _side_mean(true_side)
    mean_false = _side_mean(false_side)
    shows = (
        len(true_side) >= MIN_SIDE
        and len(false_side) >= MIN_SIDE
        and not math.isnan(t_value)
        and abs(t_value) >= SPLIT_T
        and same
    )
    return SplitResult(
        name=name,
        n_true=len(true_side),
        n_false=len(false_side),
        mean_true=float("nan") if mean_true is None else mean_true,
        mean_false=float("nan") if mean_false is None else mean_false,
        t=t_value,
        same_sign_in_halves=same,
        shows_something=shows,
    )


# --------------------------------------------------------------------------- the report


@dataclass
class Report:
    trades: list[Trade]
    open_trades: int = 0
    entries_filled: int = 0
    daily: Optional[Mapping[str, pd.DataFrame]] = None


def _share(count: int, total: int) -> str:
    return f"{count / total * 100:.0f} %" if total else "-"


def _median(values: Sequence[float]) -> str:
    return f"{float(pd.Series(list(values)).median()):.1f}" if values else "-"


def format_report(report: Report) -> str:
    """Operational numbers always; numbers about the result only from MIN_CLOSED_TRADES closed trades
    (section 16.2), the same discipline as `judge-oracle`."""
    trades = report.trades
    n = len(trades)
    lines = [
        "paper3 report (docs/strategy-research-plan.md, section 16; docs/paper-third-account.md, section 6)",
        "  an operational view of the real paper orders, not a test of profit; Alpaca's paper fills have "
        "no market impact and no queue, so they are an upper bound on execution quality.",
        f"  entries filled: {report.entries_filled}; closed trades: {n}; still open: {report.open_trades}",
    ]
    if not n:
        return "\n".join(lines)
    by_group = {g: [t for t in trades if t.group == g] for g in ("risky", "normal", "other")}
    for group, rows in by_group.items():
        if not rows:
            continue
        stops = sum(1 for t in rows if t.exit_reason == STOP)
        targets = sum(1 for t in rows if t.exit_reason == TARGET)
        slip = [
            (t.entry_price / t.limit_price - 1) * 100 for t in rows if t.limit_price is not None
        ]
        lines.append(
            f"  {group}: {len(rows)} closed, stops {_share(stops, len(rows))}, targets "
            f"{_share(targets, len(rows))}, median held {_median([t.hold_hours for t in rows])} h, "
            f"fill versus limit {sum(slip) / len(slip):+.3f} %"
            if slip
            else f"  {group}: {len(rows)} closed, stops {_share(stops, len(rows))}, targets "
            f"{_share(targets, len(rows))}, median held {_median([t.hold_hours for t in rows])} h"
        )
    if n < MIN_CLOSED_TRADES:
        lines.append(
            f"  numbers about the result: not shown until {MIN_CLOSED_TRADES} closed trades "
            f"({n} so far); a look at {MIN_CLOSED_TRADES} and one at {SECOND_LOOK}, no more."
        )
        return "\n".join(lines)

    r_values = [v for v in (r_plan(t) for t in trades) if v is not None]
    atr_values = [v for v in (r_atr(t) for t in trades) if v is not None]
    if r_values:
        lines.append(
            f"  result: mean R of the plan {mean(r_values):+.3f} over {len(r_values)} trades"
            + (
                f"; mean return in ATR units {mean(atr_values):+.3f} over {len(atr_values)}"
                if atr_values
                else ""
            )
        )
    else:
        lines.append("  result: no closed trade has its planned stop on record, so there is no R")
    if report.daily:
        splits = [
            (t, leg_split(t, report.daily[t.symbol])) for t in trades if t.symbol in report.daily
        ]
        usable = [(t, s) for t, s in splits if s is not None]
        if usable:
            nights = mean(s[0] for _, s in usable) * 100
            days = mean(s[1] for _, s in usable) * 100
            slept = sum(
                1 for t, _ in usable if t.exit_time.astimezone(MARKET_TZ).date() > t.entry_day
            )
            lines.append(
                f"  nights {nights:+.3f} % and days {days:+.3f} % a trade (log returns, {len(usable)} "
                f"trades); {_share(slept, len(usable))} slept at least one night"
            )
    lines.append("  the five splits of section 16.3 (mean R of each side; the unit is the day):")
    for name in SPLITS:
        s = run_split(name, trades)
        lines.append(
            f"    {name}: {s.n_true} / {s.n_false} trades, mean R {s.mean_true:+.3f} / "
            f"{s.mean_false:+.3f}, t {s.t:+.2f}, same sign in both halves "
            f"{'yes' if s.same_sign_in_halves else 'no'} -> "
            f"{'SHOWS SOMETHING' if s.shows_something else 'not shown'}"
        )
    return "\n".join(lines)
