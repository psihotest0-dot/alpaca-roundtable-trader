"""What prices do after a signal, against what they do anyway.

For every day on which a signal has at least a few stocks "on", look at the stocks' return from the
next day's open to the close `horizon` days later (the signal is only known at the close, so the
next open is the first price one could get), and take the difference to the average of all stocks
that day. That difference is one observation per day: stocks of one day move together, so counting
them as independent events would make every result look surer than it is. Observations of
neighbouring days overlap in time, which stats.hac_mean accounts for.

The report shows the number three ways, never alone: for the whole development part, year by year,
and by the state of the market (SPY above or below its 200-day average). It checks the result against
the criteria the owner fixed before any result was seen (docs/strategy-research-plan.md, question 4).
Passing them makes a signal a candidate for the hold-out; it is not a result.

**What this does not account for.** The stocks are today's large companies (survivorship): those that
went under or fell out of the group are missing, and those that are large today rose to it, which
flatters anything that buys past winners. Costs are a flat round-trip figure, not a simulation.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from trading_system.research.panel import HOLDOUT_START, Panel
from trading_system.research.stats import hac_mean

# The criteria for a candidate, fixed on 2026-11-05 before any result was seen. They do not change
# after a result; if they ever do, that is written down as a change with its reason.
CRITERIA_MIN_INDEPENDENT_EVENTS = 1000
CRITERIA_MIN_T = 3.0

# 0.05 % slippage on each side, the figure the backtests of the live strategies used.
ROUND_TRIP_COST = 0.001

MIN_SIGNAL_STOCKS = 5  # fewer "on" stocks on a day and that day is not an observation
MIN_UNIVERSE = 50  # fewer stocks with a forward return on a day and the day is not an observation


@dataclass(frozen=True)
class MeasureRules:
    """What differs from one market to the next; the measurement itself does not. The default is
    the stock market of the research plan's sections 8 to 9. The crypto rules are in
    research/crypto.py (plan section 15)."""

    min_signal: int  # fewer "on" than this on a day and the day is not an observation
    min_universe: int  # fewer with a forward return on a day and the day is not an observation
    min_events: int  # the criterion "enough non-overlapping events"
    need_one_off: bool  # a day on which everything is "on" compares nothing: it is left out
    asset: str  # the word in the report: "stocks", "coins"
    market: str  # what the market regime and "vs market" are read from: "SPY", "BTC"


STOCK_RULES = MeasureRules(
    min_signal=MIN_SIGNAL_STOCKS,
    min_universe=MIN_UNIVERSE,
    min_events=CRITERIA_MIN_INDEPENDENT_EVENTS,
    need_one_off=False,
    asset="stocks",
    market="SPY",
)


@dataclass(frozen=True)
class Slice:
    """The average difference per day (a fraction: 0.01 is 1 %) over some set of days."""

    label: str
    dates: int
    events: int  # stock-events on those days (windows overlap, see Measurement.independent_events)
    mean: float
    t: float
    positive: float  # share of the days with a positive difference


@dataclass(frozen=True)
class Criteria:
    enough_events: bool
    t_high_enough: bool
    not_one_year: bool  # without the year that contributed most, the mean is still above zero
    most_years_positive: bool  # at least half of the years are above zero

    @property
    def all_met(self) -> bool:
        return (
            self.enough_events
            and self.t_high_enough
            and self.not_one_year
            and self.most_years_positive
        )


@dataclass(frozen=True)
class Measurement:
    signal: str
    horizon: int
    cost: float
    first: Optional[date]
    last: Optional[date]
    independent_events: float  # stock-events divided by the horizon: windows that do not overlap
    universe_mean: float  # what all the stocks did over the same windows
    signal_mean: float  # what the stocks that were "on" did
    vs_universe: Slice
    vs_market: Slice
    net: Slice  # vs_universe minus the cost of a round trip
    by_year: tuple[Slice, ...]
    by_regime: tuple[Slice, ...]
    criteria: Criteria
    rules: MeasureRules = STOCK_RULES
    panel_days: int = 0  # the days of the panel, to say how many of them were observations

    @property
    def usable(self) -> bool:
        return self.vs_universe.dates > 0


def forward_returns(panel: Panel, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Per stock and for SPY: the return from the open of the next day to the close `horizon` days
    after the signal day. Missing where the window leaves the panel or touches a blanked day."""
    stocks = panel.log_return.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    market = panel.market_log_return.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return (
        pd.DataFrame(np.expm1(stocks - panel.overnight_gap.shift(-1))),
        pd.Series(np.expm1(market - panel.market_gap.shift(-1))),
    )


def _slice(label: str, values: np.ndarray, mask: np.ndarray, events: np.ndarray, lag: int) -> Slice:
    stat = hac_mean(values, mask, lag)
    chosen = np.asarray(mask, dtype=bool) & np.isfinite(values)
    positive = float((values[chosen] > 0).mean()) if chosen.any() else float("nan")
    return Slice(label, int(chosen.sum()), int(events[chosen].sum()), stat.mean, stat.t, positive)


def measure(
    panel: Panel,
    on: pd.DataFrame,
    *,
    signal: str,
    horizon: int,
    cost: float = ROUND_TRIP_COST,
    rules: MeasureRules = STOCK_RULES,
) -> Measurement:
    """The signal `on` (True where it is on, shaped like the panel) over the panel's days."""
    if horizon < 1:
        raise ValueError("the horizon is at least one day")
    forward, forward_market = forward_returns(panel, horizon)
    on = on.reindex_like(forward).fillna(False).astype(bool)

    has_forward = forward.notna()
    chosen = on & has_forward
    events = chosen.sum(axis=1)
    universe_mean = forward.mean(axis=1)
    signal_mean = forward.where(chosen).mean(axis=1)
    covered = has_forward.sum(axis=1)
    usable = (events >= rules.min_signal) & (covered >= rules.min_universe) & forward_market.notna()
    if rules.need_one_off:
        usable &= events < covered
    vs_universe = (signal_mean - universe_mean).where(usable)
    vs_market = (signal_mean - forward_market).where(usable)

    lag = horizon - 1
    days = usable.to_numpy()
    counts = events.to_numpy()
    spread = vs_universe.to_numpy()
    net_values = spread - cost

    def piece(label: str, values: np.ndarray, mask: np.ndarray) -> Slice:
        return _slice(label, values, mask, counts, lag)

    years = panel.dates.year.to_numpy()
    by_year = tuple(
        piece(str(year), net_values, days & (years == year)) for year in sorted(set(years[days]))
    )
    known = panel.market_known.to_numpy()
    up = panel.market_up.to_numpy()
    by_regime = (
        piece(f"{rules.market} above its 200-day average", net_values, days & known & up),
        piece(f"{rules.market} below its 200-day average", net_values, days & known & ~up),
    )

    net = piece("net of cost", net_values, days)
    independent = float(counts[days].sum()) / horizon
    dates_used = panel.dates[days]

    return Measurement(
        signal=signal,
        horizon=horizon,
        cost=cost,
        first=dates_used[0].date() if len(dates_used) else None,
        last=dates_used[-1].date() if len(dates_used) else None,
        independent_events=independent,
        universe_mean=float(universe_mean[usable].mean()) if days.any() else float("nan"),
        signal_mean=float(signal_mean[usable].mean()) if days.any() else float("nan"),
        vs_universe=piece(f"vs all {rules.asset}", spread, days),
        vs_market=piece(f"vs {rules.market}", vs_market.to_numpy(), days),
        net=net,
        by_year=by_year,
        by_regime=by_regime,
        criteria=_criteria(net, by_year, independent, net_values, days, years, rules.min_events),
        rules=rules,
        panel_days=len(panel.dates),
    )


def _criteria(
    net: Slice,
    by_year: tuple[Slice, ...],
    independent_events: float,
    net_values: np.ndarray,
    days: np.ndarray,
    years: np.ndarray,
    min_events: int = CRITERIA_MIN_INDEPENDENT_EVENTS,
) -> Criteria:
    contributions = {
        int(s.label): float(np.nansum(net_values[days & (years == int(s.label))])) for s in by_year
    }
    rest_positive = False
    if len(contributions) > 1:
        best = max(contributions, key=lambda y: contributions[y])
        rest = days & (years != best)
        rest_positive = bool(np.nanmean(net_values[rest]) > 0)
    positive_years = sum(1 for s in by_year if s.mean > 0)
    return Criteria(
        enough_events=independent_events >= min_events,
        t_high_enough=bool(net.t >= CRITERIA_MIN_T),
        not_one_year=rest_positive,
        most_years_positive=bool(by_year) and positive_years * 2 >= len(by_year),
    )


def _pct(value: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.{digits}f}%"


def _t(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.1f}"


def format_measurement(m: Measurement) -> str:
    asset, market = m.rules.asset, m.rules.market
    lines = [
        f"{m.signal}, held {m.horizon} trading day{'s' if m.horizon > 1 else ''} "
        "(in at the next open, out at the close)"
    ]
    if not m.usable:
        lines.append(f"  no day has enough {asset} on and enough {asset} to compare with")
        return "\n".join(lines)
    v = m.vs_universe
    lines += [
        f"  {m.first} to {m.last}: {v.dates:,} days with a signal, {v.events:,} "
        f"{asset.rstrip('s')}-events (about {m.independent_events:,.0f} that do not overlap)",
        f"  {asset} that were on: {_pct(m.signal_mean)} on average; all {asset}: "
        f"{_pct(m.universe_mean)}",
        f"  vs all {asset}   {_pct(v.mean)} per day  t {_t(v.t)}  "
        f"positive on {v.positive * 100:.0f}% of days",
        f"  vs {market}".ljust(18) + f"{_pct(m.vs_market.mean)} per day  t {_t(m.vs_market.t)}",
        f"  net of {m.cost * 100:.2f}% cost  {_pct(m.net.mean)} per day  t {_t(m.net.t)}",
        "  by year (net):  "
        + "  ".join(f"{s.label} {_pct(s.mean, 1)} (t {_t(s.t)})" for s in m.by_year),
    ]
    if m.rules.need_one_off and m.panel_days:
        lines.append(
            f"  days used: {v.dates:,} of {m.panel_days:,} ({v.dates / m.panel_days * 100:.0f}%); "
            f"the others had too few {asset} on, none off, or no window"
        )
    for s in m.by_regime:
        lines.append(f"  {s.label}: {_pct(s.mean)} net, t {_t(s.t)}, {s.dates:,} days")
    c = m.criteria

    def mark(ok: bool) -> str:
        return "yes" if ok else "NO"

    lines.append(
        f"  criteria on the development part: {m.rules.min_events}+ events "
        f"{mark(c.enough_events)}; net t >= {CRITERIA_MIN_T:g} {mark(c.t_high_enough)}; "
        f"without the best year still positive {mark(c.not_one_year)}; "
        f"half of the years positive {mark(c.most_years_positive)}  ->  "
        + ("meets them" if c.all_met else "does not meet them")
    )
    return "\n".join(lines)


def development_note() -> str:
    return (
        f"Development part only: the hold-out from {HOLDOUT_START} is not loaded. Stocks are "
        "today's large companies (survivorship), which flatters anything that buys past winners."
    )
