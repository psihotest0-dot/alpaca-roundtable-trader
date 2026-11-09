"""The two candidates to repair the live strategy, run forward in time (docs/strategy-research-plan.md,
section 12.1, registered on 2026-11-07 before any code and before any signal).

**What is recorded.** On every successful pass the Oracle observer (`runtime/observe.py`) also asks
`trend_cross_session` and `trend_cross_session_late` for their signals over the same stored bars it
shows the live strategy, and journals each signal as a `candidate_signal` event: the two candidates
AND the live `trend_cross` over the same symbols and bars (the control). Nothing is put to the Oracle
and no order is possible. A `candidate_cycle` event marks every pass that ran, so the number of
trading days covered is known even for a day without a single signal.

**How it is judged.** The signal is the unit, the account plays no part. Each signal is played out with
the code `judge-oracle` uses (`backtest/shadow.py: play_out`), one open trade per symbol and strategy,
and a trade that no level has reached yet is valued at the last close and COUNTED, not dropped (only
the fast ones would be left otherwise). Only signals old enough (MATURITY) are used.

**The rules are in the code, because the discipline is the point.** Per candidate: below MIN_TRADES
mature trades (times the look number) or MIN_RUN_DAYS days run, the judge prints COUNTS ONLY and returns
no return; the first look needs t >= FIRST_LOOK_T, the second (at double the trades) SECOND_LOOK_T, and
there is no third (the second wants twice the trades of the first); every look is journalled. "helps" needs ALL FOUR of: (1) a mean net trade above zero
with t against zero at the bar, (2) a mean above the control's, Welch t >= DIFFERENCE_T, (3) the same
sign in both halves of the sample, in time, for both the mean and the difference, (4) a t of the
per-day means against zero of at least DAILY_T (trades of one day are not independent). Anything else
is "not shown to help". Even "helps" is a reason to look further; the live loop is never switched by it.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_system.backtest.analysis import t_statistic, welch_t
from trading_system.backtest.shadow import BAR, SignalToPlay, play_out
from trading_system.domain.models import StrategySignal
from trading_system.observability import metrics
from trading_system.storage.repositories import AuditRepository, MarketBarRepository
from trading_system.strategy.base import BaseStrategy
from trading_system.strategy.trend import filter_signals

logger = logging.getLogger(__name__)

CANDIDATE_EVENT = "candidate_signal"
CYCLE_EVENT = "candidate_cycle"
JUDGEMENT_EVENT = "candidate_judgement"

CONTROL = "trend_cross"  # the live strategy, over the same symbols and bars
CANDIDATES = ("trend_cross_session", "trend_cross_session_late")  # section 12.1
# Added on 2026-11-09 (section 14, variant 19), registered before any code: its own bar, because the
# family of candidates the observer looks at is three, and its own count of trading days, from the
# first pass that ran it. The pair above keeps its rules unchanged.
NEW_CANDIDATES = ("dip_in_uptrend",)
RECORDED = CANDIDATES + NEW_CANDIDATES  # what the observer runs and the judge looks at

MIN_TRADES = 150  # mature trades of the candidate for the first look; double for the second
MIN_RUN_DAYS = 40  # trading days on which the candidates were run
MATURITY = timedelta(days=14)  # a signal counts once its entry is this old
FIRST_LOOK_T = 2.2  # 2.0, a little higher because two candidates are looked at
SECOND_LOOK_T = 2.5
# (first look, second look) per candidate: 2.4 and 2.7 for the one added in section 14, because
# three candidates are looked at in all (the ATR exit candidate has the same bar).
LOOK_T = {
    **{name: (FIRST_LOOK_T, SECOND_LOOK_T) for name in CANDIDATES},
    "dip_in_uptrend": (2.4, 2.7),
}
DIFFERENCE_T = 1.0  # weak on purpose: the groups overlap, which makes Welch conservative
DAILY_T = 1.5
MAX_LOOKS = 2

MARKET_TZ = ZoneInfo("America/New_York")
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------- recording


def _utc(instant: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(instant)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def journal_signal(
    audit: AuditRepository, strategy: str, signal: StrategySignal, when: datetime
) -> None:
    audit.record(
        CANDIDATE_EVENT,
        {
            "strategy": strategy,
            "version": signal.version,
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "closed_bar": _utc(signal.closed_bar_timestamp).isoformat(),
        },
        when=when,
    )


def record_pass(
    audit: AuditRepository,
    candidates: Sequence[BaseStrategy],
    control: Sequence[StrategySignal],
    bars_15m: dict[str, pd.DataFrame],
    bars_1h: dict[str, pd.DataFrame],
    now: datetime,
    bar_start: datetime,
) -> dict[str, int]:
    """Journals the signals of the control (already found by the observer) and of each candidate for
    the bar that just closed, then marks the pass. One candidate that breaks does not stop the others:
    its error is logged and counted. Returns the number of
    signals per strategy, plus `candidate_errors`."""
    counts: dict[str, int] = {}
    errors = 0
    for signal in control:
        journal_signal(audit, CONTROL, signal, now)
    counts[CONTROL] = len(control)
    for strategy in candidates:
        try:
            signals = filter_signals(strategy.generate_signals(bars_15m, bars_1h, now), set())
            for signal in signals:
                journal_signal(audit, strategy.name, signal, now)
            counts[strategy.name] = len(signals)
        except Exception as e:  # one broken candidate must not stop the others or the Oracle
            logger.error(f"Candidate {strategy.name} could not be run on this bar: {e}")
            errors += 1
            counts[strategy.name] = -1
    audit.record(
        CYCLE_EVENT,
        {"bar_start": bar_start.isoformat(), "errors": errors, **counts},
        when=now,
    )
    metrics.candidate_pass(counts, errors)  # for Grafana; never raises
    counts["candidate_errors"] = errors
    return counts


# ------------------------------------------------------------------------------ reading


def journalled_signals(audit: AuditRepository, now: datetime) -> dict[str, list[SignalToPlay]]:
    """Per strategy, the signals journalled, each (symbol, closed bar) once (a bar that was passed over
    twice must not count twice)."""
    out: dict[str, dict[tuple[str, pd.Timestamp], SignalToPlay]] = {}
    for _, details in audit.events(CANDIDATE_EVENT, EPOCH, now + timedelta(days=1)):
        strategy = str(details["strategy"])
        closed = _utc(details["closed_bar"])
        key = (str(details["symbol"]), closed)
        out.setdefault(strategy, {}).setdefault(
            key, SignalToPlay(str(details["signal_id"]), key[0], closed, strategy)
        )
    return {strategy: list(found.values()) for strategy, found in out.items()}


def run_days(
    audit: AuditRepository, now: datetime, candidate: Optional[str] = None
) -> tuple[int, int]:
    """(trading days on which the candidates were run, passes in which a candidate broke). With a
    `candidate`, only the passes that ran it count: the cycle event carries the name of every
    strategy it ran, so a candidate added later counts its days from its own first pass."""
    days: set[Any] = set()
    broken = 0
    for when, details in audit.events(CYCLE_EVENT, EPOCH, now + timedelta(days=1)):
        if candidate is not None and candidate not in details:
            continue
        days.add(when.astimezone(MARKET_TZ).date())
        broken += int(details.get("errors", 0) > 0)
    return len(days), broken


# ------------------------------------------------------------------------------ judging


@dataclass(frozen=True)
class Detail:
    mean: float  # per trade, after costs, as a fraction
    t: float  # against zero
    control_mean: float
    control_trades: int
    difference: float  # candidate minus control
    difference_t: float
    halves: tuple[float, float]  # the candidate's mean in the first and second half, in time
    half_differences: tuple[float, float]  # and its difference to the control in each
    daily_t: float
    days_with_trades: int
    open_trades: int  # valued at the last close, counted


@dataclass(frozen=True)
class CandidateResult:
    name: str
    signals: int  # mature, once each
    trades: int  # mature, after the one-open-trade-per-symbol rule
    looks_before: int
    verdict: str
    reason: str
    detail: Optional[Detail] = None


@dataclass(frozen=True)
class Judgement:
    run_days: int
    broken_passes: int
    control_trades: int
    results: tuple[CandidateResult, ...]


Trade = tuple[pd.Timestamp, float, bool]  # (entry time, net return, still open)


def _trades(
    signals: list[SignalToPlay], bars: MarketBarRepository, settings: Any
) -> tuple[list[Trade], int]:
    played = play_out(signals, bars, settings, one_at_a_time=True)
    trades = [
        (_utc(t.entry_time), t.pnl_pct, t.exit_reason == "OPEN")
        for t in sorted(played.trades, key=lambda t: t.entry_time)
    ]
    return trades, len(signals)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _detail(trades: list[Trade], control: list[Trade]) -> Detail:
    returns = [r for _, r, _ in trades]
    control_returns = [r for _, r, _ in control]
    boundary = trades[len(trades) // 2][0]
    first = [r for when, r, _ in trades if when < boundary]
    second = [r for when, r, _ in trades if when >= boundary]
    control_first = [r for when, r, _ in control if when < boundary]
    control_second = [r for when, r, _ in control if when >= boundary]
    by_day: dict[Any, list[float]] = {}
    for when, r, _ in trades:
        by_day.setdefault(when.tz_convert(MARKET_TZ).date(), []).append(r)
    daily = [float(np.mean(day)) for day in by_day.values()]
    return Detail(
        mean=_mean(returns),
        t=t_statistic(returns),
        control_mean=_mean(control_returns),
        control_trades=len(control),
        difference=_mean(returns) - _mean(control_returns),
        difference_t=welch_t(returns, control_returns),
        halves=(_mean(first), _mean(second)),
        half_differences=(
            _mean(first) - _mean(control_first),
            _mean(second) - _mean(control_second),
        ),
        daily_t=t_statistic(daily),
        days_with_trades=len(daily),
        open_trades=sum(1 for _, _, still_open in trades if still_open),
    )


def _verdict(detail: Detail, bar: float) -> tuple[str, str]:
    failed: list[str] = []
    if not (detail.mean > 0 and detail.t >= bar):
        failed.append(
            f"mean net trade {detail.mean * 100:+.2f}% with t {detail.t:.1f} (bar {bar:g})"
        )
    if not (detail.difference > 0 and detail.difference_t >= DIFFERENCE_T):
        failed.append(
            f"difference to the control {detail.difference * 100:+.2f}% with Welch t "
            f"{detail.difference_t:.1f} (bar {DIFFERENCE_T:g})"
        )
    if not (all(h > 0 for h in detail.halves) and all(d > 0 for d in detail.half_differences)):
        failed.append("not the same sign in both halves of the sample")
    if not detail.daily_t >= DAILY_T:
        failed.append(f"t of the daily means {detail.daily_t:.1f} (bar {DAILY_T:g})")
    if failed:
        return "not shown to help", "; ".join(failed)
    return (
        "helps",
        "all four conditions hold (a reason to look further, not to switch the live loop)",
    )


def mature_signals(audit: AuditRepository, now: datetime) -> dict[str, list[SignalToPlay]]:
    """The journalled signals old enough to count (their entry is MATURITY old), per strategy."""
    cutoff = now - MATURITY
    return {
        strategy: [s for s in signals if (s.closed_bar + BAR).to_pydatetime() <= cutoff]
        for strategy, signals in journalled_signals(audit, now).items()
    }


def judge_candidates(
    audit: AuditRepository, bars: MarketBarRepository, settings: Any, now: datetime
) -> Judgement:
    """The judgement of each candidate, or only the counts when it is not time. A look is journalled
    when taken."""
    days, broken = run_days(audit, now)
    mature = mature_signals(audit, now)
    control_trades: list[Trade] = []
    results: list[CandidateResult] = []
    played: dict[str, tuple[list[Trade], int]] = {}

    def play(name: str) -> tuple[list[Trade], int]:
        if name not in played:
            played[name] = _trades(mature.get(name, []), bars, settings)
        return played[name]

    for name in RECORDED:
        trades, signals = play(name)
        own_days = days if name in CANDIDATES else run_days(audit, now, name)[0]
        first_t, second_t = LOOK_T[name]
        taken = [
            d
            for _, d in audit.events(JUDGEMENT_EVENT, EPOCH, now + timedelta(days=1))
            if d.get("candidate") == name
        ]
        looks = len(taken)

        def said(verdict: str, reason: str, detail: Optional[Detail] = None) -> CandidateResult:
            return CandidateResult(name, signals, len(trades), looks, verdict, reason, detail)

        if looks >= MAX_LOOKS:
            results.append(
                said("no more looks", f"{looks} looks were taken; the rules allow {MAX_LOOKS}")
            )
            continue
        # The second look wants twice the trades the FIRST look had (at least twice the minimum): at the
        # real signal rate the first look already has several hundred, and without this the second look
        # could be taken at once on the same sample.
        need = MIN_TRADES if looks == 0 else 2 * max(MIN_TRADES, *(int(d["trades"]) for d in taken))
        if len(trades) < need or own_days < MIN_RUN_DAYS:
            results.append(
                said(
                    "no verdict",
                    f"not enough yet: {len(trades)} mature trades (need {need}) over {own_days} "
                    f"trading days (need {MIN_RUN_DAYS}). Nothing is computed below the minimum.",
                )
            )
            continue
        control, _ = play(CONTROL)
        detail = _detail(trades, control)
        verdict, reason = _verdict(detail, first_t if looks == 0 else second_t)
        audit.record(
            JUDGEMENT_EVENT,
            {
                "candidate": name,
                "look": looks + 1,
                "verdict": verdict,
                "trades": len(trades),
                "mean": detail.mean,
                "t": detail.t,
                "difference": detail.difference,
                "difference_t": detail.difference_t,
                "daily_t": detail.daily_t,
            },
            when=now,
        )
        results.append(said(verdict, reason, detail))
    control_trades, _ = play(CONTROL)
    return Judgement(days, broken, len(control_trades), tuple(results))


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.2f}%"


def format_candidates(j: Judgement) -> str:
    lines = [
        "The candidates run forward in time in the observer (no orders, no Oracle question); "
        "the control is the live trend_cross over the same symbols and days:",
        f"  trading days run: {j.run_days} (need {MIN_RUN_DAYS}); passes in which a candidate broke: "
        f"{j.broken_passes}; control: {j.control_trades} mature trades",
    ]
    for r in j.results:
        lines += [
            f"  {r.name}: {r.signals} mature signals, {r.trades} trades; looks taken "
            f"{r.looks_before} of {MAX_LOOKS}",
            f"    verdict: {r.verdict}",
            f"    {r.reason}",
        ]
        d = r.detail
        if d is not None:
            lines += [
                f"    mean net trade {_pct(d.mean)} (t {d.t:.1f}); control {_pct(d.control_mean)} "
                f"({d.control_trades} trades); difference {_pct(d.difference)} (Welch t "
                f"{d.difference_t:.1f})",
                f"    first half {_pct(d.halves[0])} (vs control {_pct(d.half_differences[0])}), second "
                f"half {_pct(d.halves[1])} (vs control {_pct(d.half_differences[1])}); t of the "
                f"daily means {d.daily_t:.1f} over {d.days_with_trades} days; {d.open_trades} "
                "trades still open, valued at the last close",
                "    This was a LOOK: it is written to the journal and cannot be repeated for the "
                "same sample.",
            ]
    return "\n".join(lines)
