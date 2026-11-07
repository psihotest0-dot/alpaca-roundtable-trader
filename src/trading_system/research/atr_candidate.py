"""The ATR exit candidate, judged forward in time (docs/strategy-research-plan.md, section 13.1,
registered on 2026-11-07 before any code and before any result).

**What is compared.** The recorded signals of the control (the live `trend_cross` over the observed
symbols, journalled by the observer, section 12.1) are played out twice: with the live exit (4 % / 5 %,
one open trade per symbol, the trades of 12.1) and with the candidate exit (stop `ATR_EXIT.stop_atr` ATR,
target `ATR_EXIT.target_atr` ATR, fixed on 2026-11-07 and never changed). The comparison is PAIRED: the
same signals with the two exits; the candidate does not add the signals that closing earlier would have
let in. Same play-out code as `judge-oracle` and `judge-candidates` (`backtest/shadow.py: play_out`); a
trade no level has reached is valued at the last close and counted.

**The rules are in the code, because the discipline is the point.** Counts only below MIN_TRADES mature
paired trades or MIN_RUN_DAYS days run; at most MAX_LOOKS looks (the second wants twice the trades of the
first, and a higher bar); every look is journalled. "helps" needs ALL FOUR of: (1) the candidate's mean
net trade above zero with t against zero of at least the bar, (2) the paired difference (candidate minus
live exit) above zero with a paired t of at least PAIRED_T, (3) the same sign in both halves of the sample,
in time, for both the mean and the difference, (4) a t of the per-day means of at least DAILY_T. Spread,
time held, the share still open and the worst trade of both exits are printed but are NOT criteria: if
the candidate does not earn more but evens out the risk, that is said and is not "helps".
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd

from trading_system.backtest.analysis import t_statistic
from trading_system.backtest.shadow import SignalToPlay, play_out
from trading_system.research.candidates import (
    CONTROL,
    DAILY_T,
    EPOCH,
    JUDGEMENT_EVENT,
    MARKET_TZ,
    MAX_LOOKS,
    MIN_RUN_DAYS,
    MIN_TRADES,
    mature_signals,
    run_days,
)
from trading_system.risk.entry_plan import AtrExit
from trading_system.storage.repositories import AuditRepository, MarketBarRepository

ATR_CANDIDATE = "trend_cross_atr_exit"
ATR_EXIT = AtrExit(stop_atr=Decimal("6"), target_atr=Decimal("7.5"))  # registered; never changed

FIRST_LOOK_T = 2.4  # three candidates are looked at in all (12.1 has two, at 2.2)
SECOND_LOOK_T = 2.7
PAIRED_T = 1.0


@dataclass(frozen=True)
class Pair:
    """One signal played out with both exits."""

    entry: pd.Timestamp
    fixed: float  # net return of the trade with the live exit, as a fraction
    atr: float
    fixed_open: bool
    atr_open: bool
    fixed_hours: float
    atr_hours: float


@dataclass(frozen=True)
class Detail:
    mean: float  # the candidate's mean net trade
    t: float  # against zero
    fixed_mean: float
    difference: float  # candidate minus live exit, per signal
    paired_t: float
    halves: tuple[float, float]  # the candidate's mean in the first and the second half, in time
    half_differences: tuple[float, float]
    daily_t: float
    days_with_trades: int
    # printed, not criteria
    spread: tuple[float, float]  # standard deviation of a trade: live exit, candidate
    median_hours: tuple[float, float]
    still_open: tuple[float, float]  # share of the trades no level had reached
    worst: tuple[float, float]


@dataclass(frozen=True)
class AtrJudgement:
    run_days: int
    signals: int  # mature control signals
    pairs: int  # trades with both exits
    no_atr: int  # signals the candidate could not play (no ATR)
    looks_before: int
    verdict: str
    reason: str
    detail: Optional[Detail] = None


def _pairs(
    signals: list[SignalToPlay], bars: MarketBarRepository, settings: Any
) -> tuple[list[Pair], int]:
    base = play_out(signals, bars, settings, one_at_a_time=True)
    taken = {t.signal_id for t in base.trades}
    candidate = play_out(
        [s for s in signals if s.signal_id in taken], bars, settings, exit_rule=ATR_EXIT
    )
    by_id = {t.signal_id: t for t in candidate.trades}
    pairs: list[Pair] = []
    for fixed in sorted(base.trades, key=lambda t: t.entry_time):
        atr = by_id.get(fixed.signal_id)
        if atr is None:
            continue
        pairs.append(
            Pair(
                entry=pd.Timestamp(fixed.entry_time),
                fixed=fixed.pnl_pct,
                atr=atr.pnl_pct,
                fixed_open=fixed.exit_reason == "OPEN",
                atr_open=atr.exit_reason == "OPEN",
                fixed_hours=(fixed.exit_time - fixed.entry_time).total_seconds() / 3600,
                atr_hours=(atr.exit_time - atr.entry_time).total_seconds() / 3600,
            )
        )
    return pairs, candidate.no_atr


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _detail(pairs: list[Pair]) -> Detail:
    atr = [p.atr for p in pairs]
    fixed = [p.fixed for p in pairs]
    diff = [p.atr - p.fixed for p in pairs]
    boundary = pairs[len(pairs) // 2].entry
    first = [p for p in pairs if p.entry < boundary]
    second = [p for p in pairs if p.entry >= boundary]
    by_day: dict[Any, list[float]] = {}
    for p in pairs:
        by_day.setdefault(p.entry.tz_convert(MARKET_TZ).date(), []).append(p.atr)
    daily = [float(np.mean(day)) for day in by_day.values()]
    return Detail(
        mean=_mean(atr),
        t=t_statistic(atr),
        fixed_mean=_mean(fixed),
        difference=_mean(diff),
        paired_t=t_statistic(diff),
        halves=(_mean([p.atr for p in first]), _mean([p.atr for p in second])),
        half_differences=(
            _mean([p.atr - p.fixed for p in first]),
            _mean([p.atr - p.fixed for p in second]),
        ),
        daily_t=t_statistic(daily),
        days_with_trades=len(daily),
        spread=(statistics.pstdev(fixed), statistics.pstdev(atr)),
        median_hours=(
            float(np.median([p.fixed_hours for p in pairs])),
            float(np.median([p.atr_hours for p in pairs])),
        ),
        still_open=(
            sum(p.fixed_open for p in pairs) / len(pairs),
            sum(p.atr_open for p in pairs) / len(pairs),
        ),
        worst=(min(fixed), min(atr)),
    )


def _verdict(detail: Detail, bar: float) -> tuple[str, str]:
    failed: list[str] = []
    if not (detail.mean > 0 and detail.t >= bar):
        failed.append(
            f"mean net trade {detail.mean * 100:+.2f}% with t {detail.t:.1f} (bar {bar:g})"
        )
    if not (detail.difference > 0 and detail.paired_t >= PAIRED_T):
        failed.append(
            f"difference to the live exit {detail.difference * 100:+.2f}% with paired t "
            f"{detail.paired_t:.1f} (bar {PAIRED_T:g})"
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


def judge_atr_exit(
    audit: AuditRepository, bars: MarketBarRepository, settings: Any, now: datetime
) -> AtrJudgement:
    """The judgement of the ATR exit, or only the counts when it is not time. A look is journalled
    when taken."""
    days, _ = run_days(audit, now)
    signals = mature_signals(audit, now).get(CONTROL, [])
    pairs, no_atr = _pairs(signals, bars, settings)
    taken = [
        d
        for _, d in audit.events(JUDGEMENT_EVENT, EPOCH, now + timedelta(days=1))
        if d.get("candidate") == ATR_CANDIDATE
    ]
    looks = len(taken)

    def said(verdict: str, reason: str, detail: Optional[Detail] = None) -> AtrJudgement:
        return AtrJudgement(days, len(signals), len(pairs), no_atr, looks, verdict, reason, detail)

    if looks >= MAX_LOOKS:
        return said("no more looks", f"{looks} looks were taken; the rules allow {MAX_LOOKS}")
    need = MIN_TRADES if looks == 0 else 2 * max(MIN_TRADES, *(int(d["trades"]) for d in taken))
    if len(pairs) < need or days < MIN_RUN_DAYS:
        return said(
            "no verdict",
            f"not enough yet: {len(pairs)} mature paired trades (need {need}) over {days} trading "
            f"days (need {MIN_RUN_DAYS}). Nothing is computed below the minimum.",
        )
    detail = _detail(pairs)
    verdict, reason = _verdict(detail, FIRST_LOOK_T if looks == 0 else SECOND_LOOK_T)
    audit.record(
        JUDGEMENT_EVENT,
        {
            "candidate": ATR_CANDIDATE,
            "look": looks + 1,
            "verdict": verdict,
            "trades": len(pairs),
            "mean": detail.mean,
            "t": detail.t,
            "difference": detail.difference,
            "paired_t": detail.paired_t,
            "daily_t": detail.daily_t,
        },
        when=now,
    )
    return said(verdict, reason, detail)


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.2f}%"


def format_atr_exit(j: AtrJudgement) -> str:
    lines = [
        f"The ATR exit candidate ({ATR_CANDIDATE}: stop {ATR_EXIT.stop_atr} ATR, target "
        f"{ATR_EXIT.target_atr} ATR), the recorded signals of the live trend_cross played out with "
        "both exits:",
        f"  {j.signals} mature control signals, {j.pairs} paired trades ({j.no_atr} without an ATR); "
        f"trading days run {j.run_days} (need {MIN_RUN_DAYS}); looks taken {j.looks_before} of "
        f"{MAX_LOOKS}",
        f"  verdict: {j.verdict}",
        f"  {j.reason}",
    ]
    d = j.detail
    if d is not None:
        lines += [
            f"  candidate {_pct(d.mean)} per trade (t {d.t:.1f}); live exit {_pct(d.fixed_mean)}; "
            f"paired difference {_pct(d.difference)} (paired t {d.paired_t:.1f})",
            f"  first half {_pct(d.halves[0])} (difference {_pct(d.half_differences[0])}), second "
            f"half {_pct(d.halves[1])} (difference {_pct(d.half_differences[1])}); t of the daily "
            f"means {d.daily_t:.1f} over {d.days_with_trades} days",
            f"  not criteria: spread of a trade {d.spread[0] * 100:.2f}% live exit, "
            f"{d.spread[1] * 100:.2f}% candidate; median hours held {d.median_hours[0]:.0f} / "
            f"{d.median_hours[1]:.0f}; still open {d.still_open[0]:.0%} / {d.still_open[1]:.0%}; "
            f"worst trade {_pct(d.worst[0])} / {_pct(d.worst[1])}",
            "  This was a LOOK: it is written to the journal and cannot be repeated for the same "
            "sample.",
        ]
    return "\n".join(lines)
