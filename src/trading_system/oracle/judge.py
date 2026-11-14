"""Judging the Oracle on its own recorded answers (docs/oracle-provider-task.md, section 5).

The question: would the bot have been better off obeying the model's VETOs? Every signal that got a
final answer in shadow mode is played out as a shadow trade (the same entry plan, stop, target and
costs as the backtest uses, `backtest/shadow.py`), approved and vetoed signals by the SAME rule, and the
two groups are compared with Welch's t. The signal is the unit, the account plays no part, so the
signals of the observer (symbols outside the watchlist, never traded) judge the model as well as the
live loop's.

**The rules are in the code, because the discipline is the point.**
* Signals count from START only (the model has seen the past), probes (`oracle-probe`) never, and only
  final answers (APPROVE or VETO) that did not fail: an outage is not a veto.
* Below MIN_PER_GROUP approvals AND vetoes, or below MIN_AVAILABILITY of the asked signals answered,
  the judge prints COUNTS ONLY. It does not compute a return, so nobody can peek.
* One PERIOD at a time: a period is a model and the backend's `prompt_version` (docs section 5). An answer
  under another model or label is another period and is never pooled with it; answers from before the label
  existed form their own "unlabelled" period. The period judged is that of the newest answer; the others
  are listed with counts only.
* One look, taken when the minimum is reached; the second only at double the minimum and with a higher
  bar (t 2.5); then no more. Looks are counted per period (every new period is a new sample), and every
  look taken is written to the journal with its period: a new prompt after a bad result is a new try, and
  the journal shows how many periods were judged.
* A random relabelling of the same signals (the control) must stay under t 2, or the verdict is "control
  suspect" and the instrument is to blame.
* "helps" needs t >= the bar in favour of the approved, the same sign in both halves of the period (by
  time) and an approved mean above zero after costs; "hurts" is the mirror image; anything else is
  "cannot be told from noise".

A second question, B (docs section 5.1, registered 2026-11-14 before any result): would skipping
everything the model did NOT approve (a VETO, or a signal left on HOLD) have been better than taking
everything? It is the question of switching `active` mode on. Its groups are the approved and the not
approved; it needs 60 + 60 signals from 20 trading days before a single return is computed; it has its
own looks per period (journal field `question`) and a higher bar (2.4, then 2.7) and a second test on the
per-day differences, because the signals of one day move together. Question A is not changed.
"""

from dataclasses import dataclass
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from trading_system.backtest.analysis import welch_t
from trading_system.backtest.shadow import SignalToPlay, play_out
from trading_system.research.signals import CONTROL_SEED
from trading_system.storage.models import StrategySignalModel
from trading_system.storage.repositories import (
    AuditRepository,
    MarketBarRepository,
    OracleDecisionRepository,
)

# The moment the Oracle was switched on in shadow mode (2026-11-06, about 16:00 UTC). Answers before it
# are probes of the road, not part of the sample.
START = datetime(2026, 11, 6, 16, 0, tzinfo=timezone.utc)
PROBE_VERSION = "oracle-probe"
JUDGEMENT_EVENT = "oracle_judgement"

MIN_PER_GROUP = 30
MIN_AVAILABILITY = 0.80
PASS_T = 2.0
SECOND_LOOK_T = 2.5
CONTROL_MAX_T = 2.0
MAX_LOOKS = 2

# Question B (docs/oracle-provider-task.md, section 5.1).
QUESTION_A = "A"
QUESTION_B = "B"
B_MIN_PER_GROUP = 60
B_MIN_DAYS = 20  # different trading days, so that one strong day cannot decide
B_PASS_T = 2.4
B_SECOND_LOOK_T = 2.7
B_DAY_T = 1.5  # the t of the per-day differences
B_MIN_PAIRED_DAYS = 10  # days with a trade in both groups, for that t
NOT_APPROVED = "NOT_APPROVED"


Period = tuple[str, str]  # (model, prompt_version); "" where the answer carries none


def period_key(period: Optional[Period]) -> str:
    """The period as the journal writes it: "model|label", "unlabelled" for no label."""
    if period is None:
        return "none"
    return f"{period[0] or 'unknown model'}|{period[1] or 'unlabelled'}"


@dataclass(frozen=True)
class Counts:
    asked: int  # signals the model was put to (shadow mode, after START, not a probe)
    approve: int
    veto: int
    availability: (
        float  # the share of the asked signals that got a REPLY: (asked - no reply) / asked
    )
    period: Optional[Period] = None  # the period these counts are of
    held: int = 0  # the model's own HOLD: a reply, not a final one, and no outage
    no_reply: int = 0  # asked, no reply at all (the backend or the model failed)

    @property
    def answered(self) -> int:
        return self.approve + self.veto


@dataclass(frozen=True)
class Detail:
    approved: int
    vetoed: int
    approved_mean: float  # per trade, after costs, as a fraction
    vetoed_mean: float
    edge: float  # approved minus vetoed
    t: float
    first_half_edge: float
    second_half_edge: float
    control_t: float
    unresolved: int  # signals whose shadow trade has not reached its stop or target yet
    no_bars: int  # signals with no bar to enter on
    days: int = 0  # question B: different trading days among the resolved trades
    day_t: float = float("nan")  # question B: t of the per-day differences (approved minus not)
    paired_days: int = 0  # question B: days with a trade in both groups


@dataclass(frozen=True)
class Judgement:
    counts: Counts
    looks_before: int  # looks taken in THIS period
    verdict: str
    reason: str
    detail: Optional[Detail] = None
    other_periods: tuple[Counts, ...] = ()  # counts only, never judged here
    question: str = QUESTION_A


def collect(
    session_factory: Any, oracle_repo: OracleDecisionRepository
) -> dict[str, dict[str, Any]]:
    """Per signal id: the model's latest word and the signal itself, for the sample only."""
    export = oracle_repo.replay_export()
    sample: dict[str, dict[str, Any]] = {}
    for signal_id, entry in export.items():
        if entry.get("mode") != "shadow":
            continue
        if datetime.fromisoformat(entry["decided_at"]) < START:
            continue
        sample[signal_id] = dict(entry)
    if not sample:
        return {}
    with session_factory() as session:
        rows = (
            session.query(StrategySignalModel)
            .filter(StrategySignalModel.signal_id.in_(list(sample)))
            .all()
        )
        for row in rows:
            if str(row.version) == PROBE_VERSION:
                sample.pop(str(row.signal_id), None)
                continue
            closed = row.closed_bar_timestamp
            sample[str(row.signal_id)].update(
                symbol=str(row.symbol),
                closed_bar=pd.Timestamp(closed).tz_localize("UTC")
                if getattr(closed, "tzinfo", None) is None
                else pd.Timestamp(closed).tz_convert("UTC"),
            )
    return {k: v for k, v in sample.items() if "symbol" in v}


def is_answered(entry: dict[str, Any]) -> bool:
    return bool(entry["final"] and not entry["failed"])


def period_of(entry: dict[str, Any]) -> Period:
    return (str(entry.get("model") or ""), str(entry.get("prompt_version") or ""))


def assign_periods(
    sample: dict[str, dict[str, Any]],
) -> tuple[dict[str, Period], Optional[Period], list[Period]]:
    """(the period of each signal, the current period, the periods oldest first).

    An answered signal belongs to the period of its answer. One that got no answer (an outage, a HOLD)
    has no label of its own: it belongs to the period that was running when it was asked, that is the
    latest period whose first answer is not later. The current period is that of the newest answer."""
    answered = {k: e for k, e in sample.items() if is_answered(e)}
    if not answered:
        return {}, None, []
    starts: dict[Period, datetime] = {}
    for entry in answered.values():
        when = datetime.fromisoformat(entry["decided_at"])
        period = period_of(entry)
        starts[period] = min(starts.get(period, when), when)
    ordered = sorted(starts, key=lambda p: starts[p])
    current = period_of(max(answered.values(), key=lambda e: e["decided_at"]))
    mapping: dict[str, Period] = {}
    for signal_id, entry in sample.items():
        if is_answered(entry):
            mapping[signal_id] = period_of(entry)
            continue
        when = datetime.fromisoformat(entry["decided_at"])
        running = [p for p in ordered if starts[p] <= when]
        mapping[signal_id] = running[-1] if running else ordered[0]
    return mapping, current, ordered


def count(sample: dict[str, dict[str, Any]], period: Optional[Period] = None) -> Counts:
    answered = [e for e in sample.values() if is_answered(e)]
    approve = sum(1 for e in answered if e["verdict"] == "APPROVE")
    veto = sum(1 for e in answered if e["verdict"] == "VETO")
    asked = len(sample)
    no_reply = sum(1 for e in sample.values() if e["failed"])
    held = sum(1 for e in sample.values() if not e["failed"] and not e["final"])
    # Availability is about the BACKEND and the model being there. A HOLD is the model answering "not
    # clear" (it is no final answer and stays out of both groups), so it is not an outage; counting it
    # as one would make the judge refuse a verdict whenever the model is cautious.
    availability = (asked - no_reply) / asked if asked else 0.0
    return Counts(asked, approve, veto, availability, period, held, no_reply)


def _shadow_pnl(
    sample: dict[str, dict[str, Any]], bars: MarketBarRepository, settings: Any
) -> tuple[list[tuple[str, float, pd.Timestamp]], int, int]:
    """(verdict, pnl, entry time) for each answered signal that has resolved; and how many have not,
    and how many had no bar to enter on."""
    answered = {k: e for k, e in sample.items() if e["final"] and not e["failed"]}
    played = play_out(
        [
            SignalToPlay(signal_id, entry["symbol"], entry["closed_bar"], entry["verdict"])
            for signal_id, entry in answered.items()
        ],
        bars,
        settings,
    )
    out: list[tuple[str, float, pd.Timestamp]] = []
    unresolved = 0
    for shadow in played.trades:
        if shadow.exit_reason == "OPEN":  # this judge counts only trades that reached a level
            unresolved += 1
            continue
        out.append((shadow.verdict, shadow.pnl_pct, pd.Timestamp(shadow.entry_time)))
    return out, unresolved, played.no_bars


def _edge(
    trades: list[tuple[str, float, pd.Timestamp]],
    positive: str = "APPROVE",
    negative: str = "VETO",
) -> float:
    a = [p for v, p, _ in trades if v == positive]
    b = [p for v, p, _ in trades if v == negative]
    return float(np.mean(a) - np.mean(b)) if a and b else float("nan")


def _looks_taken(audit: AuditRepository, key: str, question: str, now: datetime) -> int:
    """Looks already taken in this period for this question. A journal row without the field is
    from before question B existed: it is a look at A."""
    return sum(
        1
        for _, details in audit.events(
            JUDGEMENT_EVENT, datetime(2000, 1, 1, tzinfo=timezone.utc), now + timedelta(days=1)
        )
        if details.get("period") == key and details.get("question", QUESTION_A) == question
    )


def judge(
    session_factory: Any,
    oracle_repo: OracleDecisionRepository,
    bars: MarketBarRepository,
    audit: AuditRepository,
    settings: Any,
    now: datetime,
) -> Judgement:
    """The judgement of the current period, or only the counts when it is not time. A look is
    journalled (with its period) when taken."""
    everything = collect(session_factory, oracle_repo)
    mapping, current, ordered = assign_periods(everything)
    if current is None:
        sample = everything
    else:
        sample = {k: v for k, v in everything.items() if mapping[k] == current}
    counts = count(sample, current)
    others = tuple(
        count({k: v for k, v in everything.items() if mapping[k] == p}, p)
        for p in ordered
        if p != current
    )
    key = period_key(current)
    looks = _looks_taken(audit, key, QUESTION_A, now)

    def said(verdict: str, reason: str, detail: Optional[Detail] = None) -> Judgement:
        return Judgement(counts, looks, verdict, reason, detail, others)

    if looks >= MAX_LOOKS:
        return said(
            "no more looks",
            f"{looks} looks were taken in this period; the rules allow {MAX_LOOKS}",
        )
    need = MIN_PER_GROUP * (looks + 1)
    bar = PASS_T if looks == 0 else SECOND_LOOK_T
    if counts.approve < need or counts.veto < need:
        return said(
            "no verdict",
            f"not enough answers yet in this period: {counts.approve} approvals and {counts.veto} "
            f"vetoes, need {need} of each",
        )
    if counts.availability < MIN_AVAILABILITY:
        return said(
            "no verdict",
            f"availability {counts.availability:.0%} is under {MIN_AVAILABILITY:.0%}: {counts.no_reply} of "
            f"{counts.asked} signals got no reply, so the backend or the model fails too often to judge",
        )

    trades, unresolved, no_bars = _shadow_pnl(sample, bars, settings)
    approved = [p for v, p, _ in trades if v == "APPROVE"]
    vetoed = [p for v, p, _ in trades if v == "VETO"]
    if len(approved) < need or len(vetoed) < need:
        return said(
            "no verdict",
            f"not enough resolved shadow trades yet: {len(approved)} approved and {len(vetoed)} "
            f"vetoed ({unresolved} still open, {no_bars} without a bar); need {need} of each. "
            "No look was used.",
        )

    ordered_trades = sorted(trades, key=lambda x: x[2])
    middle = len(ordered_trades) // 2
    first_edge, second_edge = _edge(ordered_trades[:middle]), _edge(ordered_trades[middle:])
    t = welch_t(approved, vetoed)
    rng = np.random.default_rng(CONTROL_SEED)
    labels = np.array([v for v, _, _ in trades])
    shuffled = rng.permutation(labels)
    pnl = np.array([p for _, p, _ in trades])
    control_t = welch_t(list(pnl[shuffled == "APPROVE"]), list(pnl[shuffled == "VETO"]))
    approved_mean, vetoed_mean = float(np.mean(approved)), float(np.mean(vetoed))
    edge = approved_mean - vetoed_mean
    same_sign = np.sign(first_edge) == np.sign(second_edge) == np.sign(edge) and edge != 0
    detail = Detail(
        len(approved),
        len(vetoed),
        approved_mean,
        vetoed_mean,
        edge,
        t,
        first_edge,
        second_edge,
        control_t,
        unresolved,
        no_bars,
    )

    if abs(control_t) >= CONTROL_MAX_T:
        verdict, reason = (
            "control suspect",
            f"a random relabelling gave t {control_t:.1f}: the instrument is suspect",
        )
    elif t >= bar and same_sign and approved_mean > 0:
        verdict, reason = (
            "helps",
            "the vetoes remove worse trades (t {:.1f}, same sign in both halves)".format(t),
        )
    elif t <= -bar and same_sign:
        verdict, reason = (
            "hurts",
            "the vetoes remove better trades (t {:.1f}, same sign in both halves)".format(t),
        )
    else:
        verdict, reason = "cannot be told from noise", f"t {t:.1f} against a bar of {bar:g}"

    audit.record(
        JUDGEMENT_EVENT,
        {
            "question": QUESTION_A,
            "look": looks + 1,
            "period": key,
            "periods_so_far": len(ordered),
            "verdict": verdict,
            "approve": counts.approve,
            "veto": counts.veto,
            "t": t,
            "edge": edge,
            "control_t": control_t,
        },
        when=now,
    )
    return said(verdict, reason, detail)


# ------------------------------------------------------------------ question B (docs section 5.1)


def _new_york_day(moment: Any) -> date:
    stamp = pd.Timestamp(moment)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return cast(date, stamp.tz_convert("America/New_York").date())


def _not_approved_trades(
    sample: dict[str, dict[str, Any]], bars: MarketBarRepository, settings: Any
) -> tuple[list[tuple[str, float, pd.Timestamp]], int, int]:
    """(label, pnl, entry time) of every resolved shadow trade of a signal the model replied to:
    "APPROVE" for a final approval, NOT_APPROVED for a VETO or a signal left on HOLD. Also how many
    have not reached a level and how many had no bar to enter on."""
    replied = {k: e for k, e in sample.items() if not e["failed"]}
    played = play_out(
        [
            SignalToPlay(
                signal_id,
                entry["symbol"],
                entry["closed_bar"],
                "APPROVE" if entry["final"] and entry["verdict"] == "APPROVE" else NOT_APPROVED,
            )
            for signal_id, entry in replied.items()
        ],
        bars,
        settings,
    )
    out: list[tuple[str, float, pd.Timestamp]] = []
    unresolved = 0
    for shadow in played.trades:
        if shadow.exit_reason == "OPEN":
            unresolved += 1
            continue
        out.append((shadow.verdict, shadow.pnl_pct, pd.Timestamp(shadow.entry_time)))
    return out, unresolved, played.no_bars


def _daily_t(trades: list[tuple[str, float, pd.Timestamp]]) -> tuple[float, int]:
    """The t of the per-day differences (the approved's mean minus the not approved's mean, on each
    day that has a trade in both groups), and how many such days there are. nan below the minimum:
    the signals of one day move together, so this is the check that one strong day is not the result."""
    by_day: dict[date, dict[str, list[float]]] = {}
    for label, pnl, entry in trades:
        by_day.setdefault(_new_york_day(entry), {}).setdefault(label, []).append(pnl)
    differences = [
        float(np.mean(groups["APPROVE"]) - np.mean(groups[NOT_APPROVED]))
        for groups in by_day.values()
        if groups.get("APPROVE") and groups.get(NOT_APPROVED)
    ]
    n = len(differences)
    if n < B_MIN_PAIRED_DAYS:
        return float("nan"), n
    mean = float(np.mean(differences))
    spread = float(np.std(differences, ddof=1))
    if spread == 0:
        return (0.0 if mean == 0 else math.copysign(math.inf, mean)), n
    return mean / (spread / math.sqrt(n)), n


def judge_not_approved(
    session_factory: Any,
    oracle_repo: OracleDecisionRepository,
    bars: MarketBarRepository,
    audit: AuditRepository,
    settings: Any,
    now: datetime,
) -> Judgement:
    """Question B for the current period: would skipping every signal the model did not approve
    (a VETO, or a HOLD it never lifted) have been better? Counts only until the minimum of signals
    AND of trading days is there; one look, a second at double the minimum with a higher bar; a look
    is journalled with its period and `question: B`."""
    everything = collect(session_factory, oracle_repo)
    mapping, current, ordered = assign_periods(everything)
    if current is None:
        sample = everything
    else:
        sample = {k: v for k, v in everything.items() if mapping[k] == current}
    counts = count(sample, current)
    others = tuple(
        count({k: v for k, v in everything.items() if mapping[k] == p}, p)
        for p in ordered
        if p != current
    )
    key = period_key(current)
    looks = _looks_taken(audit, key, QUESTION_B, now)

    def said(verdict: str, reason: str, detail: Optional[Detail] = None) -> Judgement:
        return Judgement(counts, looks, verdict, reason, detail, others, QUESTION_B)

    if looks >= MAX_LOOKS:
        return said(
            "no more looks",
            f"{looks} looks were taken in this period; the rules allow {MAX_LOOKS}",
        )
    need = B_MIN_PER_GROUP * (looks + 1)
    days_need = B_MIN_DAYS * (looks + 1)
    bar = B_PASS_T if looks == 0 else B_SECOND_LOOK_T
    not_approved = counts.veto + counts.held
    if counts.approve < need or not_approved < need:
        return said(
            "no verdict",
            f"not enough answers yet in this period: {counts.approve} approved and "
            f"{not_approved} not approved ({counts.veto} vetoes, {counts.held} holds), need "
            f"{need} of each",
        )
    days = len({_new_york_day(e["decided_at"]) for e in sample.values() if not e["failed"]})
    if days < days_need:
        return said(
            "no verdict",
            f"only {days} trading days of answers in this period, need {days_need}: the signals "
            "of one day move together",
        )
    if counts.availability < MIN_AVAILABILITY:
        return said(
            "no verdict",
            f"availability {counts.availability:.0%} is under {MIN_AVAILABILITY:.0%}: "
            f"{counts.no_reply} of {counts.asked} signals got no reply, so the backend or the "
            "model fails too often to judge",
        )

    trades, unresolved, no_bars = _not_approved_trades(sample, bars, settings)
    approved = [p for label, p, _ in trades if label == "APPROVE"]
    rest = [p for label, p, _ in trades if label == NOT_APPROVED]
    resolved_days = len({_new_york_day(entry) for _, _, entry in trades})
    if len(approved) < need or len(rest) < need or resolved_days < days_need:
        return said(
            "no verdict",
            f"not enough resolved shadow trades yet: {len(approved)} approved and {len(rest)} not "
            f"approved on {resolved_days} trading days ({unresolved} still open, {no_bars} without "
            f"a bar); need {need} of each on {days_need} days. No look was used.",
        )

    ordered_trades = sorted(trades, key=lambda x: x[2])
    middle = len(ordered_trades) // 2
    first_edge = _edge(ordered_trades[:middle], "APPROVE", NOT_APPROVED)
    second_edge = _edge(ordered_trades[middle:], "APPROVE", NOT_APPROVED)
    t = welch_t(approved, rest)
    rng = np.random.default_rng(CONTROL_SEED)
    labels = np.array([label for label, _, _ in trades])
    shuffled = rng.permutation(labels)
    pnl = np.array([p for _, p, _ in trades])
    control_t = welch_t(list(pnl[shuffled == "APPROVE"]), list(pnl[shuffled == NOT_APPROVED]))
    approved_mean, rest_mean = float(np.mean(approved)), float(np.mean(rest))
    edge = approved_mean - rest_mean
    day_t, paired_days = _daily_t(trades)
    same_sign = np.sign(first_edge) == np.sign(second_edge) == np.sign(edge) and edge != 0
    detail = Detail(
        len(approved),
        len(rest),
        approved_mean,
        rest_mean,
        edge,
        t,
        first_edge,
        second_edge,
        control_t,
        unresolved,
        no_bars,
        resolved_days,
        day_t,
        paired_days,
    )

    if abs(control_t) >= CONTROL_MAX_T:
        verdict, reason = (
            "control suspect",
            f"a random relabelling gave t {control_t:.1f}: the instrument is suspect",
        )
    elif t >= bar and same_sign and approved_mean > 0 and day_t >= B_DAY_T:
        verdict, reason = (
            "helps",
            f"skipping what the model did not approve would have been better (t {t:.1f}, same "
            f"sign in both halves, per-day t {day_t:.1f} on {paired_days} days)",
        )
    elif t <= -bar and same_sign and day_t <= -B_DAY_T:
        verdict, reason = (
            "hurts",
            f"skipping what the model did not approve would have been worse (t {t:.1f}, same "
            f"sign in both halves, per-day t {day_t:.1f} on {paired_days} days)",
        )
    else:
        day_text = (
            f"per-day t {day_t:.1f} on {paired_days} days"
            if np.isfinite(day_t)
            else (f"too few days with both groups ({paired_days}) for the per-day t")
        )
        verdict, reason = (
            "cannot be told from noise",
            f"t {t:.1f} against a bar of {bar:g}; {day_text}",
        )

    audit.record(
        JUDGEMENT_EVENT,
        {
            "question": QUESTION_B,
            "look": looks + 1,
            "period": key,
            "periods_so_far": len(ordered),
            "verdict": verdict,
            "approved": len(approved),
            "not_approved": len(rest),
            "t": t,
            "day_t": day_t if np.isfinite(day_t) else None,
            "edge": edge,
            "control_t": control_t,
        },
        when=now,
    )
    return said(verdict, reason, detail)


def format_not_approved(j: Judgement) -> str:
    c = j.counts
    lines = [
        "Question B: would skipping every signal the model did not approve (a veto, or a hold it "
        "never lifted) have been better than taking everything? (docs section 5.1)",
        f"  not approved: {c.veto + c.held} ({c.veto} vetoes, {c.held} holds), approved: {c.approve}; "
        f"looks taken in this period: {j.looks_before} of {MAX_LOOKS}",
        f"  verdict: {j.verdict}",
        f"  {j.reason}",
    ]
    d = j.detail
    if d is not None:
        lines += [
            f"  shadow trades: {d.approved} approved, {d.vetoed} not approved, on {d.days} trading "
            f"days ({d.unresolved} still open, {d.no_bars} without a bar)",
            f"  approved {_pct(d.approved_mean)} per trade, not approved {_pct(d.vetoed_mean)}; "
            f"difference {_pct(d.edge)}, Welch t {d.t:.1f}",
            f"  first half {_pct(d.first_half_edge)}, second half {_pct(d.second_half_edge)}; "
            f"per-day t {d.day_t:.1f} on {d.paired_days} days; random-relabelling control t "
            f"{d.control_t:.1f}",
            "  This was a LOOK: it is written to the journal and cannot be repeated for the same "
            "sample.",
        ]
    return "\n".join(lines)


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.2f}%"


def format_judgement(j: Judgement) -> str:
    c = j.counts
    lines = [
        f"The Oracle's answers since {START:%Y-%m-%d %H:%M} UTC (shadow mode, probes left out):",
        "  period judged (model | prompt label): " + period_key(c.period),
        f"  asked about {c.asked} signals; final answers {c.answered} "
        f"({c.approve} approve, {c.veto} veto), the model held on {c.held}, no reply to "
        f"{c.no_reply}; availability {c.availability:.0%} (replies / asked)",
        f"  looks taken in this period: {j.looks_before} of {MAX_LOOKS}",
        f"  verdict: {j.verdict}",
        f"  {j.reason}",
    ]
    for other in j.other_periods:
        lines.append(
            f"  other period {period_key(other.period)}: {other.asked} asked, {other.answered} "
            f"final ({other.approve} approve, {other.veto} veto), {other.held} held, {other.no_reply} "
            "without a reply; counts only, not judged"
        )
    d = j.detail
    if d is not None:
        lines += [
            f"  shadow trades (entry plan, stop, target and costs of the backtest): "
            f"{d.approved} approved, {d.vetoed} vetoed ({d.unresolved} still open, {d.no_bars} "
            "without a bar)",
            f"  approved {_pct(d.approved_mean)} per trade, vetoed {_pct(d.vetoed_mean)}; "
            f"difference {_pct(d.edge)}, Welch t {d.t:.1f}",
            f"  first half {_pct(d.first_half_edge)}, second half {_pct(d.second_half_edge)}; "
            f"random-relabelling control t {d.control_t:.1f}",
            "  This was a LOOK: it is written to the journal and cannot be repeated for the same "
            "sample.",
        ]
    return "\n".join(lines)
