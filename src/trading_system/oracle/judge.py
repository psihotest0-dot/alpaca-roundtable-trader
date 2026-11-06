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
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd

from trading_system.backtest.analysis import welch_t
from trading_system.backtest.models import BacktestConfig
from trading_system.backtest.shadow import simulate_shadow_trade
from trading_system.research.signals import CONTROL_SEED
from trading_system.risk.entry_plan import InvalidEntryPlan, build_entry_plan
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
BAR = timedelta(minutes=15)


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
    availability: float  # (approve + veto) / asked
    period: Optional[Period] = None  # the period these counts are of

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


@dataclass(frozen=True)
class Judgement:
    counts: Counts
    looks_before: int  # looks taken in THIS period
    verdict: str
    reason: str
    detail: Optional[Detail] = None
    other_periods: tuple[Counts, ...] = ()  # counts only, never judged here


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
    return Counts(asked, approve, veto, (approve + veto) / asked if asked else 0.0, period)


def _shadow_pnl(
    sample: dict[str, dict[str, Any]], bars: MarketBarRepository, settings: Any
) -> tuple[list[tuple[str, float, pd.Timestamp]], int, int]:
    """(verdict, pnl, entry time) for each answered signal that has resolved; and how many have not,
    and how many had no bar to enter on."""
    costs = BacktestConfig()
    answered = {k: e for k, e in sample.items() if e["final"] and not e["failed"]}
    by_symbol: dict[str, list[str]] = {}
    for signal_id, entry in answered.items():
        by_symbol.setdefault(entry["symbol"], []).append(signal_id)
    out: list[tuple[str, float, pd.Timestamp]] = []
    unresolved = no_bars = 0
    for symbol, ids in by_symbol.items():
        first = min(answered[i]["closed_bar"] for i in ids) - timedelta(hours=2)
        frame = bars.load([symbol], "15m", start=first.to_pydatetime()).get(symbol)
        for signal_id in ids:
            entry_time = answered[signal_id]["closed_bar"] + BAR
            if frame is None or entry_time not in frame.index:
                no_bars += 1
                continue
            price = Decimal(str(frame.loc[entry_time, "open"]))
            try:
                plan = build_entry_plan(
                    price,
                    limit_buffer_pct=Decimal(str(settings.entry_limit_buffer_pct)),
                    stop_loss_pct=Decimal(str(settings.stop_loss_pct)),
                    take_profit_pct=Decimal(str(settings.take_profit_pct)),
                )
            except InvalidEntryPlan:
                no_bars += 1
                continue
            fill = min(plan.limit_price, price * (Decimal("1") + Decimal(str(costs.slippage_pct))))
            shadow = simulate_shadow_trade(
                symbol=symbol,
                signal_id=signal_id,
                verdict=answered[signal_id]["verdict"],
                bars=frame,
                entry_time=entry_time.to_pydatetime(),
                entry_price=fill,
                stop_loss=plan.stop_price,
                take_profit=plan.take_profit_price,
                slippage_pct=costs.slippage_pct,
                fee_pct=costs.fee_pct_per_trade,
            )
            if shadow.exit_reason == "OPEN":
                unresolved += 1
                continue
            out.append((answered[signal_id]["verdict"], shadow.pnl_pct, entry_time))
    return out, unresolved, no_bars


def _edge(trades: list[tuple[str, float, pd.Timestamp]]) -> float:
    a = [p for v, p, _ in trades if v == "APPROVE"]
    b = [p for v, p, _ in trades if v == "VETO"]
    return float(np.mean(a) - np.mean(b)) if a and b else float("nan")


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
    looks = sum(
        1
        for _, details in audit.events(
            JUDGEMENT_EVENT, datetime(2000, 1, 1, tzinfo=timezone.utc), now + timedelta(days=1)
        )
        if details.get("period") == key
    )

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
            f"availability {counts.availability:.0%} is under {MIN_AVAILABILITY:.0%}: the backend or "
            "the model fails too often to judge",
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


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.2f}%"


def format_judgement(j: Judgement) -> str:
    c = j.counts
    lines = [
        f"The Oracle's answers since {START:%Y-%m-%d %H:%M} UTC (shadow mode, probes left out):",
        "  period judged (model | prompt label): " + period_key(c.period),
        f"  asked about {c.asked} signals; answered {c.answered} "
        f"({c.approve} approve, {c.veto} veto), availability {c.availability:.0%}",
        f"  looks taken in this period: {j.looks_before} of {MAX_LOOKS}",
        f"  verdict: {j.verdict}",
        f"  {j.reason}",
    ]
    for other in j.other_periods:
        lines.append(
            f"  other period {period_key(other.period)}: {other.asked} asked, {other.answered} "
            f"answered ({other.approve} approve, {other.veto} veto); counts only, not judged"
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
