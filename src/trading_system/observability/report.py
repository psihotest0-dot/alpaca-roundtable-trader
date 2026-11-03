"""The daily report (Phase 6.2): what the system did on one New York trading day.

It is built from what the system recorded in its own database: the journal of passes over a
bar and of alerts, the risk decisions, the orders and the fills. No network, no keys. Its job is
to make a quiet failure loud the same day: "24 passes, 1 signal, 1 broke, no order" is a line
that a log of 26 identical "Generated 0 valid signals" never was."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from trading_system.domain.models import TradingState
from trading_system.risk.daily_equity import MARKET_TZ
from trading_system.storage.models import BrokerOrderModel, FillModel, RiskDecisionModel
from trading_system.storage.repositories import AuditRepository

# A full regular session has this many 15-minute bars.
BARS_IN_A_SESSION = 26


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """[start, end) of a New York calendar day, in UTC."""
    start = datetime.combine(day, time.min, tzinfo=MARKET_TZ)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=MARKET_TZ)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


@dataclass
class DailyReport:
    day: date
    passes: int = 0  # entries_cycle events, every try
    completed: int = 0  # the ones that reached a decision (not "try again later")
    not_completed: Counter[str] = field(default_factory=Counter)  # why the others did not
    skipped: Counter[str] = field(default_factory=Counter)  # completed, but nothing was decided
    signals: int = 0
    submitted: int = 0
    held: int = 0
    errors: int = 0
    approved: int = 0
    refused: Counter[str] = field(default_factory=Counter)  # reason -> how many
    orders: Counter[str] = field(default_factory=Counter)  # status -> how many
    dry_run_orders: int = 0
    fills: int = 0
    equity_first: Optional[Decimal] = None
    equity_last: Optional[Decimal] = None
    baseline: Optional[Decimal] = None
    alerts: list[tuple[datetime, str]] = field(default_factory=list)
    halted: bool = False
    halt_reason: Optional[str] = None


def _reason(text: str) -> str:
    """A refusal without its numbers: "Daily loss guard breached: 0.0212 >= 0.02" counts with
    the other daily-loss refusals."""
    return text.split(":")[0].strip() or text


def build_daily_report(
    session_factory: sessionmaker[Session], day: date, state: Optional[TradingState] = None
) -> DailyReport:
    start, end = day_bounds(day)
    report = DailyReport(day=day)
    journal = AuditRepository(session_factory)

    equities: list[Decimal] = []
    for _, details in journal.events("entries_cycle", start, end):
        report.passes += 1
        if details.get("retry_later"):
            report.not_completed[str(details.get("skipped") or "no reason given")] += 1
        else:
            report.completed += 1
            if details.get("skipped"):
                report.skipped[str(details["skipped"])] += 1
        report.signals += int(details.get("signals") or 0)
        report.submitted += int(details.get("submitted") or 0)
        report.held += int(details.get("held") or 0)
        report.errors += int(details.get("errors") or 0)
        if details.get("equity"):
            equities.append(Decimal(str(details["equity"])))
    if equities:
        report.equity_first, report.equity_last = equities[0], equities[-1]

    for when, details in journal.events("alert", start, end):
        report.alerts.append((when, str(details.get("title") or "alert")))

    with session_factory() as session:
        for decision in session.scalars(
            select(RiskDecisionModel).where(
                RiskDecisionModel.decision_timestamp >= start,  # type: ignore[arg-type]
                RiskDecisionModel.decision_timestamp < end,  # type: ignore[arg-type]
            )
        ):
            if decision.approved:
                report.approved += 1
            else:
                report.refused[_reason(str(decision.reasoning))] += 1

        for order in session.scalars(
            select(BrokerOrderModel).where(
                BrokerOrderModel.created_at >= start,  # type: ignore[arg-type]
                BrokerOrderModel.created_at < end,  # type: ignore[arg-type]
            )
        ):
            report.orders[str(getattr(order.status, "value", order.status))] += 1
            if str(order.broker_id or "").startswith("dry_run_"):
                report.dry_run_orders += 1

        report.fills = len(
            session.scalars(
                select(FillModel.id).where(
                    FillModel.timestamp >= start,  # type: ignore[arg-type]
                    FillModel.timestamp < end,  # type: ignore[arg-type]
                )
            ).all()
        )

    if state is not None:
        report.halted = state.is_trading_halted
        report.halt_reason = state.halt_reason
        if (
            state.daily_start_equity is not None
            and state.daily_equity_date is not None
            and state.daily_equity_date.date() == day
        ):
            report.baseline = state.daily_start_equity
    return report


def _counts(counter: Counter[str]) -> str:
    return ", ".join(f"{reason}: {n}" for reason, n in counter.most_common())


def format_daily_report(report: DailyReport) -> str:
    lines = [f"=== Daily report {report.day} (New York trading day) ==="]

    passes = f"{report.completed} completed of {report.passes} tries"
    if report.completed < BARS_IN_A_SESSION:
        passes += f" (a full session has {BARS_IN_A_SESSION} bars)"
    lines.append(f"Passes over a bar:  {passes}")
    if report.not_completed:
        lines.append(f"  not completed:    {_counts(report.not_completed)}")
    if report.skipped:
        lines.append(f"  decided nothing:  {_counts(report.skipped)}")

    lines.append(
        f"Signals:            {report.signals} -> {report.submitted} became an order, "
        f"{report.held} held, {report.errors} BROKE"
        if report.errors
        else f"Signals:            {report.signals} -> {report.submitted} became an order, "
        f"{report.held} held"
    )
    refused = sum(report.refused.values())
    lines.append(f"Risk decisions:     {report.approved} approved, {refused} refused")
    if report.refused:
        lines.append(f"  refused:          {_counts(report.refused)}")

    orders = sum(report.orders.values())
    lines.append(
        f"Orders:             {orders}" + (f" ({_counts(report.orders)})" if report.orders else "")
    )
    if orders:
        lines.append(f"  simulated:        {report.dry_run_orders} of {orders} (dry-run)")
    lines.append(f"Fills:              {report.fills}")

    if report.equity_first is not None and report.equity_last is not None:
        change = (
            (report.equity_last - report.equity_first) / report.equity_first * 100
            if report.equity_first
            else Decimal(0)
        )
        lines.append(
            f"Equity:             {report.equity_first:,.2f} at the first pass, "
            f"{report.equity_last:,.2f} at the last ({change:+.2f}%)"
        )
    else:
        lines.append("Equity:             not recorded on this day")
    if report.baseline is not None:
        lines.append(
            f"  start of day:     {report.baseline:,.2f} (the daily loss guard's baseline)"
        )

    if report.alerts:
        lines.append(f"ALERTS:             {len(report.alerts)}")
        for when, title in report.alerts:
            lines.append(f"  {when:%H:%M} UTC  {title}")
    else:
        lines.append("Alerts:             none")

    if report.halted:
        lines.append(f"Trading is HALTED:  {report.halt_reason or 'no reason recorded'}")
    return "\n".join(lines)
