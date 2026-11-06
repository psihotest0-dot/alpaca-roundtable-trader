"""Does our own journal of the account's equity agree with Alpaca's history of it?

The journal (`entries_cycle` events, see runtime/cycle.py) records the equity at every pass over a
bar; Alpaca keeps its own end-of-day equity (`GET /account/portfolio/history`). They are two
independent records of the same number. While the bot only dry-runs they are flat and equal, which
proves little; once real paper orders flow, a fill the journal never saw, a position nobody
recorded or a day the bot was down shows as a difference here, in a form a person can read.

Facts from probing the live API on 2026-11-06: the point of a day is stamped 20:00 New York time
(00:00 or 01:00 UTC the next calendar day, so the UTC date is a day late and a Friday lands on a
Saturday): the New York date of the stamp is the trading day. The current day has no point until
it has closed.

The tolerance is an arbitrary 0.5 %: the journal's last pass of a day is a few minutes before the
close, so some drift is normal. It is there to catch a missing fill, not a rounding; tighten it
when there is real data to tighten it against.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from trading_system.observability.report import day_bounds
from trading_system.risk.daily_equity import MARKET_TZ
from trading_system.storage.repositories import AuditRepository

TOLERANCE = Decimal("0.005")


@dataclass(frozen=True)
class DayComparison:
    day: date
    alpaca: Decimal
    journal: Decimal

    @property
    def difference(self) -> Decimal:
        """The journal's equity against Alpaca's, as a fraction of Alpaca's."""
        if self.alpaca == 0:
            return Decimal(0) if self.journal == 0 else Decimal(1)
        return (self.journal - self.alpaca) / self.alpaca


@dataclass(frozen=True)
class EquityCheck:
    compared: list[DayComparison]
    missing_days: list[date]  # Alpaca has the day, the journal has not, and the journal had begun
    before_journal: int  # days older than the journal's first entry
    tolerance: Decimal

    @property
    def differing(self) -> list[DayComparison]:
        return [c for c in self.compared if abs(c.difference) > self.tolerance]

    @property
    def ok(self) -> bool:
        """Nothing compared is not a pass: a check that looked at nothing proves nothing."""
        return bool(self.compared) and not self.differing


def alpaca_equity_by_day(history: Any) -> dict[date, Decimal]:
    """{trading day: equity} from a PortfolioHistory (daily points)."""
    days: dict[date, Decimal] = {}
    for stamp, equity in zip(history.timestamp, history.equity):
        moment = datetime.fromtimestamp(stamp, timezone.utc).astimezone(MARKET_TZ)
        days[moment.date()] = Decimal(str(equity))
    return days


def journal_equity_by_day(journal: AuditRepository, first: date, last: date) -> dict[date, Decimal]:
    """{New York day: the equity of the day's last pass} from the `entries_cycle` events."""
    start, _ = day_bounds(first)
    _, end = day_bounds(last)
    days: dict[date, Decimal] = {}
    for when, details in journal.events("entries_cycle", start, end):
        if details.get("equity"):
            days[when.astimezone(MARKET_TZ).date()] = Decimal(str(details["equity"]))
    return days


def compare_equity(
    alpaca: dict[date, Decimal],
    journal: dict[date, Decimal],
    tolerance: Decimal = TOLERANCE,
) -> EquityCheck:
    began: Optional[date] = min(journal) if journal else None
    compared: list[DayComparison] = []
    missing: list[date] = []
    before = 0
    for day in sorted(alpaca):
        if day in journal:
            compared.append(DayComparison(day, alpaca[day], journal[day]))
        elif began is None or day < began:
            before += 1
        else:
            missing.append(day)
    return EquityCheck(compared, missing, before, tolerance)


def format_equity_check(check: EquityCheck) -> str:
    lines = ["Equity: our journal against Alpaca's end-of-day history"]
    for c in check.compared:
        flag = "   <-- DIFFERS" if abs(c.difference) > check.tolerance else ""
        lines.append(
            f"  {c.day}  journal {c.journal:>12,.2f}  alpaca {c.alpaca:>12,.2f}  "
            f"{c.difference * 100:+.2f} %{flag}"
        )
    if not check.compared:
        lines.append("  Nothing to compare: the journal has no equity entry for any of the days.")
    if check.before_journal:
        lines.append(
            f"  {check.before_journal} older day(s) in Alpaca's history predate the journal."
        )
    if check.missing_days:
        lines.append(
            "  No journal entry for "
            + ", ".join(str(d) for d in check.missing_days)
            + " (the bot was down, or paused before its first pass)."
        )
    differing = check.differing
    if differing:
        lines.append(
            f"{len(differing)} of {len(check.compared)} day(s) differ by more than "
            f"{check.tolerance * 100:.1f} %: look for a fill or a position the journal missed "
            "(`reconcile`, `report --date`)."
        )
    elif check.compared:
        lines.append(
            f"All {len(check.compared)} compared day(s) agree within {check.tolerance * 100:.1f} %."
        )
    return "\n".join(lines)
