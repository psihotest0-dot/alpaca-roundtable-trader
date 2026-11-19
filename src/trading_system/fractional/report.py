"""The report of the fractional paper trial (docs/fractional-paper-trial.md, section 3), read from the
journal. It counts what the loop did and states the written criteria of operational readiness; it
says nothing about profit, which the trial does not test.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# The written criteria (section 3). Not to be changed without an entry in the document.
MIN_TRADING_DAYS = 20
MIN_ENTRIES = 15
MAX_GAP_SECONDS = 120.0  # fill to first stop
OPENING_GRACE = timedelta(minutes=5)  # after 09:30 New York the first look may take this long
GROSS_CEILING = 0.20


@dataclass
class Report:
    trading_days: int = 0
    passes: int = 0
    crashed_passes: int = 0
    entries: int = 0
    fills: int = 0
    refused: Counter[str] = field(default_factory=Counter)
    stale_cancelled: int = 0
    stops_placed: int = 0
    unprotected_episodes: int = 0
    late_gaps: list[float] = field(default_factory=list)  # gaps over the limit, outside the opening
    gaps: list[float] = field(default_factory=list)
    stops_refused: int = 0
    target_sales: int = 0
    stop_sales: int = 0
    sale_problems: int = 0
    alerts: Counter[str] = field(default_factory=Counter)
    versus_limit: list[float] = field(default_factory=list)
    versus_reference: list[float] = field(default_factory=list)
    gross_over_ceiling: int = 0
    baseline: str = "not recorded yet"
    unexplained_refusals: int = 0

    @property
    def enough(self) -> bool:
        return self.trading_days >= MIN_TRADING_DAYS and self.entries >= MIN_ENTRIES

    def failures(self) -> list[str]:
        out = []
        if self.late_gaps:
            out.append(
                f"{len(self.late_gaps)} position(s) stood without a stop for over "
                f"{MAX_GAP_SECONDS:.0f} s with the market open (longest {max(self.late_gaps):.0f} s)"
            )
        if self.unexplained_refusals:
            out.append(f"{self.unexplained_refusals} order(s) refused by Alpaca")
        if self.gross_over_ceiling:
            out.append(
                f"{self.gross_over_ceiling} pass(es) with gross exposure over 20 % of the capital"
            )
        if self.alerts.get("Fractional entries stopped"):
            out.append("the entries were halted for a position without a stop")
        if self.alerts.get("Fractional order above the hard ceiling"):
            out.append("an order reached the hard ceiling")
        if self.sale_problems:
            out.append(f"{self.sale_problems} sale(s) refused")
        return out


def _opening_grace(start: datetime) -> bool:
    local = start.astimezone(NEW_YORK)
    opened = local.replace(hour=9, minute=30, second=0, microsecond=0)
    return opened <= local <= opened + OPENING_GRACE


def build(audit: Any, now: datetime) -> Report:
    end = now + timedelta(days=1)

    def events(name: str) -> list[tuple[datetime, dict[str, Any]]]:
        return list(audit.events(name, EPOCH, end))

    report = Report()
    for _, d in events("fractional_baseline")[:1]:
        report.baseline = f"{d['equity']} (budget {d['budget']})"

    days = set()
    for when, d in events("fractional_cycle"):
        report.passes += 1
        days.add(when.astimezone(NEW_YORK).date())
        if str(d.get("skipped") or "").startswith("crashed"):
            report.crashed_passes += 1
        gross_pct = d.get("gross_pct")
        if gross_pct is not None and float(gross_pct) > GROSS_CEILING + 1e-9:
            report.gross_over_ceiling += 1
    report.trading_days = len(days)

    report.entries = len(events("fractional_entry"))
    for _, d in events("fractional_refused"):
        report.refused[str(d.get("reason", ""))[:60]] += 1
    for _, d in events("fractional_fill"):
        report.fills += 1
        report.versus_limit.append(float(d["versus_limit_pct"]))
        report.versus_reference.append(float(d["versus_reference_pct"]))
    report.stale_cancelled = len(events("fractional_entry_cancelled"))
    report.unprotected_episodes = len(events("fractional_unprotected"))
    for when, d in events("fractional_stop_placed"):
        report.stops_placed += 1
        gap = d.get("gap_seconds")
        if gap is None:
            continue
        report.gaps.append(float(gap))
        started = when - timedelta(seconds=float(gap))
        if float(gap) > MAX_GAP_SECONDS and not _opening_grace(started):
            report.late_gaps.append(float(gap))
    report.stops_refused = len(events("fractional_stop_refused"))
    report.target_sales = len(events("fractional_target_sold"))
    report.stop_sales = len(events("fractional_stop_sold"))
    report.sale_problems = len(events("fractional_sale_refused"))
    for _, d in events("fractional_alert"):
        report.alerts[str(d.get("title", ""))] += 1
    # Every refusal of a stop is Alpaca's word and is written down; none is expected (the one expected
    # case, a price already through the stop, is sold instead of sent).
    report.unexplained_refusals = report.stops_refused
    return report


def _avg(values: list[float]) -> str:
    return f"{mean(values):+.3f} %" if values else "n/a"


def format_report(r: Report) -> str:
    lines = [
        "Fractional paper trial (docs/fractional-paper-trial.md): an operational test, not a test of profit.",
        f"  baseline equity of the account: {r.baseline}",
        f"  trading days with a pass: {r.trading_days} (need {MIN_TRADING_DAYS}); entries: {r.entries} "
        f"(need {MIN_ENTRIES}); passes: {r.passes}, crashed: {r.crashed_passes}",
        f"  entries filled: {r.fills} of {r.entries}; stale entries cancelled: {r.stale_cancelled}; "
        f"refused before sending: {sum(r.refused.values())}"
        + (
            f" ({', '.join(f'{n} x {why}' for why, n in r.refused.most_common(3))})"
            if r.refused
            else ""
        ),
        f"  fill versus limit: {_avg(r.versus_limit)} on average; versus the signal bar's close: "
        f"{_avg(r.versus_reference)} (descriptive, the dry run assumed 0.05 % a side)",
        f"  stops placed: {r.stops_placed}; episodes without a stop with the market open: "
        f"{r.unprotected_episodes}"
        + (
            f"; gap from fill or notice to stop: median {median(r.gaps):.0f} s, longest {max(r.gaps):.0f} s"
            if r.gaps
            else ""
        ),
        f"  stops refused by Alpaca: {r.stops_refused}; sold at the target: {r.target_sales}; "
        f"sold below the stop: {r.stop_sales}; sales refused: {r.sale_problems}",
        f"  alerts: {sum(r.alerts.values())}"
        + (f" ({', '.join(f'{n} x {t}' for t, n in r.alerts.most_common())})" if r.alerts else ""),
    ]
    problems = r.failures()
    if not r.enough:
        lines.append(
            "  verdict: not enough yet (the written minimum is "
            f"{MIN_TRADING_DAYS} trading days and {MIN_ENTRIES} entries)."
            + (f" Already against the criteria: {'; '.join(problems)}." if problems else "")
        )
    elif problems:
        lines.append(f"  verdict: NOT ready for a next step: {'; '.join(problems)}.")
    else:
        lines.append(
            "  verdict: the criteria of operational readiness are met. What follows (a bigger "
            "budget, real money, another strategy) is the owner's decision with a new definition."
        )
    return "\n".join(lines)
