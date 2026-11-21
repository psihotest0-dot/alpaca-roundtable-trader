from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# --- the loop, judged from the outside (Phase 6.1b) ------------------------------------------


@dataclass(frozen=True)
class Verdict:
    healthy: bool
    reasons: list[str] = field(default_factory=list)


def _moment(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def evaluate_heartbeat(
    beat: Optional[dict[str, Any]],
    now: datetime,
    *,
    grace_seconds: float = 120.0,
    quiet_minutes: float = 20.0,
) -> Verdict:
    """Whether the trading loop is doing its job, from the last thing it wrote.

    Unhealthy when: there is no heartbeat; the loop has not ticked within the wait it announced
    plus a grace; it cannot read the market clock; the market is open and no entry look-up has
    completed for `quiet_minutes` (a bar every 15 minutes, a few retries); or the last look-up
    had signals that broke before they became orders. A deliberate halt is not unhealthy: it is
    what the owner or the protection check asked for, and it shows in `status`."""
    if beat is None:
        return Verdict(False, ["no heartbeat: the loop has not started, or its file is unreadable"])

    reasons: list[str] = []
    at = _moment(beat.get("at"))
    if at is None:
        return Verdict(False, ["the heartbeat has no time"])

    announced = float(beat.get("next_within_seconds") or 0.0)
    age = (now - at).total_seconds()
    if age > announced + grace_seconds:
        reasons.append(
            f"the loop has not ticked for {age:.0f} s (it announced {announced:.0f} s "
            f"and gets {grace_seconds:.0f} s of grace)"
        )

    if beat.get("clock_error"):
        reasons.append("the market clock cannot be read")

    if beat.get("market_open"):
        since = max(
            (
                m
                for m in (
                    _moment(beat.get("last_entries_at")),
                    _moment(beat.get("market_open_since")),
                    _moment(beat.get("started_at")),
                )
                if m is not None
            ),
            default=None,
        )
        if since is not None and (now - since).total_seconds() > quiet_minutes * 60:
            reasons.append(
                f"the market is open and no entry look-up has completed for "
                f"{(now - since).total_seconds() / 60:.0f} min"
            )

    errors = int(beat.get("last_entries_errors") or 0)
    if errors:
        reasons.append(
            f"the last entry look-up had {errors} signal(s) that broke before becoming an order"
        )
    return Verdict(not reasons, reasons)
