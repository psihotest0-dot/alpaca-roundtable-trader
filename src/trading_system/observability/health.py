from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class HealthStatus:
    """Tracks the health and readiness of the trading system."""

    def __init__(self) -> None:
        self._is_running = True

        # State tracking
        self.market_data_stale = False
        self.reconciliation_failed = False
        self.oracle_failed = False
        self.daily_loss_breached = False
        self.emergency_halt = False

    def check_liveness(self) -> bool:
        """
        Liveness indicates if the process is functioning.
        If this returns False, the orchestrator/container should be restarted.
        """
        return self._is_running

    def check_readiness(self) -> bool:
        """
        Readiness indicates if it is safe to process NEW inputs (e.g., new entries).
        If this returns False, new signals should be rejected, but exits/reconciliation
        can and must continue.
        """
        if not self._is_running:
            return False

        if self.emergency_halt:
            return False

        if self.market_data_stale:
            return False

        if self.reconciliation_failed:
            return False

        if self.daily_loss_breached:
            return False

        # Note: Oracle failure explicitly does NOT affect readiness.
        # Defensive exits and reconciliation mechanisms operate entirely
        # independently of the Oracle.
        # If the Oracle fails, we might reject new entries that *require* it,
        # but the system as a whole remains ready for exits and reconciliation.

        return True

    def mark_market_data_stale(self, is_stale: bool) -> None:
        self.market_data_stale = is_stale

    def mark_reconciliation_failed(self, has_failed: bool) -> None:
        self.reconciliation_failed = has_failed

    def mark_oracle_failed(self, has_failed: bool) -> None:
        self.oracle_failed = has_failed

    def mark_daily_loss_breached(self, is_breached: bool) -> None:
        self.daily_loss_breached = is_breached

    def trigger_emergency_halt(self) -> None:
        self.emergency_halt = True

    def set_running(self, is_running: bool) -> None:
        self._is_running = is_running


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
