"""The count of variants tried.

Try enough variants and one will look good by chance: that is how `strict_ema_crossover` showed
+0.38 % over 90 days and nothing over 180. The count is what lets a result be read against it.
Every measurement of a real signal is written to the journal (`audit_events`, in trading.db, which
is backed up; research.db is a cache that may be deleted). A variant is a signal, a regime filter and
a horizon; measuring the same one again is a run, not a new variant.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from trading_system.research.measure import Measurement
from trading_system.storage.repositories import AuditRepository

EVENT_TYPE = "research_trial"


def variant_key(signal: str, regime_filter: Optional[str], horizon: int) -> str:
    return f"{signal}|{regime_filter or 'no filter'}|{horizon}"


def record_trial(
    audit: AuditRepository,
    measurement: Measurement,
    *,
    regime_filter: Optional[str],
    universe_size: int,
    when: Optional[datetime] = None,
) -> None:
    net = measurement.net
    audit.record(
        EVENT_TYPE,
        {
            "variant": variant_key(measurement.signal, regime_filter, measurement.horizon),
            "signal": measurement.signal,
            "regime_filter": regime_filter,
            "horizon": measurement.horizon,
            "cost": measurement.cost,
            "universe_size": universe_size,
            "first": measurement.first,
            "last": measurement.last,
            "independent_events": round(measurement.independent_events),
            "net_mean": net.mean,
            "net_t": net.t,
            "meets_criteria": measurement.criteria.all_met,
        },
        when=when,
    )


def trial_counts(audit: AuditRepository) -> tuple[int, int]:
    """(runs, distinct variants) recorded so far."""
    events: list[tuple[datetime, dict[str, Any]]] = audit.events(
        EVENT_TYPE,
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(2100, 1, 1, tzinfo=timezone.utc),
    )
    return len(events), len({details["variant"] for _, details in events})
