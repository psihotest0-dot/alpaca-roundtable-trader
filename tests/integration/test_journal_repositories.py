"""The journal and the record of risk decisions, on a real SQLite database."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError

from trading_system.domain.models import RiskDecision, StrategySignal
from trading_system.storage.repositories import (
    AuditRepository,
    RiskDecisionRepository,
    SignalRepository,
)

UTC = timezone.utc
NOON = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


def test_events_come_back_in_order_with_their_details(session_factory: Any) -> None:
    journal = AuditRepository(session_factory)
    journal.record("entries_cycle", {"signals": 1, "skipped": None}, when=NOON)
    journal.record(
        "entries_cycle", {"signals": 0, "skipped": "x"}, when=NOON + timedelta(minutes=15)
    )

    events = journal.events("entries_cycle", NOON - timedelta(hours=1), NOON + timedelta(hours=1))

    assert [(when, details["signals"]) for when, details in events] == [
        (NOON, 1),
        (NOON + timedelta(minutes=15), 0),
    ]
    assert events[1][1]["skipped"] == "x" and events[0][1]["skipped"] is None


def test_the_range_is_closed_at_the_start_and_open_at_the_end(session_factory: Any) -> None:
    journal = AuditRepository(session_factory)
    for hours in (0, 1, 2):
        journal.record("entries_cycle", {"hour": hours}, when=NOON + timedelta(hours=hours))

    events = journal.events("entries_cycle", NOON + timedelta(hours=1), NOON + timedelta(hours=2))

    assert [details["hour"] for _, details in events] == [1]  # the one at the end is the next day's


def test_only_events_of_the_asked_type_come_back(session_factory: Any) -> None:
    journal = AuditRepository(session_factory)
    journal.record("entries_cycle", {"a": 1}, when=NOON)
    journal.record("something_else", {"b": 2}, when=NOON)

    events = journal.events("entries_cycle", NOON - timedelta(days=1), NOON + timedelta(days=1))

    assert [details for _, details in events] == [{"a": 1}]


def test_times_come_back_with_their_timezone(session_factory: Any) -> None:
    journal = AuditRepository(session_factory)
    journal.record("e", {}, when=NOON)

    [(when, _)] = journal.events("e", NOON - timedelta(days=1), NOON + timedelta(days=1))

    assert when.tzinfo is not None and when == NOON


def _signal() -> StrategySignal:
    return StrategySignal(
        version="1",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=NOON,
        direction="LONG",  # type: ignore[arg-type]
    )


def test_a_risk_decision_is_saved_against_its_signal(session_factory: Any) -> None:
    from trading_system.storage.models import RiskDecisionModel

    signal = _signal()
    SignalRepository(session_factory).save_signal(signal)

    RiskDecisionRepository(session_factory).save(
        RiskDecision(
            signal_id=signal.signal_id or "",
            approved=False,
            reasoning="Max open positions limit reached.",
            suggested_quantity=Decimal("3"),
        )
    )

    with session_factory() as session:
        [row] = session.query(RiskDecisionModel).all()
        assert (row.signal_id, row.approved) == (signal.signal_id, False)
        assert row.reasoning == "Max open positions limit reached."


def test_a_decision_for_an_unknown_signal_is_refused(session_factory: Any) -> None:
    with pytest.raises(IntegrityError):
        RiskDecisionRepository(session_factory).save(
            RiskDecision(signal_id="nobody", approved=True, reasoning="Approved.")
        )
