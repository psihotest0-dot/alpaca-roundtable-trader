"""What the cycle leaves behind (Phase 6): a journal entry for every pass over a bar, the risk
decision of every signal, and an alert when a signal breaks before it becomes an order.

The first real signal broke in the risk step and the only trace was one line in a log; nothing
said "a signal failed" and nothing recorded why a signal was not traded."""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from trading_system.storage.models import BrokerOrderModel, RiskDecisionModel
from trading_system.storage.repositories import AuditRepository

RunCycle = Callable[..., SimpleNamespace]


def _journal(session_factory: Any) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    events = AuditRepository(session_factory).events(
        "entries_cycle", now - timedelta(days=1), now + timedelta(days=1)
    )
    return [details for _, details in events]


def test_a_pass_over_a_bar_is_journalled_with_what_it_did(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle()

    [entry] = _journal(session_factory)
    assert (entry["signals"], entry["submitted"], entry["held"], entry["errors"]) == (1, 1, 0, 0)
    assert entry["skipped"] is None and entry["retry_later"] is False
    assert entry["bar_start"]


def test_a_pass_that_could_not_decide_is_journalled_with_the_reason(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle(positions_error=RuntimeError("broker down"))

    [entry] = _journal(session_factory)
    assert entry["skipped"] == "positions unavailable" and entry["retry_later"] is True
    assert entry["submitted"] == 0


def test_the_risk_decision_of_a_signal_is_kept_with_its_reason(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle()

    with session_factory() as session:
        [decision] = session.query(RiskDecisionModel).all()
        assert decision.approved is True and decision.reasoning == "Approved."
        assert decision.suggested_limit_price is not None and decision.suggested_quantity


def test_a_refusal_is_kept_with_the_reason_and_no_order_follows(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle(equity="1000")  # 4 % of 1000 is under the price of one share

    with session_factory() as session:
        [decision] = session.query(RiskDecisionModel).all()
        assert decision.approved is False and "less than 1" in str(decision.reasoning)
        assert session.query(BrokerOrderModel).count() == 0


def _break_execution(cycle: Any) -> None:
    def boom(intent: Any) -> Any:
        raise RuntimeError("the broker contract changed")

    cycle.execution.submit_order = boom


def test_a_signal_that_breaks_is_counted_journalled_and_alerted(
    run_cycle: RunCycle, session_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        cycle = run_cycle(tweak=_break_execution)

    assert (cycle.report.signals, cycle.report.submitted, cycle.report.errors) == (1, 0, 1)
    [entry] = _journal(session_factory)
    assert entry["errors"] == 1 and entry["submitted"] == 0
    alerts = [
        r for r in caplog.records if r.levelno == logging.CRITICAL and "ALERT:" in r.getMessage()
    ]
    assert len(alerts) == 1
    assert "Signals broke before they became orders" in alerts[0].getMessage()
    assert "1 of 1 signals" in alerts[0].getMessage()


def test_a_healthy_pass_raises_no_alert(
    run_cycle: RunCycle, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        run_cycle()

    assert not [r for r in caplog.records if "ALERT:" in r.getMessage()]


class _Broken:
    def save(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("disk full")

    record = save


def test_a_journal_that_cannot_be_written_does_not_stop_the_trading(
    run_cycle: RunCycle, caplog: pytest.LogCaptureFixture
) -> None:
    def break_records(cycle: Any) -> None:
        cycle.audit_repo = _Broken()
        cycle.risk_repo = _Broken()

    with caplog.at_level(logging.WARNING):
        cycle = run_cycle(tweak=break_records)

    assert cycle.report.submitted == 1  # the order went ahead
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "Could not journal the entry look-up" in messages
    assert "Could not save the risk decision" in messages


def test_a_look_up_that_crashes_is_journalled_and_alerted_before_the_error_goes_on(
    run_cycle: RunCycle,
    session_factory: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_strategy(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the strategy broke")

    monkeypatch.setattr("trading_system.runtime.cycle.generate_signals", broken_strategy)

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="the strategy broke"):
        run_cycle()

    [entry] = _journal(session_factory)
    assert entry["skipped"] == "crashed: the strategy broke" and entry["errors"] == 1
    assert any(
        r.levelno == logging.CRITICAL and "Entry look-up crashed" in r.getMessage()
        for r in caplog.records
    )
