"""The cycle acts on the Oracle's three answers (Phase 2.1a), with a scripted Oracle."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import OracleDecision, TradingState
from trading_system.oracle import OracleRequest
from trading_system.storage.models import BrokerOrderModel
from trading_system.storage.repositories import StateRepository

RunCycle = Callable[..., SimpleNamespace]
APPROVE, VETO, HOLD = OracleVerdict.APPROVE, OracleVerdict.VETO, OracleVerdict.HOLD


class ScriptedOracle:
    """Answers from a list, one per question; an Exception in the list is raised."""

    name = "scripted"

    def __init__(self, *answers: OracleVerdict | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[OracleRequest] = []

    def evaluate(self, request: OracleRequest) -> OracleDecision:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return OracleDecision(
            signal_id=request.signal.signal_id or "",
            verdict=answer,
            reason=f"scripted {answer.value}",
            provider=self.name,
        )


def _orders(session_factory: Any) -> int:
    with session_factory() as session:
        return session.query(BrokerOrderModel).count()


def test_an_approve_lets_the_signal_through_to_an_order(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    oracle = ScriptedOracle(APPROVE)

    cycle = run_cycle(oracle=oracle)

    assert cycle.report.submitted == 1 and _orders(session_factory) == 1
    (request,) = oracle.requests
    assert request.signal.symbol == "AAPL" and "close" in request.technicals


def test_a_veto_ends_the_signal_and_it_is_not_asked_about_again(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    oracle = ScriptedOracle(VETO)
    cycle = run_cycle(oracle=oracle)

    ordered = cycle.cycle.revisit_held(datetime.now(timezone.utc))

    assert (cycle.report.submitted, cycle.report.held, ordered) == (0, 0, 0)
    assert len(oracle.requests) == 1  # a VETO is final for the signal
    assert _orders(session_factory) == 0 and cycle.decisions == []  # risk never saw it


def test_a_hold_waits_and_is_asked_again_until_the_oracle_decides(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    oracle = ScriptedOracle(HOLD, HOLD, APPROVE)
    cycle = run_cycle(oracle=oracle)
    assert (cycle.report.submitted, cycle.report.held) == (0, 1)

    soon = datetime.now(timezone.utc)
    assert cycle.cycle.revisit_held(soon) == 0  # still HOLD
    assert _orders(session_factory) == 0
    assert cycle.cycle.revisit_held(soon) == 1  # now APPROVE: the order goes

    assert len(oracle.requests) == 3 and _orders(session_factory) == 1
    assert cycle.cycle.revisit_held(soon) == 0 and len(oracle.requests) == 3  # nothing left held


def test_a_held_signal_that_expires_is_dropped_without_another_question(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    oracle = ScriptedOracle(HOLD)
    cycle = run_cycle(oracle=oracle)

    ordered = cycle.cycle.revisit_held(datetime.now(timezone.utc) + timedelta(minutes=45))

    assert ordered == 0 and len(oracle.requests) == 1 and _orders(session_factory) == 0
    assert cycle.cycle._held == {}


def test_an_oracle_that_raises_is_a_hold_not_a_veto(
    run_cycle: RunCycle,
    session_factory: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """H1. An Oracle that could not answer has said nothing about the signal: the signal waits,
    and the next answer counts. (A cached veto would have killed it for good.)"""
    oracle = ScriptedOracle(ConnectionError("lemonade is down"), APPROVE)
    with caplog.at_level("INFO"):
        cycle = run_cycle(oracle=oracle)

    assert (cycle.report.submitted, cycle.report.held) == (0, 1)
    assert any("(no answer)" in r.getMessage() for r in caplog.records)
    assert cycle.cycle.revisit_held(datetime.now(timezone.utc)) == 1
    assert _orders(session_factory) == 1


def test_nothing_is_asked_while_trading_is_halted(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    oracle = ScriptedOracle(HOLD, APPROVE)
    cycle = run_cycle(oracle=oracle)
    StateRepository(session_factory).save_state(
        TradingState(is_trading_halted=True, halt_reason="manual")
    )

    ordered = cycle.cycle.revisit_held(datetime.now(timezone.utc))

    assert ordered == 0 and len(oracle.requests) == 1  # it stays held, unasked
    assert len(cycle.cycle._held) == 1
