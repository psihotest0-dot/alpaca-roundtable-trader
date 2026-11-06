"""The cycle acts on the Oracle's three answers (Phase 2.1a), with a scripted Oracle."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.oracles import ScriptedOracle
from trading_system.domain.enums import OracleVerdict
from trading_system.domain.models import TradingState
from trading_system.storage.models import BrokerOrderModel, RiskDecisionModel
from trading_system.storage.repositories import StateRepository

RunCycle = Callable[..., SimpleNamespace]
APPROVE, VETO, HOLD = OracleVerdict.APPROVE, OracleVerdict.VETO, OracleVerdict.HOLD


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
    # Risk took a first look before the Oracle was asked (and approved); that look is not
    # recorded, and the final evaluation never came: no risk decision is on record.
    assert _orders(session_factory) == 0 and len(cycle.decisions) == 1
    with session_factory() as session:
        assert session.query(RiskDecisionModel).count() == 0


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


def test_the_oracle_is_given_the_facts_to_judge_by(run_cycle: RunCycle) -> None:
    oracle = ScriptedOracle(APPROVE)

    cycle = run_cycle(oracle=oracle, live_price_factor=1.01)

    (request,) = oracle.requests
    assert request.live_price == cycle.live_price  # the price of right now, not of the bar
    assert {"close", "ema_fast", "ema_slow", "rsi", "atr", "hourly_trend_up"} <= set(
        request.technicals
    )
    assert len(request.recent_bars) == 12 and request.volume_ratio is not None
    assert request.context["open_positions"] == 0
    assert request.context["max_open_positions"] == 5
    assert request.context["stop_loss_pct"] == pytest.approx(4.0)
    assert request.context["daily_pnl_pct"] == 0.0


def test_a_signal_without_a_usable_price_is_not_put_to_the_oracle(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    """The cheap check comes first. And an unusable price is not the end of the signal: a quiet
    symbol prints again, so the signal waits for the next tick."""
    oracle = ScriptedOracle()

    cycle = run_cycle(oracle=oracle, live_age_seconds=600)

    assert oracle.requests == [] and _orders(session_factory) == 0
    assert (cycle.report.submitted, cycle.report.held) == (0, 1)
