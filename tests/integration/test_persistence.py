"""C4, C5, C6: does state survive, and can the runtime's sessions work together?"""

import itertools
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from tests.fixtures.fakes import FakeBroker
from trading_system.domain.enums import OrderSide, OrderStatus, OrderType, SignalDirection
from trading_system.domain.models import BrokerOrder, OrderIntent, StrategySignal
from trading_system.execution.engine import ExecutionEngine
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.storage.models import BrokerOrderModel, StrategySignalModel
from trading_system.storage.repositories import OrderRepository, SignalRepository, StateRepository


def _broker_order(broker_id: str, client_order_id: str, **overrides: Any) -> BrokerOrder:
    now = datetime.now(timezone.utc)
    values: dict[str, Any] = {
        "broker_id": broker_id,
        "client_order_id": client_order_id,
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "type": OrderType.LIMIT,
        "status": OrderStatus.SUBMITTED,
        "quantity": Decimal("10"),
        "filled_quantity": Decimal("0"),
        "limit_price": Decimal("150.00"),
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return BrokerOrder(**values)


def test_every_enum_survives_a_database_round_trip(session_factory: Any) -> None:
    """Guard, passes today. The original bot (Money_bot) saved OrderSide.BUY as 'buy'
    and reloaded it as SELL after a restart, which flipped its take-profit and
    stop-loss. Every value must come back as the same member."""
    combos = list(itertools.product(OrderSide, OrderType, OrderStatus))
    with session_factory() as session:
        repo = OrderRepository(session)
        for i, (side, type_, status) in enumerate(combos):
            repo.save_broker_order(
                _broker_order(f"b{i}", f"c{i}", side=side, type=type_, status=status)
            )
        for i, direction in enumerate(SignalDirection):
            session.add(
                StrategySignalModel(
                    signal_id=f"s{i}",
                    version="1",
                    symbol="AAPL",
                    timeframe="15m",
                    closed_bar_timestamp=datetime.now(timezone.utc),
                    direction=direction,
                    confidence=1.0,
                )
            )
        session.commit()

    with session_factory() as fresh:
        for i, (side, type_, status) in enumerate(combos):
            row = fresh.get(BrokerOrderModel, f"b{i}")
            assert (row.side, row.type, row.status) == (side, type_, status)
        for i, direction in enumerate(SignalDirection):
            assert fresh.get(StrategySignalModel, f"s{i}").direction == direction


@pytest.mark.xfail(
    strict=True,
    reason="C4: StateRepository.save_state only flushes and nobody commits, so the kill "
    "switch is lost when the CLI process exits (docs/operations.md promises otherwise).",
)
def test_pause_entries_survives_process_exit(settings: Any, session_factory: Any) -> None:
    orchestrator = Orchestrator(settings)
    orchestrator.pause_entries()
    # A process exit rolls back whatever was not committed. Closing the session does the same.
    orchestrator.state_repo.session.close()

    with session_factory() as fresh:
        assert StateRepository(fresh).get_state().is_trading_halted is True


@pytest.mark.xfail(
    strict=True,
    reason="C4: ExecutionEngine.halt_trading goes through the same uncommitted session, so an "
    "emergency halt is forgotten on restart.",
)
def test_emergency_halt_survives_process_exit(settings: Any, session_factory: Any) -> None:
    session = session_factory()
    engine = ExecutionEngine(
        FakeBroker(),
        OrderRepository(session),
        StateRepository(session),
        settings,
    )
    engine.halt_trading("unprotected position")
    session.close()

    with session_factory() as fresh:
        state = StateRepository(fresh).get_state()
        assert state.is_trading_halted is True
        assert state.halt_reason == "unprotected position"


@pytest.mark.xfail(
    strict=True,
    raises=OperationalError,
    reason="C5: the first get_state() inserts the default row and leaves the transaction open, "
    "holding SQLite's single write lock; the next writer fails with 'database is locked'.",
)
def test_reading_state_does_not_block_other_writers(session_factory: Any, signal: Any) -> None:
    state_session = session_factory()
    try:
        StateRepository(state_session).get_state()  # a plain read as far as the caller can tell

        with session_factory() as other:
            SignalRepository(other).save_signal(signal)
            other.commit()
    finally:
        state_session.close()


@pytest.mark.xfail(
    strict=True,
    raises=IntegrityError,
    reason="C6: ExecutionEngine saves a placeholder row with broker_id=client_order_id, then the "
    "real row with Alpaca's id and the same client_order_id: UNIQUE(client_order_id) fails.",
)
def test_broker_confirmation_can_replace_the_inflight_placeholder(session_factory: Any) -> None:
    with session_factory() as session:
        repo = OrderRepository(session)
        repo.save_broker_order(_broker_order("exec_abc", "exec_abc", status=OrderStatus.SUBMITTING))
        repo.save_broker_order(_broker_order("alpaca-uuid-1", "exec_abc"))


@pytest.mark.xfail(
    strict=True,
    reason="C6: a successful live submission is treated as a failure because of the duplicate "
    "client_order_id. The failed flush leaves the session unusable (PendingRollbackError), "
    "so the reconcile and halt steps fail too and the exception escapes submit_order.",
)
def test_successful_live_submission_does_not_halt_trading(
    settings: Any, session_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)  # submit_order sleeps 1s before reconcile
    broker = FakeBroker()
    # One shared session isolates C6 from C5 (cross-session locking).
    session = session_factory()
    signal = StrategySignal(
        version="1",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=datetime.now(timezone.utc),
        direction=SignalDirection.LONG,
    )
    assert SignalRepository(session).save_signal(signal)  # order_intents.signal_id is a FK
    state_repo = StateRepository(session)
    engine = ExecutionEngine(
        broker,
        OrderRepository(session),
        state_repo,
        settings,
        is_dry_run=False,
    )
    intent = OrderIntent(
        signal_id=signal.signal_id or "",
        symbol="AAPL",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("10"),
        limit_price=Decimal("150.00"),
        stop_price=Decimal("144.00"),
    )

    order = engine.submit_order(intent)

    assert len(broker.submitted) == 1  # the order really reached the (fake) broker
    assert order is not None
    assert state_repo.get_state().is_trading_halted is False
