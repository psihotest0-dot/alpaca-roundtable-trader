"""Does state survive, and can the runtime's repositories work side by side?

C4 and C5 are fixed (every repository call is its own committed transaction) and
C6 is fixed (an order is identified by its client_order_id).
"""

import itertools
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError

from tests.fixtures.fakes import FakeBroker
from trading_system.domain.enums import OrderSide, OrderStatus, OrderType, SignalDirection
from trading_system.domain.models import BrokerOrder, OrderIntent, StrategySignal
from trading_system.execution.engine import ExecutionEngine
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.storage.models import BrokerOrderModel, StrategySignalModel
from trading_system.storage.repositories import OrderRepository, SignalRepository, StateRepository


def _broker_order(broker_id: str | None, client_order_id: str, **overrides: Any) -> BrokerOrder:
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
    """Guard. The original bot (Money_bot) saved OrderSide.BUY as 'buy' and reloaded
    it as SELL after a restart, which flipped its take-profit and stop-loss. Every
    value must come back as the same member."""
    combos = list(itertools.product(OrderSide, OrderType, OrderStatus))
    repo = OrderRepository(session_factory)
    for i, (side, type_, status) in enumerate(combos):
        repo.save_broker_order(
            _broker_order(f"b{i}", f"c{i}", side=side, type=type_, status=status)
        )
    with session_factory.begin() as session:
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

    with session_factory() as fresh:
        for i, (side, type_, status) in enumerate(combos):
            row = fresh.get(BrokerOrderModel, f"c{i}")
            assert (row.side, row.type, row.status) == (side, type_, status)
        for i, direction in enumerate(SignalDirection):
            assert fresh.get(StrategySignalModel, f"s{i}").direction == direction


def test_pause_entries_survives_process_exit(settings: Any, session_factory: Any) -> None:
    """C4. The kill switch must still be on when a different process looks."""
    Orchestrator(settings).pause_entries()

    # session_factory is a separate engine on the same file: another process.
    assert StateRepository(session_factory).get_state().is_trading_halted is True


def test_emergency_halt_survives_process_exit(settings: Any, session_factory: Any) -> None:
    """C4. An emergency halt must also outlive the process that raised it."""
    engine = ExecutionEngine(
        FakeBroker(),
        OrderRepository(session_factory),
        StateRepository(session_factory),
        settings,
    )
    engine.halt_trading("unprotected position")

    state = StateRepository(session_factory).get_state()
    assert state.is_trading_halted is True
    assert state.halt_reason == "unprotected position"


def test_reading_state_does_not_block_other_writers(session_factory: Any, signal: Any) -> None:
    """C5. A plain read of the state must not leave a write lock behind."""
    StateRepository(session_factory).get_state()

    assert SignalRepository(session_factory).save_signal(signal) is True


def test_a_failed_write_does_not_break_the_next_one(session_factory: Any) -> None:
    """A rejected write rolls back cleanly. Before, the failed flush left the
    long-lived order session unusable (PendingRollbackError) for the rest of the run."""
    repo = OrderRepository(session_factory)
    repo.save_broker_order(_broker_order("b1", "client-1"))

    with pytest.raises(IntegrityError):  # broker id b1 already belongs to client-1
        repo.save_broker_order(_broker_order("b1", "client-2"))

    repo.save_broker_order(_broker_order("b3", "client-3"))
    with session_factory() as fresh:
        assert {r.client_order_id for r in fresh.query(BrokerOrderModel)} == {
            "client-1",
            "client-3",
        }


def test_broker_confirmation_replaces_the_inflight_placeholder(session_factory: Any) -> None:
    """C6. The placeholder (no broker id yet) and the broker's confirmation are the same
    order, so saving the confirmation updates the row instead of clashing with it."""
    repo = OrderRepository(session_factory)
    repo.save_broker_order(_broker_order(None, "exec_abc", status=OrderStatus.SUBMITTING))
    repo.save_broker_order(_broker_order("alpaca-uuid-1", "exec_abc"))

    with session_factory() as fresh:
        (row,) = fresh.query(BrokerOrderModel).all()
        assert (row.client_order_id, row.broker_id) == ("exec_abc", "alpaca-uuid-1")
        assert row.status == OrderStatus.SUBMITTED


def test_two_unconfirmed_orders_can_coexist(session_factory: Any) -> None:
    """Several placeholders have no broker id at the same time; NULLs are not duplicates."""
    repo = OrderRepository(session_factory)
    repo.save_broker_order(_broker_order(None, "exec_a", status=OrderStatus.SUBMITTING))
    repo.save_broker_order(_broker_order(None, "exec_b", status=OrderStatus.SUBMITTING))

    with session_factory() as fresh:
        assert fresh.query(BrokerOrderModel).count() == 2


def test_successful_live_submission_does_not_halt_trading(
    settings: Any, session_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6. Before, the confirmation clashed with the placeholder, the order was treated as
    failed, reconcile failed the same way and trading was halted while the order was live."""
    monkeypatch.setattr("time.sleep", lambda _s: None)  # only used on the failure path
    broker = FakeBroker()
    signal = StrategySignal(
        version="1",
        symbol="AAPL",
        timeframe="15m",
        closed_bar_timestamp=datetime.now(timezone.utc),
        direction=SignalDirection.LONG,
    )
    assert SignalRepository(session_factory).save_signal(signal)  # order_intents.signal_id is a FK
    state_repo = StateRepository(session_factory)
    engine = ExecutionEngine(
        broker, OrderRepository(session_factory), state_repo, settings, is_dry_run=False
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
    # One row, keyed by our client id, now carrying the broker's id and the confirmed status.
    with session_factory() as fresh:
        (row,) = fresh.query(BrokerOrderModel).all()
        assert row.client_order_id == order.client_order_id
        assert row.broker_id == broker.submitted[0].id
        assert row.status == OrderStatus.SUBMITTED
