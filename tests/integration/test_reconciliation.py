"""Orders and fills pulled back from the broker into a real SQLite database (task 1.7)."""

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any

import pytest
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Order

from trading_system.domain.enums import OrderSide, OrderStatus, OrderType
from trading_system.domain.models import BrokerOrder
from trading_system.execution.engine import ExecutionEngine
from trading_system.runtime.reconciliation import record_fills, reconcile_orders
from trading_system.storage.models import BrokerOrderModel, FillModel
from trading_system.storage.repositories import OrderRepository, StateRepository

NOW = dt.datetime(2026, 11, 2, 16, 0, tzinfo=dt.timezone.utc)
OLD = NOW - dt.timedelta(minutes=10)


def _not_found() -> APIError:
    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError(response=response)
    return APIError('{"code": 40410000, "message": "order not found"}', error)


def _alpaca_order(
    client_order_id: str,
    status: str,
    *,
    order_type: str = "limit",
    side: str = "buy",
    qty: str = "10",
    filled: str = "0",
    legs: list[Order] | None = None,
) -> Order:
    now = dt.datetime.now(dt.timezone.utc)
    return Order(
        id=uuid.uuid4(),
        client_order_id=client_order_id,
        created_at=now,
        updated_at=now,
        submitted_at=now,
        asset_id=uuid.uuid4(),
        symbol="AAPL",
        asset_class="us_equity",
        order_class="bracket",
        time_in_force="gtc",
        extended_hours=False,
        order_type=order_type,
        type=order_type,
        side=side,
        status=status,
        qty=qty,
        filled_qty=filled,
        legs=legs,
        limit_price="150.00",
    )


class _Broker:
    """What reconciliation asks of the broker: orders by client id, and fills."""

    def __init__(self) -> None:
        self.orders: dict[str, Order] = {}
        self.errors: dict[str, Exception] = {}
        self.fills: list[dict[str, Any]] = []
        self.lookups: list[str] = []

    def get_order_by_client_id(self, client_order_id: str) -> Order:
        self.lookups.append(client_order_id)
        if client_order_id in self.errors:
            raise self.errors[client_order_id]
        if client_order_id not in self.orders:
            raise _not_found()
        return self.orders[client_order_id]

    def get_fills(self, after: dt.datetime) -> list[dict[str, Any]]:
        return [f for f in self.fills if dt.datetime.fromisoformat(f["transaction_time"]) > after]


@pytest.fixture
def repo(session_factory: Any) -> OrderRepository:
    return OrderRepository(session_factory)


@pytest.fixture
def execution(repo: OrderRepository, session_factory: Any, settings: Any) -> ExecutionEngine:
    # Reconciliation only uses the engine to store what the broker reports.
    return ExecutionEngine(_Broker(), repo, StateRepository(session_factory), settings)  # type: ignore[arg-type]


def _db_order(
    client_order_id: str,
    status: OrderStatus,
    *,
    broker_id: str | None = None,
    updated: dt.datetime = OLD,
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_id=broker_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        status=status,
        quantity=Decimal(10),
        filled_quantity=Decimal(0),
        created_at=OLD,
        updated_at=updated,
    )


def _status_of(session_factory: Any, client_order_id: str) -> OrderStatus:
    with session_factory() as session:
        row = session.get(BrokerOrderModel, client_order_id)
        return row.status


def test_an_entry_that_filled_while_we_were_away_is_updated_with_its_legs(
    repo: OrderRepository, execution: ExecutionEngine, session_factory: Any
) -> None:
    repo.save_broker_order(_db_order("exec_a", OrderStatus.SUBMITTED, broker_id="b-1"))
    broker = _Broker()
    broker.orders["exec_a"] = _alpaca_order(
        "exec_a",
        "filled",
        filled="10",
        legs=[
            _alpaca_order("leg-tp", "new", order_type="limit", side="sell"),
            _alpaca_order("leg-stop", "held", order_type="stop", side="sell"),
        ],
    )

    report = reconcile_orders(broker, execution, repo, NOW)

    assert (report.checked, report.changed) == (1, 1)
    assert _status_of(session_factory, "exec_a") is OrderStatus.FILLED
    assert _status_of(session_factory, "leg-tp") is OrderStatus.SUBMITTED
    assert _status_of(session_factory, "leg-stop") is OrderStatus.CREATED


def test_finished_orders_are_not_asked_about_again(
    repo: OrderRepository, execution: ExecutionEngine
) -> None:
    final = (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED)
    for i, status in enumerate(final):
        repo.save_broker_order(_db_order(f"exec_done_{i}", status, broker_id=f"b-{i}"))
    broker = _Broker()

    report = reconcile_orders(broker, execution, repo, NOW)

    assert report.checked == 0 and broker.lookups == []


def test_a_leg_that_fills_later_is_followed_through_its_own_id(
    repo: OrderRepository, execution: ExecutionEngine, session_factory: Any
) -> None:
    """The entry is final (FILLED), its take-profit leg is not: the leg is what changes."""
    repo.save_broker_order(_db_order("exec_a", OrderStatus.FILLED, broker_id="b-1"))
    repo.save_broker_order(_db_order("leg-tp", OrderStatus.SUBMITTED, broker_id="b-2"))
    broker = _Broker()
    broker.orders["leg-tp"] = _alpaca_order(
        "leg-tp", "filled", order_type="limit", side="sell", filled="10"
    )

    reconcile_orders(broker, execution, repo, NOW)

    assert broker.lookups == ["leg-tp"]
    assert _status_of(session_factory, "leg-tp") is OrderStatus.FILLED


def test_an_order_that_was_never_sent_is_cancelled_once_it_is_old_enough(
    repo: OrderRepository, execution: ExecutionEngine, session_factory: Any
) -> None:
    repo.save_broker_order(_db_order("exec_old", OrderStatus.SUBMITTING, updated=OLD))
    young = NOW - dt.timedelta(seconds=10)
    repo.save_broker_order(_db_order("exec_young", OrderStatus.SUBMITTING, updated=young))

    report = reconcile_orders(_Broker(), execution, repo, NOW)

    assert _status_of(session_factory, "exec_old") is OrderStatus.CANCELED
    assert _status_of(session_factory, "exec_young") is OrderStatus.SUBMITTING  # maybe in flight
    assert report.never_sent == ["exec_old"]


def test_a_sent_order_the_broker_does_not_know_needs_a_human(
    repo: OrderRepository, execution: ExecutionEngine, session_factory: Any
) -> None:
    repo.save_broker_order(_db_order("exec_ghost", OrderStatus.SUBMITTED, broker_id="b-9"))

    reconcile_orders(_Broker(), execution, repo, NOW)

    assert _status_of(session_factory, "exec_ghost") is OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION


def test_dry_run_orders_are_left_alone(
    repo: OrderRepository, execution: ExecutionEngine, session_factory: Any
) -> None:
    simulated = _db_order("exec_sim", OrderStatus.SUBMITTED, broker_id="dry_run_exec_sim")
    repo.save_broker_order(simulated)
    broker = _Broker()

    reconcile_orders(broker, execution, repo, NOW)

    assert broker.lookups == []
    assert _status_of(session_factory, "exec_sim") is OrderStatus.SUBMITTED


def test_a_failed_lookup_changes_nothing_and_is_reported(
    repo: OrderRepository, execution: ExecutionEngine, session_factory: Any
) -> None:
    repo.save_broker_order(_db_order("exec_a", OrderStatus.SUBMITTING))
    broker = _Broker()
    broker.errors["exec_a"] = ConnectionError("down")

    report = reconcile_orders(broker, execution, repo, NOW)

    assert report.failed == ["exec_a"]
    assert _status_of(session_factory, "exec_a") is OrderStatus.SUBMITTING


def _fill(
    execution_id: str, order_id: str, moment: dt.datetime, qty: str = "4", kind: str = "fill"
) -> dict[str, Any]:
    return {
        "id": execution_id,
        "order_id": order_id,
        "activity_type": "FILL",
        "type": kind,
        "price": "150.25",
        "qty": qty,
        "side": "buy",
        "symbol": "AAPL",
        "transaction_time": moment.isoformat().replace("+00:00", "Z"),
    }


def test_fills_of_our_orders_are_stored_once(repo: OrderRepository, session_factory: Any) -> None:
    repo.save_broker_order(_db_order("exec_a", OrderStatus.PARTIALLY_FILLED, broker_id="b-1"))
    broker = _Broker()
    first, second = NOW - dt.timedelta(minutes=2), NOW - dt.timedelta(minutes=1)
    broker.fills = [_fill("f-1", "b-1", first), _fill("f-2", "b-1", second, qty="6")]
    since = NOW - dt.timedelta(hours=1)

    saved, resume = record_fills(broker, repo, since)
    again, _ = record_fills(broker, repo, since)

    assert (saved, again) == (2, 0)
    assert resume == second - dt.timedelta(seconds=1)
    with session_factory() as session:
        quantities = sorted(Decimal(f.quantity) for f in session.query(FillModel).all())
    assert quantities == [Decimal(4), Decimal(6)]


def test_fills_of_other_orders_and_other_activity_are_ignored(
    repo: OrderRepository, session_factory: Any
) -> None:
    repo.save_broker_order(_db_order("exec_a", OrderStatus.FILLED, broker_id="b-1"))
    broker = _Broker()
    moment = NOW - dt.timedelta(minutes=1)
    broker.fills = [
        _fill("f-foreign", "someone-elses-order", moment),
        _fill("f-cancel", "b-1", moment, kind="canceled"),
    ]

    saved, _ = record_fills(broker, repo, NOW - dt.timedelta(hours=1))

    assert saved == 0
    with session_factory() as session:
        assert session.query(FillModel).count() == 0
