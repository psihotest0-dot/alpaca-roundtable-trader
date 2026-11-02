import datetime
import uuid
from decimal import Decimal
from types import SimpleNamespace

import pytest
from alpaca.trading.enums import OrderStatus as AlpacaStatus
from alpaca.trading.models import Order

from trading_system.broker.order_mapping import (
    legs_of,
    map_side,
    map_status,
    map_type,
    to_broker_order,
)
from trading_system.domain.enums import OrderSide, OrderStatus, OrderType


def _order(order_type="limit", side="buy", status="new", **fields) -> Order:
    now = datetime.datetime.now(datetime.timezone.utc)
    values = dict(
        id=uuid.uuid4(),
        client_order_id="exec_abc",
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
        qty="10",
        filled_qty="0",
    )
    values.update(fields)
    return Order(**values)


@pytest.mark.parametrize(
    "alpaca, domain",
    [
        ("pending_new", OrderStatus.SUBMITTING),
        ("new", OrderStatus.SUBMITTED),
        ("accepted", OrderStatus.SUBMITTED),
        ("pending_cancel", OrderStatus.SUBMITTED),
        ("partially_filled", OrderStatus.PARTIALLY_FILLED),
        ("filled", OrderStatus.FILLED),
        ("canceled", OrderStatus.CANCELED),
        ("expired", OrderStatus.EXPIRED),
        ("done_for_day", OrderStatus.EXPIRED),
        ("rejected", OrderStatus.REJECTED),
        ("held", OrderStatus.CREATED),
        ("suspended", OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION),
        ("something_alpaca_adds_later", OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION),
    ],
)
def test_statuses_map_the_same_from_strings_and_alpaca_enums(alpaca, domain):
    assert map_status(alpaca) is domain
    assert map_status(alpaca.upper()) is domain
    if alpaca in {s.value for s in AlpacaStatus}:
        assert map_status(AlpacaStatus(alpaca)) is domain


def test_every_status_alpaca_defines_maps_to_something():
    """A new alpaca-py status must not crash the mapping."""
    for status in AlpacaStatus:
        assert isinstance(map_status(status), OrderStatus)


def test_sides_and_types():
    assert map_side("buy") is OrderSide.BUY
    assert map_side("SELL") is OrderSide.SELL
    assert [map_type(t) for t in ("market", "limit", "stop", "stop_limit", "trailing_stop")] == [
        OrderType.MARKET,
        OrderType.LIMIT,
        OrderType.STOP,
        OrderType.STOP_LIMIT,
        OrderType.TRAILING_STOP,
    ]
    with pytest.raises(ValueError, match="Unknown order type"):
        map_type("iceberg")


def test_a_stop_order_stays_a_stop_order():
    """The open-orders copy of this code turned every non-limit order into MARKET."""
    order = to_broker_order(_order("stop", "sell", stop_price="144.00"))

    assert order.type is OrderType.STOP
    assert order.side is OrderSide.SELL
    assert order.stop_price == Decimal("144.00")
    assert order.limit_price is None


def test_a_real_alpaca_order_becomes_a_broker_order():
    alpaca = _order(status="partially_filled", qty="10", filled_qty="4", limit_price="150.25")

    order = to_broker_order(alpaca)

    assert order.broker_id == str(alpaca.id)
    assert order.client_order_id == "exec_abc"
    assert (order.status, order.quantity, order.filled_quantity) == (
        OrderStatus.PARTIALLY_FILLED,
        Decimal(10),
        Decimal(4),
    )
    assert order.limit_price == Decimal("150.25")


def test_our_client_order_id_overrides_the_one_on_the_order():
    assert to_broker_order(_order(), client_order_id="exec_mine").client_order_id == "exec_mine"


def test_a_notional_order_without_qty_has_quantity_zero():
    assert to_broker_order(_order(qty=None)).quantity == Decimal(0)


def test_legs_come_back_as_a_list_whatever_the_order_looks_like():
    parent = _order(legs=[_order("stop", "sell", "held"), _order("limit", "sell", "held")])

    assert len(legs_of(parent)) == 2
    assert legs_of(_order()) == []
    assert legs_of(SimpleNamespace()) == []
