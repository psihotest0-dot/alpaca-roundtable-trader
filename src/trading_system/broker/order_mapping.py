"""Alpaca orders as the domain sees them.

One table for statuses, sides and types, and one conversion to BrokerOrder. The same code
used to exist three times (order submission, reconciliation, the orchestrator's list of open
orders) and the copies disagreed: the open-orders copy turned every stop order into MARKET.
"""

from decimal import Decimal
from typing import Any, Optional

from trading_system.domain.enums import OrderSide, OrderStatus, OrderType
from trading_system.domain.models import BrokerOrder

# Alpaca's order statuses. Anything not listed (stopped, suspended, calculated,
# pending_review, or a status Alpaca adds later) is UNKNOWN_REQUIRES_RECONCILIATION.
_STATUS = {
    "pending_new": OrderStatus.SUBMITTING,
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "pending_cancel": OrderStatus.SUBMITTED,  # still live until the cancel goes through
    "pending_replace": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,  # superseded by another order
    "expired": OrderStatus.EXPIRED,
    "done_for_day": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    # A bracket's exit leg waits in `held` until it is its turn.
    "held": OrderStatus.CREATED,
}

_TYPE = {
    "market": OrderType.MARKET,
    "limit": OrderType.LIMIT,
    "stop": OrderType.STOP,
    "stop_limit": OrderType.STOP_LIMIT,
    "trailing_stop": OrderType.TRAILING_STOP,
}


def _text(value: Any) -> str:
    """ "filled" for OrderStatus.FILLED as well as for "filled" or "FILLED"."""
    return str(getattr(value, "value", value)).lower()


def map_status(value: Any) -> OrderStatus:
    return _STATUS.get(_text(value), OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION)


def map_side(value: Any) -> OrderSide:
    return OrderSide.BUY if _text(value) == "buy" else OrderSide.SELL


def map_type(value: Any) -> OrderType:
    try:
        return _TYPE[_text(value)]
    except KeyError:
        raise ValueError(f"Unknown order type {value!r}") from None


def _decimal(value: Any) -> Optional[Decimal]:
    return None if value is None or value == "" else Decimal(str(value))


def to_broker_order(order: Any, client_order_id: Optional[str] = None) -> BrokerOrder:
    """The domain record of an Alpaca order (the order itself, not its legs).

    `client_order_id` overrides the one on the order; reconciliation asks for an order by
    our id and keeps that id."""
    return BrokerOrder(
        broker_id=str(order.id),
        client_order_id=client_order_id or order.client_order_id,
        symbol=order.symbol,
        side=map_side(order.side),
        type=map_type(order.order_type),
        status=map_status(order.status),
        quantity=_decimal(order.qty) or Decimal(0),  # a notional order has no qty
        filled_quantity=_decimal(order.filled_qty) or Decimal(0),
        limit_price=_decimal(order.limit_price),
        stop_price=_decimal(order.stop_price),
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


def legs_of(order: Any) -> list[Any]:
    """The child orders of a bracket, as returned with nested=True or by order id."""
    return list(getattr(order, "legs", None) or [])
