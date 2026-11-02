"""Keeping the database in line with the broker: order states and fills.

The database is a record of what we asked for; Alpaca decides what happened. Orders change
without us (a limit entry fills, a stop triggers, a GTC entry is cancelled), so these two
functions pull the facts back: `reconcile_orders` refreshes every order that can still
change, `record_fills` stores the executions. Both run when the system starts, and again on
every heartbeat of the runner.
"""

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from alpaca.common.exceptions import APIError

from trading_system.domain.enums import OrderStatus
from trading_system.domain.models import BrokerOrder

logger = logging.getLogger(__name__)

# An order the broker has never heard of is only declared "never sent" when its record is
# older than this: a younger one may still be on its way out.
NEVER_SENT_AFTER = dt.timedelta(seconds=60)

# Orders that exist only here: the broker id of a dry-run order starts with this.
_DRY_RUN_PREFIX = "dry_run_"

_NOT_YET_SENT = (OrderStatus.CREATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMITTING)


@dataclass
class OrderReconciliation:
    checked: int = 0
    changed: int = 0  # status or filled quantity differs from what the database had
    never_sent: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # lookup errors; tried again next time


def _is_not_found(error: Exception) -> bool:
    return isinstance(error, APIError) and error.status_code == 404


def reconcile_orders(
    broker: Any,
    execution: Any,
    order_repo: Any,
    now: Optional[dt.datetime] = None,
) -> OrderReconciliation:
    """Refreshes every order that may still change from the broker's record of it.

    An order the broker does not know is marked CANCELED if we had not sent it yet (the
    process died before the request went out), and UNKNOWN_REQUIRES_RECONCILIATION if we had.
    Dry-run orders are skipped: they never existed at the broker."""
    now = now or dt.datetime.now(dt.timezone.utc)
    report = OrderReconciliation()
    for order in order_repo.list_unsettled_orders():
        if (order.broker_id or "").startswith(_DRY_RUN_PREFIX):
            continue
        report.checked += 1
        try:
            alpaca_order = broker.get_order_by_client_id(order.client_order_id)
        except Exception as e:
            if _is_not_found(e):
                _handle_unknown_order(order, order_repo, report, now)
            else:
                logger.warning(f"Could not look up order {order.client_order_id}: {e}")
                report.failed.append(order.client_order_id)
            continue
        try:
            fresh = execution.store_broker_order(alpaca_order, order.client_order_id)
        except Exception as e:
            logger.error(f"Could not store order {order.client_order_id}: {e}")
            report.failed.append(order.client_order_id)
            continue
        if (fresh.status, fresh.filled_quantity) != (order.status, order.filled_quantity):
            report.changed += 1
            logger.info(
                f"Order {order.client_order_id} {order.symbol}: {order.status.value} "
                f"-> {fresh.status.value}, filled {fresh.filled_quantity}/{fresh.quantity}"
            )
    return report


def _handle_unknown_order(
    order: BrokerOrder, order_repo: Any, report: OrderReconciliation, now: dt.datetime
) -> None:
    updated = order.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=dt.timezone.utc)
    if now - updated < NEVER_SENT_AFTER:
        return  # perhaps still in flight; the next pass decides
    if order.status in _NOT_YET_SENT:
        new_status = OrderStatus.CANCELED
        logger.warning(f"Order {order.client_order_id} was never sent; marking it CANCELED.")
    else:
        new_status = OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION
        logger.error(
            f"Order {order.client_order_id} was sent ({order.status.value}) but the broker "
            "does not know it."
        )
    if new_status is not order.status:
        order_repo.save_broker_order(order.model_copy(update={"status": new_status}))
        report.never_sent.append(order.client_order_id)


def record_fills(broker: Any, order_repo: Any, after: dt.datetime) -> tuple[int, dt.datetime]:
    """Stores the executions since `after` and returns (how many were new, where to resume).

    Alpaca lists the whole account's fills; those of orders we do not have are ignored
    (save_fill refuses them), and one already stored is not stored twice."""
    saved = 0
    resume = after
    for row in broker.get_fills(after):
        if row.get("type") not in ("fill", "partial_fill"):
            continue
        moment = dt.datetime.fromisoformat(row["transaction_time"])
        if order_repo.save_fill(
            row["order_id"], row["id"], float(row["price"]), float(row["qty"]), moment
        ):
            saved += 1
            logger.info(f"Fill {row['symbol']} {row['side']} {row['qty']} @ {row['price']}")
        # A second of overlap: a fill stamped at the boundary must not be missed.
        resume = max(resume, moment - dt.timedelta(seconds=1))
    return saved, resume
