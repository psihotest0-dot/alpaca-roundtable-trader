"""A look at the account against our records, before anyone trusts it (Phase 6.5).

`reconcile` runs it on demand. `resume-entries` runs it before it lifts a halt: a halt is
raised because something was wrong (an unprotected position, an order nobody could account
for), and clearing it without looking again just restarts the thing that was wrong. The check
refreshes the orders from the broker (that writes to our database, like every pass of the
runner), reads the positions and the open orders, and reports. It never trades, never halts
and never closes anything."""

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional

from trading_system.domain.enums import OrderStatus
from trading_system.execution.engine import UnprotectedPosition
from trading_system.runtime.reconciliation import OrderReconciliation, reconcile_orders


@dataclass
class AccountCheck:
    orders: OrderReconciliation = field(default_factory=OrderReconciliation)
    positions: int = 0
    open_orders: int = 0
    unprotected: list[UnprotectedPosition] = field(default_factory=list)
    unresolved: list[tuple[str, str]] = field(default_factory=list)  # (client_order_id, symbol)
    broker_error: Optional[str] = None

    @property
    def problems(self) -> list[str]:
        """Reasons not to resume trading. Empty means the account agrees with our records."""
        found: list[str] = []
        if self.broker_error:
            found.append(f"the broker could not be read: {self.broker_error}")
        for gap in self.unprotected:
            found.append(
                f"{gap.symbol}: {gap.quantity} shares, only {gap.covered} covered by an active stop"
            )
        for order_id, symbol in self.unresolved:
            found.append(f"order {order_id} ({symbol}) was sent, but the broker does not know it")
        if self.orders.failed:
            found.append(f"{len(self.orders.failed)} order(s) could not be looked up")
        return found

    @property
    def ok(self) -> bool:
        return not self.problems


def check_account(
    broker: Any, execution: Any, order_repo: Any, now: dt.datetime | None = None
) -> AccountCheck:
    check = AccountCheck()
    try:
        check.orders = reconcile_orders(broker, execution, order_repo, now)
        positions = broker.get_positions()
        open_orders = broker.get_open_orders()
        check.positions = len(positions)
        check.open_orders = len(open_orders)
        check.unprotected = execution.unprotected_positions(positions, open_orders)
    except Exception as e:  # any failure to read the broker is a reason not to go on
        check.broker_error = f"{type(e).__name__}: {e}"
    check.unresolved = [
        (o.client_order_id, o.symbol)
        for o in order_repo.list_unsettled_orders()
        if o.status is OrderStatus.UNKNOWN_REQUIRES_RECONCILIATION
    ]
    return check


def format_account_check(check: AccountCheck) -> str:
    lines = [
        f"Orders refreshed from the broker: {check.orders.checked} checked, "
        f"{check.orders.changed} changed, {len(check.orders.never_sent)} found never sent",
        f"Positions: {check.positions}, open orders: {check.open_orders}",
    ]
    problems = check.problems
    if problems:
        lines.append("PROBLEMS:")
        lines.extend(f"  - {problem}" for problem in problems)
    else:
        lines.append("No problems: every position is protected and every order is accounted for.")
    return "\n".join(lines)
