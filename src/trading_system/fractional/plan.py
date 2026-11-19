"""Sizing and prices of the fractional paper trial (docs/fractional-paper-trial.md, section 2).

Pure arithmetic, no I/O. The limits are the live loop's own percentages (`Settings`), applied to a
VIRTUAL capital, because the second paper account holds a million dollars and the broker will not
limit the bot to 500: only this code does. So there is also a hard ceiling on one order that
refuses instead of sizing down, and that nothing but a bug can reach.
"""

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_FLOOR, Decimal
from typing import Any, Iterable

from trading_system.risk.entry_plan import (
    MIN_LEG_DISTANCE,
    InvalidEntryPlan,
    build_entry_plan,
    price_tick,
)

# Alpaca takes fractional quantities to nine decimals; six is a hundredth of a cent on a 100-dollar stock.
QUANTITY_STEP = Decimal("0.000001")
# The least Alpaca accepts for a fractional order.
MIN_NOTIONAL = Decimal("1")
# Quantities that differ by less than this are the same quantity (a stop that covers a position
# to within a millionth of a share covers it).
QUANTITY_TOLERANCE = Decimal("0.000001")

# The order statuses of a stop that is standing (or queued): `held` is the stop leg of a bracket whose
# entry has filled, kept for the same reason as in execution/engine.py.
LIVE_STOP_STATUSES = frozenset({"new", "accepted", "pending_new", "held"})


class Refused(ValueError):
    """The entry is not made; the message says why. It is a decision, not an error."""


class HardCapExceeded(Refused):
    """The order is above the hard ceiling. Sizing makes this impossible, so it means a bug: the
    cycle raises the alarm instead of just logging a refusal."""


@dataclass(frozen=True)
class Sized:
    quantity: Decimal
    limit_price: Decimal
    stop_price: Decimal  # of the plan; the real stop is placed from the fill (stop_price_for)
    take_profit_price: (
        Decimal  # of the plan; the real target is kept from the fill (target_price_for)
    )
    notional: Decimal  # quantity x limit, the most the entry can cost
    planned_risk: Decimal  # quantity x (limit - stop)


def virtual_equity(budget: Decimal, equity_now: Decimal, baseline: Decimal) -> Decimal:
    """The budget, moved by what this account has made or lost since the trial began."""
    return budget + (equity_now - baseline)


def floor_quantity(quantity: Decimal) -> Decimal:
    return quantity.quantize(QUANTITY_STEP, rounding=ROUND_DOWN)


def size_entry(
    *,
    equity: Decimal,
    reference_price: Decimal,
    open_positions: int,
    open_gross: Decimal,
    open_risk: Decimal,
    max_positions: int,
    position_size_pct: Decimal,
    max_gross_pct: Decimal,
    max_trade_risk_pct: Decimal,
    max_total_risk_pct: Decimal,
    limit_buffer_pct: Decimal,
    stop_loss_pct: Decimal,
    take_profit_pct: Decimal,
    max_order_notional: Decimal,
) -> Sized:
    """The quantity and prices of one entry, or `Refused`.

    `equity` is the VIRTUAL capital. `open_gross` is the value of the positions plus the pending
    entries; `open_risk` their planned risk (entry to stop). The order of the checks follows
    risk/engine.py so that the two read alike; the one difference is that the quantity is
    fractional (rounded down to a millionth of a share) instead of whole shares."""
    if equity <= 0:
        raise Refused("The virtual capital is not positive.")
    if open_positions >= max_positions:
        raise Refused("Max open positions limit reached.")
    try:
        plan = build_entry_plan(
            reference_price,
            limit_buffer_pct=limit_buffer_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    except InvalidEntryPlan as e:
        raise Refused(f"Entry plan rejected: {e}") from e

    available_gross = equity * max_gross_pct - open_gross
    if available_gross <= 0:
        raise Refused("Max gross exposure limit reached.")
    notional = min(equity * position_size_pct, available_gross)
    quantity = floor_quantity(notional / plan.limit_price)

    risk_per_share = (
        plan.limit_price - plan.stop_price
    )  # positive: the plan guarantees stop < limit
    max_trade_risk = equity * max_trade_risk_pct
    if quantity * risk_per_share > max_trade_risk:
        quantity = floor_quantity(max_trade_risk / risk_per_share)

    available_risk = equity * max_total_risk_pct - open_risk
    if available_risk <= 0:
        raise Refused("Max total planned open risk limit reached.")
    if quantity * risk_per_share > available_risk:
        quantity = floor_quantity(available_risk / risk_per_share)

    cost = quantity * plan.limit_price
    if cost < MIN_NOTIONAL:
        raise Refused(f"The order would be worth {cost:.2f} dollars, under the 1 dollar minimum.")
    if cost > max_order_notional:
        raise HardCapExceeded(
            f"The order would cost {cost:.2f} dollars, above the hard ceiling of {max_order_notional}."
        )
    return Sized(
        quantity=quantity,
        limit_price=plan.limit_price,
        stop_price=plan.stop_price,
        take_profit_price=plan.take_profit_price,
        notional=cost,
        planned_risk=quantity * risk_per_share,
    )


def stop_price_for(average_entry: Decimal, stop_loss_pct: Decimal) -> Decimal:
    """The stop of a position: the stop percentage below the REAL average entry price (not the limit,
    which the live plan measures from), rounded down to a tick. Raises `Refused` if the stop is not a
    usable price."""
    tick = price_tick(average_entry)
    stop = ((average_entry * (1 - stop_loss_pct)) / tick).to_integral_value(
        rounding=ROUND_FLOOR
    ) * tick
    if stop <= 0 or average_entry - stop < MIN_LEG_DISTANCE:
        raise Refused(f"A stop of {stop} under an entry of {average_entry} is not usable.")
    return stop


def target_price_for(average_entry: Decimal, take_profit_pct: Decimal) -> Decimal:
    """The price at which the software sells (no exchange order: a stop and a target cannot share one
    quantity)."""
    return average_entry * (1 + take_profit_pct)


def _text(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _is_live_stop(order: Any) -> bool:
    return (
        _text(order.side) == "sell"
        and _text(order.type) in ("stop", "stop_limit")
        and _text(order.status) in LIVE_STOP_STATUSES
    )


def stop_cover(orders: Iterable[Any]) -> dict[str, Decimal]:
    """Per symbol, the quantity that standing stop-sell orders cover. Orders from `get_open_orders`
    (nested), so the stop legs of a bracket are found as well; each order is counted once."""
    seen: set[str] = set()
    cover: dict[str, Decimal] = {}

    def visit(order: Any) -> None:
        key = str(getattr(order, "id", id(order)))
        if key in seen:
            return
        seen.add(key)
        if _is_live_stop(order) and order.qty is not None:
            cover[order.symbol] = cover.get(order.symbol, Decimal(0)) + Decimal(str(order.qty))
        for leg in getattr(order, "legs", None) or []:
            visit(leg)

    for order in orders:
        visit(order)
    return cover


def live_stops_of(orders: Iterable[Any], symbol: str) -> list[Any]:
    """The standing stop-sell orders of one symbol (to cancel before a market sale)."""
    found: list[Any] = []
    seen: set[str] = set()

    def visit(order: Any) -> None:
        key = str(getattr(order, "id", id(order)))
        if key in seen:
            return
        seen.add(key)
        if order.symbol == symbol and _is_live_stop(order):
            found.append(order)
        for leg in getattr(order, "legs", None) or []:
            visit(leg)

    for order in orders:
        visit(order)
    return found
