"""The entry plan: limit, stop and take-profit prices of one long bracket order (C3).

Pure arithmetic on a reference price. The risk engine sizes the position with exactly these
prices and execution sends them unchanged, so the risk that was planned is the risk that is
placed. Before, risk emitted a MARKET intent without prices and execution refused it.
"""

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

# US equity tick sizes: a cent from $1 upward, a hundredth of a cent below.
_TICK_FROM_ONE_DOLLAR = Decimal("0.01")
_TICK_BELOW_ONE_DOLLAR = Decimal("0.0001")

# The stop and the target of a bracket must differ from the entry by at least a cent
# (Alpaca rejects the order otherwise), which is more than a tick for stocks under $1.
MIN_LEG_DISTANCE = Decimal("0.01")


class InvalidEntryPlan(ValueError):
    """The prices cannot form a valid bracket; the message says why."""


@dataclass(frozen=True)
class EntryPlan:
    limit_price: Decimal
    stop_price: Decimal
    take_profit_price: Decimal


@dataclass(frozen=True)
class AtrExit:
    """Stop and target as multiples of the ATR (a CANDIDATE, docs/strategy-research-plan.md, section
    13.1). The live path does not read it: it plans with `stop_loss_pct` and `take_profit_pct`."""

    stop_atr: Decimal
    target_atr: Decimal


def atr_exit_percentages(atr: float, close: float, rule: AtrExit) -> tuple[Decimal, Decimal]:
    """(stop %, target %) as fractions of the price for an ATR exit: the multiples of the ATR as a
    share of the signal bar's close, ready for `build_entry_plan`. Raises InvalidEntryPlan for an ATR
    or a close that is not a usable positive number, or a stop of 100 % or more."""
    if not (math.isfinite(atr) and math.isfinite(close)) or atr <= 0 or close <= 0:
        raise InvalidEntryPlan(f"ATR {atr} and close {close} must be positive numbers.")
    fraction = Decimal(str(atr)) / Decimal(str(close))
    stop, target = rule.stop_atr * fraction, rule.target_atr * fraction
    if stop >= 1:
        raise InvalidEntryPlan(f"An ATR stop of {stop:.0%} of the price is not a stop.")
    return stop, target


def price_tick(price: Decimal) -> Decimal:
    return _TICK_FROM_ONE_DOLLAR if price >= 1 else _TICK_BELOW_ONE_DOLLAR


def _round_to_tick(price: Decimal, rounding: str) -> Decimal:
    tick = price_tick(price)
    return (price / tick).to_integral_value(rounding=rounding) * tick


def build_entry_plan(
    reference_price: Decimal,
    *,
    limit_buffer_pct: Decimal,
    stop_loss_pct: Decimal,
    take_profit_pct: Decimal,
) -> EntryPlan:
    """Prices for a long bracket order around `reference_price`.

    The limit sits `limit_buffer_pct` above the reference: a buy limit above the ask fills at
    the ask, so the buffer caps the price paid instead of costing anything, and it keeps the
    order from going stale when the price ticks up. Stop and target are measured from the
    limit, the worst price the entry can fill at.

    Rounding: the limit to the nearest tick, the stop and the target down. Raises
    InvalidEntryPlan unless 0 < stop < limit < target, each at least MIN_LEG_DISTANCE apart.
    """
    if reference_price <= 0:
        raise InvalidEntryPlan(f"Reference price {reference_price} is not positive.")

    limit = _round_to_tick(reference_price * (1 + limit_buffer_pct), ROUND_HALF_UP)
    stop = _round_to_tick(limit * (1 - stop_loss_pct), ROUND_FLOOR)
    target = _round_to_tick(limit * (1 + take_profit_pct), ROUND_FLOOR)

    if limit <= 0 or stop <= 0:
        raise InvalidEntryPlan(f"Limit {limit} and stop {stop} must be positive.")
    if limit - stop < MIN_LEG_DISTANCE:
        raise InvalidEntryPlan(
            f"Stop {stop} is less than {MIN_LEG_DISTANCE} below the limit {limit}."
        )
    if target - limit < MIN_LEG_DISTANCE:
        raise InvalidEntryPlan(
            f"Take-profit {target} is less than {MIN_LEG_DISTANCE} above the limit {limit}."
        )
    return EntryPlan(limit_price=limit, stop_price=stop, take_profit_price=target)
