"""How a bar closes a long position: the one rule, used by real and by shadow trades."""

from decimal import Decimal
from typing import Optional, Tuple


def resolve_exit(
    open_px: Decimal,
    high: Decimal,
    low: Decimal,
    stop_loss: Optional[Decimal],
    take_profit: Optional[Decimal],
) -> Optional[Tuple[str, Decimal]]:
    """(reason, price before slippage) if this bar hits the stop or the target, else None.

    A gap fills at the open if it is worse (a stop) or better (a target) than the level. If the
    bar reaches both levels the stop is taken: the bar's inner order is unknown and the
    cautious assumption is the loss."""
    stopped = False
    reason = ""
    price = Decimal("0")

    if stop_loss and low <= stop_loss:
        stopped = True
        reason = "STOP_LOSS"
        price = open_px if open_px < stop_loss else stop_loss

    if take_profit and high >= take_profit:
        if stopped:
            return reason, price
        return "TAKE_PROFIT", open_px if open_px > take_profit else take_profit

    return (reason, price) if stopped else None
