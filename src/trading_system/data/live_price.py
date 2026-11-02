"""The price an entry is planned around: the last trade, not the last bar close (task 1.3a).

The strategy works on closed 15-minute bars, so the close it sees is up to 15 minutes old.
The entry limit, the stop and the target are measured from the price the risk engine is
given, so that price has to be today's, not the bar's. A trade price is accepted only if it
is fresh and does not stray from the last bar; otherwise there is no entry (fail closed):
a stale or odd print would put the whole bracket in the wrong place.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

# A trade stamped slightly ahead of this machine's clock is normal; far ahead means the
# clock is wrong, and then no age can be trusted.
MAX_CLOCK_SKEW = timedelta(seconds=5)


class UnusableLivePrice(ValueError):
    """There is no live price that can be traded on; the message says why."""


@dataclass(frozen=True)
class LivePrice:
    price: Decimal
    timestamp: datetime


def check_live_price(
    live: Optional[LivePrice],
    *,
    reference_close: Decimal,
    now: datetime,
    max_age_seconds: float,
    max_deviation_pct: Decimal,
) -> Decimal:
    """The price to plan an entry around, or UnusableLivePrice.

    `reference_close` is the close of the last closed bar. It is the yardstick for an
    outlier (a bad print, or a move so large that the signal is no longer the one that was
    generated), not the price itself."""
    if live is None:
        raise UnusableLivePrice("no trade price was returned")
    if live.price <= 0:
        raise UnusableLivePrice(f"trade price {live.price} is not positive")
    if reference_close <= 0:
        raise UnusableLivePrice(f"last bar close {reference_close} is not positive")

    timestamp = live.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age = now - timestamp
    if age > timedelta(seconds=max_age_seconds):
        raise UnusableLivePrice(
            f"last trade is {age.total_seconds():.0f}s old (limit {max_age_seconds:.0f}s)"
        )
    if age < -MAX_CLOCK_SKEW:
        raise UnusableLivePrice(
            f"last trade is stamped {-age.total_seconds():.0f}s in the future (clock skew?)"
        )

    deviation = abs(live.price - reference_close) / reference_close
    if deviation > max_deviation_pct:
        raise UnusableLivePrice(
            f"trade price {live.price} is {deviation:.2%} from the last bar close "
            f"{reference_close} (limit {max_deviation_pct:.2%})"
        )
    return live.price
