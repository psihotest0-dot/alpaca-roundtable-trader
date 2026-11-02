"""When the trading cycle runs: once per closed 15-minute bar, not once per minute.

The bars change every 15 minutes, so the old loop that fetched them every minute asked for
the same data 14 times out of 15. These are the boundaries it is now tied to.
"""

from datetime import datetime, timedelta, timezone

from trading_system.data.alpaca_data import FIFTEEN_MINUTES

BAR = timedelta(minutes=FIFTEEN_MINUTES.amount_value)


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def bar_boundary(now: datetime) -> datetime:
    """The latest bar boundary: `now` rounded down to a multiple of the bar length (UTC; the
    US exchange's offset from UTC is a whole number of hours, so its boundaries agree). The bar
    that ended there is the newest closed bar."""
    step = int(BAR.total_seconds())
    seconds = int(_utc(now).timestamp())
    return datetime.fromtimestamp(seconds - seconds % step, timezone.utc)


def expected_bar_start(now: datetime) -> datetime:
    """The start time (Alpaca stamps bars with it) of the newest closed bar at `now`."""
    return bar_boundary(now) - BAR
