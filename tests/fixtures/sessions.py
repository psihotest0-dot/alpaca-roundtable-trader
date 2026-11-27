"""Bar times inside the regular session, for fixtures.

Since 2026-11-27 the backtest and the shadow simulation (`backtest/shadow.py`) take stops and
targets on regular-session bars only, as Alpaca works the bracket legs (GTC, no extended hours).
A fixture whose trades must end by a given bar therefore puts its signals and bars on weekdays
between 09:30 and 16:00 New York time, where the real ones are.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_system.runtime.schedule import regular_session_mask

NEW_YORK = ZoneInfo("America/New_York")
BAR = pd.Timedelta(minutes=15)
LAST_SLOT = 25  # 15:45, the last bar that starts in the session (09:30 is slot 0)


def weekday_open(first: datetime, day: int) -> pd.Timestamp:
    """09:30 New York (in UTC) of the `day`-th weekday counted from the New York date of `first`
    (day 0 is that date, or the next weekday if it falls on a weekend)."""
    start = pd.Timestamp(first).tz_convert(NEW_YORK).date()
    date = np.busday_offset(np.datetime64(start), day, roll="forward")
    local = pd.Timestamp(str(date)).tz_localize(NEW_YORK) + pd.Timedelta(hours=9, minutes=30)
    return local.tz_convert("UTC")


def session_bar(first: datetime, day: int, slot: int) -> pd.Timestamp:
    """The start of the `slot`-th bar (0 = 09:30) of the session of that weekday."""
    if not 0 <= slot <= LAST_SLOT:
        raise ValueError(f"slot {slot} is not a bar of the session (0 to {LAST_SLOT})")
    return weekday_open(first, day) + slot * BAR


def _session(start: pd.Timestamp, periods: int, *, backwards: bool) -> pd.DatetimeIndex:
    if backwards:
        grid = pd.date_range(end=start.floor("15min"), periods=periods, freq="15min")
    else:
        grid = pd.date_range(start.ceil("15min"), periods=periods, freq="15min")
    return grid[regular_session_mask(grid)]


def session_bars_from(
    start: datetime, count: int, *, last_slot: int = LAST_SLOT
) -> pd.DatetimeIndex:
    """The first `count` session bar starts at or after `start`, only those up to `last_slot` of
    each session (a signal's bar must leave room for the entry bar after it: `last_slot=24`)."""
    begin = pd.Timestamp(start).tz_convert("UTC")
    periods = (count // 20 + 3) * 7 * 96  # 20 bars a weekday at least: more than enough
    bars = _session(begin, periods, backwards=False)
    slots = _slots(bars)
    return bars[slots <= last_slot][:count]


def session_bars_until(end: datetime, count: int) -> pd.DatetimeIndex:
    """The last `count` session bar starts at or before `end`."""
    finish = pd.Timestamp(end).tz_convert("UTC")
    periods = (count // 20 + 3) * 7 * 96
    return _session(finish, periods, backwards=True)[-count:]


def _slots(bars: pd.DatetimeIndex) -> np.ndarray:
    local = bars.tz_convert(NEW_YORK)
    return np.asarray(((local.hour * 60 + local.minute) - (9 * 60 + 30)) // 15)
