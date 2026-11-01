"""The baseline of the daily loss guard: the account's equity at the start of the trading day.

RiskEngine compares today's equity with this number and stops new entries once the loss
reaches MAX_DAILY_LOSS_PCT. Nothing ever recorded it, so every signal was rejected with
"Missing daily start equity". This module decides what the baseline is and when it resets.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from trading_system.domain.models import TradingState

# The guard resets with the exchange's trading day, not at UTC midnight (which falls in the
# middle of the US evening session).
MARKET_TZ = ZoneInfo("America/New_York")


def trading_day(moment: datetime) -> date:
    """The exchange's calendar date at `moment`. A naive `moment` is taken as UTC."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(MARKET_TZ).date()


def day_marker(day: date) -> datetime:
    """How a trading day is stored in runtime_state.daily_equity_date: midnight UTC of that date.
    Only the date part is ever read back (SQLite returns it without a timezone)."""
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def has_start_equity_for(state: TradingState, day: date) -> bool:
    """True if `state` already holds a baseline recorded for the trading day `day`."""
    return (
        state.daily_start_equity is not None
        and state.daily_equity_date is not None
        and state.daily_equity_date.date() == day
    )


def choose_start_equity(equity: Decimal, previous_close_equity: Optional[Decimal]) -> Decimal:
    """The previous close's equity when the broker reports a positive one, else the current equity.

    The previous close is what the broker itself uses for the day's profit and loss, so the
    guard sees the same number as the account page, and it counts an overnight gap that
    already happened before the first cycle. The fallback is the equity at the first cycle."""
    if previous_close_equity is not None and previous_close_equity > 0:
        return previous_close_equity
    return equity


def with_daily_start_equity(
    state: TradingState,
    *,
    now: datetime,
    equity: Decimal,
    previous_close_equity: Optional[Decimal] = None,
) -> TradingState:
    """`state` with the baseline for the trading day of `now`.

    If a baseline for today is already there, `state` is returned unchanged (same object).
    That is deliberate: a restart in the middle of the day must not move the baseline to
    the current, already lower, equity, or a loss would be forgotten."""
    today = trading_day(now)
    if has_start_equity_for(state, today):
        return state
    return state.model_copy(
        update={
            "daily_start_equity": choose_start_equity(equity, previous_close_equity),
            "daily_equity_date": day_marker(today),
        }
    )
