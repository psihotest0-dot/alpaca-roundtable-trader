"""The baseline of the daily loss guard: what it is, and when it resets."""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from trading_system.domain.models import TradingState
from trading_system.risk.daily_equity import (
    choose_start_equity,
    day_marker,
    has_start_equity_for,
    trading_day,
    with_daily_start_equity,
)

# 2026-11-02 14:30 UTC is 09:30 in New York (EST): Monday's open.
MONDAY_OPEN = datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc)
D = Decimal


@pytest.mark.parametrize(
    "moment, expected",
    [
        # 21:00 UTC on Sunday 1 Nov is only 16:00/17:00 in New York: still Sunday there.
        (datetime(2026, 11, 1, 21, 0, tzinfo=timezone.utc), date(2026, 11, 1)),
        # Just after UTC midnight it is still the previous evening in New York.
        (datetime(2026, 11, 3, 0, 30, tzinfo=timezone.utc), date(2026, 11, 2)),
        # 05:30 UTC is 00:30 EST: the new trading day has begun.
        (datetime(2026, 11, 2, 5, 30, tzinfo=timezone.utc), date(2026, 11, 2)),
        # Summer (EDT, UTC-4): 03:59 UTC is 23:59 the day before.
        (datetime(2026, 7, 15, 3, 59, tzinfo=timezone.utc), date(2026, 7, 14)),
        (datetime(2026, 7, 15, 4, 0, tzinfo=timezone.utc), date(2026, 7, 15)),
    ],
)
def test_the_trading_day_is_the_new_york_date(moment: datetime, expected: date) -> None:
    assert trading_day(moment) == expected


def test_a_naive_time_is_taken_as_utc() -> None:
    assert trading_day(datetime(2026, 11, 3, 0, 30)) == date(2026, 11, 2)


def test_the_first_cycle_of_the_day_fixes_the_baseline_at_the_current_equity() -> None:
    state = with_daily_start_equity(TradingState(), now=MONDAY_OPEN, equity=D("100000"))

    assert state.daily_start_equity == D("100000")
    assert state.daily_equity_date == day_marker(date(2026, 11, 2))


def test_the_previous_close_is_preferred_because_the_broker_measures_the_day_from_it() -> None:
    state = with_daily_start_equity(
        TradingState(), now=MONDAY_OPEN, equity=D("98500"), previous_close_equity=D("100000")
    )

    assert state.daily_start_equity == D("100000")  # the overnight gap already counts


@pytest.mark.parametrize("previous", [None, D("0"), D("-5")])
def test_without_a_usable_previous_close_the_current_equity_is_used(
    previous: Decimal | None,
) -> None:
    assert choose_start_equity(D("98500"), previous) == D("98500")


def test_a_baseline_already_recorded_today_is_never_moved() -> None:
    """A restart at 15:00 must not reset the baseline to the already lower equity."""
    morning = with_daily_start_equity(TradingState(), now=MONDAY_OPEN, equity=D("100000"))
    afternoon = datetime(2026, 11, 2, 20, 0, tzinfo=timezone.utc)

    again = with_daily_start_equity(morning, now=afternoon, equity=D("95000"))

    assert again is morning
    assert again.daily_start_equity == D("100000")


def test_the_next_trading_day_gets_a_new_baseline() -> None:
    monday = with_daily_start_equity(TradingState(), now=MONDAY_OPEN, equity=D("100000"))
    tuesday_open = datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc)

    tuesday = with_daily_start_equity(monday, now=tuesday_open, equity=D("96000"))

    assert tuesday.daily_start_equity == D("96000")
    assert tuesday.daily_equity_date == day_marker(date(2026, 11, 3))


def test_a_new_utc_date_inside_the_same_new_york_day_does_not_reset_the_guard() -> None:
    """Midnight UTC is 19:00/20:00 in New York, in the middle of the extended session."""
    monday = with_daily_start_equity(TradingState(), now=MONDAY_OPEN, equity=D("100000"))
    late = datetime(2026, 11, 3, 0, 30, tzinfo=timezone.utc)  # still Monday in New York

    assert with_daily_start_equity(monday, now=late, equity=D("90000")) is monday


def test_the_input_state_is_left_alone() -> None:
    original = TradingState(is_trading_halted=True, halt_reason="manual")

    updated = with_daily_start_equity(original, now=MONDAY_OPEN, equity=D("100000"))

    assert original.daily_start_equity is None
    assert (updated.is_trading_halted, updated.halt_reason) == (True, "manual")


def test_a_date_read_back_from_sqlite_without_a_timezone_still_matches() -> None:
    state = TradingState(
        daily_start_equity=D("100000"),
        daily_equity_date=datetime(2026, 11, 2),  # naive, as stored
    )

    assert has_start_equity_for(state, date(2026, 11, 2))
    assert not has_start_equity_for(state, date(2026, 11, 3))


def test_a_baseline_without_a_date_counts_as_not_recorded() -> None:
    assert not has_start_equity_for(TradingState(daily_start_equity=D("100000")), date(2026, 11, 2))
    assert not has_start_equity_for(TradingState(), date(2026, 11, 2))
