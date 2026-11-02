from datetime import datetime, timedelta, timezone

import pytest

from trading_system.runtime.schedule import BAR, bar_boundary, expected_bar_start

UTC = timezone.utc


@pytest.mark.parametrize(
    "now, boundary",
    [
        (datetime(2026, 11, 2, 15, 30, 0, tzinfo=UTC), datetime(2026, 11, 2, 15, 30, tzinfo=UTC)),
        (datetime(2026, 11, 2, 15, 30, 1, tzinfo=UTC), datetime(2026, 11, 2, 15, 30, tzinfo=UTC)),
        (datetime(2026, 11, 2, 15, 44, 59, tzinfo=UTC), datetime(2026, 11, 2, 15, 30, tzinfo=UTC)),
        (datetime(2026, 11, 2, 15, 45, 0, tzinfo=UTC), datetime(2026, 11, 2, 15, 45, tzinfo=UTC)),
        (datetime(2026, 11, 2, 0, 0, 3, tzinfo=UTC), datetime(2026, 11, 2, 0, 0, tzinfo=UTC)),
    ],
)
def test_the_boundary_is_the_latest_multiple_of_the_bar(now: datetime, boundary: datetime) -> None:
    assert bar_boundary(now) == boundary


def test_the_newest_closed_bar_started_one_bar_before_the_boundary() -> None:
    now = datetime(2026, 11, 2, 15, 31, tzinfo=UTC)

    assert expected_bar_start(now) == datetime(2026, 11, 2, 15, 15, tzinfo=UTC)
    assert BAR == timedelta(minutes=15)


def test_boundaries_agree_in_new_york_time() -> None:
    """The exchange's offset from UTC is whole hours, so 09:45 ET is a boundary in UTC too."""
    from zoneinfo import ZoneInfo

    ny = datetime(2026, 11, 2, 9, 47, tzinfo=ZoneInfo("America/New_York"))

    assert bar_boundary(ny).astimezone(ZoneInfo("America/New_York")).strftime("%H:%M") == "09:45"


def test_a_naive_datetime_is_read_as_utc() -> None:
    assert bar_boundary(datetime(2026, 11, 2, 15, 31)) == datetime(2026, 11, 2, 15, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    "moment, expected",
    [
        # November, New York on standard time (UTC-5): the session is 14:30 to 21:00 UTC.
        (datetime(2026, 11, 2, 14, 29, tzinfo=UTC), False),
        (datetime(2026, 11, 2, 14, 30, tzinfo=UTC), True),
        (datetime(2026, 11, 2, 20, 59, tzinfo=UTC), True),
        (datetime(2026, 11, 2, 21, 0, tzinfo=UTC), False),
        # July, daylight time (UTC-4): 13:30 to 20:00 UTC. A fixed UTC window would be wrong.
        (datetime(2026, 7, 1, 13, 29, tzinfo=UTC), False),
        (datetime(2026, 7, 1, 13, 30, tzinfo=UTC), True),
        (datetime(2026, 7, 1, 19, 59, tzinfo=UTC), True),
        (datetime(2026, 7, 1, 20, 0, tzinfo=UTC), False),
        # The weekend, at an hour that is inside the session on a weekday.
        (datetime(2026, 11, 7, 16, 0, tzinfo=UTC), False),
        (datetime(2026, 11, 8, 16, 0, tzinfo=UTC), False),
        # A naive time is taken as UTC.
        (datetime(2026, 11, 2, 16, 0), True),
    ],
)
def test_the_regular_session_is_nine_thirty_to_four_new_york_time_on_weekdays(
    moment: datetime, expected: bool
) -> None:
    from trading_system.runtime.schedule import is_regular_session

    assert is_regular_session(moment) is expected
