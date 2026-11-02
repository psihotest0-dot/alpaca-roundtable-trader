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
