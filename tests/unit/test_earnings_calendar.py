"""The earnings record (data/earnings_calendar.py, plan section 16.1): what the page said, kept in a
small shared file, with a page that cannot be read being "not known" and never "no report"."""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from trading_system.data.earnings_calendar import (
    AFTER,
    PRE,
    UNKNOWN,
    EarningsCalendar,
    fetch_day,
    next_weekdays,
)

FRIDAY = date(2026, 11, 20)
MONDAY, TUESDAY = date(2026, 11, 23), date(2026, 11, 24)
NOON = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc)


class Pages:
    """A fake fetch: day -> rows (or an exception), and a log of the days asked."""

    def __init__(self, pages: dict[date, Any]) -> None:
        self.pages = pages
        self.asked: list[date] = []

    def __call__(self, day: date) -> dict[str, str]:
        self.asked.append(day)
        page = self.pages.get(day, {})
        if isinstance(page, Exception):
            raise page
        return dict(page)


def calendar(tmp_path: Path, pages: Pages, now: datetime = NOON) -> EarningsCalendar:
    clock = {"now": now}
    cal = EarningsCalendar(tmp_path / "earnings.json", fetch=pages, clock=lambda: clock["now"])
    cal._clock_box = clock  # type: ignore[attr-defined]
    return cal


def test_next_weekdays_skip_the_weekend() -> None:
    assert next_weekdays(FRIDAY, 2) == [MONDAY, TUESDAY]
    assert next_weekdays(date(2026, 11, 18), 2) == [date(2026, 11, 19), FRIDAY]


def test_a_report_after_the_close_today_or_on_either_of_the_next_two_sessions_counts(
    tmp_path: Path,
) -> None:
    pages = Pages({FRIDAY: {"AAA": AFTER}, MONDAY: {"BBB": PRE}, TUESDAY: {"CCC": AFTER}})
    cal = calendar(tmp_path, pages)

    assert cal.lookup("AAA", FRIDAY).in_2_sessions is True
    assert cal.lookup("BBB", FRIDAY).dates == [{"date": "2026-11-23", "timing": PRE}]
    assert cal.lookup("CCC", FRIDAY).in_2_sessions is True
    clear = cal.lookup("ZZZ", FRIDAY)
    assert clear.known is True and clear.in_2_sessions is False and clear.dates == []


def test_a_report_before_the_open_today_is_already_in_the_price(tmp_path: Path) -> None:
    cal = calendar(tmp_path, Pages({FRIDAY: {"AAA": PRE, "BBB": UNKNOWN}}))

    assert cal.lookup("AAA", FRIDAY).in_2_sessions is False
    assert cal.lookup("BBB", FRIDAY).in_2_sessions is True  # not said: treated like after the close


def test_a_page_that_cannot_be_read_is_not_known_and_never_no_report(tmp_path: Path) -> None:
    cal = calendar(tmp_path, Pages({MONDAY: RuntimeError("down"), FRIDAY: {}, TUESDAY: {}}))

    info = cal.lookup("AAA", FRIDAY)

    assert info.known is False and info.in_2_sessions is None


def test_a_hit_on_a_readable_day_stands_even_when_another_day_is_unreadable(
    tmp_path: Path,
) -> None:
    cal = calendar(tmp_path, Pages({FRIDAY: {"AAA": AFTER}, MONDAY: RuntimeError("down")}))

    info = cal.lookup("AAA", FRIDAY)

    assert info.known is False and info.in_2_sessions is True


def test_a_day_is_read_once_and_a_failure_is_not_asked_again_at_once(tmp_path: Path) -> None:
    pages = Pages({FRIDAY: {"AAA": AFTER}, MONDAY: RuntimeError("down")})
    cal = calendar(tmp_path, pages)

    cal.lookup("AAA", FRIDAY)
    cal.lookup("BBB", FRIDAY)
    cal.lookup("CCC", FRIDAY)

    assert pages.asked.count(FRIDAY) == 1 and pages.asked.count(TUESDAY) == 1
    assert pages.asked.count(MONDAY) == 1  # the failure waits RETRY_AFTER_FAILURE


def test_today_is_read_again_after_six_hours_and_the_past_never(tmp_path: Path) -> None:
    pages = Pages({FRIDAY: {"AAA": AFTER}})
    cal = calendar(tmp_path, pages)
    cal.reports(FRIDAY)

    cal._clock_box["now"] = NOON + timedelta(hours=7)  # type: ignore[attr-defined]
    cal.reports(FRIDAY)
    assert pages.asked.count(FRIDAY) == 2

    cal._clock_box["now"] = NOON + timedelta(days=3)  # type: ignore[attr-defined]
    cal.reports(FRIDAY)
    assert pages.asked.count(FRIDAY) == 2  # Friday is the past by now: kept as it was


def test_the_cache_file_is_shared_between_instances_and_survives_a_bad_file(
    tmp_path: Path,
) -> None:
    pages = Pages({FRIDAY: {"AAA": AFTER}})
    calendar(tmp_path, pages).reports(FRIDAY)

    other = Pages({})
    assert calendar(tmp_path, other).reports(FRIDAY) == {"AAA": AFTER}
    assert other.asked == []  # read from the file

    (tmp_path / "earnings.json").write_text("not json", encoding="utf-8")
    assert calendar(tmp_path, Pages({FRIDAY: {"BBB": PRE}})).reports(FRIDAY) == {"BBB": PRE}


def test_a_symbol_with_a_dot_is_found_under_the_pages_slash(tmp_path: Path) -> None:
    cal = calendar(tmp_path, Pages({FRIDAY: {"BRK/B": AFTER}}))

    assert cal.lookup("BRK.B", FRIDAY).in_2_sessions is True


def test_the_page_is_parsed_into_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "status": {"rCode": 200},
        "data": {
            "rows": [
                {"symbol": "aaa", "time": "time-pre-market"},
                {"symbol": "BBB", "time": "time-after-hours"},
                {"symbol": "CCC", "time": "time-not-supplied"},
                {"symbol": "DDD", "time": "something new"},
                {"symbol": "", "time": "time-pre-market"},
            ]
        },
    }

    class Reply:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

        def __enter__(self) -> "Reply":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self, *args: Any) -> bytes:
            return json.dumps(self.payload).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Reply(body))
    assert fetch_day(FRIDAY) == {"AAA": PRE, "BBB": AFTER, "CCC": UNKNOWN, "DDD": UNKNOWN}

    body["data"] = {"rows": None}  # a weekend: a real answer, nobody reports
    assert fetch_day(FRIDAY) == {}

    body["status"] = {"rCode": 400}
    with pytest.raises(ValueError):
        fetch_day(FRIDAY)
