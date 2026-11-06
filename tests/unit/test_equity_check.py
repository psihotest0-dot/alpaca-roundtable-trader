"""The journal of the account's equity against Alpaca's own history of it.

The stamps are the ones the live API sent on 2026-11-06 (20:00 New York, a day late in UTC, a
Friday on a Saturday); the history is the real alpaca-py model."""

from datetime import date, datetime, timezone
from decimal import Decimal

from alpaca.trading.models import PortfolioHistory

from trading_system.observability.equity_check import (
    alpaca_equity_by_day,
    compare_equity,
    format_equity_check,
    journal_equity_by_day,
)
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.models import Base
from trading_system.storage.repositories import AuditRepository


def _history(points: list[tuple[int, float]]) -> PortfolioHistory:
    return PortfolioHistory(
        timestamp=[t for t, _ in points],
        equity=[e for _, e in points],
        profit_loss=[0.0 for _ in points],
        profit_loss_pct=[0.0 for _ in points],
        base_value=100000.0,
        timeframe="1D",
    )


# 2026-10-30 (a Friday) is stamped Saturday 00:00 UTC = Friday 20:00 in New York;
# 2026-11-02 (a Monday) is stamped Tuesday 01:00 UTC (daylight saving ended on the 1st).
FRIDAY_STAMP = 1793404800
MONDAY_STAMP = 1793667600


def test_a_point_belongs_to_the_new_york_date_of_its_stamp_not_the_utc_one():
    days = alpaca_equity_by_day(_history([(FRIDAY_STAMP, 101569.59), (MONDAY_STAMP, 102382.1)]))

    assert days == {
        date(2026, 10, 30): Decimal("101569.59"),
        date(2026, 11, 2): Decimal("102382.1"),
    }


def test_days_that_agree_pass():
    check = compare_equity(
        {date(2026, 11, 3): Decimal("102382.10"), date(2026, 11, 4): Decimal("102382.10")},
        {date(2026, 11, 3): Decimal("102382.10"), date(2026, 11, 4): Decimal("102382.10")},
    )

    assert check.ok and len(check.compared) == 2 and check.differing == []


def test_a_difference_beyond_the_tolerance_fails_and_a_small_one_does_not():
    alpaca = {date(2026, 11, 3): Decimal("100000"), date(2026, 11, 4): Decimal("100000")}
    journal = {
        date(2026, 11, 3): Decimal("100200"),
        date(2026, 11, 4): Decimal("101000"),
    }  # 0.2 %, 1 %

    check = compare_equity(alpaca, journal)

    assert not check.ok
    assert [c.day for c in check.differing] == [date(2026, 11, 4)]


def test_nothing_compared_is_not_a_pass():
    """A check that looked at no day proves nothing and must not look green."""
    check = compare_equity({date(2026, 11, 3): Decimal("100000")}, {})

    assert check.compared == [] and not check.ok


def test_a_day_without_a_journal_entry_is_before_the_journal_or_missing_inside_it():
    alpaca = {
        d: Decimal("100000")
        for d in (date(2026, 10, 30), date(2026, 11, 3), date(2026, 11, 4), date(2026, 11, 5))
    }
    journal = {date(2026, 11, 3): Decimal("100000"), date(2026, 11, 5): Decimal("100000")}

    check = compare_equity(alpaca, journal)

    assert check.before_journal == 1  # 10-30 is older than the journal
    assert check.missing_days == [date(2026, 11, 4)]  # the bot was down inside the journal
    assert check.ok  # a gap is reported, not failed: the bot may have been stopped on purpose


def test_the_text_marks_the_day_that_differs_and_says_where_to_look():
    check = compare_equity(
        {date(2026, 11, 3): Decimal("100000")}, {date(2026, 11, 3): Decimal("98000")}
    )

    text = format_equity_check(check)

    assert "2026-11-03" in text and "-2.00 %" in text and "DIFFERS" in text
    assert "reconcile" in text


def test_the_journal_gives_the_last_equity_of_each_new_york_day(tmp_path):
    url = f"sqlite:///{(tmp_path / 't.db').as_posix()}"
    Base.metadata.create_all(get_engine(url))
    journal = AuditRepository(get_session_factory(get_engine(url)))
    for hour, equity in ((14, "100000"), (19, "100500")):  # two passes on 2026-11-03 (UTC)
        journal.record(
            "entries_cycle",
            {"equity": equity},
            when=datetime(2026, 11, 3, hour, 30, tzinfo=timezone.utc),
        )
    journal.record(
        "entries_cycle", {"equity": None}, when=datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)
    )

    days = journal_equity_by_day(journal, date(2026, 11, 3), date(2026, 11, 4))

    assert days == {date(2026, 11, 3): Decimal("100500")}  # the later pass wins; None is no entry
