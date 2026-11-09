"""The weekly signal tested forward in time (docs/strategy-research-plan.md, section 10).

What is shown: the picks look only at the past and can be written down once a day; a record that
comes too late, a day that was missed and anything before the start never count; an effect planted
in the future is found by the same measurement as on the development part, with the control quiet;
and no verdict is given before the time that was fixed in advance.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.research import noise_returns, stored_bars
from trading_system.cli import main
from trading_system.research.forward import (
    EVENT_TYPE,
    FORWARD_START,
    MIN_ELAPSED_DAYS,
    SIGNAL_START,
    SIGNALS,
    close_time,
    evaluate,
    format_report,
    last_complete_day,
    open_time,
    picks_on,
    random_tenth_on,
    record_picks,
)
from trading_system.research.panel import build_panel
from trading_system.storage.repositories import AuditRepository, MarketBarRepository
from trading_system.storage.research import open_research_store
from tests.fixtures.research import frames_from_returns

UTC = timezone.utc
FETCHED = datetime(2026, 11, 5, tzinfo=UTC)


def _fill(store: MarketBarRepository, returns: pd.DataFrame) -> None:
    for symbol, bars in stored_bars(returns).items():
        store.replace(symbol, "1d", bars, adjustment="all", fetched_at=FETCHED)


@pytest.fixture
def research_url(tmp_path: Any) -> str:
    return f"sqlite:///{(tmp_path / 'research.db').as_posix()}"


@pytest.fixture
def store(research_url: str, db_url: str) -> MarketBarRepository:
    return open_research_store(research_url, db_url)


@pytest.fixture
def audit(session_factory: Any) -> AuditRepository:
    return AuditRepository(session_factory)


def _returns(end: str, *, days: int = 700, planted: float = 0.0, seed: int = 7) -> pd.DataFrame:
    """60 stocks and SPY; the first six drift by `planted` a day, all through the series."""
    returns = noise_returns(days=days, stocks=60, seed=seed, end=end)
    returns.iloc[:, :6] += planted
    return returns


def _evening(day: date, minutes: int = 40) -> datetime:
    return close_time(day) + timedelta(minutes=minutes)


# ------------------------------------------------------------------------------ the picks


def test_the_picks_are_the_stocks_that_rose_most_over_the_last_year_without_the_last_month() -> (
    None
):
    returns = _returns("2026-11-20", planted=0.003)
    panel = build_panel(frames_from_returns(returns))
    day = panel.dates[-1].date()

    picks = picks_on(panel, "momentum_12_1", day)

    assert picks == sorted(returns.columns[:6])  # the six planted risers, the top tenth of 60


def test_a_pick_cannot_see_a_later_price() -> None:
    returns = _returns("2026-11-20", planted=0.003)
    day = returns.index[-30].date()
    known = picks_on(build_panel(frames_from_returns(returns)), "momentum_12_1", day)

    changed = returns.copy()
    changed.iloc[-29:, :] = np.random.default_rng(99).normal(0.2, 0.3, changed.iloc[-29:, :].shape)
    after = picks_on(build_panel(frames_from_returns(changed)), "momentum_12_1", day)

    assert after == known  # the future was rewritten; the pick of that day did not move


def test_the_placebo_depends_on_the_date_alone() -> None:
    symbols = [f"S{i:03d}" for i in range(200)]
    day = date(2026, 11, 6)

    first = random_tenth_on(day, symbols)

    assert first == random_tenth_on(day, list(reversed(symbols)))  # not on the order given
    assert first != random_tenth_on(date(2026, 11, 9), symbols)
    assert 8 <= len(first) <= 35  # about a tenth of 200


def test_a_day_with_too_few_stocks_gives_no_picks() -> None:
    returns = noise_returns(days=400, stocks=10, end="2026-11-20")
    panel = build_panel(frames_from_returns(returns))

    assert picks_on(panel, "momentum_12_1", panel.dates[-1].date()) == []


# ---------------------------------------------------------------------- recording each evening


def test_the_complete_day_is_the_one_whose_bar_has_had_time_to_settle() -> None:
    dates = list(pd.bdate_range("2026-11-02", "2026-11-06"))  # Monday to Friday
    friday = date(2026, 11, 6)

    assert last_complete_day(dates, close_time(friday) + timedelta(minutes=10)) == date(2026, 11, 5)
    assert last_complete_day(dates, close_time(friday) + timedelta(minutes=31)) == friday


def test_one_evening_one_record_per_signal_and_a_second_call_adds_nothing(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    _fill(store, _returns("2026-11-06"))
    friday = date(2026, 11, 6)

    first = record_picks(audit, store, _evening(friday))
    again = record_picks(audit, store, _evening(friday, 120))

    # Friday 2026-11-06 is before the start of high_52w (2026-11-09): it is not recorded for it.
    older = [s for s in SIGNALS if SIGNAL_START[s] <= friday]
    assert older == ["momentum_12_1", "random_tenth"]
    assert first.day == friday and first.recorded == older and not first.skipped
    assert again.recorded == [] and again.already == older
    events = audit.events(
        EVENT_TYPE, datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC)
    )
    assert len(events) == len(older)
    details = {d["signal"]: d for _, d in events}["momentum_12_1"]
    assert details["day"] == "2026-11-06" and len(details["symbols"]) == 6
    assert details["market_known"] is True and details["stocks"] == 60


def test_high_52w_is_recorded_from_its_own_start_and_the_report_shows_what_it_shares(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    returns = _returns("2026-11-10", planted=0.003)
    others = [c for c in returns.columns[6:] if c != "SPY"]
    returns[others] -= 0.0005  # none of the rest is near its high
    _fill(store, returns)
    monday, tuesday = date(2026, 11, 9), date(2026, 11, 10)

    first = record_picks(audit, store, _evening(monday))
    second = record_picks(audit, store, _evening(tuesday))

    assert SIGNAL_START["high_52w"] == monday
    assert first.recorded == list(SIGNALS) and second.recorded == list(SIGNALS)
    events = audit.events(
        EVENT_TYPE, datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC)
    )
    picks = {(d["signal"], d["day"]): d["symbols"] for _, d in events}
    assert picks[("high_52w", "2026-11-09")] == sorted(returns.columns[:6])
    assert picks[("high_52w", "2026-11-10")] == sorted(returns.columns[:6])

    report = evaluate(audit, store, _evening(tuesday, 60))

    assert report.recorded_days["high_52w"] == 2
    # one variant only: 20 days, no filter
    assert [(v.regime_filter, v.horizon) for v in report.variants if v.signal == "high_52w"] == [
        (None, 20)
    ]
    assert report.overlap == (2, 1.0)  # the six planted risers are also the weekly signal's six
    assert "high_52w picks that momentum_12_1 also picked: 100%" in format_report(report)


def test_the_clock_of_high_52w_counts_from_its_own_start_not_from_the_older_two(
    store: MarketBarRepository, audit: AuditRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_system.research import forward as forward_module

    # The matured-days rule is set aside so that only the elapsed-days rule decides: sixty trading
    # days since 2026-11-06 is the older signals' minimum, and one day less for high_52w, which
    # started on 2026-11-09 (the next trading day).
    monkeypatch.setattr(forward_module, "MIN_MATURED_DAYS", 1)
    days = list(pd.bdate_range("2026-11-06", periods=MIN_ELAPSED_DAYS))
    _fill(store, _returns(str(days[-1].date()), planted=0.003))
    for day in days:
        record_picks(audit, store, _evening(day.date()))

    report = evaluate(audit, store, _evening(days[-1].date(), 60))

    verdict = {(v.signal, v.horizon): v.verdict for v in report.variants if v.regime_filter is None}
    assert report.elapsed_days == MIN_ELAPSED_DAYS
    assert verdict[("momentum_12_1", 10)] in ("pass", "no pass")  # its sixty days are there
    assert verdict[("high_52w", 20)] == "too early"  # its own are fifty-nine
    assert report.recorded_days["high_52w"] == report.recorded_days["momentum_12_1"] - 1


def test_nothing_is_recorded_before_the_days_bar_is_final(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    """At 16:10 the bar of the day is still forming. The day before is complete, but its record is
    too late (the open of the day came and went), so nothing can be written honestly."""
    _fill(store, _returns("2026-11-10"))
    tuesday = date(2026, 11, 10)

    result = record_picks(audit, store, close_time(tuesday) + timedelta(minutes=10))

    assert result.recorded == [] and result.skipped is not None and "too late" in result.skipped


def test_a_record_after_the_next_open_is_refused(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    _fill(store, _returns("2026-11-10"))
    next_morning = open_time(date(2026, 11, 11)) + timedelta(minutes=1)

    result = record_picks(audit, store, next_morning)

    assert result.recorded == [] and "too late" in (result.skipped or "")


def test_nothing_is_recorded_for_days_before_the_start(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    _fill(store, _returns("2026-11-05"))

    result = record_picks(audit, store, _evening(date(2026, 11, 5)))

    assert result.recorded == [] and "before the start" in (result.skipped or "")
    assert FORWARD_START == date(2026, 11, 6)


def test_too_few_stocks_in_the_store_is_said_not_recorded(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    _fill(store, noise_returns(days=400, stocks=10, end="2026-11-06"))

    result = record_picks(audit, store, _evening(date(2026, 11, 6)))

    assert result.recorded == [] and "fetch-daily" in (result.skipped or "")


# ----------------------------------------------------------------------- judging them later


def _record_every_evening(
    store: MarketBarRepository, audit: AuditRepository, through: date, skip: set[date] = frozenset()
) -> None:
    for day in pd.bdate_range(FORWARD_START, through):
        if day.date() not in skip:
            record_picks(audit, store, _evening(day.date()))


def test_an_effect_planted_in_the_future_is_found_and_the_control_stays_quiet(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    through = date(2027, 2, 5)  # about 66 trading days: just past the minimum
    _fill(store, _returns(str(through), days=420, planted=0.003))
    _record_every_evening(store, audit, through)

    report = evaluate(audit, store, _evening(through, 60))

    assert report.elapsed_days >= MIN_ELAPSED_DAYS and report.missed_days == 0 and report.late == 0
    assert set(report.recorded_days) == set(SIGNALS)
    momentum = {
        (v.regime_filter, v.horizon): v for v in report.variants if v.signal == "momentum_12_1"
    }
    plain = momentum[(None, 10)]
    assert plain.measurement.vs_universe.mean > 0.01 and plain.measurement.net.t > 3
    assert plain.verdict == "pass"
    controls = [v for v in report.variants if v.signal == "random_tenth"]
    assert len(controls) == 6
    assert {v.verdict for v in controls} <= {"control ok", "too early"}  # never "suspect"
    assert [v.verdict for v in controls if v.regime_filter is None and v.horizon == 10] == [
        "control ok"
    ]
    text = format_report(report)
    assert "momentum_12_1, no filter, 10 days" in text and "[pass]" in text
    assert "Nothing here places an order." in text


def test_before_the_minimum_time_there_is_no_verdict_whatever_the_numbers_say(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    through = date(2026, 12, 4)  # about four weeks
    _fill(store, _returns(str(through), days=420, planted=0.003))
    _record_every_evening(store, audit, through)

    report = evaluate(audit, store, _evening(through, 60))

    assert report.elapsed_days < MIN_ELAPSED_DAYS
    assert {v.verdict for v in report.variants} == {"too early"}
    assert report.variants[0].measurement.vs_universe.dates >= 0  # numbers are shown, not judged


def test_a_missed_day_stays_missed_and_is_counted(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    through = date(2026, 12, 11)
    _fill(store, _returns(str(through), days=420))
    skipped = {date(2026, 11, 12), date(2026, 11, 13)}
    _record_every_evening(store, audit, through, skip=skipped)

    report = evaluate(audit, store, _evening(through, 60))

    assert report.missed_days == 2
    assert report.recorded_days["momentum_12_1"] == report.elapsed_days - 2


def test_a_record_made_after_the_entry_open_is_kept_but_never_counted(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    through = date(2026, 12, 4)
    _fill(store, _returns(str(through), days=420, planted=0.003))
    friday, monday = date(2026, 11, 6), date(2026, 11, 9)
    # written by hand, the way a bug or a late run could have: after Monday's open
    audit.record(
        EVENT_TYPE,
        {
            "signal": "momentum_12_1",
            "day": friday.isoformat(),
            "symbols": ["AAA"],
            "stocks": 60,
            "market_up": True,
            "market_known": True,
        },
        when=open_time(monday) + timedelta(hours=2),
    )

    report = evaluate(audit, store, _evening(through, 60))

    assert report.late == 1 and report.recorded_days["momentum_12_1"] == 0


def test_picks_dated_before_the_start_are_ignored_by_the_judgement(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    _fill(store, _returns("2026-11-27", days=420, planted=0.003))
    audit.record(
        EVENT_TYPE,
        {
            "signal": "momentum_12_1",
            "day": "2026-10-30",
            "symbols": ["AAPL"],
            "stocks": 60,
            "market_up": True,
            "market_known": True,
        },
        when=datetime(2026, 11, 9, 12, tzinfo=UTC),  # written after the start, for a day before it
    )

    report = evaluate(audit, store, _evening(date(2026, 11, 27), 60))

    assert report.recorded_days["momentum_12_1"] == 0 and report.late == 0


def test_the_regime_filter_uses_what_was_recorded_that_evening(
    store: MarketBarRepository, audit: AuditRepository
) -> None:
    through = date(2026, 12, 18)
    _fill(store, _returns(str(through), days=420, planted=0.003))
    for day in pd.bdate_range(FORWARD_START, through):
        record_picks(audit, store, _evening(day.date()))
    # rewrite history in the record itself: pretend the market was down on every one of those days
    from sqlalchemy import text

    with audit._session_factory.begin() as session:  # the journal of the test database
        rows = session.execute(
            text("select id, details from audit_events where event_type = :t"), {"t": EVENT_TYPE}
        ).all()
        import json

        for row_id, raw in rows:
            details = json.loads(raw)
            details["market_up"] = False
            session.execute(
                text("update audit_events set details = :d where id = :i"),
                {"d": json.dumps(details), "i": row_id},
            )

    report = evaluate(audit, store, _evening(through, 60))

    filtered = [v for v in report.variants if v.regime_filter == "spy200"]
    plain = [v for v in report.variants if v.regime_filter is None]
    assert all(not v.measurement.usable for v in filtered)  # market "down" every day: no day left
    assert any(v.measurement.usable for v in plain)


# ------------------------------------------------------------------------------ the commands


def test_the_commands_record_and_report_without_placing_an_order(
    research_url: str,
    db_url: str,
    make_settings: Any,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = open_research_store(research_url, db_url)
    _fill(store, _returns("2026-11-06", days=420, planted=0.003))
    settings = make_settings(research_database_url=research_url)
    now = _evening(date(2026, 11, 6))

    class FrozenNow(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "FrozenNow":
            return cls.fromtimestamp(now.timestamp(), tz or UTC)  # type: ignore[return-value]

    with (
        patch("trading_system.cli.Settings", return_value=settings),
        patch("datetime.datetime", FrozenNow),
    ):
        assert main(["forward-record"]) == 0
    recorded = capsys.readouterr().out
    assert "Picks for the close of 2026-11-06" in recorded and "recorded momentum_12_1" in recorded

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["forward-report"]) == 0
    out = capsys.readouterr().out
    assert "Forward test since 2026-11-06" in out and "too early" in out
    with session_factory() as session:
        from trading_system.storage.models import BrokerOrderModel

        assert session.query(BrokerOrderModel).count() == 0


def test_the_command_refuses_when_the_research_file_is_the_trading_database(
    db_url: str, make_settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(research_database_url=db_url)

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["forward-report"]) == 1

    assert "must not go into the trading database" in capsys.readouterr().out


# ---------------------------------------------------------------------------------- the verdict


def _stub(matured: int, mean: float, t: float, gross_t: float = 0.0) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        vs_universe=SimpleNamespace(dates=matured, t=gross_t),
        net=SimpleNamespace(mean=mean, t=t),
    )


@pytest.mark.parametrize(
    "signal, elapsed, matured, mean, t, gross_t, expected",
    [
        ("momentum_12_1", 59, 100, 0.01, 5.0, 5.0, "too early"),  # time is not up
        ("momentum_12_1", 60, 39, 0.01, 5.0, 5.0, "too early"),  # not enough matured days
        ("momentum_12_1", 60, 40, 0.001, 1.0, 1.5, "pass"),  # the bar is net t of 1, inclusive
        ("momentum_12_1", 60, 40, 0.001, 0.99, 1.5, "no pass"),
        ("momentum_12_1", 90, 70, -0.001, 1.5, 1.5, "no pass"),  # same direction means above zero
        ("momentum_12_1", 90, 70, 0.002, float("nan"), 1.5, "no pass"),
        ("random_tenth", 90, 70, 0.0, 0.3, 1.9, "control ok"),
        ("random_tenth", 90, 70, 0.0, 0.3, -2.0, "control suspect"),  # |t| of 2 or more
        ("random_tenth", 90, 70, 0.0, 0.3, float("nan"), "control suspect"),
        ("random_tenth", 30, 70, 0.0, 0.3, 5.0, "too early"),
    ],
)
def test_the_verdict_follows_the_rules_fixed_in_advance(
    signal: str, elapsed: int, matured: int, mean: float, t: float, gross_t: float, expected: str
) -> None:
    from trading_system.research.forward import _verdict

    assert _verdict(signal, _stub(matured, mean, t, gross_t), elapsed) == expected
