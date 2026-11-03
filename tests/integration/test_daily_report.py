"""The daily report: built from what the system recorded, for one New York trading day."""

import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from trading_system.cli import main
from trading_system.domain.enums import OrderSide, OrderStatus, OrderType
from trading_system.domain.models import BrokerOrder, RiskDecision, StrategySignal, TradingState
from trading_system.observability.report import (
    BARS_IN_A_SESSION,
    DailyReport,
    build_daily_report,
    day_bounds,
    format_daily_report,
)
from trading_system.storage.repositories import (
    AuditRepository,
    OrderRepository,
    RiskDecisionRepository,
    SignalRepository,
)

UTC = timezone.utc
DAY = date(2026, 11, 3)  # a Tuesday in standard time: New York is UTC-5
OPEN = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)  # 09:30 in New York
RunCycle = Callable[..., SimpleNamespace]


# --- the day ----------------------------------------------------------------------------------


def test_a_new_york_day_is_five_hours_into_the_utc_day_in_winter() -> None:
    assert day_bounds(DAY) == (
        datetime(2026, 11, 3, 5, 0, tzinfo=UTC),
        datetime(2026, 11, 4, 5, 0, tzinfo=UTC),
    )


def test_a_new_york_day_is_four_hours_into_the_utc_day_in_summer() -> None:
    start, end = day_bounds(date(2026, 7, 1))

    assert start == datetime(2026, 7, 1, 4, 0, tzinfo=UTC) and end - start == timedelta(hours=24)


def test_the_day_the_clocks_change_is_25_hours_long() -> None:
    start, end = day_bounds(date(2026, 11, 1))  # summer time ends in the night

    assert end - start == timedelta(hours=25)


def _cycle(journal: AuditRepository, when: datetime, **details: Any) -> None:
    base = {
        "skipped": None,
        "retry_later": False,
        "signals": 0,
        "submitted": 0,
        "held": 0,
        "errors": 0,
        "equity": None,
    }
    journal.record("entries_cycle", {**base, **details}, when=when)


def test_passes_signals_and_equity_of_the_day_are_added_up(session_factory: Any) -> None:
    journal = AuditRepository(session_factory)
    _cycle(journal, OPEN, equity="100000")
    _cycle(journal, OPEN + timedelta(minutes=15), retry_later=True, skipped="bar not available yet")
    _cycle(
        journal,
        OPEN + timedelta(minutes=16),
        signals=2,
        submitted=1,
        held=0,
        errors=1,
        equity="100500",
    )
    _cycle(journal, OPEN + timedelta(minutes=30), skipped="trading halted", equity="101000.50")

    report = build_daily_report(session_factory, DAY)

    assert (report.passes, report.completed) == (4, 3)
    assert dict(report.not_completed) == {"bar not available yet": 1}
    assert dict(report.skipped) == {"trading halted": 1}
    assert (report.signals, report.submitted, report.held, report.errors) == (2, 1, 0, 1)
    assert (report.equity_first, report.equity_last) == (Decimal("100000"), Decimal("101000.50"))


def test_only_the_asked_new_york_day_is_counted(session_factory: Any) -> None:
    journal = AuditRepository(session_factory)
    _cycle(journal, datetime(2026, 11, 3, 4, 59, tzinfo=UTC), signals=7)  # 23:59 the day before
    _cycle(journal, datetime(2026, 11, 3, 5, 0, tzinfo=UTC), signals=1)  # midnight, in
    _cycle(journal, datetime(2026, 11, 4, 4, 59, tzinfo=UTC), signals=1)  # 23:59, in
    _cycle(journal, datetime(2026, 11, 4, 5, 0, tzinfo=UTC), signals=7)  # midnight after, out

    assert build_daily_report(session_factory, DAY).signals == 2


def test_alerts_of_the_day_are_listed_with_their_time(session_factory: Any) -> None:
    journal = AuditRepository(session_factory)
    journal.record("alert", {"title": "Signals broke before they became orders"}, when=OPEN)
    journal.record("alert", {"title": "yesterday's"}, when=OPEN - timedelta(days=1))

    report = build_daily_report(session_factory, DAY)

    assert report.alerts == [(OPEN, "Signals broke before they became orders")]


def _signal(session_factory: Any, n: int) -> str:
    signal = StrategySignal(
        version="1",
        symbol=f"S{n}",
        timeframe="15m",
        closed_bar_timestamp=OPEN + timedelta(minutes=15 * n),
        direction="LONG",  # type: ignore[arg-type]
    )
    SignalRepository(session_factory).save_signal(signal)
    return signal.signal_id or ""


def test_risk_decisions_are_counted_and_refusals_grouped_by_reason_without_numbers(
    session_factory: Any,
) -> None:
    repo = RiskDecisionRepository(session_factory)
    for n, (approved, reason) in enumerate(
        [
            (True, "Approved."),
            (False, "Max open positions limit reached."),
            (False, "Max open positions limit reached."),
            (False, "Daily loss guard breached: 0.0212 >= 0.02"),
            (False, "Daily loss guard breached: 0.0300 >= 0.02"),
        ]
    ):
        repo.save(
            RiskDecision(
                signal_id=_signal(session_factory, n),
                approved=approved,
                reasoning=reason,
                decision_timestamp=OPEN + timedelta(minutes=n),
            )
        )

    report = build_daily_report(session_factory, DAY)

    assert report.approved == 1
    assert dict(report.refused) == {
        "Max open positions limit reached.": 2,
        "Daily loss guard breached": 2,
    }


def _order(client_id: str, broker_id: str, status: OrderStatus, when: datetime) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_id,
        broker_id=broker_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        status=status,
        quantity=Decimal(10),
        filled_quantity=Decimal(0),
        limit_price=Decimal(100),
        created_at=when,
        updated_at=when,
    )


def test_orders_are_counted_by_status_and_simulated_ones_are_told_apart(
    session_factory: Any,
) -> None:
    orders = OrderRepository(session_factory)
    orders.save_broker_order(_order("a", "dry_run_a", OrderStatus.SUBMITTED, OPEN))
    orders.save_broker_order(_order("b", "dry_run_b", OrderStatus.SUBMITTED, OPEN))
    orders.save_broker_order(_order("c", "real-1", OrderStatus.FILLED, OPEN))
    orders.save_broker_order(
        _order("d", "dry_run_d", OrderStatus.SUBMITTED, OPEN - timedelta(days=2))
    )
    orders.save_fill("real-1", "exec-1", 100.5, 10, OPEN + timedelta(minutes=1))
    orders.save_fill("real-1", "exec-0", 99.0, 1, OPEN - timedelta(days=2))

    report = build_daily_report(session_factory, DAY)

    assert dict(report.orders) == {"SUBMITTED": 2, "FILLED": 1}
    assert report.dry_run_orders == 2
    assert report.fills == 1


def test_the_baseline_is_shown_only_for_its_own_day_and_a_halt_is_carried(
    session_factory: Any,
) -> None:
    state = TradingState(
        is_trading_halted=True,
        halt_reason="Unprotected position",
        daily_start_equity=Decimal("102382.1"),
        daily_equity_date=datetime(2026, 11, 3, tzinfo=UTC),
    )

    today = build_daily_report(session_factory, DAY, state)
    another = build_daily_report(session_factory, DAY - timedelta(days=1), state)

    assert today.baseline == Decimal("102382.1") and another.baseline is None
    assert today.halted and today.halt_reason == "Unprotected position"


# --- the text ---------------------------------------------------------------------------------


def _text(**fields: Any) -> str:
    return format_daily_report(DailyReport(day=DAY, **fields))


def test_a_quiet_clean_day_reads_as_such() -> None:
    text = _text(passes=BARS_IN_A_SESSION, completed=BARS_IN_A_SESSION)

    assert "26 completed of 26 tries" in text and "a full session" not in text
    assert "BROKE" not in text and "Alerts:             none" in text
    assert "not recorded on this day" in text and "HALTED" not in text


def test_a_short_day_says_how_long_a_full_one_is() -> None:
    assert "(a full session has 26 bars)" in _text(passes=10, completed=10)


def test_broken_signals_and_alerts_are_loud() -> None:
    text = _text(
        passes=1,
        completed=1,
        signals=1,
        errors=1,
        alerts=[(OPEN, "Signals broke before they became orders")],
    )

    assert "1 BROKE" in text
    assert "ALERTS:             1" in text and "14:30 UTC  Signals broke" in text


def test_equity_change_is_shown_as_a_percentage() -> None:
    text = _text(
        equity_first=Decimal("100000"), equity_last=Decimal("101500"), baseline=Decimal("99900")
    )

    assert "100,000.00 at the first pass, 101,500.00 at the last (+1.50%)" in text
    assert "99,900.00 (the daily loss guard's baseline)" in text


def test_a_halt_is_in_the_report_with_its_reason() -> None:
    assert "Trading is HALTED:  Unprotected position" in _text(
        halted=True, halt_reason="Unprotected position"
    )


def test_refusals_and_orders_are_listed() -> None:
    from collections import Counter

    text = _text(
        approved=1,
        refused=Counter({"Max open positions limit reached.": 3}),
        orders=Counter({"submitted": 2}),
        dry_run_orders=2,
    )

    assert "1 approved, 3 refused" in text
    assert "refused:          Max open positions limit reached.: 3" in text
    assert "Orders:             2 (submitted: 2)" in text and "simulated:        2 of 2" in text


# --- the cycle feeds the journal the report reads -----------------------------------------------


def test_a_pass_journals_the_equity_of_the_account(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle(equity="123456.7")

    [(_, details)] = AuditRepository(session_factory).events(
        "entries_cycle",
        datetime.now(UTC) - timedelta(days=1),
        datetime.now(UTC) + timedelta(days=1),
    )
    assert details["equity"] == "123456.7"


def test_an_alert_is_journalled_so_that_it_outlives_the_log(
    run_cycle: RunCycle, session_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    def break_execution(cycle: Any) -> None:
        def boom(intent: Any) -> Any:
            raise RuntimeError("the broker contract changed")

        cycle.execution.submit_order = boom

    with caplog.at_level(logging.INFO):
        run_cycle(tweak=break_execution)

    [(_, alert)] = AuditRepository(session_factory).events(
        "alert", datetime.now(UTC) - timedelta(days=1), datetime.now(UTC) + timedelta(days=1)
    )
    assert alert["title"] == "Signals broke before they became orders"


def test_a_healthy_day_leaves_no_alert_in_the_journal(
    run_cycle: RunCycle, session_factory: Any
) -> None:
    run_cycle()

    assert (
        AuditRepository(session_factory).events(
            "alert", datetime.now(UTC) - timedelta(days=1), datetime.now(UTC) + timedelta(days=1)
        )
        == []
    )


# --- the command --------------------------------------------------------------------------------


def test_the_report_command_prints_the_day(
    settings: Any, session_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _cycle(AuditRepository(session_factory), OPEN, signals=1, submitted=1, equity="100000")

    with patch("trading_system.cli.Settings", return_value=settings):
        code = main(["report", "--date", "2026-11-03"])

    out = capsys.readouterr().out
    assert code == 0 and "=== Daily report 2026-11-03" in out
    assert "Signals:            1 -> 1 became an order" in out


def test_the_report_command_defaults_to_today(
    settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        code = main(["report"])

    assert code == 0 and "=== Daily report" in capsys.readouterr().out


def test_the_report_command_refuses_a_bad_date(
    settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=settings):
        code = main(["report", "--date", "3 November"])

    assert code == 1 and "YYYY-MM-DD" in capsys.readouterr().out
