"""The fractional trial's wiring: it refuses the live loop's account, resumes only with stops standing,
reports from the journal, and has its commands (docs/fractional-paper-trial.md)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trading_system.cli import main
from trading_system.fractional.report import build, format_report
from trading_system.runtime.orchestrator import Orchestrator
from trading_system.storage.repositories import AuditRepository

NOW = datetime.now(timezone.utc)
LIVE_KEY = "PKLIVELOOPKEYID000000"
SECOND_KEY = "PKSECONDACCOUNTKEY000"


# ------------------------------------------------------------------------- the live loop's key


def test_the_loop_refuses_to_start_without_the_live_loops_key_id(make_settings: Any) -> None:
    settings = make_settings(alpaca_api_key=SECOND_KEY)  # nothing in FRACTIONAL_FORBIDDEN_KEY_ID

    with pytest.raises(RuntimeError, match="FRACTIONAL_FORBIDDEN_KEY_ID is empty"):
        Orchestrator(settings).build_fractional_cycle()


def test_the_loop_refuses_to_start_on_the_live_loops_key(make_settings: Any) -> None:
    settings = make_settings(alpaca_api_key=LIVE_KEY, fractional_forbidden_key_id=LIVE_KEY)

    with pytest.raises(RuntimeError, match="live loop's key"):
        Orchestrator(settings).build_fractional_cycle()


def test_the_loop_is_built_on_another_key(make_settings: Any) -> None:
    settings = make_settings(alpaca_api_key=SECOND_KEY, fractional_forbidden_key_id=LIVE_KEY)

    cycle = Orchestrator(settings).build_fractional_cycle()

    assert cycle.halt_path == Path(settings.fractional_halt_path)


def test_the_command_says_why_it_did_not_start(
    make_settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(alpaca_api_key=LIVE_KEY, fractional_forbidden_key_id=LIVE_KEY)

    with patch("trading_system.cli.Settings", return_value=settings):
        assert main(["fractional-run"]) == 1

    assert "live loop's key" in capsys.readouterr().out


# --------------------------------------------------------------------------------------- report


def _journal(session_factory: Any) -> AuditRepository:
    return AuditRepository(session_factory)


def _fill_days(audit: AuditRepository, days: int, entries: int) -> None:
    audit.record("fractional_baseline", {"equity": "1000000", "budget": "500.0"}, when=NOW)
    day = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)  # a Monday, 10:00 New York
    for n in range(days):
        audit.record(
            "fractional_cycle",
            {"skipped": None, "signals": 1, "submitted": 0, "errors": 0, "gross_pct": "0.08"},
            when=day + timedelta(days=n),
        )
    for n in range(entries):
        audit.record("fractional_entry", {"symbol": "KO"}, when=day + timedelta(hours=n))


def test_the_report_says_not_enough_yet_below_the_written_minimum(session_factory: Any) -> None:
    audit = _journal(session_factory)
    _fill_days(audit, days=3, entries=2)

    text = format_report(build(audit, NOW + timedelta(days=30)))

    assert "trading days with a pass: 3 (need 20)" in text and "entries: 2 (need 15)" in text
    assert "verdict: not enough yet" in text


@pytest.mark.parametrize("days, entries", [(20, 14), (19, 15), (40, 0), (0, 40)])
def test_both_minimums_are_needed_before_the_report_judges(
    session_factory: Any, days: int, entries: int
) -> None:
    audit = _journal(session_factory)
    _fill_days(audit, days=days, entries=entries)

    report = build(audit, NOW + timedelta(days=90))

    assert not report.enough
    assert "verdict: not enough yet" in format_report(report)


def test_the_report_calls_the_trial_ready_only_when_every_criterion_holds(
    session_factory: Any,
) -> None:
    audit = _journal(session_factory)
    _fill_days(audit, days=20, entries=15)
    audit.record(
        "fractional_stop_placed",
        {"symbol": "KO", "gap_seconds": 40.0},
        when=NOW,
    )

    report = build(audit, NOW + timedelta(days=60))

    assert report.enough and report.failures() == []
    assert "criteria of operational readiness are met" in format_report(report)


def test_a_long_gap_outside_the_opening_fails_the_trial_but_one_at_the_opening_does_not(
    session_factory: Any,
) -> None:
    audit = _journal(session_factory)
    _fill_days(audit, days=20, entries=15)
    # 14:34 UTC in November = 09:34 New York: the opening grace (the first five minutes after 09:30)
    opening = datetime(2026, 11, 9, 14, 38, tzinfo=timezone.utc)
    audit.record("fractional_stop_placed", {"symbol": "KO", "gap_seconds": 250.0}, when=opening)
    assert build(audit, NOW + timedelta(days=60)).failures() == []

    midday = datetime(2026, 11, 9, 17, 0, tzinfo=timezone.utc)
    audit.record("fractional_stop_placed", {"symbol": "PEP", "gap_seconds": 200.0}, when=midday)
    report = build(audit, NOW + timedelta(days=60))

    assert report.late_gaps == [200.0]
    assert "NOT ready" in format_report(report) and "without a stop" in format_report(report)


def test_a_halt_a_refusal_a_sale_problem_or_too_much_gross_each_fail_the_trial(
    session_factory: Any,
) -> None:
    audit = _journal(session_factory)
    _fill_days(audit, days=20, entries=15)
    audit.record(
        "fractional_alert", {"title": "Fractional entries stopped", "message": "x"}, when=NOW
    )
    audit.record("fractional_stop_refused", {"symbol": "KO", "message": "no"}, when=NOW)
    audit.record("fractional_sale_refused", {"symbol": "KO", "message": "no"}, when=NOW)
    audit.record(
        "fractional_cycle", {"skipped": None, "gross_pct": "0.31"}, when=NOW + timedelta(days=1)
    )

    problems = " | ".join(build(audit, NOW + timedelta(days=60)).failures())

    assert "halted" in problems and "refused by Alpaca" in problems
    assert "sale(s) refused" in problems and "over 20 %" in problems


def test_an_entry_alpaca_rejected_is_no_entry_and_fails_the_trial(session_factory: Any) -> None:
    audit = _journal(session_factory)
    _fill_days(audit, days=20, entries=16)
    audit.record("fractional_entry_rejected", {"symbol": "KO"}, when=NOW)

    report = build(audit, NOW + timedelta(days=60))

    assert report.entries == 15  # sent 16 times, entered 15
    assert report.unexplained_refusals == 1
    assert "1 order(s) refused by Alpaca" in " | ".join(report.failures())


def test_the_report_counts_fills_and_how_far_they_were_from_the_limit(session_factory: Any) -> None:
    audit = _journal(session_factory)
    audit.record(
        "fractional_fill",
        {"versus_limit_pct": "-0.10", "versus_reference_pct": "0.02"},
        when=NOW,
    )
    audit.record(
        "fractional_fill",
        {"versus_limit_pct": "0.00", "versus_reference_pct": "0.10"},
        when=NOW,
    )

    text = format_report(build(audit, NOW + timedelta(days=1)))

    assert "entries filled: 2 of 0" in text and "-0.050 %" in text


def test_the_report_command_prints_the_report(
    make_settings: Any, session_factory: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch("trading_system.cli.Settings", return_value=make_settings()):
        assert main(["fractional-report"]) == 0

    assert "Fractional paper trial" in capsys.readouterr().out
