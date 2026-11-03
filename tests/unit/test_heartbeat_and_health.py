"""The loop's pulse and how it is judged from outside (Phase 6.1b)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from trading_system.cli import main
from trading_system.observability.health import evaluate_heartbeat
from trading_system.runtime.heartbeat import Heartbeat, read_heartbeat

UTC = timezone.utc
NOW = datetime(2026, 11, 3, 16, 0, 0, tzinfo=UTC)


# --- the file -------------------------------------------------------------------------------


def test_a_beat_is_written_with_its_fields_the_time_and_the_process(tmp_path: Path) -> None:
    beat = Heartbeat(tmp_path / "var" / "heartbeat.json")  # the folder does not exist yet

    beat.beat({"market_open": True, "last_entries_errors": 0}, now=NOW)

    data = read_heartbeat(tmp_path / "var" / "heartbeat.json")
    assert data is not None
    assert data["market_open"] is True and data["at"] == NOW.isoformat() and data["pid"] > 0


def test_a_new_beat_replaces_the_old_one_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    beat = Heartbeat(path)

    beat.beat({"n": 1}, now=NOW)
    beat.beat({"n": 2}, now=NOW + timedelta(seconds=60))

    data = read_heartbeat(path)
    assert data is not None and data["n"] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["heartbeat.json"]


@pytest.mark.parametrize("content", [None, "", "not json {", "[1, 2, 3]", '"text"'])
def test_a_missing_or_unreadable_file_reads_as_no_heartbeat(
    tmp_path: Path, content: str | None
) -> None:
    path = tmp_path / "heartbeat.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    assert read_heartbeat(path) is None


# --- the judgement --------------------------------------------------------------------------


def _beat(**overrides: Any) -> dict[str, Any]:
    """A loop that is fine: ticked a moment ago, said it would tick within a minute, market
    open since the morning, looked at the last bar five minutes ago."""
    beat: dict[str, Any] = {
        "at": (NOW - timedelta(seconds=10)).isoformat(),
        "next_within_seconds": 60.0,
        "market_open": True,
        "market_open_since": (NOW - timedelta(hours=2)).isoformat(),
        "started_at": (NOW - timedelta(hours=3)).isoformat(),
        "last_entries_at": (NOW - timedelta(minutes=5)).isoformat(),
        "last_entries_errors": 0,
        "clock_error": False,
    }
    beat.update(overrides)
    return beat


def test_a_loop_that_is_ticking_and_looking_at_bars_is_healthy() -> None:
    verdict = evaluate_heartbeat(_beat(), NOW)

    assert verdict.healthy and verdict.reasons == []


def test_no_heartbeat_is_unhealthy() -> None:
    verdict = evaluate_heartbeat(None, NOW)

    assert not verdict.healthy and "no heartbeat" in verdict.reasons[0]


def test_a_loop_that_stopped_ticking_is_unhealthy_once_the_announced_wait_and_the_grace_pass() -> (
    None
):
    within = evaluate_heartbeat(_beat(at=(NOW - timedelta(seconds=170)).isoformat()), NOW)
    beyond = evaluate_heartbeat(_beat(at=(NOW - timedelta(seconds=190)).isoformat()), NOW)

    assert within.healthy  # 60 s announced + 120 s of grace = 180 s
    assert not beyond.healthy and "has not ticked for 190 s" in beyond.reasons[0]


def test_a_closed_market_may_be_quiet_for_the_longer_wait_it_announced() -> None:
    closed = _beat(market_open=False, next_within_seconds=300.0)

    assert evaluate_heartbeat(
        {**closed, "at": (NOW - timedelta(seconds=400)).isoformat()}, NOW
    ).healthy
    assert not evaluate_heartbeat(
        {**closed, "at": (NOW - timedelta(seconds=430)).isoformat()}, NOW
    ).healthy


def test_an_open_market_with_no_entry_look_up_for_too_long_is_unhealthy() -> None:
    quiet = _beat(last_entries_at=(NOW - timedelta(minutes=25)).isoformat())

    verdict = evaluate_heartbeat(quiet, NOW)

    assert not verdict.healthy and "no entry look-up has completed for 25 min" in verdict.reasons[0]


def test_the_quiet_time_starts_when_the_market_opened_not_at_yesterdays_look_up() -> None:
    just_opened = _beat(
        market_open_since=(NOW - timedelta(minutes=3)).isoformat(),
        last_entries_at=(NOW - timedelta(hours=18)).isoformat(),  # yesterday's last bar
    )

    assert evaluate_heartbeat(just_opened, NOW).healthy


def test_the_quiet_time_starts_at_the_start_of_the_process_when_it_has_never_looked() -> None:
    fresh = _beat(
        last_entries_at=None,
        market_open_since=(NOW - timedelta(hours=2)).isoformat(),
        started_at=(NOW - timedelta(minutes=2)).isoformat(),
    )
    stale = _beat(
        last_entries_at=None,
        market_open_since=(NOW - timedelta(hours=2)).isoformat(),
        started_at=(NOW - timedelta(minutes=30)).isoformat(),
    )

    assert evaluate_heartbeat(fresh, NOW).healthy
    assert not evaluate_heartbeat(stale, NOW).healthy


def test_a_closed_market_is_not_judged_on_entry_look_ups() -> None:
    closed = _beat(
        market_open=False,
        last_entries_at=(NOW - timedelta(hours=18)).isoformat(),
        next_within_seconds=300.0,
    )

    assert evaluate_heartbeat(closed, NOW).healthy


def test_signals_that_broke_make_the_loop_unhealthy_until_a_clean_look_up() -> None:
    broken = evaluate_heartbeat(_beat(last_entries_errors=2), NOW)

    assert not broken.healthy and "2 signal(s) that broke" in broken.reasons[0]
    assert evaluate_heartbeat(_beat(last_entries_errors=0), NOW).healthy


def test_an_unreadable_market_clock_is_unhealthy() -> None:
    verdict = evaluate_heartbeat(_beat(market_open=None, clock_error=True), NOW)

    assert not verdict.healthy and "market clock" in verdict.reasons[0]


def test_every_reason_is_given_not_only_the_first() -> None:
    verdict = evaluate_heartbeat(
        _beat(
            at=(NOW - timedelta(minutes=10)).isoformat(),
            last_entries_errors=1,
            last_entries_at=(NOW - timedelta(minutes=40)).isoformat(),
        ),
        NOW,
    )

    assert len(verdict.reasons) == 3


def test_a_heartbeat_without_a_time_is_unhealthy() -> None:
    assert not evaluate_heartbeat({"market_open": True}, NOW).healthy


# --- the command ----------------------------------------------------------------------------


def test_the_health_command_says_healthy_for_a_fresh_beat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("HEARTBEAT_PATH", str(path))
    Heartbeat(path).beat(
        {"market_open": False, "next_within_seconds": 300.0, "last_entries_errors": 0},
        now=datetime.now(UTC),
    )

    assert main(["health"]) == 0
    assert capsys.readouterr().out.strip() == "healthy"


def test_the_health_command_fails_and_says_why_for_a_stale_beat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "heartbeat.json"
    monkeypatch.setenv("HEARTBEAT_PATH", str(path))
    Heartbeat(path).beat(
        {"market_open": False, "next_within_seconds": 60.0},
        now=datetime.now(UTC) - timedelta(hours=1),
    )

    assert main(["health"]) == 1
    assert "UNHEALTHY: the loop has not ticked" in capsys.readouterr().out


def test_the_health_command_fails_without_a_heartbeat_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "nothing.json"))

    assert main(["health"]) == 1
    assert "no heartbeat" in capsys.readouterr().out


def test_the_health_command_needs_no_alpaca_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here to supply keys
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HEARTBEAT_PATH", str(tmp_path / "nothing.json"))

    # a verdict (unhealthy: there is no heartbeat), not a crash about missing keys
    assert main(["health"]) == 1
    assert "no heartbeat" in capsys.readouterr().out
