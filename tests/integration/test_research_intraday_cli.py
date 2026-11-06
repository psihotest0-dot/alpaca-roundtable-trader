"""`measure-intraday`: measures the hours signal on the observation store, counts the variant, and
says what to do when the bars are not there."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.unit.test_research_intraday import planted_sessions
from trading_system.cli import main
from trading_system.research.trials import trial_counts
from trading_system.storage.repositories import AuditRepository
from trading_system.storage.research import open_observe_store


@pytest.fixture
def observe_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'observe.db').as_posix()}"


def _run(settings: Any, *argv: str) -> int:
    with patch("trading_system.cli.Settings", return_value=settings):
        return main(["measure-intraday", *argv])


def test_the_command_measures_the_signal_and_counts_one_variant(
    observe_url: str,
    db_url: str,
    make_settings: Any,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = open_observe_store(observe_url, db_url)
    for symbol, frame in planted_sessions().items():
        store.replace(
            symbol,
            "15m",
            frame,
            adjustment="all",
            fetched_at=datetime(2026, 11, 6, tzinfo=timezone.utc),
        )
    settings = make_settings(observe_database_url=observe_url)

    assert _run(settings, "--signal", "orb_30m_long") == 0
    first = capsys.readouterr().out
    assert _run(settings, "--signal", "orb_30m_long") == 0  # the same variant again is a run
    capsys.readouterr()
    assert _run(settings, "--signal", "random_entry") == 0
    control = capsys.readouterr().out

    assert "hold-out from 2026-09-08 is not loaded" in first
    assert "orb_30m_long" in first and "meets them" in first
    assert "Variants tried so far: 1 (1 runs)." in first
    assert "Control, not counted as a variant." in control
    runs, distinct = trial_counts(AuditRepository(session_factory))
    assert (runs, distinct) == (2, 1)  # two runs of one variant; the control is not counted


def test_without_the_bars_it_says_to_fetch_them(
    observe_url: str, make_settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(observe_database_url=observe_url)

    assert _run(settings, "--signal", "orb_30m_long") == 1

    assert "fetch-observe" in capsys.readouterr().out


def test_an_unknown_signal_is_refused_and_the_trading_database_is_not_a_store(
    db_url: str, make_settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(make_settings(), "--signal", "gap_and_go") == 1
    assert "unknown signal" in capsys.readouterr().out

    assert _run(make_settings(observe_database_url=db_url), "--signal", "orb_30m_long") == 1
    assert "must not go into the trading database" in capsys.readouterr().out
