"""The research on the stored daily bars: the hold-out stays closed, the command measures and
counts what it tried (Phase 3b, step C2)."""

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from tests.fixtures.research import noise_returns, stored_bars
from trading_system.cli import main
from trading_system.research.panel import HOLDOUT_START, load_panel
from trading_system.research.trials import trial_counts
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.repositories import AuditRepository, MarketBarRepository
from trading_system.storage.research import open_research_store

NOW = datetime(2026, 11, 5, tzinfo=timezone.utc)


def _fill(store: MarketBarRepository, returns: pd.DataFrame, *, shift_hours: int = 0) -> None:
    for symbol, bars in stored_bars(returns).items():
        if shift_hours:
            bars.index = (bars.index + pd.Timedelta(hours=shift_hours)).rename("timestamp")
        store.replace(symbol, "1d", bars, adjustment="all", fetched_at=NOW)


@pytest.fixture
def research_url(tmp_path: Any) -> str:
    return f"sqlite:///{(tmp_path / 'research.db').as_posix()}"


@pytest.fixture
def filled(research_url: str, db_url: str) -> pd.DataFrame:
    """A research file with 60 stocks and SPY, 600 days that run past the hold-out."""
    returns = noise_returns(days=600, stocks=60, seed=11, end="2026-09-30")
    _fill(open_research_store(research_url, db_url), returns)
    return returns


# ---------------------------------------------------------------------------- the hold-out


def test_the_hold_out_is_never_loaded(research_url: str, db_url: str) -> None:
    returns = noise_returns(days=300, stocks=5, end="2026-09-30")  # runs well past the hold-out
    store = open_research_store(research_url, db_url)
    _fill(store, returns)

    panel = load_panel(store, list(returns.columns))

    assert panel.dates[-1].date() < HOLDOUT_START
    assert panel.dates[-1] >= pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=5)  # right up to it
    with pytest.raises(ValueError, match="hold-out"):
        load_panel(store, list(returns.columns), end=date(2026, 6, 1))


def test_bars_are_dated_by_the_new_york_day_in_winter_and_in_summer(
    research_url: str, db_url: str
) -> None:
    store = open_research_store(research_url, db_url)
    returns = noise_returns(days=3, stocks=2, end="2025-07-03")
    _fill(store, returns, shift_hours=-1)  # 04:00 UTC: how Alpaca stamps a summer day

    panel = load_panel(store, list(returns.columns))

    assert list(panel.dates) == list(returns.index)  # the same days, none slipped


# ------------------------------------------------------------------------------ the command


def _measure(settings: Any, *argv: str) -> int:
    with patch("trading_system.cli.Settings", return_value=settings):
        return main(["measure-signals", *argv])


def test_the_command_measures_prints_and_counts_the_variants(
    filled: pd.DataFrame,
    make_settings: Any,
    research_url: str,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = make_settings(research_database_url=research_url)

    assert _measure(settings, "--signal", "reversal_1w", "--horizons", "1,2") == 0

    out = capsys.readouterr().out
    assert "Development part only" in out and str(HOLDOUT_START) in out
    assert "60 stocks" in out
    assert "reversal_1w, held 1 trading day" in out and "reversal_1w, held 2 trading days" in out
    assert "Variants tried so far: 2 (2 runs)." in out
    # the hold-out stayed closed: the last day printed is before it
    last_day = out.split("60 stocks, ")[1].split(" to ")[1].split("\n")[0]
    assert date.fromisoformat(last_day) < HOLDOUT_START


def test_the_same_variant_again_is_a_run_not_a_new_variant(
    filled: pd.DataFrame,
    make_settings: Any,
    research_url: str,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = make_settings(research_database_url=research_url)
    _measure(settings, "--signal", "reversal_1w", "--horizons", "1")
    _measure(settings, "--signal", "reversal_1w", "--horizons", "1")
    _measure(settings, "--signal", "reversal_1w", "--horizons", "1", "--regime-filter", "spy200")

    assert trial_counts(AuditRepository(session_factory)) == (3, 2)  # the filter is a new variant
    assert "Variants tried so far: 2 (3 runs)." in capsys.readouterr().out


def test_a_control_is_measured_but_not_counted(
    filled: pd.DataFrame,
    make_settings: Any,
    research_url: str,
    session_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = make_settings(research_database_url=research_url)

    assert _measure(settings, "--signal", "random_tenth", "--horizons", "1") == 0

    assert "Control, not counted as a variant." in capsys.readouterr().out
    assert trial_counts(AuditRepository(session_factory)) == (0, 0)


def test_the_count_survives_in_the_trading_database_and_not_in_the_research_file(
    filled: pd.DataFrame, make_settings: Any, research_url: str, db_url: str
) -> None:
    settings = make_settings(research_database_url=research_url)
    _measure(settings, "--signal", "reversal_1w", "--horizons", "1")

    from_trading = AuditRepository(get_session_factory(get_engine(db_url)))
    assert trial_counts(from_trading) == (1, 1)
    with get_engine(research_url).connect() as connection:
        tables = connection.exec_driver_sql(
            "select name from sqlite_master where type = 'table'"
        ).fetchall()
    assert {row[0] for row in tables} == {"market_bars"}  # a cache and nothing else


def test_an_empty_research_file_says_to_fetch_first(
    make_settings: Any, research_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(research_database_url=research_url)

    assert _measure(settings, "--signal", "reversal_1w") == 1

    assert "fetch-daily" in capsys.readouterr().out


def test_too_few_stocks_is_refused_with_the_number(
    make_settings: Any, research_url: str, db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    store = open_research_store(research_url, db_url)
    _fill(store, noise_returns(days=300, stocks=10, seed=12))
    settings = make_settings(research_database_url=research_url)

    assert _measure(settings, "--signal", "reversal_1w") == 1

    assert "only 10 stocks" in capsys.readouterr().out


def test_an_unknown_signal_and_an_unknown_filter_are_errors(
    filled: pd.DataFrame, make_settings: Any, research_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(research_database_url=research_url)

    assert _measure(settings, "--signal", "astrology") == 1
    assert "unknown signal" in capsys.readouterr().out
    assert _measure(settings, "--signal", "reversal_1w", "--regime-filter", "vix") == 1
    assert "unknown regime filter" in capsys.readouterr().out


def test_the_command_refuses_to_use_the_trading_database_as_the_research_file(
    make_settings: Any, db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings(research_database_url=db_url)

    assert _measure(settings, "--signal", "reversal_1w") == 1

    assert "must not go into the trading database" in capsys.readouterr().out
